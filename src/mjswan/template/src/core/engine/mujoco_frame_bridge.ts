
import * as THREE from 'three';
import { mjcToThreeCoordinate } from '../scene/coordinate';

export type StereoFrameBridgeMeta = {
  x: number;
  y: number;
  z: number;
  yaw: number;
  nav_state: 'idle' | 'moving' | 'arrived';
  step: number;
};

export type StereoFrameBridgeRuntimeHooks = {
  scene: THREE.Scene;
  renderer: THREE.WebGLRenderer;
  getRootObject(): THREE.Object3D | null;
  getMeta(): StereoFrameBridgeMeta | null;
  onNavGoal(x: number, y: number, z?: number): void;
  onNavCancel(): void;
  onCamLook(yaw: number, pitch: number): void;
  getCameraLook(): { yaw: number; pitch: number };
};

type BridgeCommand = {
  cmd?: string;
  type?: string;
  x?: number;
  y?: number;
  z?: number;
  yaw?: number;
  pitch?: number;
};

const WIDTH = 1280;
const HEIGHT = 720;
const FOVY = 46.8;
const HALF_BASELINE = 0.060057;
const MOUNT_POS = [0.02, 0.0, 0.57] as const;
const DEFAULT_FPS = 8;
const JPEG_QUALITY = 0.8;

function getBridgeUrl(): string | null {
  const params = new URLSearchParams(window.location.search);
  if (params.get('frameBridge') === '0') {
    return null;
  }
  const explicit = params.get('frameWsUrl') || import.meta.env.VITE_MUJOCO_FRAME_WS_URL;
  if (explicit) {
    return explicit;
  }
  const port = params.get('frameWsPort') || import.meta.env.VITE_MUJOCO_FRAME_WS_PORT || '9877';
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.hostname}:${port}`;
}

function encodeUtf8(text: string): Uint8Array {
  return new TextEncoder().encode(text);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export class MujocoFrameBridgeClient {
  private hooks: StereoFrameBridgeRuntimeHooks;
  private socket: WebSocket | null = null;
  private reconnectTimer: number | null = null;
  private captureTimer: number | null = null;
  private captureBusy = false;
  private seq = 0;
  private readonly url: string | null;
  private readonly cameraLeft: THREE.PerspectiveCamera;
  private readonly cameraRight: THREE.PerspectiveCamera;
  private readonly renderTarget: THREE.WebGLRenderTarget;
  private readonly pixels = new Uint8Array(WIDTH * HEIGHT * 4);
  private readonly rgba = new Uint8ClampedArray(WIDTH * HEIGHT * 4);
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;

  constructor(hooks: StereoFrameBridgeRuntimeHooks) {
    this.hooks = hooks;
    this.url = getBridgeUrl();
    this.cameraLeft = new THREE.PerspectiveCamera(FOVY, WIDTH / HEIGHT, 0.01, 100);
    this.cameraRight = new THREE.PerspectiveCamera(FOVY, WIDTH / HEIGHT, 0.01, 100);
    this.renderTarget = new THREE.WebGLRenderTarget(WIDTH, HEIGHT, {
      depthBuffer: true,
      stencilBuffer: false,
    });
    this.canvas = document.createElement('canvas');
    this.canvas.width = WIDTH;
    this.canvas.height = HEIGHT;
    const ctx = this.canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Unable to create stereo frame canvas context');
    }
    this.ctx = ctx;
    if (this.url) {
      this.connect();
    }
  }

  dispose(): void {
    if (this.reconnectTimer !== null) {
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.captureTimer !== null) {
      window.clearInterval(this.captureTimer);
      this.captureTimer = null;
    }
    this.socket?.close();
    this.socket = null;
    this.renderTarget.dispose();
  }

  private connect(): void {
    if (!this.url) {
      return;
    }
    try {
      this.socket = new WebSocket(this.url);
      this.socket.binaryType = 'arraybuffer';
    } catch (error) {
      console.warn('[mjswan-frame] unable to create websocket:', error);
      this.scheduleReconnect();
      return;
    }
    this.socket.onopen = () => {
      console.info(`[mjswan-frame] connected ${this.url}`);
      this.startCaptureLoop();
    };
    this.socket.onmessage = (event) => this.handleMessage(event.data);
    this.socket.onclose = () => {
      this.stopCaptureLoop();
      this.scheduleReconnect();
    };
    this.socket.onerror = () => {
      this.socket?.close();
    };
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer !== null || !this.url) {
      return;
    }
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, 1500);
  }

  private startCaptureLoop(): void {
    this.stopCaptureLoop();
    const params = new URLSearchParams(window.location.search);
    const fps = clamp(Number(params.get('frameFps') || import.meta.env.VITE_MUJOCO_FRAME_FPS || DEFAULT_FPS), 1, 30);
    this.captureTimer = window.setInterval(() => void this.captureAndSend(), 1000 / fps);
  }

  private stopCaptureLoop(): void {
    if (this.captureTimer !== null) {
      window.clearInterval(this.captureTimer);
      this.captureTimer = null;
    }
  }

  private handleMessage(data: unknown): void {
    if (typeof data !== 'string') {
      return;
    }
    let command: BridgeCommand;
    try {
      command = JSON.parse(data) as BridgeCommand;
    } catch {
      return;
    }
    const cmd = command.cmd || command.type;
    if (cmd === 'nav_goal' && typeof command.x === 'number' && typeof command.y === 'number') {
      this.hooks.onNavGoal(command.x, command.y, command.z);
    } else if (cmd === 'nav_cancel') {
      this.hooks.onNavCancel();
    } else if (cmd === 'cam_look') {
      this.hooks.onCamLook(Number(command.yaw ?? 0), Number(command.pitch ?? 0));
    }
  }

  private async captureAndSend(): Promise<void> {
    if (this.captureBusy || !this.socket || this.socket.readyState !== WebSocket.OPEN) {
      return;
    }
    const meta = this.hooks.getMeta();
    const root = this.hooks.getRootObject();
    if (!meta || !root) {
      return;
    }
    this.captureBusy = true;
    try {
      this.updateStereoCameras(root);
      const [left, right] = await Promise.all([
        this.renderCameraToJpeg(this.cameraLeft),
        this.renderCameraToJpeg(this.cameraRight),
      ]);
      if (!left || !right || this.socket.readyState !== WebSocket.OPEN) {
        return;
      }
      const seq = this.seq++;
      const header = encodeUtf8(JSON.stringify({
        type: 'stereo_frame',
        seq,
        left_size: left.byteLength,
        right_size: right.byteLength,
        meta,
      }));
      const payload = new Uint8Array(4 + header.byteLength + left.byteLength + right.byteLength);
      new DataView(payload.buffer).setUint32(0, header.byteLength, true);
      payload.set(header, 4);
      payload.set(left, 4 + header.byteLength);
      payload.set(right, 4 + header.byteLength + left.byteLength);
      this.socket.send(payload);
    } catch (error) {
      console.warn('[mjswan-frame] stereo capture failed:', error);
    } finally {
      this.captureBusy = false;
    }
  }

  private updateStereoCameras(root: THREE.Object3D): void {
    const rootPosition = new THREE.Vector3();
    const rootQuaternion = new THREE.Quaternion();
    root.getWorldPosition(rootPosition);
    root.getWorldQuaternion(rootQuaternion);

    const mount = mjcToThreeCoordinate(MOUNT_POS);
    const leftOffset = mjcToThreeCoordinate([0, HALF_BASELINE, 0]);
    const rightOffset = mjcToThreeCoordinate([0, -HALF_BASELINE, 0]);
    const look = this.hooks.getCameraLook();
    const yaw = clamp(look.yaw, -Math.PI, Math.PI);
    const pitch = clamp(-0.16 + look.pitch, -1.0, 0.8);
    const localForward = new THREE.Vector3(
      Math.cos(pitch) * Math.cos(yaw),
      Math.sin(pitch),
      -Math.cos(pitch) * Math.sin(yaw)
    ).normalize();
    const localUp = new THREE.Vector3(0, 1, 0);

    this.placeCamera(this.cameraLeft, rootPosition, rootQuaternion, mount.clone().add(leftOffset), localForward, localUp);
    this.placeCamera(this.cameraRight, rootPosition, rootQuaternion, mount.clone().add(rightOffset), localForward, localUp);
  }

  private placeCamera(
    camera: THREE.PerspectiveCamera,
    rootPosition: THREE.Vector3,
    rootQuaternion: THREE.Quaternion,
    localOffset: THREE.Vector3,
    localForward: THREE.Vector3,
    localUp: THREE.Vector3
  ): void {
    const worldPosition = rootPosition.clone().add(localOffset.applyQuaternion(rootQuaternion));
    const forward = localForward.clone().applyQuaternion(rootQuaternion);
    const up = localUp.clone().applyQuaternion(rootQuaternion);
    camera.position.copy(worldPosition);
    camera.up.copy(up);
    camera.lookAt(worldPosition.clone().add(forward));
    camera.updateMatrixWorld(true);
  }

  private async renderCameraToJpeg(camera: THREE.PerspectiveCamera): Promise<Uint8Array | null> {
    const renderer = this.hooks.renderer;
    const oldTarget = renderer.getRenderTarget();
    const oldXr = renderer.xr.enabled;
    renderer.xr.enabled = false;
    renderer.setRenderTarget(this.renderTarget);
    renderer.render(this.hooks.scene, camera);
    renderer.readRenderTargetPixels(this.renderTarget, 0, 0, WIDTH, HEIGHT, this.pixels);
    renderer.setRenderTarget(oldTarget);
    renderer.xr.enabled = oldXr;

    const rowSize = WIDTH * 4;
    for (let y = 0; y < HEIGHT; y++) {
      const src = (HEIGHT - y - 1) * rowSize;
      const dst = y * rowSize;
      this.rgba.set(this.pixels.subarray(src, src + rowSize), dst);
    }
    this.ctx.putImageData(new ImageData(this.rgba, WIDTH, HEIGHT), 0, 0);
    const blob = await new Promise<Blob | null>((resolve) => this.canvas.toBlob(resolve, 'image/jpeg', JPEG_QUALITY));
    if (!blob) {
      return null;
    }
    return new Uint8Array(await blob.arrayBuffer());
  }
}


import * as THREE from 'three';
import { mjcToThreeCoordinate } from '../scene/coordinate';

export type StereoFrameBridgeMeta = {
  x: number;
  y: number;
  z: number;
  yaw: number;
  nav_state: 'idle' | 'moving' | 'arrived';
  step: number;
  pose_source?: 'robot' | 'free_camera';
  camera_x?: number;
  camera_y?: number;
  camera_z?: number;
  camera_qx?: number;
  camera_qy?: number;
  camera_qz?: number;
  camera_qw?: number;
  camera_forward_x?: number;
  camera_forward_y?: number;
  camera_forward_z?: number;
  camera_right_x?: number;
  camera_right_y?: number;
  camera_right_z?: number;
  camera_up_x?: number;
  camera_up_y?: number;
  camera_up_z?: number;
};

export type StereoCameraPose = {
  position: THREE.Vector3;
  quaternion: THREE.Quaternion;
};

export type StereoFrameBridgeRuntimeHooks = {
  scene: THREE.Scene;
  renderer: THREE.WebGLRenderer;
  getRootObject(): THREE.Object3D | null;
  getMeta(): StereoFrameBridgeMeta | null;
  onNavGoal(x: number, y: number, z?: number): void;
  onNavCancel(): void;
  onCamLook(yaw: number, pitch: number): void;
  onCamDrive(forward: number, right: number, up: number, yaw: number, pitch: number, roll: number): void;
  onCamReset(): void;
  onCamMode(mode: 'robot' | 'free'): void;
  getCameraLook(): { yaw: number; pitch: number };
  getStereoMode(): 'robot' | 'free';
  getFreeCameraPose(): StereoCameraPose;
};

type BridgeCommand = {
  cmd?: string;
  type?: string;
  x?: number;
  y?: number;
  z?: number;
  yaw?: number;
  pitch?: number;
  roll?: number;
  forward?: number;
  right?: number;
  up?: number;
  mode?: 'robot' | 'free';
};

const WIDTH = 1280;
const HEIGHT = 720;
const FOVY = 46.8;
const HALF_BASELINE = 0.060057;
const MOUNT_POS = [0.02, 0.0, 0.57] as const;
const DEFAULT_FPS = 24;
const JPEG_QUALITY = 0.92;
const JPEG_TIMEOUT_MS = 1500;
const MAX_SOCKET_BUFFERED_BYTES = 16 * 1024 * 1024;

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

function threePointToMjc(v: THREE.Vector3): [number, number, number] {
  return [v.x, -v.z, v.y];
}

function threeDirectionToMjc(v: THREE.Vector3): [number, number, number] {
  return [v.x, -v.z, v.y];
}

function enrichMetaWithCameraPose(meta: StereoFrameBridgeMeta, camera: THREE.PerspectiveCamera, poseSource: 'robot' | 'free_camera'): StereoFrameBridgeMeta {
  camera.updateMatrixWorld(true);
  const position = new THREE.Vector3();
  camera.getWorldPosition(position);
  const quaternion = new THREE.Quaternion();
  camera.getWorldQuaternion(quaternion);
  const forward = new THREE.Vector3();
  camera.getWorldDirection(forward);
  const right = new THREE.Vector3(1, 0, 0).applyQuaternion(quaternion).normalize();
  const up = new THREE.Vector3(0, 1, 0).applyQuaternion(quaternion).normalize();
  const [cameraX, cameraY, cameraZ] = threePointToMjc(position);
  const [forwardX, forwardY, forwardZ] = threeDirectionToMjc(forward);
  const [rightX, rightY, rightZ] = threeDirectionToMjc(right);
  const [upX, upY, upZ] = threeDirectionToMjc(up);
  return {
    ...meta,
    pose_source: poseSource,
    camera_x: cameraX,
    camera_y: cameraY,
    camera_z: cameraZ,
    camera_qx: quaternion.x,
    camera_qy: quaternion.y,
    camera_qz: quaternion.z,
    camera_qw: quaternion.w,
    camera_forward_x: forwardX,
    camera_forward_y: forwardY,
    camera_forward_z: forwardZ,
    camera_right_x: rightX,
    camera_right_y: rightY,
    camera_right_z: rightZ,
    camera_up_x: upX,
    camera_up_y: upY,
    camera_up_z: upZ,
  };
}

export class MujocoFrameBridgeClient {
  private hooks: StereoFrameBridgeRuntimeHooks;
  private socket: WebSocket | null = null;
  private reconnectTimer: number | null = null;
  private captureTimer: number | null = null;
  private captureBusy = false;
  private seq = 0;
  private lastHealthLogAt = 0;
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
    } else if (cmd === 'cam_drive') {
      this.hooks.onCamDrive(
        Number(command.forward ?? 0),
        Number(command.right ?? 0),
        Number(command.up ?? 0),
        Number(command.yaw ?? 0),
        Number(command.pitch ?? 0),
        Number(command.roll ?? 0)
      );
    } else if (cmd === 'cam_reset') {
      this.hooks.onCamReset();
    } else if (cmd === 'cam_mode') {
      this.hooks.onCamMode(command.mode === 'robot' ? 'robot' : 'free');
    }
  }

  private async captureAndSend(): Promise<void> {
    if (this.captureBusy || !this.socket || this.socket.readyState !== WebSocket.OPEN) {
      return;
    }
    if (this.socket.bufferedAmount > MAX_SOCKET_BUFFERED_BYTES) {
      const bufferedMb = (this.socket.bufferedAmount / (1024 * 1024)).toFixed(1);
      console.warn(`[mjswan-frame] skipping stereo capture, websocket buffered=${bufferedMb} MB`);
      return;
    }
    const meta = this.hooks.getMeta();
    const root = this.hooks.getRootObject();
    if (!meta || (this.hooks.getStereoMode() === 'robot' && !root)) {
      return;
    }
    this.captureBusy = true;
    try {
      const poseSource = this.updateStereoCameras(root);
      const enrichedMeta = enrichMetaWithCameraPose(meta, this.cameraLeft, poseSource);
      const left = await this.renderCameraToJpeg(this.cameraLeft);
      const right = await this.renderCameraToJpeg(this.cameraRight);
      if (!left || !right || this.socket.readyState !== WebSocket.OPEN) {
        return;
      }
      const seq = this.seq++;
      const header = encodeUtf8(JSON.stringify({
        type: 'stereo_frame',
        seq,
        left_size: left.byteLength,
        right_size: right.byteLength,
        meta: enrichedMeta,
      }));
      const payload = new Uint8Array(4 + header.byteLength + left.byteLength + right.byteLength);
      new DataView(payload.buffer).setUint32(0, header.byteLength, true);
      payload.set(header, 4);
      payload.set(left, 4 + header.byteLength);
      payload.set(right, 4 + header.byteLength + left.byteLength);
      this.socket.send(payload);
      const now = performance.now();
      if (now - this.lastHealthLogAt >= 5000) {
        this.lastHealthLogAt = now;
        const leftKb = (left.byteLength / 1024).toFixed(0);
        const rightKb = (right.byteLength / 1024).toFixed(0);
        const bufferedKb = (this.socket.bufferedAmount / 1024).toFixed(0);
        console.info(`[mjswan-frame] sent stereo seq=${seq} source=${poseSource} left=${leftKb}KB right=${rightKb}KB buffered=${bufferedKb}KB`);
      }
    } catch (error) {
      console.warn('[mjswan-frame] stereo capture failed:', error);
    } finally {
      this.captureBusy = false;
    }
  }

  private updateStereoCameras(root: THREE.Object3D | null): 'robot' | 'free_camera' {
    const leftOffset = mjcToThreeCoordinate([0, HALF_BASELINE, 0]);
    const rightOffset = mjcToThreeCoordinate([0, -HALF_BASELINE, 0]);

    if (this.hooks.getStereoMode() === 'free') {
      const pose = this.hooks.getFreeCameraPose();
      // Baseline is along camera's local X axis (right direction)
      const cameraRight = new THREE.Vector3(1, 0, 0).applyQuaternion(pose.quaternion);
      this.cameraLeft.position.copy(pose.position.clone().addScaledVector(cameraRight, -HALF_BASELINE));
      this.cameraRight.position.copy(pose.position.clone().addScaledVector(cameraRight, HALF_BASELINE));
      this.cameraLeft.quaternion.copy(pose.quaternion);
      this.cameraRight.quaternion.copy(pose.quaternion);
      this.cameraLeft.updateMatrixWorld(true);
      this.cameraRight.updateMatrixWorld(true);
      return 'free_camera';
    }

    if (!root) {
      return 'robot';
    }
    const rootPosition = new THREE.Vector3();
    const rootQuaternion = new THREE.Quaternion();
    root.getWorldPosition(rootPosition);
    root.getWorldQuaternion(rootQuaternion);

    const mount = mjcToThreeCoordinate(MOUNT_POS);
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
    return 'robot';
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
    const hiddenObjects: Array<{ object: THREE.Object3D; visible: boolean }> = [];
    this.hooks.scene.traverse((object) => {
      if (object.userData.hideFromStereoCapture) {
        hiddenObjects.push({ object, visible: object.visible });
        object.visible = false;
      }
    });
    renderer.xr.enabled = false;
    try {
      renderer.setRenderTarget(this.renderTarget);
      renderer.render(this.hooks.scene, camera);
      renderer.readRenderTargetPixels(this.renderTarget, 0, 0, WIDTH, HEIGHT, this.pixels);
    } finally {
      renderer.setRenderTarget(oldTarget);
      renderer.xr.enabled = oldXr;
      for (const { object, visible } of hiddenObjects) {
        object.visible = visible;
      }
    }

    const rowSize = WIDTH * 4;
    for (let y = 0; y < HEIGHT; y++) {
      const src = (HEIGHT - y - 1) * rowSize;
      const dst = y * rowSize;
      this.rgba.set(this.pixels.subarray(src, src + rowSize), dst);
    }
    this.ctx.putImageData(new ImageData(this.rgba, WIDTH, HEIGHT), 0, 0);
    const blob = await new Promise<Blob | null>((resolve) => {
      let settled = false;
      const timer = window.setTimeout(() => {
        if (settled) return;
        settled = true;
        console.warn('[mjswan-frame] JPEG encode timed out');
        resolve(null);
      }, JPEG_TIMEOUT_MS);
      this.canvas.toBlob((value) => {
        if (settled) return;
        settled = true;
        window.clearTimeout(timer);
        resolve(value);
      }, 'image/jpeg', JPEG_QUALITY);
    });
    if (!blob) {
      return null;
    }
    return new Uint8Array(await blob.arrayBuffer());
  }
}

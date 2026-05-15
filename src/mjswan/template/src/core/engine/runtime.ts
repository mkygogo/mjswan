import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { VRButton } from 'three/addons/webxr/VRButton.js';
import type { MainModule, MjData, MjModel } from 'mujoco';
import {
  downloadExampleScenesFolder,
  getPosition,
  getQuaternion,
  loadSceneFromURL,
} from '../scene/scene';
import { type SplatConfig, type SplatMesh, loadSplat, disposeSplat, applySplatTransform } from '../scene/splat';
import { loadCollider, disposeCollider } from '../scene/collider';
import { DragStateManager } from '../utils/dragStateManager';
import { createTendonState, updateTendonGeometry, updateTendonRendering } from '../scene/tendons';
import { updateHeadlightFromCamera, updateLightsFromData } from '../scene/lights';
import { mjcToThreeCoordinate, threeToMjcCoordinate } from '../scene/coordinate';
import {
  type ViewerConfig,
  type ViewerState,
  applyViewerConfig,
  updateCameraFromData,
} from './viewer_config';
import { SceneCacheManager } from '../cache/sceneCacheManager';
import { SceneResourceTracker } from '../cache/resourceTracker';
import { MemoryMonitor } from '../cache/memoryMonitor';
import { Observations } from '../observation/observations';
import { TerminationManager } from '../termination/TerminationManager';
import { Terminations } from '../termination/terminations';
import * as ort from 'onnxruntime-web';
import { PolicyRunner } from '../policy/PolicyRunner';
import { OnnxModule } from '../policy/OnnxModule';
import { PolicyStateBuilder } from '../policy/PolicyStateBuilder';
import type { PolicyConfig } from '../policy/types';
import { TrackingPolicy } from '../policy/modules/TrackingPolicy';
import { LocomotionPolicy } from '../policy/modules/LocomotionPolicy';
import { getCommandManager, type CommandTermContext, type CommandsConfig } from '../command';
import { EventManager } from '../event/EventManager';
import { Events } from '../event/events';
import type { TerrainData } from '../event/EventBase';
import { MujocoFrameBridgeClient, type StereoFrameBridgeMeta } from './mujoco_frame_bridge';

type RuntimeOptions = {
  baseUrl?: string;
};

type MotionCommandTerm = {
  setSelectedMotion(name: string | null): Promise<boolean> | boolean;
  setReferenceVisible?(visible: boolean): void;
  getSelectedMotionName?(): string | null;
};

function isMotionCommandTerm(term: unknown): term is MotionCommandTerm {
  return (
    typeof term === 'object' &&
    term !== null &&
    typeof (term as MotionCommandTerm).setSelectedMotion === 'function'
  );
}

/** Thrown when a scene exceeds the browser's 2 GB WebAssembly memory limit. */
export class WasmMemoryLimitError extends Error {
  constructor() {
    super(
      "This scene cannot be loaded because it exceeds the browser's " +
        'WebAssembly 2 GB memory limit. ' +
        'Try closing other browser tabs or reloading the page to free memory.'
    );
    this.name = 'WasmMemoryLimitError';
  }
}

// mj_loadXML surfaces OOM via four distinct paths depending on which internal
// allocator hits the 2 GB ceiling first (null return, MuJoCo error string,
// lodepng error string, or raw bad_alloc).
function isWasmOom(error: unknown): boolean {
  const msg = error instanceof Error ? error.message : String(error);
  return (
    msg.includes('MjModel loading returned null') ||
    msg.includes('Could not allocate memory') ||
    msg.includes('memory allocation failed') ||
    msg.includes('bad_alloc')
  );
}

type BodyState = {
  position: THREE.Vector3;
  quaternion: THREE.Quaternion;
};

type ClickNavigationTarget = {
  mjcX: number;
  mjcY: number;
  threePosition: THREE.Vector3;
};

type StereoCameraVisualization = {
  group: THREE.Group;
  mount: THREE.Mesh;
  left: THREE.Mesh;
  right: THREE.Mesh;
};

export class mjswanRuntime {
  private mujoco: MainModule;
  private container: HTMLElement;
  private baseUrl: string;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private controls: OrbitControls;
  private mjModel: MjModel | null;
  private mjData: MjData | null;
  private bodies: Record<number, THREE.Group> | null;
  private lights: THREE.Light[];
  private mujocoRoot: THREE.Group | null;
  private lastSimState: {
    bodies: Map<number, BodyState>;
    tendons: ReturnType<typeof createTendonState>;
  };
  private dynamicBodyIds: Set<number> | null;
  private loopPromise: Promise<void> | null;
  private running: boolean;
  private timestep: number;
  private decimation: number;
  private loadingScene: Promise<void> | null;
  private resizeObserver: ResizeObserver | null;
  private dragStateManager: DragStateManager | null;
  private dragForceScale: number;
  private sceneCacheManager: SceneCacheManager;
  private resourceTracker: SceneResourceTracker;
  private memoryMonitor: MemoryMonitor;
  private policyRunner: PolicyRunner | null;
  private policyStateBuilder: PolicyStateBuilder | null;
  private policyConfigPath: string | null;
  private policyDebugCounter: number;
  private policyControl: Array<{
    controlType: string;
    ctrlAdr: number[];
    qposAdr: number[];
    qvelAdr: number[];
    // Indices into the flat policy action vector for this term's joints.
    actionIndices: number[];
    actionScale: Float32Array;
    actionOffset: Float32Array;
    defaultJointPos: Float32Array;
    encoderBias: Float32Array;
    // Per-actuator flag: true = position actuator (ctrl=target_pos, PD internal),
    // false = motor actuator (ctrl=torque, PD computed in browser from kp/kd).
    positionActuator: boolean[];
    kp: Float32Array;
    kd: Float32Array;
  }> | null;
  private onnxModule: OnnxModule | null;
  private onnxInputDict: Record<string, ort.Tensor> | null;
  private onnxInferencing: boolean;
  private onnxTimeStep: number;
  private terminationManager: TerminationManager | null;
  private eventManager: EventManager | null;
  private terrainData: TerrainData | null;
  private vrButton: HTMLElement | null;
  private splatMesh: SplatMesh | null;
  private colliderMesh: THREE.Group | null;
  private cameraState: ViewerState;
  private clickNavTarget: ClickNavigationTarget | null;
  private clickNavMarker: THREE.Group | null;
  private clickNavRaycaster: THREE.Raycaster;
  private clickNavPointerDown: { x: number; y: number; time: number; button: number } | null;
  private frameBridge: MujocoFrameBridgeClient | null;
  private stereoLookYaw: number;
  private stereoLookPitch: number;
  private navState: 'idle' | 'moving' | 'arrived';
  private navArrivedUntil: number;
  private simStepCount: number;
  private stereoCameraViz: StereoCameraVisualization | null;

  constructor(mujoco: MainModule, container: HTMLElement, options: RuntimeOptions = {}) {
    this.mujoco = mujoco;
    this.container = container;
    this.baseUrl = options.baseUrl || '/';

    const workingPath = '/working';
    try {
      this.mujoco.FS.mkdir(workingPath);
    } catch (error: unknown) {
      if (error && typeof error === 'object' && 'code' in error && error.code !== 'EEXIST') {
        console.warn('Failed to create /working directory:', error);
      }
    }
    try {
      this.mujoco.FS.mount(this.mujoco.MEMFS, { root: '.' }, workingPath);
    } catch (error: unknown) {
      if (error && typeof error === 'object' && 'code' in error && error.code !== 'EEXIST' && error.code !== 'EBUSY') {
        console.warn('Failed to mount MEMFS at /working:', error);
      }
    }

    const { width, height } = this.getSize();

    this.scene = new THREE.Scene();
    this.scene.name = 'scene';

    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.001, 1000);
    this.camera.name = 'PerspectiveCamera';
    this.camera.position.set(2.0, 1.7, 1.7);
    this.scene.add(this.camera);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.xr.enabled = true;
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(width, height);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.container.appendChild(this.renderer.domElement);

    this.vrButton = null;
    navigator.xr?.isSessionSupported('immersive-vr').then((supported) => {
      if (supported) {
        this.vrButton = VRButton.createButton(this.renderer);
        document.body.appendChild(this.vrButton);
      }
    });

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0.2, 0);
    this.controls.panSpeed = 2;
    this.controls.zoomSpeed = 1;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.1;
    this.controls.screenSpacePanning = true;
    this.controls.update();

    this.renderer.setAnimationLoop(this.render);
    window.addEventListener('resize', this.onWindowResize);

    if ('ResizeObserver' in window) {
      this.resizeObserver = new ResizeObserver(() => this.onWindowResize());
      this.resizeObserver.observe(this.container);
    } else {
      this.resizeObserver = null;
    }

    this.lastSimState = {
      bodies: new Map(),
      tendons: createTendonState(),
    };
    this.dynamicBodyIds = null;

    this.mjModel = null;
    this.mjData = null;
    this.bodies = null;
    this.lights = [];
    this.mujocoRoot = null;
    this.loopPromise = null;
    this.running = false;
    this.timestep = 0.001;
    this.decimation = 1;
    this.loadingScene = null;
    this.dragStateManager = null;
    this.dragForceScale = 100.0;
    this.policyRunner = null;
    this.policyStateBuilder = null;
    this.policyConfigPath = null;
    this.policyDebugCounter = 0;
    this.policyControl = null;
    this.onnxModule = null;
    this.onnxInputDict = null;
    this.onnxInferencing = false;
    this.onnxTimeStep = 0;
    this.terminationManager = null;
    this.eventManager = null;
    this.terrainData = null;
    this.splatMesh = null;
    this.colliderMesh = null;
    this.cameraState = { trackBodyId: null, prevBodyPos: null };
    this.clickNavTarget = null;
    this.clickNavMarker = null;
    this.clickNavRaycaster = new THREE.Raycaster();
    this.clickNavPointerDown = null;
    this.stereoLookYaw = 0;
    this.stereoLookPitch = 0;
    this.navState = 'idle';
    this.navArrivedUntil = 0;
    this.simStepCount = 0;
    this.stereoCameraViz = null;

    // Initialize cache system (singleton shared across runtime instances)
    this.sceneCacheManager = SceneCacheManager.getInstance(this.mujoco);
    this.resourceTracker = new SceneResourceTracker();
    this.memoryMonitor = new MemoryMonitor();

    this.container.addEventListener('pointerdown', this.onClickNavPointerDown, true);
    this.container.addEventListener('pointerup', this.onClickNavPointerUp, true);

    this.frameBridge = new MujocoFrameBridgeClient({
      scene: this.scene,
      renderer: this.renderer,
      getRootObject: () => this.getStereoRootObject(),
      getMeta: () => this.getFrameBridgeMeta(),
      onNavGoal: (x, y, z) => this.setNavigationTargetMjc(x, y, z),
      onNavCancel: () => this.cancelNavigation(),
      onCamLook: (yaw, pitch) => this.setStereoCameraLook(yaw, pitch),
      getCameraLook: () => ({ yaw: this.stereoLookYaw, pitch: this.stereoLookPitch }),
    });
  }

  async loadEnvironment(
    scenePath: string,
    policyConfigPath: string | null = null,
    splatConfig: SplatConfig | null = null,
    cameraConfig: ViewerConfig | null = null,
    eventsConfig: import('../event/EventBase').EventConfig[] | null = null,
    terrainData: TerrainData | null = null
  ): Promise<void> {
    this.terrainData = terrainData;
    if (eventsConfig && eventsConfig.length > 0) {
      this.eventManager = new EventManager(eventsConfig, Events);
      console.log(`[EventManager] ${this.eventManager.size} reset event(s) loaded`);
    } else {
      this.eventManager = null;
    }
    await this.stop();

    // Dispose previous splat/collider before switching scenes
    if (this.splatMesh) {
      disposeSplat(this.splatMesh, this.scene);
      this.splatMesh = null;
    }
    if (this.colliderMesh) {
      disposeCollider(this.colliderMesh, this.scene);
      this.colliderMesh = null;
    }

    const startTime = performance.now();

    // Initialize CommandManager with default velocity commands
    this.initializeCommands();

    // Check cache first
    if (this.sceneCacheManager.has(scenePath)) {
      await this.restoreFromCache(scenePath);
      const elapsed = performance.now() - startTime;
      this.memoryMonitor.logCacheOperation('hit', scenePath, { elapsedMs: elapsed });
    } else {
      this.memoryMonitor.logCacheOperation('miss', scenePath);

      // Prepare cache for new scene (may trigger eviction)
      await this.sceneCacheManager.prepareForNewScene();

      // Clear current references before loading new scene
      // This prevents loadSceneFromURL from deleting cached objects
      this.mjModel = null;
      this.mjData = null;
      this.bodies = null;
      this.lights = [];
      this.mujocoRoot = null;
      this.dynamicBodyIds = null;

      // Start tracking resources
      this.resourceTracker.startTracking(this.mujoco);

      // Normal load
      await downloadExampleScenesFolder(this.mujoco, scenePath, this.baseUrl);
      await this.loadSceneWithOomRetry(scenePath);

      // Capture and cache resources
      await this.captureAndCacheResources(scenePath);
    }

    // Load splat and optional collider
    if (splatConfig) {
      this.splatMesh = loadSplat(splatConfig, this.scene);
      if (splatConfig.colliderUrl) {
        this.colliderMesh = await loadCollider(
          this.resolveAssetUrl(splatConfig.colliderUrl),
          this.scene
        );
      }
    }

    await this.loadPolicyConfig(policyConfigPath);

    this.applyViewerConfig(cameraConfig);

    this.running = true;
    void this.startLoop();
  }

  /**
   * Initialize the CommandManager (clear and set up reset callback)
   * Commands are registered from policy config in loadPolicyConfig()
   */
  private initializeCommands(): void {
    const commandManager = getCommandManager();
    commandManager.clear();
    commandManager.setResetCallback(() => this.resetSimulation());
  }

  /**
   * Initialize commands from policy config
   */
  private initializeCommandsFromConfig(
    commands: CommandsConfig,
    context: CommandTermContext
  ): void {
    const commandManager = getCommandManager();
    commandManager.initialize(commands, context);
    console.log('[mjswanRuntime] Commands loaded from policy config:', Object.keys(commands));
  }

  /**
   * Public method to reset the simulation state
   * Can be called from UI components via the CommandManager
   */
  resetSimulation(): void {
    this.resetSimulationState();
    if (this.policyRunner && this.policyStateBuilder) {
      const state = this.policyStateBuilder.build();
      this.policyRunner.reset(state);
    }
    console.log('[mjswanRuntime] Simulation reset');
  }

  async setSelectedMotion(motionName: string | null): Promise<boolean> {
    const term = getCommandManager().getTerm('motion');
    if (!isMotionCommandTerm(term)) {
      return false;
    }
    const accepted = await term.setSelectedMotion(motionName);
    if (accepted) {
      this.resetSimulation();
    }
    return accepted;
  }

  getSelectedMotionName(): string | null {
    const term = getCommandManager().getTerm('motion');
    if (!isMotionCommandTerm(term)) {
      return null;
    }
    return term.getSelectedMotionName?.() ?? null;
  }

  setReferenceVisible(visible: boolean): void {
    const term = getCommandManager().getTerm('motion');
    if (!isMotionCommandTerm(term) || typeof term.setReferenceVisible !== 'function') {
      return;
    }
    term.setReferenceVisible(visible);
  }

  async loadScene(scenePath: string): Promise<void> {
    if (this.loadingScene) {
      await this.loadingScene;
    }

    this.loadingScene = (async () => {
      const existingRoot = this.scene.getObjectByName('MuJoCo Root');
      if (existingRoot) {
        this.scene.remove(existingRoot);
      }

      const parent = {
        mjModel: this.mjModel,
        mjData: this.mjData,
        scene: this.scene,
      };

      [this.mjModel, this.mjData, this.bodies, this.lights] = await loadSceneFromURL(
        this.mujoco,
        scenePath,
        parent
      );

      if (!this.mjModel || !this.mjData) {
        throw new Error('Failed to load MuJoCo model.');
      }

      this.mujocoRoot = this.scene.getObjectByName('MuJoCo Root') as THREE.Group | null;

      this.mujoco.mj_forward(this.mjModel, this.mjData);
      updateLightsFromData(this.mujoco, this.mjData, this.lights);
      updateHeadlightFromCamera(this.camera, this.lights);
      this.dynamicBodyIds = this.computeDynamicBodyIds(this.mjModel);
      this.syncStaticBodiesFromData();

      this.timestep = this.mjModel.opt.timestep || 0.001;
      this.decimation = Math.max(1, Math.round(0.02 / this.timestep));

      this.lastSimState.bodies.clear();
      this.updateCachedState();

      // Initialize DragStateManager
      if (!this.dragStateManager) {
        this.dragStateManager = new DragStateManager({
          scene: this.scene,
          renderer: this.renderer,
          camera: this.camera,
          container: this.container,
          controls: this.controls,
          draggableBodyIds: this.dynamicBodyIds,
        });
      } else {
        this.dragStateManager.setDraggableBodyIds(this.dynamicBodyIds);
      }

      this.loadingScene = null;
    })();

    await this.loadingScene;
  }

  // On OOM, evict all cached scenes to reclaim the WASM malloc free-list and
  // retry once.  If the retry also fails, throw WasmMemoryLimitError.
  private async loadSceneWithOomRetry(scenePath: string): Promise<void> {
    try {
      await this.loadScene(scenePath);
    } catch (error) {
      if (!isWasmOom(error)) {
        throw error;
      }
      console.warn('[mjswanRuntime] OOM — clearing cache and retrying...');
      this.loadingScene = null;
      await this.sceneCacheManager.clear();
      try {
        await this.loadScene(scenePath);
      } catch (retryError) {
        this.loadingScene = null;
        if (isWasmOom(retryError)) {
          throw new WasmMemoryLimitError();
        }
        throw retryError;
      }
    }
  }

  async startLoop(): Promise<void> {
    if (this.loopPromise) {
      return this.loopPromise;
    }
    this.running = true;
    this.loopPromise = this.mainLoop();
    return this.loopPromise;
  }

  async setSplat(config: SplatConfig | null): Promise<void> {
    if (this.splatMesh) {
      disposeSplat(this.splatMesh, this.scene);
      this.splatMesh = null;
    }
    if (this.colliderMesh) {
      disposeCollider(this.colliderMesh, this.scene);
      this.colliderMesh = null;
    }
    if (config) {
      this.splatMesh = loadSplat(config, this.scene);
      if (config.colliderUrl) {
        this.colliderMesh = await loadCollider(
          this.resolveAssetUrl(config.colliderUrl),
          this.scene
        );
      }
    }
  }

  /** Update transform of the existing splat without disposing/reloading. */
  calibrateSplat(config: SplatConfig): void {
    if (this.splatMesh) {
      applySplatTransform(this.splatMesh, config);
    }
  }

  setSplatVisible(visible: boolean): void {
    if (this.splatMesh) {
      this.splatMesh.visible = visible;
    }
  }

  async stop(): Promise<void> {
    this.running = false;
    const pending = this.loopPromise;
    if (pending) {
      await pending;
    }
    this.loopPromise = null;
  }

  private async mainLoop(): Promise<void> {
    while (this.running) {
      const loopStart = performance.now();
      const target = this.timestep * this.decimation;

      if (this.mjModel && this.mjData) {
        this.mujoco.mj_forward(this.mjModel, this.mjData);
        if (this.policyRunner && this.policyStateBuilder) {
          const state = this.policyStateBuilder.build();
          const obs = this.policyRunner.collectObservationsByKey(state);
          await this.runOnnxInference(obs);
          if (this.policyDebugCounter % 60 === 0) {
            const debugKey =
              'policy' in obs
                ? 'policy'
                : 'observation' in obs
                  ? 'observation'
                  : Object.keys(obs)[0];
            const debugObs = debugKey ? obs[debugKey] : null;
            const preview = debugObs ? Array.from(debugObs.slice(0, 8)) : [];
            console.log('[PolicyRunner] obs', {
              key: debugKey,
              size: debugObs ? debugObs.length : 0,
              sample: preview,
            });
          }
          this.policyDebugCounter += 1;
        }
        this.executeSimulationSteps();
        this.updateCachedState();

        // Evaluate termination conditions after simulation step
        if (this.terminationManager && this.policyStateBuilder) {
          const postState = this.policyStateBuilder.build();
          const result = this.terminationManager.evaluate(postState);
          if (result.done) {
            this.resetSimulationState();
            this.terminationManager.reset();
            if (this.policyRunner) {
              const resetState = this.policyStateBuilder.build();
              this.policyRunner.reset(resetState);
            }
          }
        }

        // Commands are updated after the physics step to match mjlab training-time semantics:
        // the policy sees command values from the previous step, consistent with how mjlab
        // computes observations before stepping the environment.
        this.updateClickNavigationCommand();
        getCommandManager().update(target);
        getCommandManager().updateDebugVisuals();
      }

      const elapsed = (performance.now() - loopStart) / 1000;
      const sleepTime = Math.max(0, target - elapsed);
      if (sleepTime > 0) {
        await new Promise((resolve) => setTimeout(resolve, sleepTime * 1000));
      }
    }
    this.loopPromise = null;
  }

  private async loadPolicyConfig(policyConfigPath: string | null): Promise<void> {
    const previousPolicyConfigPath = this.policyConfigPath;
    this.policyConfigPath = policyConfigPath;
    this.policyRunner = null;
    this.policyStateBuilder = null;
    this.policyDebugCounter = 0;
    this.policyControl = null;
    this.onnxModule = null;
    this.onnxInputDict = null;
    this.onnxInferencing = false;
    this.onnxTimeStep = 0;
    this.terminationManager = null;
    // Note: eventManager and terrainData are scene-level state set in loadEnvironment; do not clear here.

    // Clear existing commands when switching policies
    const commandManager = getCommandManager();
    commandManager.clear();
    commandManager.setResetCallback(() => this.resetSimulation());

    if (!policyConfigPath) {
      return;
    }

    if (!this.mjModel || !this.mjData) {
      console.warn('Policy config loaded before MuJoCo model is ready.');
      return;
    }

    if (policyConfigPath !== previousPolicyConfigPath) {
      this.resetSimulationState();
    }

    try {
      const { config } = await this.fetchPolicyConfig(policyConfigPath);
      if (Array.isArray(config.motions)) {
        config.motions = config.motions.map((motion) => ({
          ...motion,
          path: this.resolveAssetUrl(
            this.resolvePolicyAssetPath(policyConfigPath, motion.path)
          ),
        }));
      }
      if (config.commands?.motion && Array.isArray(config.motions)) {
        config.commands.motion = {
          ...config.commands.motion,
          motions: config.motions,
        };
      }
      this.resetSimulationState();
      this.mujoco.mj_forward(this.mjModel, this.mjData);
      this.updateCachedState();

      // Initialize commands from policy config if present
      if (config.commands && typeof config.commands === 'object') {
        this.initializeCommandsFromConfig(config.commands as CommandsConfig, {
          mujoco: this.mujoco,
          mjModel: this.mjModel,
          mjData: this.mjData,
          scene: this.scene,
          bodies: this.bodies,
          mujocoRoot: this.mujocoRoot,
          requestReset: () => this.resetSimulation(),
        });
        getCommandManager().resetTerms();
        const motionTerm = getCommandManager().getTerm('motion');
        if (isMotionCommandTerm(motionTerm)) {
          await motionTerm.setSelectedMotion(
            config.motions?.find((motion) => motion.default)?.name
              ?? config.motions?.[0]?.name
              ?? null
          );
          motionTerm.setReferenceVisible?.(true);
        }
        this.mujoco.mj_forward(this.mjModel, this.mjData);
        this.updateCachedState();
      }

      if (!config.policy_joint_names || config.policy_joint_names.length === 0) {
        throw new Error('Policy config missing policy_joint_names.');
      }

      const runner = new PolicyRunner(config, {
        policyModules: {
          tracking: TrackingPolicy,
          locomotion: LocomotionPolicy,
        },
        observations: Observations,
      });

      await runner.init({
        mujoco: this.mujoco,
        mjModel: this.mjModel,
        mjData: this.mjData,
        scene: this.scene,
      });

      this.policyRunner = runner;
      this.policyStateBuilder = new PolicyStateBuilder(
        this.mujoco,
        this.mjModel,
        this.mjData,
        runner.getPolicyJointNames()
      );

      const state = this.policyStateBuilder.build();
      this.policyRunner.reset(state);
      this.policyControl = this.buildPolicyControl(config, runner, this.policyStateBuilder);

      // Initialize termination manager if termination config is present
      if (config.terminations && Object.keys(config.terminations).length > 0) {
        this.terminationManager = new TerminationManager(config.terminations, Terminations);
        console.log(`[TerminationManager] ${this.terminationManager.size} termination term(s) loaded`);
      }

      if (config.onnx?.path) {
        const onnxPath = this.resolvePolicyAssetPath(policyConfigPath, config.onnx.path);
        const onnxUrl = this.resolveAssetUrl(onnxPath);
        const onnxConfig = { ...config.onnx, path: onnxUrl };
        const module = new OnnxModule(onnxConfig);
        await module.init();
        this.onnxModule = module;
        this.onnxInputDict = module.initInput();
      }

      console.log('[PolicyRunner] config loaded', {
        obsSize: runner.getObservationSize(),
        obsLayout: runner.getObservationLayout(),
        pdEnabled: this.policyControl !== null,
      });
    } catch (error) {
      console.warn('Failed to load policy config:', error);
    }
  }

  private async fetchPolicyConfig(
    policyConfigPath: string
  ): Promise<{ config: PolicyConfig; resolvedUrl: string }> {
    const resolved = this.resolveAssetUrl(policyConfigPath);
    const response = await fetch(resolved, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`Failed to fetch policy config: ${response.status}`);
    }
    const payload = await response.json();
    return { config: payload as PolicyConfig, resolvedUrl: resolved };
  }

  private resolveAssetUrl(assetPath: string): string {
    if (/^[a-z]+:\/\//i.test(assetPath)) {
      return assetPath;
    }
    const base = (this.baseUrl || '/').replace(/\/+$/, '/');
    const baseUrl = new URL(base, window.location.origin + '/').toString();
    return new URL(assetPath.replace(/^\/+/, ''), baseUrl).toString();
  }

  private resolvePolicyAssetPath(configPath: string, assetPath: string): string {
    const normalizedConfig = configPath.replace(/\\/g, '/');
    const lastSlash = normalizedConfig.lastIndexOf('/');
    if (lastSlash >= 0) {
      const dir = normalizedConfig.slice(0, lastSlash + 1);
      return `${dir}${assetPath}`.replace(/\/+/g, '/');
    }
    return assetPath;
  }

  private buildPolicyControl(
    config: PolicyConfig,
    runner: PolicyRunner,
    stateBuilder: PolicyStateBuilder
  ): Array<{
    controlType: string;
    ctrlAdr: number[];
    qposAdr: number[];
    qvelAdr: number[];
    actionIndices: number[];
    actionScale: Float32Array;
    actionOffset: Float32Array;
    defaultJointPos: Float32Array;
    encoderBias: Float32Array;
    positionActuator: boolean[];
    kp: Float32Array;
    kd: Float32Array;
  }> | null {
    const jointNames = runner.getPolicyJointNames();
    const affineBiasValue = this.mujoco.mjtBias?.mjBIAS_AFFINE?.value ?? 1;

    const buildEntry = (
      termKey: string,
      controlType: string,
      mapping: { ctrlAdr: number[]; qposAdr: number[]; qvelAdr: number[]; actionIndices: number[] },
      configScale: number[] | number | Record<string, number> | undefined,
      configOffset: number[] | number | Record<string, number> | undefined,
      configStiffness: number[] | number | Record<string, number> | undefined,
      configDamping: number[] | number | Record<string, number> | undefined,
      useDefaultOffset: boolean
    ) => {
      const n = mapping.qposAdr.length;
      const subsetJointNames = mapping.actionIndices.map((i) => jointNames[i]);

      const actionScale = this.normalizeControlArray(configScale, n, 1.0, subsetJointNames);
      const actionOffset = this.normalizeControlArray(configOffset, n, 0.0, subsetJointNames);

      const allDefaultJointPos = useDefaultOffset
        ? runner.getDefaultJointPos()
        : new Float32Array(jointNames.length);
      const defaultJointPos = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        defaultJointPos[i] = allDefaultJointPos[mapping.actionIndices[i]];
      }

      const allEncoderBias = Array.isArray(config.encoder_bias)
        ? Float32Array.from(config.encoder_bias)
        : new Float32Array(jointNames.length);
      const encoderBias = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        encoderBias[i] = allEncoderBias[mapping.actionIndices[i]] ?? 0.0;
      }

      const kp = this.normalizeControlArray(configStiffness, n, 0.0, subsetJointNames);
      const kd = this.normalizeControlArray(configDamping, n, 0.0, subsetJointNames);

      // Detect per-actuator whether the scene uses position actuators (biastype=affine,
      // ctrl=target_pos, PD handled internally by MuJoCo) or motor actuators
      // (biastype=none, ctrl=torque, PD must be computed externally from kp/kd).
      const positionActuator: boolean[] = mapping.ctrlAdr.map((adr) => {
        if (adr < 0 || !this.mjModel) return false;
        return this.mjModel.actuator_biastype[adr] === affineBiasValue;
      });

      const isPosition = positionActuator.some(Boolean);
      const isMotor = positionActuator.some((v) => !v);
      if (isPosition && isMotor) {
        console.warn(`[PolicyRunner] Action term "${termKey}": mixed actuator types detected.`);
      }
      console.log(
        `[PolicyRunner] Action term "${termKey}" (${controlType}): ${n} joint(s), ` +
        `mode: ${isPosition ? 'position (ctrl=target_pos)' : 'motor (ctrl=torque, external PD)'}`
      );

      return {
        controlType,
        ctrlAdr: mapping.ctrlAdr,
        qposAdr: mapping.qposAdr,
        qvelAdr: mapping.qvelAdr,
        actionIndices: mapping.actionIndices,
        actionScale,
        actionOffset,
        defaultJointPos,
        encoderBias,
        positionActuator,
        kp,
        kd,
      };
    };

    // ── Legacy path: no `actions` block, use flat top-level fields ──────────
    const actionsConfig = config.actions;
    if (!actionsConfig || Object.keys(actionsConfig).length === 0) {
      const controlType = config.control_type ?? 'joint_position';
      if (controlType !== 'joint_position' && controlType !== 'torque') {
        console.warn(`[PolicyRunner] Unsupported control_type: ${controlType}`);
        return null;
      }
      const baseMapping = stateBuilder.getControlMapping();
      if (!baseMapping) {
        console.warn('[PolicyRunner] Failed to build control mapping.');
        return null;
      }
      const mapping = {
        ...baseMapping,
        actionIndices: Array.from({ length: baseMapping.qposAdr.length }, (_, i) => i),
      };
      return [buildEntry(
        'legacy',
        controlType,
        mapping,
        config.action_scale,
        undefined,
        config.stiffness,
        config.damping,
        true
      )];
    }

    // ── Multi-term path: iterate every entry in the `actions` block ─────────
    const results: Array<ReturnType<typeof buildEntry>> = [];

    for (const [termKey, actionTerm] of Object.entries(actionsConfig)) {
      const controlType = actionTerm.type ?? 'joint_position';
      if (controlType !== 'joint_position' && controlType !== 'torque') {
        console.warn(`[PolicyRunner] Action term "${termKey}": unsupported type "${controlType}", skipping.`);
        continue;
      }

      // If actuator_names is absent or [".*"], match all joints (backward-compatible).
      const patterns = actionTerm.actuator_names ?? ['.*'];
      const isMatchAll = patterns.length === 1 && patterns[0] === '.*';

      let mapping: { ctrlAdr: number[]; qposAdr: number[]; qvelAdr: number[]; actionIndices: number[] } | null;

      if (isMatchAll) {
        const baseMapping = stateBuilder.getControlMapping();
        if (!baseMapping) {
          console.warn(`[PolicyRunner] Action term "${termKey}": failed to build control mapping, skipping.`);
          continue;
        }
        mapping = {
          ...baseMapping,
          actionIndices: Array.from({ length: baseMapping.qposAdr.length }, (_, i) => i),
        };
      } else {
        mapping = stateBuilder.getControlMappingFor(patterns, jointNames);
        if (!mapping) {
          console.warn(`[PolicyRunner] Action term "${termKey}": no joints matched patterns [${patterns.join(', ')}], skipping.`);
          continue;
        }
      }

      const useDefaultOffset = actionTerm.use_default_offset !== undefined
        ? actionTerm.use_default_offset
        : controlType === 'joint_position';

      results.push(buildEntry(
        termKey,
        controlType,
        mapping,
        actionTerm.scale as number[] | number | Record<string, number> | undefined,
        actionTerm.offset as number[] | number | Record<string, number> | undefined,
        actionTerm.stiffness as number[] | number | Record<string, number> | undefined,
        actionTerm.damping as number[] | number | Record<string, number> | undefined,
        useDefaultOffset
      ));
    }

    if (results.length === 0) {
      console.warn('[PolicyRunner] No valid action terms found in config.actions.');
      return null;
    }
    return results;
  }

  private normalizeControlArray(
    values: number[] | number | Record<string, number> | undefined,
    length: number,
    fallback: number,
    jointNames?: string[]
  ): Float32Array {
    const output = new Float32Array(length);
    output.fill(fallback);
    if (typeof values === 'number') {
      output.fill(values);
      return output;
    }
    if (Array.isArray(values)) {
      for (let i = 0; i < length; i++) {
        output[i] = typeof values[i] === 'number' ? values[i] : fallback;
      }
      return output;
    }
    if (values !== null && typeof values === 'object' && jointNames) {
      for (const [name, val] of Object.entries(values)) {
        const idx = jointNames.indexOf(name);
        if (idx >= 0 && idx < length) {
          output[idx] = val;
        } else {
          console.warn(`[PolicyRunner] Joint name "${name}" not found in policy_joint_names; skipping.`);
        }
      }
      return output;
    }
    return output;
  }

  private resetSimulationState(): void {
    if (!this.mjModel || !this.mjData) {
      return;
    }
    if (this.mjModel.nkey > 0) {
      this.mujoco.mj_resetDataKeyframe(this.mjModel, this.mjData, 0);
    } else {
      this.mujoco.mj_resetData(this.mjModel, this.mjData);
    }
    if (this.eventManager) {
      this.eventManager.onReset({
        mjModel: this.mjModel,
        mjData: this.mjData,
        terrainData: this.terrainData,
      });
    }
    getCommandManager().resetTerms();
    if (this.onnxModule) {
      this.onnxInputDict = this.onnxModule.initInput();
    }
    this.onnxTimeStep = 0;
    this.simStepCount = 0;
    this.navState = 'idle';
    this.navArrivedUntil = 0;
    this.mujoco.mj_forward(this.mjModel, this.mjData);
    this.lastSimState.bodies.clear();
    this.updateCachedState();
  }

  private executeSimulationSteps(): void {
    if (!this.mjModel || !this.mjData) {
      return;
    }
    // Apply drag forces
    this.applyDragForces();

    for (let substep = 0; substep < this.decimation; substep++) {
      this.applyPolicyControl();
      this.mujoco.mj_step(this.mjModel, this.mjData);
      this.simStepCount += 1;
    }
  }

  private applyPolicyControl(): void {
    if (!this.policyControl || !this.mjData) {
      return;
    }

    const ctrl = this.mjData.ctrl;
    ctrl.fill(0.0);

    // Fetch the full action vector once; each term reads its own slice via actionIndices.
    const allActions = this.policyRunner?.getLastActions() ?? new Float32Array(0);

    for (const term of this.policyControl) {
      const {
        controlType,
        ctrlAdr,
        qposAdr,
        qvelAdr,
        actionIndices,
        actionScale,
        actionOffset,
        defaultJointPos,
        encoderBias,
        positionActuator,
        kp,
        kd,
      } =
        term;
      const numJoints = ctrlAdr.length;

      if (controlType === 'joint_position') {
        for (let i = 0; i < numJoints; i++) {
          const ctrlIndex = ctrlAdr[i];
          if (ctrlIndex < 0) continue;
          const actionValue = allActions[actionIndices[i]] ?? 0;
          const target =
            defaultJointPos[i] +
            actionOffset[i] +
            actionScale[i] * actionValue -
            encoderBias[i];

          if (positionActuator[i]) {
            // Position actuator (biastype=affine): ctrl = target joint position.
            // MuJoCo computes force = kp*(ctrl - qpos) - kd*qvel internally.
            ctrl[ctrlIndex] = target;
          } else {
            // Motor actuator (biastype=none): ctrl = torque.
            // PD must be computed externally using kp/kd from the policy config.
            const qpos = this.mjData.qpos[qposAdr[i]];
            const qvel = this.mjData.qvel[qvelAdr[i]];
            ctrl[ctrlIndex] = kp[i] * (target - qpos) + kd[i] * (0 - qvel);
          }
        }
      } else if (controlType === 'torque') {
        for (let i = 0; i < numJoints; i++) {
          const ctrlIndex = ctrlAdr[i];
          if (ctrlIndex >= 0) {
            ctrl[ctrlIndex] = actionScale[i] * (allActions[actionIndices[i]] ?? 0);
          }
        }
      }
    }
  }

  private async runOnnxInference(obs: Record<string, Float32Array>): Promise<void> {
    if (!this.onnxModule || !this.policyRunner || this.onnxInferencing) {
      return;
    }

    this.onnxInferencing = true;
    try {
      if (!this.onnxInputDict) {
        this.onnxInputDict = this.onnxModule.initInput();
      }
      const input: Record<string, ort.Tensor> = { ...this.onnxInputDict };
      if (this.onnxModule.inKeys.includes('time_step')) {
        input.time_step = new ort.Tensor('float32', new Float32Array([this.onnxTimeStep]), [1, 1]);
      }
      for (const [key, value] of Object.entries(obs)) {
        input[key] = new ort.Tensor('float32', value, [1, value.length]);
      }
      for (const key of this.onnxModule.inKeys) {
        if (!input[key]) {
          console.warn('[PolicyRunner] Missing ONNX input:', {
            key,
            available: Object.keys(input),
          });
          return;
        }
      }

      const [result, carry] = await this.onnxModule.runInference(input);
      if (Object.keys(carry).length > 0) {
        this.onnxInputDict = { ...this.onnxInputDict, ...carry };
      }
      if (this.onnxModule.inKeys.includes('time_step')) {
        this.onnxTimeStep += 1;
      }

      const outKey = this.onnxModule.outKeys[0];
      const actionTensor = result.action ?? (outKey ? result[outKey] : null) ?? result.policy ?? null;
      if (!actionTensor) {
        return;
      }

      const raw = actionTensor.data as Float32Array | number[];
      const action = ArrayBuffer.isView(raw) ? new Float32Array(raw) : Float32Array.from(raw);
      const expectedActionCount = this.policyRunner?.getNumActions() ?? 0;
      if (this.policyControl && action.length !== expectedActionCount) {
        console.warn('[PolicyRunner] Action size mismatch:', {
          expected: expectedActionCount,
          got: action.length,
        });
        return;
      }
      this.policyRunner.setLastActions(action);
    } catch (error) {
      console.warn('[PolicyRunner] ONNX inference failed:', error);
    } finally {
      this.onnxInferencing = false;
    }
  }

  private applyDragForces(): void {
    if (!this.dragStateManager || !this.mjModel || !this.mjData || !this.bodies) {
      return;
    }

    // Clear xfrc_applied (reset to zero at each step)
    for (let i = 0; i < this.mjData.xfrc_applied.length; i++) {
      this.mjData.xfrc_applied[i] = 0.0;
    }

    const dragged = this.dragStateManager.physicsObject;
    if (!dragged || !('bodyID' in dragged) || typeof dragged.bodyID !== 'number' || dragged.bodyID <= 0) {
      return;
    }

    const bodyId = dragged.bodyID as number;
    if (this.dynamicBodyIds && !this.dynamicBodyIds.has(bodyId)) {
      return;
    }

    // Update body positions (for drag calculation)
    for (let b = 0; b < this.mjModel.nbody; b++) {
      if (this.bodies[b]) {
        getPosition(this.mjData.xpos, b, this.bodies[b].position);
        getQuaternion(this.mjData.xquat, b, this.bodies[b].quaternion);
        this.bodies[b].updateWorldMatrix(true, false);
      }
    }

    // Update offset
    this.dragStateManager.update();

    // Calculate force (Three.js coordinate system → MuJoCo coordinate system)
    const forceThree = this.dragStateManager.offset
      .clone()
      .multiplyScalar(this.dragForceScale);
    const force = threeToMjcCoordinate(forceThree);

    // Point where force is applied (world coordinates)
    const pointThree = this.dragStateManager.worldHit.clone();
    const point = threeToMjcCoordinate(pointThree);
    // Body position
    const bodyPos = new THREE.Vector3(
      this.mjData.xpos[bodyId * 3 + 0],
      this.mjData.xpos[bodyId * 3 + 1],
      this.mjData.xpos[bodyId * 3 + 2]
    );

    // Calculate torque: τ = r × F
    const r = new THREE.Vector3(
      point.x - bodyPos.x,
      point.y - bodyPos.y,
      point.z - bodyPos.z
    );
    const f = new THREE.Vector3(force.x, force.y, force.z);
    const torque = new THREE.Vector3().crossVectors(r, f);

    // Set xfrc_applied
    // xfrc_applied: (nbody, 6) = [fx, fy, fz, tx, ty, tz] for each body
    const offset = bodyId * 6;
    this.mjData.xfrc_applied[offset + 0] = force.x;
    this.mjData.xfrc_applied[offset + 1] = force.y;
    this.mjData.xfrc_applied[offset + 2] = force.z;
    this.mjData.xfrc_applied[offset + 3] = torque.x;
    this.mjData.xfrc_applied[offset + 4] = torque.y;
    this.mjData.xfrc_applied[offset + 5] = torque.z;
  }

  private updateCachedState(): void {
    if (!this.mjModel || !this.mjData || !this.bodies) {
      return;
    }
    const dynamicBodyIds = this.dynamicBodyIds;
    for (let b = 0; b < this.mjModel.nbody; b++) {
      if (dynamicBodyIds && !dynamicBodyIds.has(b)) {
        continue;
      }
      if (this.bodies[b]) {
        if (!this.lastSimState.bodies.has(b)) {
          this.lastSimState.bodies.set(b, {
            position: new THREE.Vector3(),
            quaternion: new THREE.Quaternion(),
          });
        }
        const state = this.lastSimState.bodies.get(b) as BodyState;
        getPosition(this.mjData.xpos, b, state.position);
        getQuaternion(this.mjData.xquat, b, state.quaternion);
      }
    }

    if (this.mujocoRoot && this.mujocoRoot.cylinders) {
      updateTendonGeometry(
        this.mjModel,
        this.mjData,
        {
          cylinders: this.mujocoRoot.cylinders,
          spheres: this.mujocoRoot.spheres!,
        },
        this.lastSimState.tendons
      );
    }
  }

  private applyViewerConfig(config: ViewerConfig | null): void {
    this.cameraState = applyViewerConfig(config, this.camera, this.controls, this.mjModel, this.mjData);
  }

  private onClickNavPointerDown = (event: PointerEvent): void => {
    if (event.button !== 0) {
      return;
    }
    this.clickNavPointerDown = {
      x: event.clientX,
      y: event.clientY,
      time: performance.now(),
      button: event.button,
    };
  };

  private onClickNavPointerUp = (event: PointerEvent): void => {
    const down = this.clickNavPointerDown;
    this.clickNavPointerDown = null;
    if (!down || down.button !== 0 || event.button !== 0) {
      return;
    }
    const dx = event.clientX - down.x;
    const dy = event.clientY - down.y;
    const moved = Math.hypot(dx, dy);
    const elapsed = performance.now() - down.time;
    if (moved > 6 || elapsed > 750 || !this.hasPlanarVelocityCommand()) {
      return;
    }

    const target = this.pickNavigationTarget(event.clientX, event.clientY);
    if (!target) {
      return;
    }
    this.clickNavTarget = target;
    this.navState = 'moving';
    this.navArrivedUntil = 0;
    this.showClickNavigationMarker(target.threePosition);
  };

  private hasPlanarVelocityCommand(): boolean {
    const commandManager = getCommandManager();
    return (
      Boolean(commandManager.getCommandById('velocity:lin_vel_x')) &&
      Boolean(commandManager.getCommandById('velocity:lin_vel_y')) &&
      Boolean(commandManager.getCommandById('velocity:ang_vel_z'))
    );
  }

  private pickNavigationTarget(clientX: number, clientY: number): ClickNavigationTarget | null {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1
    );
    this.clickNavRaycaster.setFromCamera(ndc, this.camera);

    const sceneHit = this.clickNavRaycaster.intersectObjects(this.scene.children, true)
      .find(hit => !this.isNavigationIgnoredObject(hit.object));
    if (sceneHit && this.isDynamicBodyObject(sceneHit.object)) {
      return null;
    }

    const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    const threePosition = new THREE.Vector3();
    if (!this.clickNavRaycaster.ray.intersectPlane(groundPlane, threePosition)) {
      return null;
    }
    if (!Number.isFinite(threePosition.x) || !Number.isFinite(threePosition.z)) {
      return null;
    }

    const mjcTarget = threeToMjcCoordinate(threePosition);
    return {
      mjcX: mjcTarget.x,
      mjcY: mjcTarget.y,
      threePosition: threePosition.clone(),
    };
  }

  private isNavigationIgnoredObject(object: THREE.Object3D): boolean {
    let current: THREE.Object3D | null = object;
    while (current) {
      if (current.userData?.ignoreClickNavigation === true) {
        return true;
      }
      current = current.parent;
    }
    return false;
  }

  private isDynamicBodyObject(object: THREE.Object3D): boolean {
    let current: THREE.Object3D | null = object;
    while (current) {
      if ('bodyID' in current && typeof current.bodyID === 'number') {
        return this.dynamicBodyIds?.has(current.bodyID) ?? current.bodyID > 0;
      }
      current = current.parent;
    }
    return false;
  }

  private showClickNavigationMarker(position: THREE.Vector3): void {
    if (!this.clickNavMarker) {
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(0.12, 0.18, 40),
        new THREE.MeshBasicMaterial({
          color: 0x31d0aa,
          transparent: true,
          opacity: 0.9,
          side: THREE.DoubleSide,
          depthWrite: false,
        })
      );
      ring.rotation.x = -Math.PI / 2;
      const center = new THREE.Mesh(
        new THREE.CircleGeometry(0.035, 24),
        new THREE.MeshBasicMaterial({
          color: 0xffffff,
          transparent: true,
          opacity: 0.85,
          side: THREE.DoubleSide,
          depthWrite: false,
        })
      );
      center.rotation.x = -Math.PI / 2;
      this.clickNavMarker = new THREE.Group();
      this.clickNavMarker.name = 'Click Navigation Target';
      this.clickNavMarker.userData.ignoreDragForce = true;
      this.clickNavMarker.userData.ignoreClickNavigation = true;
      this.clickNavMarker.add(ring, center);
      this.scene.add(this.clickNavMarker);
    }
    this.clickNavMarker.position.copy(position);
    this.clickNavMarker.position.y += 0.01;
    this.clickNavMarker.visible = true;
  }



  private ensureStereoCameraVisualization(): StereoCameraVisualization {
    if (this.stereoCameraViz) {
      return this.stereoCameraViz;
    }
    const group = new THREE.Group();
    group.name = 'Virtual Stereo Camera';
    group.userData.ignoreClickNavigation = true;
    group.userData.ignoreDragForce = true;

    const mount = new THREE.Mesh(
      new THREE.BoxGeometry(0.024, 0.026, 0.135),
      new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.72, metalness: 0.18 })
    );
    mount.name = 'Stereo Camera Bar';

    const lensMaterial = new THREE.MeshStandardMaterial({ color: 0x020202, roughness: 0.42, metalness: 0.32 });
    const left = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.018, 0.026, 32), lensMaterial.clone());
    left.name = 'cam_left';
    left.rotation.z = -Math.PI / 2;
    left.position.set(0.02, 0, -0.050);

    const right = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.018, 0.026, 32), lensMaterial.clone());
    right.name = 'cam_right';
    right.rotation.z = -Math.PI / 2;
    right.position.set(0.02, 0, 0.050);

    group.add(mount, left, right);
    this.scene.add(group);
    this.stereoCameraViz = { group, mount, left, right };
    return this.stereoCameraViz;
  }

  private updateStereoCameraVisualization(): void {
    const root = this.getStereoRootObject();
    if (!root) {
      if (this.stereoCameraViz) {
        this.stereoCameraViz.group.visible = false;
      }
      return;
    }
    const viz = this.ensureStereoCameraVisualization();
    viz.group.visible = true;

    const rootPosition = new THREE.Vector3();
    root.getWorldPosition(rootPosition);

    const yaw = this.getRootYaw() + Math.min(Math.PI, Math.max(-Math.PI, this.stereoLookYaw));
    const mountWorld = mjcToThreeCoordinate([
      (this.mjData?.qpos[0] ?? 0) + Math.cos(yaw) * 0.02,
      (this.mjData?.qpos[1] ?? 0) + Math.sin(yaw) * 0.02,
      rootPosition.y + 0.05,
    ]);

    viz.group.position.copy(mountWorld);
    viz.group.quaternion.setFromEuler(new THREE.Euler(0, -yaw, 0));
  }

  private setNavigationTargetMjc(x: number, y: number, z = 0): void {
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      console.warn('[mjswan-frame] invalid nav_goal', { x, y, z });
      return;
    }
    if (!this.hasPlanarVelocityCommand()) {
      console.warn('[mjswan-frame] nav_goal ignored: velocity command terms are missing');
      return;
    }
    console.info('[mjswan-frame] nav_goal accepted', { x, y, z });
    const threePosition = mjcToThreeCoordinate([x, y, Number.isFinite(z) ? z : 0]);
    this.clickNavTarget = {
      mjcX: x,
      mjcY: y,
      threePosition,
    };
    this.navState = 'moving';
    this.navArrivedUntil = 0;
    this.showClickNavigationMarker(threePosition);
  }

  private cancelNavigation(): void {
    const commandManager = getCommandManager();
    commandManager.setValue('velocity:lin_vel_x', 0.0);
    commandManager.setValue('velocity:lin_vel_y', 0.0);
    commandManager.setValue('velocity:ang_vel_z', 0.0);
    this.clickNavTarget = null;
    this.navState = 'idle';
    this.navArrivedUntil = 0;
    if (this.clickNavMarker) {
      this.clickNavMarker.visible = false;
    }
  }

  private setStereoCameraLook(yaw: number, pitch: number): void {
    this.stereoLookYaw = Math.min(Math.PI, Math.max(-Math.PI, yaw));
    this.stereoLookPitch = Math.min(0.8, Math.max(-0.8, pitch));
    console.info('[mjswan-frame] cam_look accepted', { yaw: this.stereoLookYaw, pitch: this.stereoLookPitch });
  }

  private getStereoRootObject(): THREE.Object3D | null {
    if (!this.mjModel || !this.bodies) {
      return null;
    }
    const preferredNames = ['torso_link', 'base_link', 'pelvis', 'trunk'];
    for (let bodyId = 0; bodyId < this.mjModel.nbody; bodyId++) {
      const name = this.mjModel.body(bodyId).name;
      if (preferredNames.includes(name) && this.bodies[bodyId]) {
        return this.bodies[bodyId];
      }
    }
    return this.bodies[1] ?? this.bodies[0] ?? null;
  }

  private getFrameBridgeMeta(): StereoFrameBridgeMeta | null {
    if (!this.mjData) {
      return null;
    }
    if (this.navState === 'arrived' && this.navArrivedUntil > 0 && performance.now() > this.navArrivedUntil) {
      this.navState = 'idle';
      this.navArrivedUntil = 0;
    }
    return {
      x: this.mjData.qpos[0] ?? 0,
      y: this.mjData.qpos[1] ?? 0,
      z: this.mjData.qpos[2] ?? 0,
      yaw: this.getRootYaw(),
      nav_state: this.navState,
      step: this.simStepCount,
    };
  }

  private updateClickNavigationCommand(): void {
    if (!this.clickNavTarget || !this.mjData) {
      return;
    }

    const rootX = this.mjData.qpos[0] ?? 0;
    const rootY = this.mjData.qpos[1] ?? 0;
    const dx = this.clickNavTarget.mjcX - rootX;
    const dy = this.clickNavTarget.mjcY - rootY;
    const distance = Math.hypot(dx, dy);
    const commandManager = getCommandManager();

    if (distance < 0.22) {
      commandManager.setValue('velocity:lin_vel_x', 0.0);
      commandManager.setValue('velocity:lin_vel_y', 0.0);
      commandManager.setValue('velocity:ang_vel_z', 0.0);
      this.clickNavTarget = null;
      this.navState = 'arrived';
      this.navArrivedUntil = performance.now() + 1000;
      if (this.clickNavMarker) {
        this.clickNavMarker.visible = false;
      }
      return;
    }

    const yaw = this.getRootYaw();
    const cosYaw = Math.cos(yaw);
    const sinYaw = Math.sin(yaw);
    const bodyX = cosYaw * dx + sinYaw * dy;
    const bodyY = -sinYaw * dx + cosYaw * dy;
    const targetYaw = Math.atan2(dy, dx);
    const yawError = Math.atan2(Math.sin(targetYaw - yaw), Math.cos(targetYaw - yaw));
    const slowDown = Math.min(1.0, Math.max(0.25, distance / 1.2));

    commandManager.setValue('velocity:lin_vel_x', bodyX * 0.9 * slowDown);
    commandManager.setValue('velocity:lin_vel_y', bodyY * 0.9 * slowDown);
    commandManager.setValue('velocity:ang_vel_z', yawError * 1.8);
  }

  private getRootYaw(): number {
    if (!this.mjData) {
      return 0.0;
    }
    const w = this.mjData.qpos[3] ?? 1;
    const x = this.mjData.qpos[4] ?? 0;
    const y = this.mjData.qpos[5] ?? 0;
    const z = this.mjData.qpos[6] ?? 0;
    return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  }

  private computeDynamicBodyIds(mjModel: MjModel): Set<number> {
    const dynamic = new Set<number>();
    for (let bodyId = 1; bodyId < mjModel.nbody; bodyId++) {
      let current = bodyId;
      while (current > 0) {
        if (mjModel.body_jntnum[current] > 0) {
          dynamic.add(bodyId);
          break;
        }
        current = mjModel.body_parentid[current];
      }
    }
    return dynamic;
  }

  private syncStaticBodiesFromData(): void {
    if (!this.mjModel || !this.mjData || !this.bodies) {
      return;
    }
    const dynamicBodyIds = this.dynamicBodyIds;
    for (let bodyId = 0; bodyId < this.mjModel.nbody; bodyId++) {
      if (dynamicBodyIds?.has(bodyId)) {
        continue;
      }
      const body = this.bodies[bodyId];
      if (!body) {
        continue;
      }
      getPosition(this.mjData.xpos, bodyId, body.position);
      getQuaternion(this.mjData.xquat, bodyId, body.quaternion);
    }
  }

  private render = (): void => {
    getCommandManager().updateDebugVisuals();

    if (this.mjData) {
      updateCameraFromData(this.mjData, this.camera, this.controls, this.cameraState);
    }
    this.controls.update();

    if (this.mjModel && this.mjData && this.bodies) {
      updateHeadlightFromCamera(this.camera, this.lights);

      for (const [b, state] of this.lastSimState.bodies) {
        const body = this.bodies[b];
        if (body) {
          body.position.copy(state.position);
          body.quaternion.copy(state.quaternion);
        }
      }

      updateLightsFromData(this.mujoco, this.mjData, this.lights);

      if (this.mujocoRoot && this.mujocoRoot.cylinders) {
        updateTendonRendering(
          {
            cylinders: this.mujocoRoot.cylinders,
            spheres: this.mujocoRoot.spheres!,
          },
          this.lastSimState.tendons
        );
      }
    }

    this.updateStereoCameraVisualization();
    this.renderer.render(this.scene, this.camera);
  };

  private onWindowResize = (): void => {
    const { width, height } = this.getSize();
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  };

  dispose(): void {
    this.stop();
    this.frameBridge?.dispose();
    this.frameBridge = null;
    this.policyRunner = null;
    this.policyStateBuilder = null;
    this.policyConfigPath = null;
    this.container.removeEventListener('pointerdown', this.onClickNavPointerDown, true);
    this.container.removeEventListener('pointerup', this.onClickNavPointerUp, true);

    if (this.dragStateManager) {
      this.dragStateManager.dispose();
      this.dragStateManager = null;
    }

    // NOTE: Do NOT delete mjData/mjModel here as they may be cached
    // The cache manager will handle their disposal when evicting
    // Just clear references
    this.mjData = null;
    this.mjModel = null;

    // NOTE: Do NOT dispose Three.js resources here as they may be cached
    // The cache manager will handle their disposal when evicting
    // Just clear references
    // this.disposeThreeJSResources();

    window.removeEventListener('resize', this.onWindowResize);
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;

    this.controls.dispose();
    this.renderer.setAnimationLoop(null);
    this.renderer.dispose();

    if (this.renderer.domElement.parentElement) {
      this.renderer.domElement.parentElement.removeChild(this.renderer.domElement);
    }

    if (this.vrButton?.parentElement) {
      this.vrButton.parentElement.removeChild(this.vrButton);
      this.vrButton = null;
    }

    if (this.stereoCameraViz?.group.parent) {
      this.stereoCameraViz.group.parent.remove(this.stereoCameraViz.group);
    }
    if (this.stereoCameraViz) {
      this.stereoCameraViz.group.traverse((object) => {
        if ('geometry' in object && object.geometry) {
          (object.geometry as THREE.BufferGeometry).dispose();
        }
        if ('material' in object && object.material) {
          const material = object.material as THREE.Material | THREE.Material[];
          if (Array.isArray(material)) {
            material.forEach((entry) => entry.dispose());
          } else {
            material.dispose();
          }
        }
      });
      this.stereoCameraViz = null;
    }

    if (this.clickNavMarker?.parent) {
      this.clickNavMarker.parent.remove(this.clickNavMarker);
    }
    if (this.clickNavMarker) {
      this.clickNavMarker.traverse((object) => {
        if ('geometry' in object && object.geometry) {
          (object.geometry as THREE.BufferGeometry).dispose();
        }
        if ('material' in object && object.material) {
          const material = object.material as THREE.Material | THREE.Material[];
          if (Array.isArray(material)) {
            material.forEach((entry) => entry.dispose());
          } else {
            material.dispose();
          }
        }
      });
      this.clickNavMarker = null;
    }

    this.bodies = null;
    this.lights = [];
    this.mujocoRoot = null;
    this.dynamicBodyIds = null;
    this.lastSimState.bodies.clear();
  }

  private disposeThreeJSResources(): void {
    if (!this.scene) {
      return;
    }

    this.scene.traverse((object) => {
      if ('geometry' in object && object.geometry) {
        (object.geometry as THREE.BufferGeometry).dispose();
      }
      if ('material' in object && object.material) {
        if (Array.isArray(object.material)) {
          object.material.forEach((material) => this.disposeMaterial(material));
        } else {
          this.disposeMaterial(object.material as THREE.Material);
        }
      }
    });

    while (this.scene.children.length > 0) {
      this.scene.remove(this.scene.children[0]);
    }
  }

  private disposeMaterial(material: THREE.Material): void {
    const anyMaterial = material as THREE.MeshStandardMaterial & {
      map?: THREE.Texture;
      aoMap?: THREE.Texture;
      emissiveMap?: THREE.Texture;
      metalnessMap?: THREE.Texture;
      normalMap?: THREE.Texture;
      roughnessMap?: THREE.Texture;
    };

    if (anyMaterial.map) {
      anyMaterial.map.dispose();
    }
    if (anyMaterial.aoMap) {
      anyMaterial.aoMap.dispose();
    }
    if (anyMaterial.emissiveMap) {
      anyMaterial.emissiveMap.dispose();
    }
    if (anyMaterial.metalnessMap) {
      anyMaterial.metalnessMap.dispose();
    }
    if (anyMaterial.normalMap) {
      anyMaterial.normalMap.dispose();
    }
    if (anyMaterial.roughnessMap) {
      anyMaterial.roughnessMap.dispose();
    }
    material.dispose();
  }

  private getSize(): { width: number; height: number } {
    const width = this.container.clientWidth || window.innerWidth;
    const height = this.container.clientHeight || window.innerHeight;
    return {
      width: Math.max(1, width),
      height: Math.max(1, height),
    };
  }

  /**
   * Restore scene from cache
   */
  private async restoreFromCache(scenePath: string): Promise<void> {
    const resources = this.sceneCacheManager.get(scenePath);
    if (!resources) {
      throw new Error(`Scene ${scenePath} not found in cache`);
    }

    // Remove existing root if present
    const existingRoot = this.scene.getObjectByName('MuJoCo Root');
    if (existingRoot) {
      this.scene.remove(existingRoot);
    }

    // Restore MuJoCo objects
    this.mjModel = resources.mjModel;
    this.mjData = resources.mjData;

    // Restore Three.js resources
    this.bodies = resources.bodies;
    this.lights = resources.lights;
    this.mujocoRoot = resources.mujocoRoot;

    // Re-add root to scene
    this.scene.add(this.mujocoRoot);

    // Restore skybox background
    this.scene.background = resources.skybox;

    // Run forward dynamics
    this.mujoco.mj_forward(this.mjModel, this.mjData);
    this.dynamicBodyIds = this.computeDynamicBodyIds(this.mjModel);
    this.syncStaticBodiesFromData();

    // Update runtime parameters
    this.timestep = this.mjModel.opt.timestep || 0.001;
    this.decimation = Math.max(1, Math.round(0.02 / this.timestep));

    // Clear and update cached state
    this.lastSimState.bodies.clear();
    this.updateCachedState();

    // Initialize DragStateManager if needed
    if (!this.dragStateManager) {
      this.dragStateManager = new DragStateManager({
        scene: this.scene,
        renderer: this.renderer,
        camera: this.camera,
        container: this.container,
        controls: this.controls,
        draggableBodyIds: this.dynamicBodyIds,
      });
    } else {
      this.dragStateManager.setDraggableBodyIds(this.dynamicBodyIds);
    }
  }

  /**
   * Capture resources and add to cache
   */
  private async captureAndCacheResources(scenePath: string): Promise<void> {
    // Stop tracking and get FS files
    const fsFiles = this.resourceTracker.stopTracking(this.mujoco);

    if (!this.mjModel || !this.mjData || !this.bodies || !this.mujocoRoot) {
      console.warn('[SceneCache] Cannot cache scene: missing resources');
      return;
    }

    // Estimate memory usage
    const estimatedMemoryBytes = this.resourceTracker.estimateSceneMemory({
      mjModel: this.mjModel,
      mjData: this.mjData,
      bodies: this.bodies,
      meshes: {}, // Meshes are part of the scene
      mujocoRoot: this.mujocoRoot,
    });

    // Create cache entry
    await this.sceneCacheManager.set(scenePath, {
      scenePath,
      lastAccessed: Date.now(),
      loadedAt: Date.now(),
      mjModel: this.mjModel,
      mjData: this.mjData,
      bodies: this.bodies,
      lights: this.lights,
      meshes: {},
      mujocoRoot: this.mujocoRoot,
      skybox: this.scene.background instanceof THREE.CubeTexture ? this.scene.background : null,
      fsFiles,
      estimatedMemoryBytes,
    });

    // Log the cache operation
    const metrics = this.sceneCacheManager.getMetrics();
    this.memoryMonitor.logCacheOperation('load', scenePath, {
      memoryMB: estimatedMemoryBytes / 1048576,
      totalScenes: metrics.totalScenes,
      totalMemoryMB: metrics.totalMemoryBytes / 1048576,
    });
  }
}

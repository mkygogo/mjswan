import * as THREE from 'three';
import type { MjModel } from 'mujoco';
import type { CommandConfigEntry, CommandTerm, CommandTermContext } from 'mjswan/command';
import { mjcToThreeCoordinate } from 'mjswan/coordinate';

type Range2 = [number, number];

type PositionRange = {
  x?: Range2;
  y?: Range2;
  z?: Range2;
};

type ObjectPoseRange = PositionRange & {
  yaw?: Range2;
};

function sampleRange(range: unknown, fallback: number): number {
  if (Array.isArray(range) && range.length >= 2) {
    const min = typeof range[0] === 'number' ? range[0] : fallback;
    const max = typeof range[1] === 'number' ? range[1] : fallback;
    return min + Math.random() * (max - min);
  }
  return fallback;
}

function quatFromYaw(yaw: number): [number, number, number, number] {
  const half = yaw * 0.5;
  return [Math.cos(half), 0.0, 0.0, Math.sin(half)];
}

export class LiftingCommand implements CommandTerm {
  private readonly context: CommandTermContext;
  private readonly targetPos = new Float32Array(3);
  private readonly resamplingTimeRange: Range2;
  private readonly difficulty: 'fixed' | 'dynamic';
  private readonly targetPositionRange: PositionRange;
  private readonly objectPoseRange: ObjectPoseRange | null;
  private readonly debugVisEnabled: boolean;
  private readonly objectQposAdr: number | null;
  private readonly objectQvelAdr: number | null;
  private readonly marker: THREE.Mesh | null;
  private timeLeft = 0.0;

  constructor(
    termName: string,
    config: CommandConfigEntry,
    context: CommandTermContext
  ) {
    this.context = context;
    const range = Array.isArray(config.resampling_time_range)
      ? config.resampling_time_range
      : [1.0, 1.0];
    this.resamplingTimeRange = [
      typeof range[0] === 'number' ? range[0] : 1.0,
      typeof range[1] === 'number' ? range[1] : 1.0,
    ];
    this.difficulty = config.difficulty === 'dynamic' ? 'dynamic' : 'fixed';
    this.targetPositionRange = (config.target_position_range as PositionRange | undefined) ?? {};
    this.objectPoseRange = (config.object_pose_range as ObjectPoseRange | undefined) ?? null;
    this.debugVisEnabled = Boolean(config.debug_vis);

    const entityName = typeof config.entity_name === 'string' ? config.entity_name : termName;
    const mjModel = context.mjModel;
    if (mjModel === null) {
      this.objectQposAdr = null;
      this.objectQvelAdr = null;
    } else {
      const bodyIdx = this.resolveEntityBodyIdx(mjModel, entityName);
      const jointIdx = bodyIdx >= 0 ? this.resolveFreeJointIdx(mjModel, bodyIdx) : -1;
      this.objectQposAdr = jointIdx >= 0 ? mjModel.jnt_qposadr[jointIdx] : null;
      this.objectQvelAdr = jointIdx >= 0 ? mjModel.jnt_dofadr[jointIdx] : null;
      if (jointIdx < 0) {
        console.warn(`[LiftingCommand] free joint not found for entity "${entityName}"`);
      }
    }

    this.marker = this.createMarker(config);
  }

  getCommand(): Float32Array {
    return new Float32Array(this.targetPos);
  }

  reset(): void {
    this.resample();
  }

  update(dt: number): void {
    this.timeLeft -= dt;
    if (this.timeLeft <= 0.0) {
      this.resample();
    }
  }

  updateDebugVisuals(): void {
    if (!this.marker) {
      return;
    }
    this.marker.visible = this.debugVisEnabled;
    if (!this.marker.visible) {
      return;
    }
    this.marker.position.copy(mjcToThreeCoordinate(this.targetPos));
  }

  dispose(): void {
    if (!this.marker) {
      return;
    }
    this.context.scene.remove(this.marker);
    this.marker.geometry.dispose();
    const material = this.marker.material;
    if (Array.isArray(material)) {
      for (const entry of material) {
        entry.dispose();
      }
    } else {
      material.dispose();
    }
  }

  private resample(): void {
    this.timeLeft = sampleRange(this.resamplingTimeRange, 1.0);

    if (this.difficulty === 'fixed') {
      this.targetPos[0] = 0.4;
      this.targetPos[1] = 0.0;
      this.targetPos[2] = 0.3;
    } else {
      this.targetPos[0] = sampleRange(this.targetPositionRange.x, 0.4);
      this.targetPos[1] = sampleRange(this.targetPositionRange.y, 0.0);
      this.targetPos[2] = sampleRange(this.targetPositionRange.z, 0.3);
    }

    if (this.objectPoseRange && this.objectQposAdr !== null && this.objectQvelAdr !== null) {
      const { mjData, mujoco, mjModel } = this.context;
      if (!mjData || !mjModel) {
        return;
      }
      const x = sampleRange(this.objectPoseRange.x, 0.3);
      const y = sampleRange(this.objectPoseRange.y, 0.0);
      const z = sampleRange(this.objectPoseRange.z, 0.03);
      const yaw = sampleRange(this.objectPoseRange.yaw, 0.0);
      const quat = quatFromYaw(yaw);

      const qposAdr = this.objectQposAdr;
      mjData.qpos[qposAdr] = x;
      mjData.qpos[qposAdr + 1] = y;
      mjData.qpos[qposAdr + 2] = z;
      mjData.qpos[qposAdr + 3] = quat[0];
      mjData.qpos[qposAdr + 4] = quat[1];
      mjData.qpos[qposAdr + 5] = quat[2];
      mjData.qpos[qposAdr + 6] = quat[3];

      const qvelAdr = this.objectQvelAdr;
      for (let i = 0; i < 6; i++) {
        mjData.qvel[qvelAdr + i] = 0.0;
      }
      mujoco.mj_forward(mjModel, mjData);
    }
  }

  private resolveEntityBodyIdx(mjModel: MjModel, entityName: string): number {
    const namesArray = new Uint8Array(mjModel.names);
    const decoder = new TextDecoder();
    const bodyNames: string[] = [];
    for (let i = 0; i < mjModel.nbody; i++) {
      let start = mjModel.name_bodyadr[i];
      let end = start;
      while (end < namesArray.length && namesArray[end] !== 0) {
        end++;
      }
      bodyNames.push(decoder.decode(namesArray.subarray(start, end)));
    }

    for (let i = 0; i < bodyNames.length; i++) {
      const name = bodyNames[i];
      if (name === entityName || name === `${entityName}/${entityName}`) {
        return i;
      }
    }
    for (let i = 0; i < bodyNames.length; i++) {
      if (bodyNames[i].startsWith(`${entityName}/`)) {
        return i;
      }
    }
    return -1;
  }

  private resolveFreeJointIdx(mjModel: MjModel, bodyIdx: number): number {
    for (let i = 0; i < mjModel.njnt; i++) {
      if (mjModel.jnt_bodyid[i] === bodyIdx && mjModel.jnt_type[i] === 0) {
        return i;
      }
    }
    return -1;
  }

  private createMarker(config: CommandConfigEntry): THREE.Mesh | null {
    if (!this.debugVisEnabled) {
      return null;
    }

    const targetColor = Array.isArray((config.viz as { target_color?: unknown } | undefined)?.target_color)
      ? (config.viz as { target_color?: number[] }).target_color ?? [1.0, 0.5, 0.0, 0.3]
      : [1.0, 0.5, 0.0, 0.3];

    const geometry = new THREE.SphereGeometry(0.03, 20, 12);
    const material = new THREE.MeshBasicMaterial({
      color: new THREE.Color(targetColor[0] ?? 1.0, targetColor[1] ?? 0.5, targetColor[2] ?? 0.0),
      transparent: true,
      opacity: targetColor[3] ?? 0.3,
      depthWrite: false,
    });
    const marker = new THREE.Mesh(geometry, material);
    marker.name = 'mjswan-command-lift-height-target';
    marker.visible = false;
    this.context.scene.add(marker);
    return marker;
  }
}

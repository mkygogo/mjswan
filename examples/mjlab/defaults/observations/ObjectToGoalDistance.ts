import * as THREE from 'three';
import type { MjModel } from 'mujoco';
import { ObservationBase } from 'mjswan/observation';
import type { ObservationConfig } from 'mjswan/observation';
import type { PolicyRunner } from 'mjswan/types';
import { getCommandManager } from 'mjswan/command';

/**
 * Distance vector from object to goal, expressed in the robot base frame.
 *
 * Mirrors mjlab's ``object_to_goal_distance`` observation function.
 * Output shape: [3] (x, y, z).
 *
 * The goal position is read from the browser command manager via
 * ``command_name``. If the command is not registered (e.g. the mjlab
 * ``lift_height`` command has no browser equivalent), the goal position
 * falls back to the world origin and the distance equals the object's
 * world-frame position negated into the base frame.
 *
 * Config params:
 *   object_name    — MuJoCo body name of the target object (e.g. "cube").
 *   command_name   — Name of the command group that carries the 3-element
 *                    [x, y, z] goal position in world frame.
 *   base_body_name — (optional) MuJoCo body name of the robot root for frame
 *                    transformation. Defaults to body 1 (first non-world body).
 */
export class ObjectToGoalDistance extends ObservationBase {
  private objectBodyIdx: number;
  private baseBodyIdx: number;
  private commandName: string;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    const mjModel = runner.getContext()?.mjModel ?? null;
    this.commandName = (config.command_name as string | undefined) ?? '';
    if (mjModel === null) {
      this.objectBodyIdx = 1;
      this.baseBodyIdx = 1;
    } else {
      const objectName = config.object_name as string | undefined;
      this.objectBodyIdx = objectName !== undefined
        ? this.resolveBodyIdx(mjModel, objectName)
        : 1;

      const baseBodyName = config.base_body_name as string | undefined;
      this.baseBodyIdx = baseBodyName !== undefined
        ? this.resolveBodyIdx(mjModel, baseBodyName)
        : 1;
    }
  }

  get size(): number {
    return 3;
  }

  compute(): Float32Array {
    const ctx = this.runner.getContext();
    const mjData = ctx?.mjData;
    if (mjData == null) {
      return new Float32Array(3);
    }

    // Object body position in world frame (xpos: 3 floats per body)
    const oi = this.objectBodyIdx * 3;
    const objPosW = new THREE.Vector3(
      mjData.xpos[oi],
      mjData.xpos[oi + 1],
      mjData.xpos[oi + 2],
    );

    // Goal position from command manager, or world origin if unavailable
    const cmd = this.commandName
      ? getCommandManager().getCommand(this.commandName)
      : null;
    const goalPosW = (cmd !== null && cmd.length >= 3)
      ? new THREE.Vector3(cmd[0], cmd[1], cmd[2])
      : new THREE.Vector3();

    // Distance vector in world frame: goal - object
    const distW = goalPosW.sub(objPosW);

    // Robot base quaternion in world frame (xquat: w,x,y,z per body)
    const bi = this.baseBodyIdx * 4;
    const baseQuat = new THREE.Quaternion(
      mjData.xquat[bi + 1], // x
      mjData.xquat[bi + 2], // y
      mjData.xquat[bi + 3], // z
      mjData.xquat[bi],     // w
    );

    // Rotate distance into base frame: quat_apply(quat_inv(base_quat_w), dist_w)
    distW.applyQuaternion(baseQuat.invert());

    return new Float32Array([distW.x, distW.y, distW.z]);
  }

  private resolveBodyIdx(mjModel: MjModel, bodyName: string): number {
    const namesArray = new Uint8Array(mjModel.names);
    const decoder = new TextDecoder();
    const allNames: string[] = [];
    for (let b = 0; b < mjModel.nbody; b++) {
      let start = mjModel.name_bodyadr[b];
      let end = start;
      while (namesArray[end] !== 0) end++;
      const name = decoder.decode(namesArray.slice(start, end));
      // Exact match or mjlab entity-namespaced match ({entity}/{body})
      if (name === bodyName || name === `${bodyName}/${bodyName}`) return b;
      allNames.push(name);
    }
    // Fallback: first body whose name starts with "{bodyName}/"
    for (let b = 0; b < allNames.length; b++) {
      if (allNames[b].startsWith(`${bodyName}/`)) return b + (allNames[0] === '' ? 1 : 0);
    }
    console.warn(`ObjectToGoalDistance: body "${bodyName}" not found. Available bodies: ${allNames.join(', ')}`);
    return 1;
  }
}

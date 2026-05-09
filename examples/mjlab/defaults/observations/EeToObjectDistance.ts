import * as THREE from 'three';
import type { MjModel } from 'mujoco';
import { ObservationBase } from 'mjswan/observation';
import type { ObservationConfig } from 'mjswan/observation';
import type { PolicyRunner } from 'mjswan/types';

/**
 * Distance vector from end-effector to object, expressed in the robot base frame.
 *
 * Mirrors mjlab's ``ee_to_object_distance`` observation function.
 * Output shape: [3] (x, y, z).
 *
 * Config params:
 *   object_name    — MuJoCo body name of the target object (e.g. "cube").
 *   site_name      — MuJoCo site name of the end-effector (e.g. "grasp_site").
 *   base_body_name — (optional) MuJoCo body name of the robot root for frame
 *                    transformation. Defaults to body 1 (first non-world body).
 */
export class EeToObjectDistance extends ObservationBase {
  private siteIdx: number;
  private objectBodyIdx: number;
  private baseBodyIdx: number;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    const mjModel = runner.getContext()?.mjModel ?? null;
    if (mjModel === null) {
      this.siteIdx = 0;
      this.objectBodyIdx = 1;
      this.baseBodyIdx = 1;
    } else {
      const siteName = config.site_name as string | undefined;
      this.siteIdx = siteName !== undefined
        ? this.resolveSiteIdx(mjModel, siteName)
        : 0;

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

    // End-effector position in world frame (site_xpos: 3 floats per site)
    const si = this.siteIdx * 3;
    const eePosW = new THREE.Vector3(
      mjData.site_xpos[si],
      mjData.site_xpos[si + 1],
      mjData.site_xpos[si + 2],
    );

    // Object body position in world frame (xpos: 3 floats per body)
    const oi = this.objectBodyIdx * 3;
    const objPosW = new THREE.Vector3(
      mjData.xpos[oi],
      mjData.xpos[oi + 1],
      mjData.xpos[oi + 2],
    );

    // Distance vector in world frame
    const distW = objPosW.sub(eePosW);

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

  private resolveSiteIdx(mjModel: MjModel, siteName: string): number {
    const namesArray = new Uint8Array(mjModel.names);
    const decoder = new TextDecoder();
    const allNames: string[] = [];
    for (let s = 0; s < mjModel.nsite; s++) {
      let start = mjModel.name_siteadr[s];
      let end = start;
      while (namesArray[end] !== 0) end++;
      const name = decoder.decode(namesArray.slice(start, end));
      if (name === siteName) return s;
      allNames.push(name);
    }
    console.warn(`EeToObjectDistance: site "${siteName}" not found. Available sites: ${allNames.join(', ')}`);
    return 0;
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
    console.warn(`EeToObjectDistance: body "${bodyName}" not found. Available bodies: ${allNames.join(', ')}`);
    return 1;
  }
}

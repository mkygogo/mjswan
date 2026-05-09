import * as THREE from 'three';
import type { MjData } from 'mujoco';
import { ObservationBase } from 'mjswan/observation';
import type { ObservationConfig } from 'mjswan/observation';
import type { PolicyRunner } from 'mjswan/types';
import { mjcToThreeCoordinate } from 'mjswan/coordinate';

type FrameType = 'body' | 'site' | 'geom';

type FramePose = {
  pos: [number, number, number];
  mat: [number, number, number, number, number, number, number, number, number];
};

type HFieldConfig = {
  geomId: number;
  bodyId: number;
  localPos: [number, number, number];
  localMat: [number, number, number, number, number, number, number, number, number];
  halfX: number;
  halfY: number;
  zScale: number;
  nrow: number;
  ncol: number;
  data: Float32Array;
};

/**
 * Browser-side approximation of mjlab's `height_scan` observation.
 *
 * This reads the configured attachment frame directly from MuJoCo state,
 * reproduces mjlab's yaw-aligned grid pattern, then raycasts against the
 * rendered terrain meshes to estimate vertical clearance.
 */
export class HeightScan extends ObservationBase {
  private readonly frameType: FrameType;
  private readonly frameIndex: number;
  private readonly rayAlignment: 'base' | 'yaw' | 'world';
  private readonly localOffsets: Array<[number, number, number]>;
  private readonly localDirections: Array<[number, number, number]>;
  private readonly maxDistance: number;
  private readonly missValue: number;
  private readonly offset: number;
  private readonly terrainBodyName: string;
  private readonly raycaster: THREE.Raycaster;
  private readonly scale: Float32Array | null;
  private readonly clipRange: [number, number] | null;
  private readonly hfields: HFieldConfig[];
  private terrainTargets: THREE.Object3D[] | null = null;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    const mjModel = runner.getContext()?.mjModel ?? null;

    this.frameType = this.normalizeFrameType(config.frame_type);
    this.frameIndex = mjModel !== null
      ? this.resolveFrameIndex(mjModel, this.frameType, config.frame_ref_name)
      : 0;
    this.rayAlignment = this.normalizeAlignment(config.ray_alignment);
    this.localOffsets = this.buildGridOffsets(config.pattern_size, config.pattern_resolution);
    this.localDirections = this.buildDirections(config.pattern_direction, this.localOffsets.length);
    this.maxDistance = this.normalizeNumber(config.max_distance, 5.0);
    this.missValue = this.normalizeNumber(config.miss_value, this.maxDistance);
    this.offset = this.normalizeNumber(config.offset, 0.0);
    this.terrainBodyName = typeof config.terrain_body_name === 'string'
      ? config.terrain_body_name
      : 'terrain';
    this.hfields = this.resolveHFields(runner, this.terrainBodyName);
    this.scale = this.normalizeScale(config.scale, this.localOffsets.length);
    this.clipRange = this.normalizeClipRange(config.clip);
    this.raycaster = new THREE.Raycaster();
    this.raycaster.near = 0;
    this.raycaster.far = this.maxDistance;
  }

  get size(): number {
    return this.localOffsets.length;
  }

  compute(): Float32Array {
    const ctx = this.runner.getContext();
    const mjData = ctx?.mjData ?? null;
    if (mjData === null) {
      return this.fillMisses();
    }

    const framePose = this.getFramePose(mjData, this.frameType, this.frameIndex);
    if (framePose === null) {
      return this.fillMisses();
    }

    const rotation = this.computeAlignmentRotation(framePose.mat);
    const output = new Float32Array(this.size);
    const hfieldPoses = this.hfields.map((hfield) => ({
      hfield,
      pose: this.getHFieldPose(mjData, hfield),
    })).filter((entry): entry is { hfield: HFieldConfig; pose: FramePose } => entry.pose !== null);
    const targets = this.getTerrainTargets();

    for (let i = 0; i < this.localOffsets.length; i++) {
      const originMj = this.addVec3(framePose.pos, this.mulMat3Vec3(rotation, this.localOffsets[i]));
      const hfieldHeight = this.sampleAnyHFieldHeight(originMj, hfieldPoses);
      if (hfieldHeight !== null) {
        output[i] = hfieldHeight;
      } else {
        if (targets.length === 0) {
          output[i] = this.missValue;
          continue;
        }
        const dirMj = this.mulMat3Vec3(rotation, this.localDirections[i]);
        const originThree = mjcToThreeCoordinate(originMj);
        const dirThree = mjcToThreeCoordinate(dirMj).normalize();

        this.raycaster.set(originThree, dirThree);
        this.raycaster.far = this.maxDistance;

        const hits = this.raycaster.intersectObjects(targets, true);
        if (hits.length === 0) {
          output[i] = this.missValue;
          continue;
        }

        const hit = hits[0];
        const hitZj = hit.point.y;
        output[i] = Math.max(0.0, originMj[2] - hitZj - this.offset);
      }
    }

    if (this.scale !== null) {
      for (let i = 0; i < output.length; i++) {
        output[i] *= this.scale[i] ?? 1.0;
      }
    }
    if (this.clipRange !== null) {
      const [clipMin, clipMax] = this.clipRange;
      for (let i = 0; i < output.length; i++) {
        output[i] = Math.min(clipMax, Math.max(clipMin, output[i]));
      }
    }

    return output;
  }

  private fillMisses(): Float32Array {
    const output = new Float32Array(this.size);
    output.fill(this.missValue);
    return output;
  }

  private getTerrainTargets(): THREE.Object3D[] {
    if (this.terrainTargets !== null) {
      return this.terrainTargets;
    }
    const scene = this.runner.getContext()?.scene ?? null;
    if (scene === null) {
      this.terrainTargets = [];
      return this.terrainTargets;
    }

    const targets: THREE.Object3D[] = [];
    scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) {
        return;
      }
      if (this.hasAncestorNamed(object, this.terrainBodyName)) {
        targets.push(object);
      }
    });
    this.terrainTargets = targets;
    return targets;
  }

  private hasAncestorNamed(object: THREE.Object3D, targetName: string): boolean {
    let current: THREE.Object3D | null = object;
    while (current !== null) {
      if (current.name === targetName) {
        return true;
      }
      current = current.parent;
    }
    return false;
  }

  private resolveHFields(runner: PolicyRunner, terrainBodyName: string): HFieldConfig[] {
    const ctx = runner.getContext();
    const mjModel = ctx?.mjModel ?? null;
    const mujoco = ctx?.mujoco;
    if (mjModel === null || mujoco == null) {
      return [];
    }

    const hfieldType = mujoco.mjtGeom?.mjGEOM_HFIELD?.value;
    if (hfieldType === undefined) {
      return [];
    }

    const bodyNames = this.getModelBodyNames(mjModel);
    const hfields: HFieldConfig[] = [];
    for (let geomId = 0; geomId < mjModel.ngeom; geomId++) {
      if (mjModel.geom_type[geomId] !== hfieldType) {
        continue;
      }
      const bodyId = mjModel.geom_bodyid[geomId];
      const bodyName = bodyNames[bodyId] ?? '';
      if (
        bodyName !== terrainBodyName &&
        bodyName !== `${terrainBodyName}/${terrainBodyName}` &&
        !bodyName.startsWith(`${terrainBodyName}/`)
      ) {
        continue;
      }

      const hfieldId = mjModel.geom_dataid[geomId];
      if (hfieldId < 0) {
        continue;
      }
      const nrow = mjModel.hfield_nrow[hfieldId];
      const ncol = mjModel.hfield_ncol[hfieldId];
      const adr = mjModel.hfield_adr[hfieldId];
      hfields.push({
        geomId,
        bodyId,
        localPos: this.readVec3(mjModel.geom_pos, geomId) ?? [0, 0, 0],
        localMat: this.quatToMat(this.readQuat(mjModel.geom_quat, geomId) ?? [1, 0, 0, 0]),
        halfX: mjModel.hfield_size[hfieldId * 4 + 0],
        halfY: mjModel.hfield_size[hfieldId * 4 + 1],
        zScale: mjModel.hfield_size[hfieldId * 4 + 2],
        nrow,
        ncol,
        data: Float32Array.from(mjModel.hfield_data.subarray(adr, adr + nrow * ncol)),
      });
    }

    return hfields;
  }

  private getHFieldPose(
    mjData: MjData,
    hfield: HFieldConfig
  ): FramePose | null {
    const bodyPose = this.getFramePose(mjData, 'body', hfield.bodyId);
    if (bodyPose === null) {
      return null;
    }
    return {
      pos: this.addVec3(bodyPose.pos, this.mulMat3Vec3(bodyPose.mat, hfield.localPos)),
      mat: this.mulMat3(bodyPose.mat, hfield.localMat),
    };
  }

  private sampleHFieldHeight(
    originMj: [number, number, number],
    hfieldPose: FramePose,
    hfield: HFieldConfig
  ): number | null {
    const localOrigin = this.worldToLocal(hfieldPose.pos, hfieldPose.mat, originMj);
    if (
      localOrigin[0] < -hfield.halfX ||
      localOrigin[0] > hfield.halfX ||
      localOrigin[1] < -hfield.halfY ||
      localOrigin[1] > hfield.halfY
    ) {
      return null;
    }

    const col = ((localOrigin[0] + hfield.halfX) / (2 * hfield.halfX)) * (hfield.ncol - 1);
    const row = ((localOrigin[1] + hfield.halfY) / (2 * hfield.halfY)) * (hfield.nrow - 1);
    const col0 = Math.max(0, Math.min(hfield.ncol - 1, Math.floor(col)));
    const row0 = Math.max(0, Math.min(hfield.nrow - 1, Math.floor(row)));
    const col1 = Math.min(hfield.ncol - 1, col0 + 1);
    const row1 = Math.min(hfield.nrow - 1, row0 + 1);
    const tx = col - col0;
    const ty = row - row0;

    const h00 = hfield.data[row0 * hfield.ncol + col0];
    const h10 = hfield.data[row0 * hfield.ncol + col1];
    const h01 = hfield.data[row1 * hfield.ncol + col0];
    const h11 = hfield.data[row1 * hfield.ncol + col1];
    const height =
      ((1 - tx) * (1 - ty) * h00 +
        tx * (1 - ty) * h10 +
        (1 - tx) * ty * h01 +
        tx * ty * h11) *
      hfield.zScale;

    return Math.max(0.0, localOrigin[2] - height - this.offset);
  }

  private sampleAnyHFieldHeight(
    originMj: [number, number, number],
    hfieldPoses: Array<{ hfield: HFieldConfig; pose: FramePose }>
  ): number | null {
    for (const entry of hfieldPoses) {
      const height = this.sampleHFieldHeight(originMj, entry.pose, entry.hfield);
      if (height !== null) {
        return height;
      }
    }
    return null;
  }

  private getFramePose(mjData: MjData, frameType: FrameType, frameIndex: number): FramePose | null {
    const pos = this.readVec3(
      frameType === 'body'
        ? mjData.xpos
        : frameType === 'site'
          ? mjData.site_xpos
          : (mjData as MjData & { geom_xpos?: Float64Array }).geom_xpos,
      frameIndex
    );
    const mat = this.readMat3(
      frameType === 'body'
        ? mjData.xmat
        : frameType === 'site'
          ? mjData.site_xmat
          : (mjData as MjData & { geom_xmat?: Float64Array }).geom_xmat,
      frameIndex
    );

    if (pos === null || mat === null) {
      return null;
    }
    return { pos, mat };
  }

  private readVec3(buffer: ArrayLike<number> | undefined, index: number): [number, number, number] | null {
    if (buffer === undefined || buffer.length < index * 3 + 3) {
      return null;
    }
    return [
      Number(buffer[index * 3]),
      Number(buffer[index * 3 + 1]),
      Number(buffer[index * 3 + 2]),
    ];
  }

  private readQuat(
    buffer: ArrayLike<number> | undefined,
    index: number
  ): [number, number, number, number] | null {
    if (buffer === undefined || buffer.length < index * 4 + 4) {
      return null;
    }
    return [
      Number(buffer[index * 4]),
      Number(buffer[index * 4 + 1]),
      Number(buffer[index * 4 + 2]),
      Number(buffer[index * 4 + 3]),
    ];
  }

  private readMat3(
    buffer: ArrayLike<number> | undefined,
    index: number
  ): [number, number, number, number, number, number, number, number, number] | null {
    if (buffer === undefined || buffer.length < index * 9 + 9) {
      return null;
    }
    return [
      Number(buffer[index * 9]),
      Number(buffer[index * 9 + 1]),
      Number(buffer[index * 9 + 2]),
      Number(buffer[index * 9 + 3]),
      Number(buffer[index * 9 + 4]),
      Number(buffer[index * 9 + 5]),
      Number(buffer[index * 9 + 6]),
      Number(buffer[index * 9 + 7]),
      Number(buffer[index * 9 + 8]),
    ];
  }

  private normalizeFrameType(value: unknown): FrameType {
    return value === 'site' || value === 'geom' ? value : 'body';
  }

  private normalizeAlignment(value: unknown): 'base' | 'yaw' | 'world' {
    return value === 'base' || value === 'world' ? value : 'yaw';
  }

  private normalizeNumber(value: unknown, fallback: number): number {
    return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
  }

  private getModelBodyNames(mjModel: import('mujoco').MjModel): string[] {
    const namesArray = new Uint8Array(mjModel.names);
    const decoder = new TextDecoder();
    const names: string[] = [];
    for (let bodyId = 0; bodyId < mjModel.nbody; bodyId++) {
      let start = mjModel.name_bodyadr[bodyId];
      let end = start;
      while (end < namesArray.length && namesArray[end] !== 0) {
        end++;
      }
      names.push(decoder.decode(namesArray.subarray(start, end)));
    }
    return names;
  }

  private normalizeScale(value: unknown, size: number): Float32Array | null {
    if (typeof value === 'number') {
      const out = new Float32Array(size);
      out.fill(value);
      return out;
    }
    if (!Array.isArray(value)) {
      return null;
    }
    const out = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      out[i] = typeof value[i] === 'number' ? value[i] : 1.0;
    }
    return out;
  }

  private normalizeClipRange(value: unknown): [number, number] | null {
    if (!Array.isArray(value) || value.length < 2) {
      return null;
    }
    const clipMin = this.normalizeNumber(value[0], -Infinity);
    const clipMax = this.normalizeNumber(value[1], Infinity);
    return [clipMin, clipMax];
  }

  private buildGridOffsets(patternSize: unknown, resolutionValue: unknown): Array<[number, number, number]> {
    const size = Array.isArray(patternSize) ? patternSize : [1.6, 1.0];
    const sizeX = this.normalizeNumber(size[0], 1.6);
    const sizeY = this.normalizeNumber(size[1], 1.0);
    const resolution = Math.max(1e-6, this.normalizeNumber(resolutionValue, 0.1));
    const xValues = this.buildAxis(-sizeX / 2, sizeX / 2, resolution);
    const yValues = this.buildAxis(-sizeY / 2, sizeY / 2, resolution);
    const offsets: Array<[number, number, number]> = [];
    for (const y of yValues) {
      for (const x of xValues) {
        offsets.push([x, y, 0.0]);
      }
    }
    return offsets;
  }

  private buildAxis(start: number, end: number, step: number): number[] {
    const values: number[] = [];
    for (let value = start; value <= end + step * 0.5; value += step) {
      values.push(value);
    }
    return values;
  }

  private buildDirections(directionValue: unknown, count: number): Array<[number, number, number]> {
    const direction = Array.isArray(directionValue) ? directionValue : [0.0, 0.0, -1.0];
    const x = this.normalizeNumber(direction[0], 0.0);
    const y = this.normalizeNumber(direction[1], 0.0);
    const z = this.normalizeNumber(direction[2], -1.0);
    const norm = Math.hypot(x, y, z) || 1.0;
    const base: [number, number, number] = [x / norm, y / norm, z / norm];
    return Array.from({ length: count }, () => base);
  }

  private resolveFrameIndex(
    mjModel: import('mujoco').MjModel,
    frameType: FrameType,
    frameRefNameValue: unknown
  ): number {
    const frameRefName = typeof frameRefNameValue === 'string' ? frameRefNameValue : '';
    const candidateNames = this.buildCandidateNames(frameRefName);
    if (candidateNames.length === 0) {
      return 0;
    }

    const addresses = frameType === 'body'
      ? mjModel.name_bodyadr
      : frameType === 'site'
        ? mjModel.name_siteadr
        : mjModel.name_geomadr;
    const count = frameType === 'body'
      ? mjModel.nbody
      : frameType === 'site'
        ? mjModel.nsite
        : mjModel.ngeom;

    const namesArray = new Uint8Array(mjModel.names);
    const decoder = new TextDecoder();
    for (let i = 0; i < count; i++) {
      let start = addresses[i];
      let end = start;
      while (end < namesArray.length && namesArray[end] !== 0) {
        end++;
      }
      const name = decoder.decode(namesArray.subarray(start, end));
      if (candidateNames.includes(name)) {
        return i;
      }
    }
    return 0;
  }

  private buildCandidateNames(frameRefName: string): string[] {
    if (!frameRefName) {
      return [];
    }
    const names = new Set<string>([frameRefName]);
    const slash = frameRefName.lastIndexOf('/');
    if (slash >= 0) {
      names.add(frameRefName.slice(slash + 1));
    }
    return Array.from(names);
  }

  private computeAlignmentRotation(
    frameMat: [number, number, number, number, number, number, number, number, number]
  ): [number, number, number, number, number, number, number, number, number] {
    if (this.rayAlignment === 'base') {
      return frameMat;
    }
    if (this.rayAlignment === 'world') {
      return [1, 0, 0, 0, 1, 0, 0, 0, 1];
    }

    let xProj: [number, number, number] = [frameMat[0], frameMat[3], 0.0];
    let xNorm = Math.hypot(xProj[0], xProj[1]);
    if (xNorm < 0.1) {
      const yProjNorm = Math.max(Math.hypot(frameMat[1], frameMat[4]), 1e-6);
      xProj = [frameMat[4] / yProjNorm, -frameMat[1] / yProjNorm, 0.0];
      xNorm = 1.0;
    }

    const invNorm = 1.0 / Math.max(xNorm, 1e-6);
    xProj = [xProj[0] * invNorm, xProj[1] * invNorm, 0.0];

    return [
      xProj[0], -xProj[1], 0.0,
      xProj[1], xProj[0], 0.0,
      0.0, 0.0, 1.0,
    ];
  }

  private mulMat3Vec3(
    mat: [number, number, number, number, number, number, number, number, number],
    vec: [number, number, number]
  ): [number, number, number] {
    return [
      mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2],
      mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2],
      mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2],
    ];
  }

  private mulMat3(
    a: [number, number, number, number, number, number, number, number, number],
    b: [number, number, number, number, number, number, number, number, number]
  ): [number, number, number, number, number, number, number, number, number] {
    return [
      a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
      a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
      a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
      a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
      a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
      a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
      a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
      a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
      a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
    ];
  }

  private worldToLocal(
    pos: [number, number, number],
    mat: [number, number, number, number, number, number, number, number, number],
    point: [number, number, number]
  ): [number, number, number] {
    const dx = point[0] - pos[0];
    const dy = point[1] - pos[1];
    const dz = point[2] - pos[2];
    return [
      mat[0] * dx + mat[3] * dy + mat[6] * dz,
      mat[1] * dx + mat[4] * dy + mat[7] * dz,
      mat[2] * dx + mat[5] * dy + mat[8] * dz,
    ];
  }

  private quatToMat(
    quat: [number, number, number, number]
  ): [number, number, number, number, number, number, number, number, number] {
    const [w, x, y, z] = quat;
    const xx = x * x;
    const yy = y * y;
    const zz = z * z;
    const xy = x * y;
    const xz = x * z;
    const yz = y * z;
    const wx = w * x;
    const wy = w * y;
    const wz = w * z;
    return [
      1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
      2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
      2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
    ];
  }

  private addVec3(
    a: [number, number, number],
    b: [number, number, number]
  ): [number, number, number] {
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  }

}

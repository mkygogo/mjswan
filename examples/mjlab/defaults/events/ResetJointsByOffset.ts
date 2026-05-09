import { EventBase, type EventConfig, type EventContext } from 'mjswan/event';

type ScalarRange = [number, number];

export class ResetJointsByOffset extends EventBase {
  private readonly entityName: string | null;
  private readonly jointNames: string[] | null;
  private readonly jointIds: number[] | null;
  private readonly positionRange: ScalarRange;
  private readonly velocityRange: ScalarRange;

  constructor(config: EventConfig) {
    super(config);
    this.entityName = typeof config.params?.entity_name === 'string' ? config.params.entity_name : null;
    this.jointNames = Array.isArray(config.params?.joint_names)
      ? config.params.joint_names.filter((value): value is string => typeof value === 'string')
      : null;
    this.jointIds = Array.isArray(config.params?.joint_ids)
      ? config.params.joint_ids.filter((value): value is number => typeof value === 'number')
      : null;
    this.positionRange = this.normalizeRange(config.params?.position_range);
    this.velocityRange = this.normalizeRange(config.params?.velocity_range);
  }

  onReset(context: EventContext): void {
    const { mjModel, mjData } = context;
    if (!mjModel || !mjData) return;

    const jointIndices = this.resolveJointIndices(mjModel);
    for (const jointIdx of jointIndices) {
      const jointType = mjModel.jnt_type[jointIdx];
      if (jointType !== 2 && jointType !== 3) {
        continue;
      }

      const qposAdr = mjModel.jnt_qposadr[jointIdx];
      const qvelAdr = mjModel.jnt_dofadr[jointIdx];
      mjData.qpos[qposAdr] += this.sampleRange(this.positionRange);
      mjData.qvel[qvelAdr] += this.sampleRange(this.velocityRange);

      if (mjModel.jnt_limited[jointIdx]) {
        const rangeAdr = jointIdx * 2;
        const lower = mjModel.jnt_range[rangeAdr];
        const upper = mjModel.jnt_range[rangeAdr + 1];
        mjData.qpos[qposAdr] = Math.min(Math.max(mjData.qpos[qposAdr], lower), upper);
      }
    }
  }

  private normalizeRange(value: unknown): ScalarRange {
    if (
      Array.isArray(value) &&
      value.length >= 2 &&
      typeof value[0] === 'number' &&
      typeof value[1] === 'number'
    ) {
      return [value[0], value[1]];
    }
    return [0, 0];
  }

  private sampleRange(range: ScalarRange): number {
    const [min, max] = range;
    return min + Math.random() * (max - min);
  }

  private resolveJointIndices(mjModel: import('mujoco').MjModel): number[] {
    if (this.jointIds && this.jointIds.length > 0) {
      return this.jointIds.filter((idx) => idx >= 0 && idx < mjModel.njnt);
    }

    const modelJointNames = this.getModelJointNames(mjModel);
    if (this.jointNames && this.jointNames.length > 0) {
      return this.jointNames
        .map((name) => this.findJointIndex(modelJointNames, name))
        .filter((idx): idx is number => idx !== null);
    }

    return modelJointNames
      .map((name, idx) => ({ name, idx }))
      .filter(({ name, idx }) => {
        if (mjModel.jnt_type[idx] === 0) {
          return false;
        }
        if (!this.entityName) {
          return true;
        }
        return name === this.entityName || name.startsWith(`${this.entityName}/`);
      })
      .map(({ idx }) => idx);
  }

  private findJointIndex(modelJointNames: string[], targetName: string): number | null {
    const exactIdx = modelJointNames.indexOf(targetName);
    if (exactIdx >= 0) {
      return exactIdx;
    }

    if (this.entityName) {
      const qualified = `${this.entityName}/${targetName}`;
      const qualifiedIdx = modelJointNames.indexOf(qualified);
      if (qualifiedIdx >= 0) {
        return qualifiedIdx;
      }
    }

    const suffixMatches = modelJointNames
      .map((name, idx) => ({ name, idx }))
      .filter(({ name }) => name === targetName || name.endsWith(`/${targetName}`));
    if (suffixMatches.length === 1) {
      return suffixMatches[0].idx;
    }
    if (this.entityName) {
      const scoped = suffixMatches.find(
        ({ name }) => name === this.entityName || name.startsWith(`${this.entityName}/`)
      );
      if (scoped) {
        return scoped.idx;
      }
    }
    return null;
  }

  private getModelJointNames(mjModel: import('mujoco').MjModel): string[] {
    const namesArray = new Uint8Array(mjModel.names);
    const decoder = new TextDecoder();
    const names: string[] = [];
    for (let jointIdx = 0; jointIdx < mjModel.njnt; jointIdx++) {
      let start = mjModel.name_jntadr[jointIdx];
      let end = start;
      while (end < namesArray.length && namesArray[end] !== 0) {
        end++;
      }
      names.push(decoder.decode(namesArray.subarray(start, end)));
    }
    return names;
  }
}

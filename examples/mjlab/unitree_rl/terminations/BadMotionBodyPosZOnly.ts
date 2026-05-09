import { getCommandManager, isTrackingSource } from 'mjswan/command';
import { TerminationBase, type TerminationConfig } from 'mjswan/termination';
import type { PolicyState } from 'mjswan/types';
import { getBodyIdByName } from './utils';

export class BadMotionBodyPosZOnly extends TerminationBase {
  private readonly threshold: number;
  private readonly bodyNames: string[] | null;

  constructor(config: TerminationConfig) {
    super(config);
    this.threshold = (config.params?.threshold as number | undefined) ?? Infinity;
    this.bodyNames = Array.isArray(config.params?.body_names)
      ? config.params!.body_names.filter((value): value is string => typeof value === 'string')
      : null;
  }

  evaluate(_state: PolicyState): boolean {
    const tracking = getCommandManager().getTerm('motion');
    const context = getCommandManager().getContext();
    const mjModel = context?.mjModel ?? null;
    const mjData = context?.mjData ?? null;
    if (!isTrackingSource(tracking) || !tracking.isReady() || !mjModel || !mjData) {
      return false;
    }

    const bodyNames = this.bodyNames && this.bodyNames.length > 0
      ? this.bodyNames
      : tracking.getBodyNames();
    if (bodyNames.length === 0) {
      return false;
    }

    const refBodyPosW = tracking.getBodyPosW();
    if (!refBodyPosW) {
      return false;
    }

    const allTrackingBodies = tracking.getBodyNames();
    for (const bodyName of bodyNames) {
      const bodySlot = allTrackingBodies.indexOf(bodyName);
      const bodyId = getBodyIdByName(mjModel, bodyName);
      if (bodySlot < 0 || bodyId < 0) {
        continue;
      }

      const refOffset = bodySlot * 3;
      const refZ = refBodyPosW[refOffset + 2] ?? 0.0;
      const currentZ = mjData.xpos[bodyId * 3 + 2] ?? 0.0;

      if (Math.abs(refZ - currentZ) > this.threshold) {
        return true;
      }
    }

    return false;
  }
}

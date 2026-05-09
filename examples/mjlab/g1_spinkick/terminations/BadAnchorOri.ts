import { getCommandManager, isTrackingSource } from 'mjswan/command';
import { TerminationBase, type TerminationConfig } from 'mjswan/termination';
import type { PolicyState } from 'mjswan/types';
import { getBodyIdByName, quatApplyInv } from './utils';

export class BadAnchorOri extends TerminationBase {
  private readonly threshold: number;

  constructor(config: TerminationConfig) {
    super(config);
    this.threshold = (config.params?.threshold as number | undefined) ?? Infinity;
  }

  evaluate(_state: PolicyState): boolean {
    const tracking = getCommandManager().getTerm('motion');
    const context = getCommandManager().getContext();
    const mjModel = context?.mjModel ?? null;
    const mjData = context?.mjData ?? null;
    if (!isTrackingSource(tracking) || !tracking.isReady() || !mjModel || !mjData) {
      return false;
    }
    const anchorQuat = tracking.getAnchorQuat();
    const anchorName = tracking.getAnchorBodyName();
    if (!anchorQuat || anchorQuat.length < 4 || !anchorName) {
      return false;
    }
    const anchorId = getBodyIdByName(mjModel, anchorName);
    if (anchorId < 0) {
      return false;
    }
    const currentAnchorQuat = mjData.xquat.slice(anchorId * 4, anchorId * 4 + 4);
    const gravity: [number, number, number] = [0.0, 0.0, -1.0];
    const motionGravity = quatApplyInv(anchorQuat, gravity);
    const robotGravity = quatApplyInv(currentAnchorQuat, gravity);
    return Math.abs(motionGravity[2] - robotGravity[2]) > this.threshold;
  }
}

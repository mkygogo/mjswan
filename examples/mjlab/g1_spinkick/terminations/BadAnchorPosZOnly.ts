import { getCommandManager, isTrackingSource } from 'mjswan/command';
import { TerminationBase, type TerminationConfig } from 'mjswan/termination';
import type { PolicyState } from 'mjswan/types';
import { getBodyIdByName } from './utils';

export class BadAnchorPosZOnly extends TerminationBase {
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
    const anchorPos = tracking.getAnchorPos();
    const anchorName = tracking.getAnchorBodyName();
    if (!anchorPos || anchorPos.length < 3 || !anchorName) {
      return false;
    }
    const anchorId = getBodyIdByName(mjModel, anchorName);
    if (anchorId < 0) {
      return false;
    }
    const currentAnchorZ = mjData.xpos[anchorId * 3 + 2] ?? 0.0;
    return Math.abs(anchorPos[2] - currentAnchorZ) > this.threshold;
  }
}

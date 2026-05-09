import { TerminationBase, type TerminationConfig } from 'mjswan/termination';
import type { PolicyState } from 'mjswan/types';

export class BaseAngVelExceed extends TerminationBase {
  private readonly threshold: number;

  constructor(config: TerminationConfig) {
    super(config);
    this.threshold = (config.params?.threshold as number | undefined) ?? Infinity;
  }

  evaluate(state: PolicyState): boolean {
    const rootAngVel = state.rootAngVel;
    if (!rootAngVel || rootAngVel.length < 3) {
      return false;
    }
    return Math.abs(rootAngVel[0]) > this.threshold
      || Math.abs(rootAngVel[1]) > this.threshold
      || Math.abs(rootAngVel[2]) > this.threshold;
  }
}

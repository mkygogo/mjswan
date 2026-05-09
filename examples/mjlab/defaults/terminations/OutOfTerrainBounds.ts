import { TerminationBase, type TerminationConfig } from 'mjswan/termination';
import type { PolicyState } from 'mjswan/types';

/**
 * Terminate when the robot leaves the generated terrain footprint.
 *
 * Mirrors mjlab's ``out_of_terrain_bounds``.  The terrain bounds are
 * pre-computed at Python build time from the ``TerrainGeneratorCfg``
 * (``num_rows``, ``num_cols``, ``size``, ``margin``) and injected as
 * ``limit_x`` / ``limit_y`` in the config params.
 *
 * If the params are absent (e.g. no terrain generator), the termination
 * never fires.
 *
 * mjlab: tasks/velocity/mdp/terminations.out_of_terrain_bounds
 */
export class OutOfTerrainBounds extends TerminationBase {
  private readonly limitX: number;
  private readonly limitY: number;

  constructor(config: TerminationConfig) {
    super(config);
    this.limitX = (config.params?.limit_x as number | undefined) ?? Infinity;
    this.limitY = (config.params?.limit_y as number | undefined) ?? Infinity;
  }

  evaluate(state: PolicyState): boolean {
    const pos = state.rootPos;
    if (!pos || pos.length < 3) return false;
    return Math.abs(pos[0]) > this.limitX || Math.abs(pos[1]) > this.limitY;
  }
}

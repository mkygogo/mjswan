import { TerminationBase, type TerminationConfig } from 'mjswan/termination';
import type { PolicyState } from 'mjswan/types';

/**
 * Terminate when the robot's displacement from its spawn origin exceeds
 * the sub-terrain boundary (intended as a curriculum success signal).
 *
 * Mirrors mjlab's ``terrain_edge_reached``. The sub-terrain half-extents are
 * pre-computed at Python build time from ``TerrainGeneratorCfg.size`` and
 * injected as ``half_x`` / ``half_y``. The threshold fraction is read from the
 * config params when present.
 *
 * The spawn origin is captured from the first post-reset root position, so the
 * browser-side implementation does not need Python to build an extra scene just
 * to resolve env origins.
 *
 * The first two steps after reset are skipped to avoid stale-position
 * triggers — matching the mjlab implementation.
 *
 * If the params are absent (e.g. no terrain generator), the termination
 * never fires.
 *
 * mjlab: tasks/velocity/mdp/terminations.terrain_edge_reached
 */
export class TerrainEdgeReached extends TerminationBase {
  private readonly halfX: number;
  private readonly halfY: number;
  private readonly thresholdFraction: number;
  private spawnX: number | null = null;
  private spawnY: number | null = null;
  private stepCount = 0;

  constructor(config: TerminationConfig) {
    super(config);
    this.halfX = (config.params?.half_x as number | undefined) ?? Infinity;
    this.halfY = (config.params?.half_y as number | undefined) ?? Infinity;
    this.thresholdFraction = (config.params?.threshold_fraction as number | undefined) ?? 0.95;
  }

  evaluate(state: PolicyState): boolean {
    this.stepCount++;
    const pos = state.rootPos;
    if (!pos || pos.length < 3) return false;
    if (this.spawnX === null || this.spawnY === null) {
      this.spawnX = pos[0];
      this.spawnY = pos[1];
    }
    if (this.stepCount <= 2) return false;
    return (
      Math.abs(pos[0] - this.spawnX) > this.halfX * this.thresholdFraction ||
      Math.abs(pos[1] - this.spawnY) > this.halfY * this.thresholdFraction
    );
  }

  reset(): void {
    this.stepCount = 0;
    this.spawnX = null;
    this.spawnY = null;
  }
}

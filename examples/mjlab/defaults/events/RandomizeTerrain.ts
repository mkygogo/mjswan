import { EventBase, type EventConfig, type EventContext } from 'mjswan/event';

/**
 * mjlab's randomize_terrain mutates terrain state during reset.
 *
 * In mjswan the terrain mesh is baked into the exported MuJoCo scene, so there
 * is no browser-side terrain generator to resample here. The event is kept as
 * an explicit no-op to preserve config compatibility without warning spam.
 */
export class RandomizeTerrain extends EventBase {
  constructor(config: EventConfig) {
    super(config);
  }

  onReset(_context: EventContext): void {}
}

# AGENTS.md

## Project

mjswan packages browser-based MuJoCo simulations with real-time policy control into an interactive static web app. It has two sides:

- **Python** ([src/mjswan/](src/mjswan/)) — `Builder` / `Project` / `Scene` API that bundles models, policies, and UI config into a static site.
- **Frontend template** ([src/mjswan/template/](src/mjswan/template/)) — TypeScript + three.js + mujoco-wasm client that the Python build step bundles.

See [CONTEXT.md](CONTEXT.md) for a full codebase map, object model, module descriptions, and tooling reference.

## Philosophy

- Write clean, readable, maintainable code.
- Don't reinvent what already exists upstream. Prefer mjlab ([GitHub](https://github.com/mujocolab/mjlab), [local](.venv/lib/python3.12/site-packages/mjlab)) or other dependencies over new boilerplate in mjswan.

## Layout

- [src/mjswan/](src/mjswan/) — package source (builder, project, scene, adapters, CLI, managers).
- [src/mjswan/template/](src/mjswan/template/) — frontend source (Vite + React + three.js + mujoco-wasm).
- [examples/](examples/) — `demo`, `mjlab`, `colab`, `tutorial` runnable examples.
- [tests/](tests/) — pytest suite. `slow`-marked tests are opt-out (see below).
- [docs/](docs/) — zensical (MkDocs-based) site published to Read the Docs. Build with `make docs-build`; serve locally with `make docs-serve`.

## Python workflow

Use `uv` instead of bare `python`/`pip`. Prefer the [Makefile](Makefile) targets.

Tests are configured with `slow` opt-out: `make test` runs everything, but pre-commit runs `pytest -m "not slow"` for speed.

## StereoSpatial / FFS integration

This repo currently provides the MuJoCo simulation side of the StereoSpatial robot-control loop.
Keep the first-pass integration compatible with the existing Fast-FoundationStereo `camera_mujoco.py` protocol unless a coordinated protocol migration is being made.

### Frame bridge

When `FRAME_PORT` or `MJSWAN_FRAME_PORT` is set, `mjswanApp.launch()` starts `src/mjswan/frame_bridge.py`:

- TCP server: `FRAME_PORT` (normally `9876`) for FFS `--camera mujoco`.
- Browser WebSocket: `FRAME_WS_PORT` (default `FRAME_PORT + 1`, normally `9877`).
- Server-to-FFS frame protocol: little-endian `<III` header followed by left JPEG, right JPEG, and meta JSON.
- FFS-to-server commands: newline-delimited JSON (`nav_goal`, `nav_cancel`, `cam_look`).
- The browser runtime renders stereo JPEG frames and streams them to the bridge; the bridge forwards FFS commands back to the browser runtime.

Typical run command:

```bash
PORT=8080 FRAME_PORT=9876 ./start_demo.sh
```

Use `./start_demo.sh --rebuild` only after frontend/template or demo asset changes. A rebuild is slow because it rebuilds the Vite bundle and demo assets.

### Virtual stereo camera

The stereo camera implementation lives in `src/mjswan/template/src/core/engine/mujoco_frame_bridge.ts` and the visible rig lives in `runtime.ts`.

Current G1 defaults:

- image size: `1280x720`
- baseline: `0.120114m` (`0.060057m` half baseline)
- fovy: `46.8`
- logical views: `cam_left` / `cam_right`
- mount: top/head area, currently `MOUNT_POS = [0.02, 0.0, 0.57]` in MuJoCo coordinates
- `cam_look.yaw` is full 360 degrees (`[-pi, pi]`); pitch remains clamped to the prior safe range

The visible rig is intentionally simple: a small horizontal black bar with two black cylindrical lenses. It should remain level for visual inspection unless the user explicitly asks for head-attached pitch/roll visualization.

### Control flow

The minimal closed loop is:

1. mjswan browser renders left/right virtual camera JPEGs.
2. `frame_bridge.py` exposes them to FFS over TCP `9876`.
3. FFS computes point clouds and publishes them to StereoSpatial scene relay.
4. StereoSpatial RobotControl sends `nav_goal`, `nav_cancel`, `cam_look`, and `refresh_scene` through the relay.
5. FFS forwards navigation/camera commands to `camera_mujoco.py`.
6. `camera_mujoco.py` writes newline JSON commands to mjswan `frame_bridge.py`.
7. mjswan browser runtime applies velocity commands through the existing `velocity:lin_vel_x`, `velocity:lin_vel_y`, `velocity:ang_vel_z` command terms.

Keep this FFS-mediated control path for the minimal integration. A future optimization may connect StereoSpatial directly to a mjswan control WebSocket, but do not mix that into small fixes.

### Verification checklist

- `cd src/mjswan/template && npm run build`
- `python3 -m py_compile src/mjswan/frame_bridge.py src/mjswan/app.py`
- Start `PORT=8080 FRAME_PORT=9876 ./start_demo.sh`.
- Confirm `ss -ltnp` shows `8080`, `9876`, and `9877`.
- Open the mjswan page and confirm `9877` has an established browser connection.
- Start FFS with `--camera mujoco --mujoco-port 9876` and confirm frames are read.


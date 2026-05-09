# CONTEXT.md

## What mjswan is

mjswan is a Python framework (v0.5.6, Apache-2.0) that packages browser-based MuJoCo simulations with real-time ONNX policy control into interactive static web apps. It is published on both PyPI and npm, and demos are hosted on GitHub Pages.

Stack: Python (builder/config) + TypeScript/React/three.js/mujoco-wasm (browser client) + ONNX Runtime Web (policy inference).


## Repository layout

```
src/mjswan/          Python package source
  builder.py           Builder — top-level entry point
  app.py               mjswanApp — launch / serve built apps
  project.py           ProjectConfig / ProjectHandle
  scene.py             SceneConfig / SceneHandle
  policy.py            PolicyConfig / PolicyHandle
  motion.py            MotionConfig / MotionHandle
  splat.py             SplatConfig / SplatHandle (Gaussian Splat)
  command.py           Command terms (Slider, Button, Checkbox, velocity_command)
  viewer_config.py     ViewerConfig
  _cli.py              CLI entry points (main, simple, mjlab, serve)
  _build_client.py     Frontend build orchestration (npm/vite)
  adapters/            mjlab soft-dependency adapter + compat helpers
  envs/mdp/            MDP building blocks (actions, events, obs, terminations)
  managers/            Observation / event / action / termination managers
  template/            TypeScript frontend (Vite + React + three.js + mujoco-wasm)

examples/            Runnable examples
  demo/                main demo (deployed to GitHub Pages)
  mjlab/               mjlab-compatible examples (G1, MyoSuite, unitree_rl, defaults)
  colab/               Google Colab notebook example
  tutorial/            hello_world quickstart

tests/               pytest suite
docs/                zensical (MkDocs-based) site — published to Read the Docs
typings/             MuJoCo stub generator script
scripts/             Maintenance scripts (e.g., sync_contributors.py)
assets/              Demo GIF and banner SVG
```


## Python object model (fluent API)

```
Builder(base_path, gtm_id, mt, debug)
  └── .add_project(name, id) → ProjectHandle
        └── .add_scene(name, model|spec, metadata) → SceneHandle
              ├── .add_policy(name, policy, ...) → PolicyHandle
              │     ├── .add_velocity_command(...) → PolicyHandle
              │     └── .add_motion(...) / .add_motion_from_wandb(...) → MotionHandle
              ├── .add_splat(name, source|url, ...) → SplatHandle
              └── .set_viewer_config(ViewerConfig)

builder.build(output_dir) → mjswanApp
mjswanApp.launch(host, port, open_browser)   # blocking; Colab-aware
```

Each `*Handle` wraps a `*Config` dataclass. Handles expose a fluent API; configs hold the serializable state that `Builder.build()` turns into a static site.


## Key modules

### `builder.py` — `Builder`
Main entry point. Accumulates `ProjectConfig` objects and calls `ClientBuilder` to invoke the Vite frontend build, then writes `config.json` + model/policy/motion/splat assets into the output directory as a ZIP-based static bundle.

### `app.py` — `mjswanApp`
Wraps a built `dist/` directory. `launch()` starts a stdlib HTTP server (COOP/COEP headers required for SharedArrayBuffer / MuJoCo WASM threading); detects Google Colab and displays an inline iframe instead.

### `policy.py` — `PolicyConfig` / `PolicyHandle`
Holds an `onnx.ModelProto` plus observation groups, action terms, termination terms, commands, and motion references. Compatible with mjlab config classes via the adapter layer. Serialized to a per-policy `<name>.json` at build time.

### `command.py`
Defines command terms consumed by policies: `SliderConfig`, `ButtonConfig`, `CheckboxConfig`, `CommandTermConfig`, `CommandTermSpec`, `CommandUiConfig`. `velocity_command()` is a convenience factory for the standard locomotion 3-DoF velocity command. Custom command terms can be registered with `register_command_term`.

### `scene.py` — `SceneConfig` / `SceneHandle`
A scene owns one MuJoCo model (as `MjModel` → binary `.mjb` or `MjSpec` → XML), zero or more policies, and zero or more Gaussian splat backgrounds.

### `splat.py` — `SplatConfig` / `SplatHandle`
Configures a 3D Gaussian Splat (`.spz` format) background: scale, position offsets, Euler rotations, optional collider mesh URL.

### `viewer_config.py` — `ViewerConfig`
Camera parameters (lookat, distance, fovy, elevation, azimuth) + tracking mode (`OriginType`: AUTO / WORLD / ASSET_ROOT / ASSET_BODY). `ViewerConfig.from_position()` computes spherical params from a Cartesian viewer position.

### `adapters/`
- `mjlab_adapter.py`: Converts mjlab types (observations, actions, terminations, events, commands) to mjswan equivalents by name-based dynamic lookup — no hardcoded registries, no hard import of mjlab.
- `mjlab_compat.py`: Monkey-patches `MujocoCfg.apply_to_spec()` onto mjlab when needed.

### `envs/mdp/` and `managers/`
mjlab-compatible MDP layer for building observation groups, action terms (joint position / effort), event functions, and termination functions. Custom obs/event/termination functions are registered via `register_obs_func` / `register_event_func` / `register_termination_func`.

### `_build_client.py`
Orchestrates the Node.js / Vite frontend build. Manages a local `nodeenv` if Node isn't available system-wide.

### `wandb_utils.py`
Downloads motion `.npz` artifacts from Weights & Biases runs. Used by `PolicyHandle.add_motion_from_wandb()`.


## Frontend (`src/mjswan/template/`)

TypeScript + React + Vite + three.js. Built by `Builder.build()` via `_build_client.py`. The browser client:
- Loads the MuJoCo WASM module and runs physics in a Web Worker.
- Runs ONNX policies via onnxruntime-web.
- Renders via three.js (reflections, shadows, Gaussian Splat background).
- Supports WebXR (VR).
- Reads `config.json` to discover projects/scenes/policies at runtime.

Multi-threaded mode (`Builder(mt=True)`) requires COOP/COEP headers; the builder writes a `_headers` file (Netlify / Cloudflare Pages / Vercel) and a service-worker script (required for GitHub Pages).


## CLI entry points

| Command | Description |
|---------|-------------|
| `main`  | Run `examples/demo/main.py` |
| `simple`| Run `examples/demo/simple.py` |
| `mjlab` | Run `examples/mjlab/defaults/main.py` |
| `serve <dist-dir>` | Serve a pre-built `dist/` directory |


## Tooling and workflow

| Tool | Purpose |
|------|---------|
| `uv` | Dependency management and script runner — use instead of bare `python`/`pip` |
| `hatchling` | Build backend |
| `ruff` | Linting and formatting |
| `pyright` / `ty` | Type checking |
| `pytest` | Tests (`make test`) |
| `pre-commit` | Hooks: trailing-whitespace, ruff, pytest (not slow), eslint |
| `eslint` | TypeScript linting (frontend) |
| `zensical` | Docs site builder (MkDocs-based), built/served via `make docs-build` / `make docs-serve` |

Key Makefile targets: `sync`, `format`, `type`, `check`, `test`, `test-all`, `docs-build`, `docs-serve`.


## Test markers

- No marker — fast unit tests, always run.
- `@pytest.mark.slow` — triggers a full frontend (npm + Vite) build; excluded from pre-commit (`pytest -m "not slow"`). Run with `make test`.

CI (`pytest.yml`) runs `pytest -m "not slow"` across Python 3.10 / 3.11 / 3.12.


## Dependencies

Core: `mujoco==3.7.0`, `onnx>=1.20.0`, `nodeenv>=1.9.1`, `rich>=13.0.0`, `wandb>=0.23.1`.  
Dev extras: `pyright`, `ruff`, `pre-commit`, `pytest`.  
Examples extras: `mjlab`, `torch`, `robot-descriptions`, `playground`, `myosuite`, `gymnasium`.

Python 3.10–3.12 only (3.13+ excluded due to a transitive `labmaze` wheel gap).


## Deployment

The demo app is built by `examples/demo/main.py` and deployed to GitHub Pages via the `deploy.yml` workflow on every push to `main` that touches relevant paths. The `MJSWAN_BASE_PATH` and `MJSWAN_NO_LAUNCH` env vars control the build.

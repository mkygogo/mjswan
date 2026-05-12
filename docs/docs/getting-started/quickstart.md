---
icon: octicons/play-16
---

# Quickstart

This page gets you from zero to a running simulation in about two minutes.

## 1. Install

```bash
pip install mjswan
```

## 2. Write a build script

Create `hello_world.py`:

```python
import mujoco
import mjswan

builder = mjswan.Builder()
project = builder.add_project(name="Hello World")

spec = mujoco.MjSpec.from_string("""
<mujoco>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="box" size=".1 .2 .3" rgba="0 .9 0 1"/>
    </body>
  </worldbody>
</mujoco>
""")

project.add_scene(spec=spec, name="Falling Box")

app = builder.build()
app.launch()
```

## 3. Run it

```bash
python hello_world.py
```

mjswan will:

1. Build the application into a `dist/` directory next to your script
2. Start a local web server on `http://localhost:8080`
3. Open the URL in your default browser

You should see an interactive 3D view of a green box falling onto a red plane. Click and drag to orbit the camera. Press `Ctrl-C` in the terminal to stop the server.

<!-- MEDIA: suggest a screenshot of the browser showing the falling-box scene with camera orbit controls visible -->

## Next steps

- [Core Concepts](core-concepts.md) — understand the Builder → Project → Scene → Policy hierarchy
- [Examples](examples.md) — copy-paste patterns for policies, multiple projects, and more
- [CLI](cli.md) — the `mjswan` command: `view`, `serve`, `new`, `demo`, `info`
- [Deployment](../guides/deployment.md) — host your app on GitHub Pages or Netlify

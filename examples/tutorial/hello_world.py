"""Hello World

This is a simple "Hello World" example that demonstrates how to create a basic MuJoCo scene
using the mjswan viewer. It sets up a scene with a plane and a box above it, and launches the viewer.
"""

import mujoco

import mjswan


def main():
    builder = mjswan.Builder()
    hello_world_project = builder.add_project(name="Hello World")
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
    hello_world_project.add_scene(
        spec=spec,
        name="Box over Plane",
    )

    app = builder.build()
    app.launch()


if __name__ == "__main__":
    main()

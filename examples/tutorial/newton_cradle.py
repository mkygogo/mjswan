"""Newton's Cradle

A passive MuJoCo scene that demonstrates conservation of momentum and energy.
Five mirror-finish balls hang from two parallel U-shaped frames; each ball is
suspended by two strings (one to each frame) so it swings only along the row.
The leftmost ball starts pulled back and sets off the chain on load.
"""

import mujoco

import mjswan


def main():
    builder = mjswan.Builder()
    project = builder.add_project(name="Physics Experience")
    spec = mujoco.MjSpec.from_string("""
    <mujoco>
      <option timestep="0.0005" density="0" viscosity="0" integrator="RK4"
              impratio="10" cone="elliptic"/>

      <asset>
        <!-- Off-white/light-gray gradient skybox for the background. -->
        <texture type="skybox" builtin="gradient"
                 rgb1="0.92 0.92 0.94" rgb2="0.85 0.85 0.88"
                 width="256" height="256"/>
        <!-- Mirror-polished silver for the balls. -->
        <material name="mirror" rgba="0.95 0.95 0.97 1"
                  specular="1" shininess="1" reflectance="0.9"/>
        <!-- Brushed silver for the U-frames. -->
        <material name="frame_mat" rgba="0.78 0.78 0.82 1"
                  specular="0.7" shininess="0.6" reflectance="0.3"/>
        <!-- Bright silver for the strings. High diffuse + emission so thin
             capsules stay visible under a single overhead light. -->
        <material name="string_mat" rgba="0.85 0.85 0.88 1"
                  specular="0.5" shininess="0.4" reflectance="0.1"
                  emission="0.3"/>
      </asset>

      <default>
        <default class="ball">
          <!-- For maximum elasticity (closest MuJoCo gets to coefficient of
               restitution = 1): stiff solref (~2 timesteps), solimp dmax=1,
               zero friction, zero joint damping. RK4 integrator preserves
               energy better than the default semi-implicit Euler. -->
          <geom type="sphere" size="0.05" material="mirror"
                friction="0 0 0" solref="-10000 100" solimp="0.999 1 0.0001"/>
          <!-- Hinge axis along Y so balls swing along the row (X direction). -->
          <joint type="hinge" axis="0 1 0" damping="0"/>
        </default>
        <default class="frame">
          <geom type="capsule" size="0.015" material="frame_mat"
                contype="0" conaffinity="0"/>
        </default>
        <default class="string">
          <geom type="capsule" size="0.005" material="string_mat"
                contype="0" conaffinity="0"/>
        </default>
      </default>

      <worldbody>
        <light diffuse=".8 .8 .8" pos="0 0 2" dir="0 0 -1"/>
        <geom type="plane" size="2 2 0.1" rgba=".9 .9 .9 1"/>

        <!-- Two U-shaped ("コ") frames, opening downward, parallel along Y.
             Front frame at y=-0.30, back frame at y=+0.30 (2x previous spacing).
             Each: two vertical posts at x=±0.3 plus a top bar at z=0.5
             (half the previous height). -->
        <geom class="frame" fromto="-0.3 -0.30 0    -0.3 -0.30 0.5"/>
        <geom class="frame" fromto=" 0.3 -0.30 0     0.3 -0.30 0.5"/>
        <geom class="frame" fromto="-0.3 -0.30 0.5   0.3 -0.30 0.5"/>
        <geom class="frame" fromto="-0.3  0.30 0    -0.3  0.30 0.5"/>
        <geom class="frame" fromto=" 0.3  0.30 0     0.3  0.30 0.5"/>
        <geom class="frame" fromto="-0.3  0.30 0.5   0.3  0.30 0.5"/>

        <!-- Five balls, spaced exactly 2*radius apart so they touch at rest.
             Each body's frame sits on the row axis at z=0.5; the ball geom
             hangs 0.25 below. Two "strings" go from the ball up to the two
             frame top bars (at y=±0.30, z=0.5) for visual V-suspension. -->
        <body name="b1" pos="-0.2010 0 0.5">
          <joint name="j1" class="ball"/>
          <geom class="ball" pos="0 0 -0.25"/>
          <geom class="string" fromto="0 0 -0.25  0 -0.30 0"/>
          <geom class="string" fromto="0 0 -0.25  0  0.30 0"/>
        </body>
        <body name="b2" pos="-0.1005 0 0.5">
          <joint name="j2" class="ball"/>
          <geom class="ball" pos="0 0 -0.25"/>
          <geom class="string" fromto="0 0 -0.25  0 -0.30 0"/>
          <geom class="string" fromto="0 0 -0.25  0  0.30 0"/>
        </body>
        <body name="b3" pos=" 0.00 0 0.5">
          <joint name="j3" class="ball"/>
          <geom class="ball" pos="0 0 -0.25"/>
          <geom class="string" fromto="0 0 -0.25  0 -0.30 0"/>
          <geom class="string" fromto="0 0 -0.25  0  0.30 0"/>
        </body>
        <body name="b4" pos=" 0.1005 0 0.5">
          <joint name="j4" class="ball"/>
          <geom class="ball" pos="0 0 -0.25"/>
          <geom class="string" fromto="0 0 -0.25  0 -0.30 0"/>
          <geom class="string" fromto="0 0 -0.25  0  0.30 0"/>
        </body>
        <body name="b5" pos=" 0.2010 0 0.5">
          <joint name="j5" class="ball"/>
          <geom class="ball" pos="0 0 -0.25"/>
          <geom class="string" fromto="0 0 -0.25  0 -0.30 0"/>
          <geom class="string" fromto="0 0 -0.25  0  0.30 0"/>
        </body>
      </worldbody>

      <!-- Start with the leftmost ball pulled back ~46 degrees (0.8 rad).
           Larger angle than before because the arm is now half as long, so a
           bigger angle is needed to give the ball comparable swing energy. -->
      <keyframe>
        <key name="start" qpos="0.8 0 0 0 0"/>
      </keyframe>
    </mujoco>
    """)
    project.add_scene(spec=spec, name="Newton's Cradle")

    app = builder.build()
    app.launch()


if __name__ == "__main__":
    main()

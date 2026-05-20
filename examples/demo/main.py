"""mjswan Demo Application

This is a demo application showcasing the usage of mjswan.
The demo app is hosted on GitHub Pages: https://ttktjmt.github.io/mjswan/
"""

import os
import posixpath
from pathlib import Path

import gymnasium.logger as gym_logger
import mujoco
import mujoco.mjx as _mjx
import onnx
from mujoco_playground import registry

# Suppress gymnasium logger output from myosuite
_prev_gym_level = gym_logger.min_level
gym_logger.min_level = gym_logger.ERROR

from myosuite import gym_registry_specs  # noqa: E402
from myosuite.envs.myo import myochallenge  # noqa: E402, F401 - for env registration

gym_logger.min_level = _prev_gym_level

from robot_descriptions._descriptions import DESCRIPTIONS  # noqa: E402

import mjswan  # noqa: E402
from mjswan.envs.mdp import observations as obs_fns  # noqa: E402
from mjswan.envs.mdp import terminations as term_fns  # noqa: E402
from mjswan.envs.mdp.actions import (  # noqa: E402
    JointEffortActionCfg,
    JointPositionActionCfg,
)
from mjswan.managers.observation_manager import (  # noqa: E402
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjswan.managers.termination_manager import TerminationTermCfg  # noqa: E402


def _fix_unitree_mujoco_macos() -> None:
    """Pre-fix the unitree_mujoco cache on macOS to avoid case-sensitivity errors.

    On macOS (case-insensitive filesystem), robot_descriptions fails to checkout
    the unitree_mujoco repo because git history contains a rename from
    terrain.STL -> terrain.stl, which macOS treats as the same file.

    Fix: clone with --no-checkout so no files exist in the working tree before
    the target commit is checked out, and set core.ignorecase=false so git
    handles the case-rename correctly.
    """
    import platform
    import shutil
    import subprocess

    if platform.system() != "Darwin":
        return

    cache_dir = Path.home() / ".cache/robot_descriptions/unitree_mujoco"

    if cache_dir.exists():
        result = subprocess.run(
            ["git", "config", "core.ignorecase"],
            cwd=cache_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout.strip() == "false":
            return  # Already correctly configured
        shutil.rmtree(cache_dir)

    print("Preparing unitree_mujoco cache for macOS (one-time setup)...")
    subprocess.run(
        [
            "git",
            "clone",
            "--no-checkout",
            "https://github.com/unitreerobotics/unitree_mujoco.git",
            str(cache_dir),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "config", "core.ignorecase", "false"],
        cwd=cache_dir,
        check=True,
    )


# fmt: off
_G1_JOINT_SCALE = {
    "left_hip_pitch_joint":      0.5475464629911068,
    "left_hip_roll_joint":       0.35066146637882434,
    "left_hip_yaw_joint":        0.5475464629911068,
    "left_knee_joint":           0.35066146637882434,
    "left_ankle_pitch_joint":    0.43857731392336724,
    "left_ankle_roll_joint":     0.43857731392336724,
    "right_hip_pitch_joint":     0.5475464629911068,
    "right_hip_roll_joint":      0.35066146637882434,
    "right_hip_yaw_joint":       0.5475464629911068,
    "right_knee_joint":          0.35066146637882434,
    "right_ankle_pitch_joint":   0.43857731392336724,
    "right_ankle_roll_joint":    0.43857731392336724,
    "waist_yaw_joint":           0.5475464629911068,
    "waist_roll_joint":          0.43857731392336724,
    "waist_pitch_joint":         0.43857731392336724,
    "left_shoulder_pitch_joint": 0.43857731392336724,
    "left_shoulder_roll_joint":  0.43857731392336724,
    "left_shoulder_yaw_joint":   0.43857731392336724,
    "left_elbow_joint":          0.43857731392336724,
    "left_wrist_roll_joint":     0.43857731392336724,
    "left_wrist_pitch_joint":    0.07450087032950714,
    "left_wrist_yaw_joint":      0.07450087032950714,
    "right_shoulder_pitch_joint": 0.43857731392336724,
    "right_shoulder_roll_joint": 0.43857731392336724,
    "right_shoulder_yaw_joint":  0.43857731392336724,
    "right_elbow_joint":         0.43857731392336724,
    "right_wrist_roll_joint":    0.43857731392336724,
    "right_wrist_pitch_joint":   0.07450087032950714,
    "right_wrist_yaw_joint":     0.07450087032950714,
}
_G1_JOINT_STIFFNESS = {
    "left_hip_pitch_joint":      40.17923863450712,
    "left_hip_roll_joint":       99.09842777666111,
    "left_hip_yaw_joint":        40.17923863450712,
    "left_knee_joint":           99.09842777666111,
    "left_ankle_pitch_joint":    28.50124619574858,
    "left_ankle_roll_joint":     28.50124619574858,
    "right_hip_pitch_joint":     40.17923863450712,
    "right_hip_roll_joint":      99.09842777666111,
    "right_hip_yaw_joint":       40.17923863450712,
    "right_knee_joint":          99.09842777666111,
    "right_ankle_pitch_joint":   28.50124619574858,
    "right_ankle_roll_joint":    28.50124619574858,
    "waist_yaw_joint":           40.17923863450712,
    "waist_roll_joint":          28.50124619574858,
    "waist_pitch_joint":         28.50124619574858,
    "left_shoulder_pitch_joint": 14.25062309787429,
    "left_shoulder_roll_joint":  14.25062309787429,
    "left_shoulder_yaw_joint":   14.25062309787429,
    "left_elbow_joint":          14.25062309787429,
    "left_wrist_roll_joint":     14.25062309787429,
    "left_wrist_pitch_joint":    16.77832748089279,
    "left_wrist_yaw_joint":      16.77832748089279,
    "right_shoulder_pitch_joint": 14.25062309787429,
    "right_shoulder_roll_joint": 14.25062309787429,
    "right_shoulder_yaw_joint":  14.25062309787429,
    "right_elbow_joint":         14.25062309787429,
    "right_wrist_roll_joint":    14.25062309787429,
    "right_wrist_pitch_joint":   16.77832748089279,
    "right_wrist_yaw_joint":     16.77832748089279,
}
_G1_JOINT_DAMPING = {
    "left_hip_pitch_joint":      2.557889775413375,
    "left_hip_roll_joint":       6.308801853496639,
    "left_hip_yaw_joint":        2.557889775413375,
    "left_knee_joint":           6.308801853496639,
    "left_ankle_pitch_joint":    1.814445686584846,
    "left_ankle_roll_joint":     1.814445686584846,
    "right_hip_pitch_joint":     2.557889775413375,
    "right_hip_roll_joint":      6.308801853496639,
    "right_hip_yaw_joint":       2.557889775413375,
    "right_knee_joint":          6.308801853496639,
    "right_ankle_pitch_joint":   1.814445686584846,
    "right_ankle_roll_joint":    1.814445686584846,
    "waist_yaw_joint":           2.557889775413375,
    "waist_roll_joint":          1.814445686584846,
    "waist_pitch_joint":         1.814445686584846,
    "left_shoulder_pitch_joint": 0.907222843292423,
    "left_shoulder_roll_joint":  0.907222843292423,
    "left_shoulder_yaw_joint":   0.907222843292423,
    "left_elbow_joint":          0.907222843292423,
    "left_wrist_roll_joint":     0.907222843292423,
    "left_wrist_pitch_joint":    1.06814150219,
    "left_wrist_yaw_joint":      1.06814150219,
    "right_shoulder_pitch_joint": 0.907222843292423,
    "right_shoulder_roll_joint": 0.907222843292423,
    "right_shoulder_yaw_joint":  0.907222843292423,
    "right_elbow_joint":         0.907222843292423,
    "right_wrist_roll_joint":    0.907222843292423,
    "right_wrist_pitch_joint":   1.06814150219,
    "right_wrist_yaw_joint":     1.06814150219,
}
# fmt: on


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        print(f"[mjswan] Ignoring invalid {name}={value!r}; using {default}")
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _splat_name_from_url(url: str) -> str:
    name = url.rstrip("/").rsplit("/", 1)[-1]
    for suffix in (".ply", ".spz", ".splat", ".ksplat"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name or "CloudTwin"


def _splat_transform_for_name(name: str) -> dict[str, float]:
    presets = {
        # Center the public CloudTwin samples around the robot and put the
        # reconstruction's dominant height close to MuJoCo z=0. These are
        # only visual alignment defaults; the splat is not a physics collider.
        "sciart": {"scale": 0.08, "x": 33.46, "y": -63.96, "z": 44.67},
        "train": {"scale": 0.08, "x": 0.11, "y": 0.84, "z": -1.35},
        "building": {"scale": 0.08, "x": 0.0, "y": 0.0, "z": 0.0},
        "indoor_3dgs_compressed": {"scale": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }
    return presets.get(name.lower(), {"scale": 1.0, "x": 0.0, "y": 0.0, "z": 0.0})


def _iter_env_splat_urls() -> list[tuple[str, str]]:
    """Return external splats configured through env vars.

    MJSWAN_EXTRA_SPLAT_URL keeps the previous single-file behavior.
    MJSWAN_EXTRA_SPLAT_URLS accepts a comma-separated list.
    MJSWAN_EXTRA_SPLAT_BASE_URL auto-adds named files from one static server.
    """
    splats: list[tuple[str, str]] = []

    single_url = os.environ.get("MJSWAN_EXTRA_SPLAT_URL") or os.environ.get(
        "MJSWAN_G1_SPLAT_URL"
    )
    if single_url:
        splats.append(
            (
                os.environ.get(
                    "MJSWAN_EXTRA_SPLAT_NAME", _splat_name_from_url(single_url)
                ),
                single_url,
            )
        )

    for raw_url in os.environ.get("MJSWAN_EXTRA_SPLAT_URLS", "").split(","):
        url = raw_url.strip()
        if url:
            splats.append((_splat_name_from_url(url), url))

    base_url = os.environ.get("MJSWAN_EXTRA_SPLAT_BASE_URL")
    names = os.environ.get("MJSWAN_EXTRA_SPLAT_FILES")
    if base_url and names:
        base = base_url.rstrip("/")
        for raw_name in names.split(","):
            filename = raw_name.strip()
            if filename:
                splats.append((_splat_name_from_url(filename), f"{base}/{filename}"))

    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for name, url in splats:
        if url in seen:
            continue
        seen.add(url)
        unique.append((name, url))
    return unique


def _add_g1_splats(g1_scene) -> None:
    if not _env_bool("MJSWAN_DISABLE_STREET_SPLAT"):
        g1_scene.add_splat(
            name="Street",
            source="assets/unitree_g1/street.spz",
            scale=3.275,
            z_offset=0.708,
            yaw=40,
            control=True,
        )

    if not _env_bool("MJSWAN_DISABLE_INDOOR_SPLAT"):
        _splat_host = os.environ.get("MJSWAN_SPLAT_HOST", "192.168.3.38")
        _splat_port = os.environ.get("MJSWAN_SPLAT_PORT", "8091")
        g1_scene.add_splat(
            name="Indoor",
            url=f"http://{_splat_host}:{_splat_port}/splats/indoor_3dgs_compressed.ply",
            scale=1.0,
            x_offset=0.0,
            y_offset=0.0,
            z_offset=0.0,
            control=True,
        )

    for name, url in _iter_env_splat_urls():
        preset = _splat_transform_for_name(name)
        print(f"[mjswan] Adding external G1 splat {name}: {url}")
        g1_scene.add_splat(
            name=name,
            url=url,
            scale=_env_float("MJSWAN_EXTRA_SPLAT_SCALE", preset["scale"]),
            x_offset=_env_float("MJSWAN_EXTRA_SPLAT_X", preset["x"]),
            y_offset=_env_float("MJSWAN_EXTRA_SPLAT_Y", preset["y"]),
            z_offset=_env_float("MJSWAN_EXTRA_SPLAT_Z", preset["z"]),
            roll=_env_float("MJSWAN_EXTRA_SPLAT_ROLL", 0.0),
            pitch=_env_float("MJSWAN_EXTRA_SPLAT_PITCH", 0.0),
            yaw=_env_float("MJSWAN_EXTRA_SPLAT_YAW", 0.0),
            control=True,
        )

    g1_scene.add_splat_section()


def _add_g1_scene(project) -> None:
    g1_scene = project.add_scene(
        spec=mujoco.MjSpec.from_file("assets/unitree_g1/scene.xml"),
        name="G1",
    ).set_viewer_config(
        mjswan.ViewerConfig(
            lookat=(0.0, 0.0, 0.7),
            distance=4.3,
            elevation=-33.0,
            azimuth=-34.0,
            origin_type=mjswan.ViewerConfig.OriginType.ASSET_BODY,
            body_name="torso_link",
        )
    )
    _add_g1_splats(g1_scene)

    g1_actions = {
        "joint_pos": JointPositionActionCfg(
            scale=_G1_JOINT_SCALE,
            stiffness=_G1_JOINT_STIFFNESS,
            damping=_G1_JOINT_DAMPING,
        )
    }
    g1_terminations = {
        "bad_orientation": TerminationTermCfg(
            func=term_fns.bad_orientation, params={"limit_angle": 1.0}
        ),
        "root_height_below_minimum": TerminationTermCfg(
            func=term_fns.root_height_below_minimum, params={"minimum_height": 0.3}
        ),
    }

    g1_loco_policy = g1_scene.add_policy(
        policy=onnx.load("assets/unitree_g1/locomotion.onnx"),
        name="Locomotion",
        config_path="assets/unitree_g1/locomotion.json",
        actions=g1_actions,
        terminations=g1_terminations,
        observations={
            "policy": ObservationGroupCfg(
                terms={
                    "base_lin_vel": ObservationTermCfg(func=obs_fns.base_lin_vel),
                    "base_ang_vel": ObservationTermCfg(func=obs_fns.base_ang_vel),
                    "projected_gravity": ObservationTermCfg(
                        func=obs_fns.projected_gravity
                    ),
                    "joint_pos": ObservationTermCfg(
                        func=obs_fns.joint_pos_rel, params={"pos_steps": [0]}
                    ),
                    "joint_vel": ObservationTermCfg(func=obs_fns.joint_vel_rel),
                    "last_action": ObservationTermCfg(func=obs_fns.last_action),
                    "velocity_cmd": ObservationTermCfg(
                        func=obs_fns.generated_commands,
                        params={"command_name": "velocity"},
                    ),
                }
            )
        },
    )
    g1_loco_policy.add_velocity_command(
        lin_vel_x=(-1.5, 1.5),
        lin_vel_y=(-0.5, 0.5),
        default_lin_vel_x=0.5,
        default_lin_vel_y=0.0,
    )

    g1_scene.add_policy(
        policy=onnx.load("assets/unitree_g1/balance.onnx"),
        name="Balance",
        config_path="assets/unitree_g1/balance.json",
        actions=g1_actions,
        terminations=g1_terminations,
        observations={
            "observation": ObservationGroupCfg(
                terms={
                    "base_ang_vel": ObservationTermCfg(
                        func=obs_fns.base_ang_vel, history_length=1
                    ),
                    "projected_gravity": ObservationTermCfg(
                        func=obs_fns.projected_gravity_isaac,
                        history_length=1,
                        params={"gravity": [0, 0, -1.0]},
                    ),
                    "joint_pos": ObservationTermCfg(
                        func=obs_fns.joint_positions_isaac, history_length=1
                    ),
                    "joint_vel": ObservationTermCfg(
                        func=obs_fns.joint_vel_rel,
                        params={"joint_names": "isaac"},
                        history_length=1,
                    ),
                    "prev_actions": ObservationTermCfg(func=obs_fns.previous_actions),
                }
            )
        },
    )


def _add_go2_scene(project) -> None:
    go2_scene = project.add_scene(
        name="Go2",
        spec=mujoco.MjSpec.from_file("assets/unitree_go2/scene.xml"),
    ).set_viewer_config(
        mjswan.ViewerConfig(
            lookat=(0.0, 0.0, 0.7),
            distance=3.8,
            elevation=-20.0,
            azimuth=34.0,
            origin_type=mjswan.ViewerConfig.OriginType.ASSET_BODY,
            body_name="base",
        )
    )

    go2_actions = {
        "joint_pos": JointPositionActionCfg(
            scale=0.5,
            stiffness=25.0,
            damping=0.5,
        )
    }
    go2_velocity_obs = {
        "policy": ObservationGroupCfg(
            terms={
                "projected_gravity": ObservationTermCfg(
                    func=obs_fns.projected_gravity_isaac, history_length=3
                ),
                "joint_pos": ObservationTermCfg(
                    func=obs_fns.joint_positions_isaac, history_length=3
                ),
                "joint_vel": ObservationTermCfg(
                    func=obs_fns.joint_vel_rel,
                    params={"joint_names": "isaac"},
                    history_length=3,
                ),
                "prev_actions": ObservationTermCfg(
                    func=obs_fns.previous_actions,
                    history_length=3,
                    params={"transpose": True},
                ),
            }
        ),
        "command_": ObservationGroupCfg(
            terms={
                "velocity_cmd": ObservationTermCfg(
                    func=obs_fns.velocity_command_with_oscillators
                ),
            }
        ),
    }

    go2_scene.add_policy(
        name="Facet",
        policy=onnx.load("assets/unitree_go2/facet.onnx"),
        config_path="assets/unitree_go2/facet.json",
        actions=go2_actions,
        observations={
            "policy": ObservationGroupCfg(
                terms={
                    "projected_gravity": ObservationTermCfg(
                        func=obs_fns.projected_gravity_isaac, history_length=3
                    ),
                    "joint_pos": ObservationTermCfg(
                        func=obs_fns.joint_positions_isaac,
                        history_length=3,
                        params={"subtract_default": False},
                    ),
                    "joint_vel": ObservationTermCfg(
                        func=obs_fns.joint_vel_rel,
                        params={"joint_names": "isaac"},
                        history_length=3,
                    ),
                    "prev_actions": ObservationTermCfg(
                        func=obs_fns.previous_actions,
                        history_length=3,
                        params={"transpose": True},
                    ),
                }
            ),
            "command": ObservationGroupCfg(
                terms={
                    "impedance_cmd": ObservationTermCfg(func=obs_fns.impedance_command),
                }
            ),
        },
    ).add_velocity_command()

    go2_scene.add_policy(
        policy=onnx.load("assets/unitree_go2/vanilla.onnx"),
        name="Vanilla",
        config_path="assets/unitree_go2/vanilla.json",
        actions=go2_actions,
        observations=go2_velocity_obs,
    ).add_velocity_command()

    go2_scene.add_policy(
        policy=onnx.load("assets/unitree_go2/robust.onnx"),
        name="Robust",
        config_path="assets/unitree_go2/robust.json",
        actions=go2_actions,
        observations=go2_velocity_obs,
    ).add_velocity_command()


def _add_go1_scene(project) -> None:
    go1_scene = project.add_scene(
        spec=mujoco.MjSpec.from_file("assets/unitree_go1/go1.xml"),
        name="Go1",
    ).set_viewer_config(
        mjswan.ViewerConfig(
            lookat=(0.0, 0.0, 0.2),
            distance=3.1,
            elevation=-25.0,
            azimuth=-45.0,
            origin_type=mjswan.ViewerConfig.OriginType.ASSET_BODY,
            body_name="trunk",
        )
    )

    # NOTE: himloco uses an interleaved history format (dict with "interleaved": true)
    # that is not yet expressible via ObservationGroupCfg. observations remains in himloco.json.
    go1_scene.add_policy(
        policy=onnx.load("assets/unitree_go1/himloco.onnx"),
        name="HiMLoco",
        config_path="assets/unitree_go1/himloco.json",
        actions={
            "joint_pos": JointPositionActionCfg(
                scale=0.25,
                stiffness=40.0,
                damping=1.0,
            )
        },
    ).add_velocity_command()

    go1_scene.add_policy(
        policy=onnx.load("assets/unitree_go1/decap.onnx"),
        name="Decap",
        config_path="assets/unitree_go1/decap.json",
        actions={
            "effort": JointEffortActionCfg(
                scale=8.0,
                stiffness=20.0,
                damping=0.5,
            )
        },
        observations={
            "obs_history": ObservationGroupCfg(
                terms={
                    "projected_gravity": ObservationTermCfg(
                        func=obs_fns.projected_gravity_isaac, history_length=1
                    ),
                    "velocity_cmd": ObservationTermCfg(
                        func=obs_fns.generated_commands,
                        params={"command_name": "velocity"},
                        scale=(2.0, 2.0, 0.25),
                    ),
                    "joint_pos": ObservationTermCfg(
                        func=obs_fns.joint_positions_isaac, history_length=1
                    ),
                    "joint_vel": ObservationTermCfg(
                        func=obs_fns.joint_vel_rel,
                        params={"joint_names": "isaac"},
                        scale=0.05,
                        history_length=1,
                    ),
                    "prev_actions": ObservationTermCfg(func=obs_fns.previous_actions),
                }
            )
        },
    ).add_velocity_command()


def _add_anymal_c_scene(project) -> None:
    anymal_c_scene = project.add_scene(
        name="ANYmal C Velocity",
        spec=mujoco.MjSpec.from_zip("assets/anymal_c_velocity/scene.mjz"),
    )
    anymal_c_scene.add_policy(
        name="velocity 3000 iters",
        policy=onnx.load(
            "assets/anymal_c_velocity/Mjlab-Velocity-Flat-Anymal-C.3000.onnx"
        ),
        config_path="assets/anymal_c_velocity/Mjlab-Velocity-Flat-Anymal-C.3000.json",
        actions={
            "joint_pos": JointPositionActionCfg(
                scale=1.013,
                stiffness=19.739,
                damping=1.257,
            )
        },
        observations={
            "obs": ObservationGroupCfg(
                terms={
                    "base_lin_vel": ObservationTermCfg(func=obs_fns.base_lin_vel),
                    "base_ang_vel": ObservationTermCfg(func=obs_fns.base_ang_vel),
                    "projected_gravity": ObservationTermCfg(
                        func=obs_fns.projected_gravity
                    ),
                    "joint_pos": ObservationTermCfg(
                        func=obs_fns.joint_pos_rel, params={"pos_steps": [0]}
                    ),
                    "joint_vel": ObservationTermCfg(func=obs_fns.joint_vel_rel),
                    "last_action": ObservationTermCfg(func=obs_fns.last_action),
                    "velocity_cmd": ObservationTermCfg(
                        func=obs_fns.generated_commands,
                        params={"command_name": "velocity"},
                    ),
                }
            )
        },
    ).add_velocity_command(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-0.5, 0.5),
        default_lin_vel_x=0.5,
        default_lin_vel_y=0.0,
        default_ang_vel_z=0.0,
    )


def _add_mjswan_demo_project(builder: mjswan.Builder) -> None:
    project = builder.add_project(name="mjswan Demo")
    _add_g1_scene(project)
    _add_go2_scene(project)
    _add_go1_scene(project)


def _add_robot_descriptions_project(builder: mjswan.Builder) -> None:
    project = builder.add_project(name="Robot Descriptions", id="robotdesc")

    # ANYmal C Velocity from https://github.com/mujocolab/anymal_c_velocity
    _add_anymal_c_scene(project)

    def _rd_spec(module_name: str) -> mujoco.MjSpec:
        from importlib import import_module

        mjcf_path = Path(import_module(f"robot_descriptions.{module_name}").MJCF_PATH)
        # Prefer scene.xml (floor + lights) over the robot-only MJCF when available.
        scene_path = mjcf_path.parent / "scene.xml"
        return mujoco.MjSpec.from_file(
            str(scene_path if scene_path.exists() else mjcf_path)
        )

    for module, desc in DESCRIPTIONS.items():
        if desc.has_mjcf:
            scene_name = module.replace("_mj_description", "")
            scene_name = " ".join([word.capitalize() for word in scene_name.split("_")])
            project.add_scene(name=scene_name, spec=_rd_spec(module))


def _add_playground_project(builder: mjswan.Builder) -> None:
    project = builder.add_project(name="MuJoCo Playground", id="playground")

    # TEMPORARY PATCH:
    # Force JAX backend for all environments: mujoco_playground inconsistently
    # migrated to warp as the default — some envs pass impl via config, others
    # call mjx.put_model() with no impl at all. Patching here covers both cases.
    # TODO: Once mujoco_playground fixes all envs to respect config_overrides,
    # replace this patch with simply: registry.load(env_name, config_overrides={"impl": "jax"})
    _orig_put_model = _mjx.put_model
    _mjx.put_model = lambda m, **kw: _orig_put_model(m, **{**kw, "impl": "jax"})
    try:
        for env_name in registry.ALL_ENVS:
            if "Sparse" in env_name:
                continue

            env = registry.load(env_name)
            with open(env.xml_path) as f:
                xml_content = f.read()
            spec = mujoco.MjSpec.from_string(xml_content, env.model_assets)

            # model_assets is consumed at parse time but not stored in spec.assets.
            # Remap basename keys (as in env.model_assets) to the effective paths
            # that spec.to_xml() looks up: dir/file (or just file when dir is empty).
            mesh_dir = spec.meshdir or ""
            tex_dir = spec.texturedir or ""

            def _add(directory: str, filename: str) -> None:
                if not filename:
                    return
                key = posixpath.join(directory, filename) if directory else filename
                basename = os.path.basename(key)
                if basename in env.model_assets:
                    spec.assets[key] = env.model_assets[basename]

            for mesh in spec.meshes:
                _add(mesh_dir, mesh.file)
            for texture in spec.textures:
                _add(tex_dir, texture.file)
                for cf in texture.cubefiles:
                    _add(tex_dir, cf)
            for hfield in spec.hfields:
                _add("", hfield.file)

            project.add_scene(name=env_name, spec=spec)
    finally:
        _mjx.put_model = _orig_put_model


def _add_myosuite_project(builder: mjswan.Builder) -> None:
    project = builder.add_project(name="MyoSuite", id="myosuite")

    registry_specs = gym_registry_specs()

    # (display_name, lookat, distance, elevation, azimuth)
    target_envs = {
        "myoChallengeDieReorientP2-v0": (
            "mc22 Die Reorient",
            (-0.1, -0.5, 1.4),
            1.3,
            -9.0,
            -61.0,
        ),
        "myoChallengeBaodingP2-v1": (
            "mc22 Baoding",
            (-0.1, -0.5, 1.4),
            1.3,
            -9.0,
            -61.0,
        ),
        "myoChallengeRelocateP2-v0": (
            "mc23 Relocate",
            (0.0, -0.1, 1.4),
            1.7,
            -7.0,
            -90.0,
        ),
        "myoChallengeChaseTagP2-v0": (
            "mc23 Chase Tag",
            (0.0, 0.0, 1.4),
            10.0,
            -15.0,
            -62.0,
        ),
        "myoChallengeBimanual-v0": (
            "mc24 Bimanual",
            (-0.1, -0.5, 1.4),
            1.3,
            -9.0,
            -61.0,
        ),
        "myoChallengeOslRunRandom-v0": (
            "mc24 OSL Run",
            (0.0, 0.0, 1.4),
            10.0,
            -15.0,
            -62.0,
        ),
        "myoChallengeTableTennisP2-v0": (
            "mc25 Table Tennis",
            (0.0, -1.0, 1.4),
            3.3,
            -11.0,
            -129.0,
        ),
        "myoChallengeSoccerP2-v0": (
            "mc25 Soccer",
            (0.0, -3.0, 2.0),
            14.7,
            -16.0,
            -172.0,
        ),
    }

    for env_name, (
        display_name,
        lookat,
        distance,
        elevation,
        azimuth,
    ) in target_envs.items():
        model_path = registry_specs[env_name].kwargs["model_path"]
        mjspec = mujoco.MjSpec.from_file(model_path)
        project.add_scene(name=display_name, spec=mjspec).set_viewer_config(
            mjswan.ViewerConfig(
                lookat=lookat,
                distance=distance,
                elevation=elevation,
                azimuth=azimuth,
                origin_type=mjswan.ViewerConfig.OriginType.WORLD,
            )
        )


def setup_builder() -> mjswan.Builder:
    """Set up and return the builder with all demo projects configured.

    This function creates the builder and adds all projects, scenes, and policies
    but does not build or launch the application. Useful for testing.

    Returns:
        Configured Builder instance ready to be built.
    """
    _fix_unitree_mujoco_macos()
    # Ensure asset-relative paths resolve regardless of current working directory.
    os.chdir(Path(__file__).resolve().parent)
    base_path = os.getenv("MJSWAN_BASE_PATH", "/")
    builder = mjswan.Builder(base_path=base_path, gtm_id="GTM-W79HQ38W")

    _add_mjswan_demo_project(builder)
    _add_robot_descriptions_project(builder)
    _add_playground_project(builder)
    _add_myosuite_project(builder)

    return builder


def _copy_licenses(output_dir: Path) -> None:
    """Copy LICENSE and NOTICE files into the built output.

    - robot_descriptions (robotdesc): copies per scene from each repo's REPOSITORY_PATH.
    - myosuite / mujoco_playground: copies to the project root from the dist-info licenses/.
    """
    import importlib.metadata
    import shutil
    from importlib import import_module

    # Per-scene for robot_descriptions (robotdesc project)
    robotdesc_assets = output_dir / "robotdesc" / "assets"
    if robotdesc_assets.exists():
        for module, desc in DESCRIPTIONS.items():
            if not desc.has_mjcf:
                continue
            scene_id = module.replace("_mj_description", "")
            scene_dir = robotdesc_assets / scene_id
            if not scene_dir.exists():
                continue
            mod = import_module(f"robot_descriptions.{module}")
            if not hasattr(mod, "REPOSITORY_PATH"):
                continue
            for fname in ["LICENSE", "NOTICE"]:
                src = Path(mod.REPOSITORY_PATH) / fname
                if src.exists():
                    shutil.copy2(src, scene_dir / fname)

    # Project-level for myosuite and mujoco_playground
    for project_id, pkg_name in [
        ("myosuite", "myosuite"),
        ("playground", "playground"),
    ]:
        project_dir = output_dir / project_id
        if not project_dir.exists():
            continue
        try:
            dist = importlib.metadata.Distribution.from_name(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        for fname in ["LICENSE", "NOTICE"]:
            matches = [
                f
                for f in (dist.files or [])
                if Path(str(f)).name == fname and "dist-info" in str(f)
            ]
            for f in matches:
                src = Path(str(dist.locate_file(f)))
                if src.exists():
                    shutil.copy2(src, project_dir / fname)
                    break


def main():
    """Main entry point for the demo application.

    Environment variables:
        MJSWAN_BASE_PATH: Base path for deployment (default: '/')
        MJSWAN_NO_LAUNCH: Set to '1' to skip launching the browser
        MJSWAN_SKIP_BUILD: Set to '1' to skip build and launch the pre-built app
    """
    dist_dir = Path(__file__).resolve().parent / "dist"
    if os.getenv("MJSWAN_SKIP_BUILD") == "1":
        app = mjswan.mjswanApp(dist_dir)
    else:
        builder = setup_builder()
        app = builder.build()
        _copy_licenses(dist_dir)
    if os.getenv("MJSWAN_NO_LAUNCH") != "1":
        app.launch()


if __name__ == "__main__":
    main()

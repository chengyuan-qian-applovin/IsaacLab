# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load a scene USDA, add the FR3 Duo + SharpaWave rig, and teleoperate it via XR.

The pipeline in one line: your USDA becomes the environment, the duo rig is
placed into it at ``--robot_pos``/``--robot_rot``, and your tracked hands drive
it — wrists through per-arm differential IK, all fingers through DexPilot
retargeting. No recording; XR controls are Play (start), Stop (pause), and
Reset (reset the scene).

Example (headless is required for XR without a desktop display):

    ./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/make_teleop_scene.py \\
        --scene_usda ~/sim_benchmark/scene/taco_hoi_178_023.usda --headless

Defaults (robot pose, XR anchor) are the placement validated on the TACO
tabletop scenes: the rig's torso stands south of the table facing +y, and the
anchor puts you at the torso with the tabletop at a comfortable height. For
other scenes, override:

- ``--robot_pos/--robot_rot``: where the rig's torso goes, in the scene's frame.
- ``--anchor_pos/--anchor_rot``: where the XR session's origin goes; standing at
  the rig's torso with the same yaw makes the robot arms line up with yours.

Smoke test without a headset (creates the env, holds the ready pose for N
control steps, reports flange drift):

    ./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/make_teleop_scene.py \\
        --scene_usda <scene.usda> --smoke 120 --headless
"""

# isort: skip_file
import argparse
import functools
import os

print = functools.partial(print, flush=True)  # noqa: A001

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Teleoperate the FR3 Duo + SharpaWave rig in a USDA scene.")
parser.add_argument("--scene_usda", type=str, required=True, help="Path to the scene USD/USDA file to load.")
parser.add_argument(
    "--robot_pos",
    type=float,
    nargs=3,
    default=(0.0, -0.7, 1.0),
    help="Rig torso position in the scene frame [m]. Default matches the TACO tabletop placement.",
)
parser.add_argument(
    "--robot_rot",
    type=float,
    nargs=4,
    default=(0.7071068, 0.0, 0.0, 0.7071068),
    help="Rig torso orientation quaternion (w x y z). Default: +90 deg yaw (facing +y).",
)
parser.add_argument(
    "--anchor_pos",
    type=float,
    nargs=3,
    default=(0.0, -0.7, -0.21),
    help="XR anchor position: default stands you at the rig torso with the TACO tabletop at ~0.75 m.",
)
parser.add_argument(
    "--anchor_rot",
    type=float,
    nargs=4,
    default=(0.7071068, 0.0, 0.0, 0.7071068),
    help="XR anchor rotation (w x y z): should match the robot yaw so the arms line up with yours.",
)
parser.add_argument(
    "--no_track_objects",
    action="store_true",
    help="Do not register the scene's rigid bodies with the env (their poses then survive resets).",
)
parser.add_argument(
    "--render_frequency",
    type=float,
    default=30.0,
    help="Target render rate in Hz; the physics-substep render interval is computed from the sim dt.",
)
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=300.0,
    help="Episode length before the scene auto-resets [s].",
)
parser.add_argument(
    "--cloudxr_env",
    type=str,
    default="cloudxrjs",
    help=(
        "CloudXR .env file path, or a shorthand: 'cloudxrjs' (Quest/Pico, default) or 'avp'"
        " (Apple Vision Pro). 'none' disables the CloudXR auto-launch."
    ),
)
parser.add_argument(
    "--smoke",
    type=int,
    default=None,
    metavar="N",
    help="No-XR validation: hold the ready pose for N control steps, report flange drift, exit.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.smoke is None:
    args_cli.xr = True
    if not args_cli.headless and not os.environ.get("DISPLAY"):
        print("[WARNING] XR in GUI mode without a DISPLAY: the AR session will never start. Add --headless.")

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import joint_pos_rel, reset_scene_to_default, time_out
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_physx.physics import PhysxCfg

from duo_robot import DuoIKActionsCfg, duo_robot_cfg
from usda_scene import add_usda_scene


# -- Environment config -----------------------------------------------------------


@configclass
class DuoTeleopSceneCfg(InteractiveSceneCfg):
    """The rig plus a dome light; the USDA scene and its objects are added at runtime."""

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.95, 0.95, 0.92)),
    )
    robot = None  # placed from CLI args in main()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=joint_pos_rel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    reset = EventTermCfg(func=reset_scene_to_default, mode="reset")


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=time_out, time_out=True)


@configclass
class RewardsCfg:
    pass


@configclass
class DuoTeleopEnvCfg(ManagerBasedRLEnvCfg):
    scene: DuoTeleopSceneCfg = DuoTeleopSceneCfg(num_envs=1, env_spacing=3.0)
    actions: DuoIKActionsCfg = DuoIKActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        # Teleop timing: 240 Hz physics, 60 Hz control (decimation 4).
        self.decimation = 4
        self.sim.dt = 1 / 240
        self.sim.physics = PhysxCfg(
            solver_type=1,  # TGS
            max_position_iteration_count=32,
            bounce_threshold_velocity=0.2,
            # Guards fast hand motions against tunneling objects through tables
            # (CPU pipeline only; silently ignored on GPU).
            enable_ccd=True,
        )


def build_env_cfg() -> ManagerBasedRLEnvCfg:
    """Assemble the env config from the CLI arguments."""
    env_cfg = DuoTeleopEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.episode_length_s = args_cli.episode_length_s
    env_cfg.scene.robot = duo_robot_cfg(pos=tuple(args_cli.robot_pos), rot=tuple(args_cli.robot_rot))
    add_usda_scene(env_cfg.scene, args_cli.scene_usda, track_objects=not args_cli.no_track_objects)
    # Render every Nth physics substep, as close to the requested rate as the dt allows.
    interval = max(1, round(1.0 / (env_cfg.sim.dt * args_cli.render_frequency)))
    env_cfg.sim.render_interval = interval
    print(f"[INFO] render interval {interval} substeps -> {1.0 / (env_cfg.sim.dt * interval):.1f} Hz")
    return env_cfg


# -- Teleop -----------------------------------------------------------------------


def to_root_frame(env: ManagerBasedRLEnv, action: torch.Tensor) -> torch.Tensor:
    """Convert the two world-frame wrist poses to the robot-root frame.

    The differential-IK action terms expect root-frame commands and do no frame
    conversion themselves; the finger slices (14:58) pass through untouched.
    """
    robot = env.scene["robot"]
    root_pos = robot.data.root_pos_w.torch[0] - env.scene.env_origins[0]
    root_quat = robot.data.root_quat_w.torch[0]
    out = action.clone()
    for base in (0, 7):
        pos, quat = subtract_frame_transforms(
            root_pos.unsqueeze(0),
            root_quat.unsqueeze(0),
            action[base : base + 3].unsqueeze(0),
            action[base + 3 : base + 7].unsqueeze(0),
        )
        out[base : base + 3] = pos.squeeze(0)
        out[base + 3 : base + 7] = quat.squeeze(0)
    return out


def run_teleop(env: ManagerBasedRLEnv) -> None:
    """Drive the env from XR hand tracking until the app closes."""
    from isaaclab_teleop import CLOUDXR_AVP_ENV, CLOUDXR_JS_ENV, create_isaac_teleop_device, poll_control_events
    from isaaclab_teleop.isaac_teleop_cfg import IsaacTeleopCfg
    from isaaclab_teleop.xr_cfg import XrCfg

    from duo_teleop_pipeline import build_duo_pipeline

    pipeline, retargeters = build_duo_pipeline()
    teleop_cfg = IsaacTeleopCfg(
        xr_cfg=XrCfg(anchor_pos=tuple(args_cli.anchor_pos), anchor_rot=tuple(args_cli.anchor_rot)),
        pipeline_builder=lambda: pipeline,
        retargeters_to_tune=lambda: retargeters,
        sim_device=env.device,
    )

    cloudxr_env = {"cloudxrjs": CLOUDXR_JS_ENV, "avp": CLOUDXR_AVP_ENV, "none": None}.get(
        args_cli.cloudxr_env, args_cli.cloudxr_env
    )

    teleop_active = False
    reset_requested = False

    def _start():
        nonlocal teleop_active
        teleop_active = True

    def _stop():
        nonlocal teleop_active
        teleop_active = False

    def _reset():
        nonlocal reset_requested
        reset_requested = True

    teleop = create_isaac_teleop_device(
        teleop_cfg,
        sim_device=env.device,
        callbacks={"START": _start, "STOP": _stop, "RESET": _reset},
        cloudxr_env_file=cloudxr_env,
        auto_launch_cloudxr=cloudxr_env is not None,
    )

    print("[INFO] Teleop loop started. Headset: Play = start, Stop = pause, Reset = reset the scene.")
    with teleop, torch.inference_mode():
        env.reset()
        teleop.reset()
        while simulation_app.is_running():
            action = teleop.advance()
            ctrl = poll_control_events(teleop)
            if ctrl.is_active is not None:
                teleop_active = ctrl.is_active
            if ctrl.should_reset:
                reset_requested = True

            if reset_requested:
                env.reset()
                teleop.reset()
                reset_requested = False
            # action is None until the XR session has started.
            if action is None or not teleop_active:
                env.sim.render()
                continue
            action = to_root_frame(env, action.to(env.device))
            env.step(action.unsqueeze(0))  # time_out auto-resets the scene


def run_smoke(env: ManagerBasedRLEnv, num_steps: int) -> None:
    """Hold the ready pose for ``num_steps`` control steps and report flange drift."""
    from isaaclab.utils.math import quat_error_magnitude

    env.reset()
    robot = env.scene["robot"]
    flange_ids = [robot.body_names.index(f"{side}_panda_link8") for side in ("left", "right")]

    def flange_poses_root() -> torch.Tensor:
        """Both flange poses in the root frame, flattened to (14,)."""
        chunks = []
        for body_id in flange_ids:
            pos, quat = subtract_frame_transforms(
                robot.data.root_pos_w.torch,
                robot.data.root_quat_w.torch,
                robot.data.body_pos_w.torch[:, body_id],
                robot.data.body_quat_w.torch[:, body_id],
            )
            chunks += [pos[0], quat[0]]
        return torch.cat(chunks)

    action_dim = env.action_manager.total_action_dim
    assert action_dim == 58, f"expected the 58-D duo action space, got {action_dim}"

    with torch.inference_mode():
        # Phase 1: hold the ready pose.
        start = flange_poses_root()
        action = torch.zeros(action_dim, device=env.device)
        action[:14] = start  # fingers stay at zero
        for _ in range(num_steps):
            env.step(action.unsqueeze(0))
        end = flange_poses_root()
        # Phase 2: command both flanges 3 cm up (in the root frame) and check they follow.
        target = start.clone()
        target[2] += 0.03
        target[9] += 0.03
        action[:14] = target
        for _ in range(num_steps):
            env.step(action.unsqueeze(0))
        moved = flange_poses_root()

    ok = True
    for i, side in enumerate(("left", "right")):
        pos_drift = (end[7 * i : 7 * i + 3] - start[7 * i : 7 * i + 3]).norm().item()
        rot_drift = quat_error_magnitude(
            start[7 * i + 3 : 7 * i + 7].unsqueeze(0), end[7 * i + 3 : 7 * i + 7].unsqueeze(0)
        ).item()
        track_err = (moved[7 * i : 7 * i + 3] - target[7 * i : 7 * i + 3]).norm().item()
        print(
            f"[SMOKE] {side} flange: hold drift {pos_drift * 1000:.2f} mm / {rot_drift:.4f} rad,"
            f" +3 cm tracking error {track_err * 1000:.2f} mm"
        )
        ok = ok and pos_drift < 0.005 and track_err < 0.01
    print("[SMOKE] OK" if ok else "[SMOKE] FAILED")
    if not ok:
        raise SystemExit(1)


def main():
    env = ManagerBasedRLEnv(cfg=build_env_cfg())
    try:
        if args_cli.smoke is not None:
            run_smoke(env, args_cli.smoke)
        else:
            run_teleop(env)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # SimulationApp.close() can terminate the process before an in-flight
        # exception is reported; print it first.
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()

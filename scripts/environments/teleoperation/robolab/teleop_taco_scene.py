# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleoperate the sim_benchmark TACO scene (brush + bowl on a table) with the SharpaWave duo.

Composes three sources:
- **Scene** — sim_benchmark's ``scene/taco_hoi_178_023.usda`` (table, brush ``taco_178``,
  bowl ``taco_023``, lights), loaded through its own ``TacoSceneCfg`` so the two objects
  stay individually tracked rigid bodies.
- **Robot** — sim_benchmark's ``franka_duo`` placement (torso at (0, -0.7, 1.0), +90° yaw,
  facing the table across +y) with the rig's vendored USD.
- **Control** — the RoboLab fork's 58-D ``FrankaDuoSharpaIKActionCfg`` (dual absolute
  wrist IK + per-finger targets) driven by our AVP dual-hand retargeters. sim_benchmark
  itself ships only joint-position actions; the IK action space plugs in because both
  repos build the same articulation with the same prim/joint names.

The XR anchor defaults stand you at the torso *and yaw you to face the table* — the
anchor rotation must equal the robot root yaw for the calibrated wrist offsets to hold.

Run (sim_benchmark is expected at <IsaacLab>/sim_benchmark):

    ./isaaclab.sh -p scripts/environments/teleoperation/robolab/teleop_taco_scene.py --headless
"""

# isort: skip_file
import argparse
import functools
import os
import sys

import cv2  # noqa: F401  Must import before isaaclab/omni modules.
import pinocchio  # noqa: F401  Must import before AppLauncher (dex_retargeting builds in-kit).

print = functools.partial(print, flush=True)  # noqa: A001

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Teleoperate the TACO benchmark scene with the SharpaWave duo.")
parser.add_argument(
    "--anchor_pos", type=float, nargs=3, default=(0.0, -0.7, -0.21),
    help="XR anchor position: default stands you at the torso with the tabletop at ~0.75 m.",
)
parser.add_argument(
    "--anchor_rot", type=float, nargs=4, default=(0.7071068, 0.0, 0.0, 0.7071068),
    help="XR anchor rotation (w x y z): must match the robot root yaw (+90°) so you face the table.",
)
parser.add_argument(
    "--profile", action="store_true",
    help="Print rolling per-stage loop timings (retarget / frame / step) once per second.",
)
parser.add_argument(
    "--visualize_hands", action="store_true",
    help="Draw red spheres on the tracked OpenXR hand joints (like the default GR1T2 teleop).",
)
parser.add_argument(
    "--render_interval", type=int, default=4,
    help="Render every N physics substeps (default 4 = 30 Hz at dt=1/120).",
)
AppLauncher.add_app_launcher_args(parser)
# Default AppLauncher's --rendering_mode to the cheap preset (still overridable on the
# CLI: --rendering_mode balanced|quality). Profiling showed ~24 ms/render on "balanced"
# vs ~13 ms on "performance", and 4 renders happen inside every control step.
parser.set_defaults(rendering_mode="balanced")
args_cli = parser.parse_args()
args_cli.xr = True

if not args_cli.headless and not os.environ.get("DISPLAY"):
    print("[WARNING] XR in GUI mode without a DISPLAY: the AR session will never start. Add --headless.")

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

"""Rest everything follows."""

# sim_benchmark is not pip-installed; it lives inside the IsaacLab checkout.
_SIM_BENCHMARK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "sim_benchmark"))
sys.path.insert(0, _SIM_BENCHMARK_ROOT)

import torch

from sim_benchmark.franka_duo import FRANKA_DUO_USD  # noqa: E402
from sim_benchmark.taco_hoi import TacoSceneCfg, robot_spawn_props  # noqa: E402

from robolab.robots.franka_duo_sharpa_wave import (  # noqa: E402
    FrankaDuoSharpaIKActionCfg,
    LEFT_HAND_JOINTS_ORDERED,
    RIGHT_HAND_JOINTS_ORDERED,
)

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import joint_pos_rel, reset_scene_to_default, time_out
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from robolab_teleop_common import LoopProfiler, OncePerStepDiffIKAction
from sharpa_duo_retargeters import FrankaDuoSharpaRetargeterCfg

# Arm ready pose shared by both repos (IK-solved: fingers forward, palms down).
_ARM_INIT = {
    "left_panda_joint1": 1.145, "left_panda_joint2": 1.048, "left_panda_joint3": -0.464,
    "left_panda_joint4": -1.516, "left_panda_joint5": -2.540, "left_panda_joint6": 2.045,
    "left_panda_joint7": 0.108,
    "right_panda_joint1": -1.144, "right_panda_joint2": 1.047, "right_panda_joint3": 0.462,
    "right_panda_joint4": -1.517, "right_panda_joint5": 2.541, "right_panda_joint6": 2.044,
    "right_panda_joint7": -0.107,
}


@configclass
class TacoTeleopSceneCfg(TacoSceneCfg):
    """sim_benchmark's TACO scene (table + brush + bowl + lights) plus the duo rig.

    Robot placement mirrors sim_benchmark's franka_duo: torso at (0, -0.7, 1.0),
    +90° yaw — standing south of the table (top at z = 0.5421), facing +y.
    """

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(FRANKA_DUO_USD), **robot_spawn_props()),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, -0.70, 1.0),
            rot=(0.7071068, 0.0, 0.0, 0.7071068),
            joint_pos={**_ARM_INIT, "(left|right)_(thumb|index|middle|ring|pinky)_.*": 0.0},
        ),
        soft_joint_pos_limit_factor=1.05,
        actuators={
            "shoulders": ImplicitActuatorCfg(
                joint_names_expr=["(left|right)_panda_joint[1-4]"],
                effort_limit=87.0, velocity_limit=2.175, stiffness=400.0, damping=80.0,
            ),
            "forearms": ImplicitActuatorCfg(
                joint_names_expr=["(left|right)_panda_joint[5-7]"],
                effort_limit=12.0, velocity_limit=2.61, stiffness=400.0, damping=80.0,
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=["(left|right)_(thumb|index|middle|ring|pinky)_.*"],
                stiffness=None, damping=None,  # keep Sharpa's USD-calibrated gains
            ),
        },
    )


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
class TacoTeleopEnvCfg(ManagerBasedRLEnvCfg):
    scene: TacoTeleopSceneCfg = TacoTeleopSceneCfg(num_envs=1, env_spacing=3.0)
    actions: FrankaDuoSharpaIKActionCfg = FrankaDuoSharpaIKActionCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        # honor --device (without this, SimulationCfg's default cuda:0 silently wins)
        self.sim.device = args_cli.device
        self.episode_length_s = 120.0
        self.decimation = 4
        self.sim.dt = 1 / 120           # teleop timing, not sim_benchmark's 20 Hz clip timing
        self.sim.render_interval = args_cli.render_interval  # 2 = 60 Hz for the headset
        self.sim.physx.solver_type = 1  # TGS (sim_benchmark's solver_type=2 is invalid on this stack)
        self.sim.physx.max_position_iteration_count = 32
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.enable_ccd = True                   # CCD
        # solve arm IK once per control step instead of once per physics substep
        # (decimation× cheaper; the stock per-substep resolve cost ~65 ms/step on this
        # scene at decimation 8 — see profiler).
        self.actions.left_arm.class_type = OncePerStepDiffIKAction
        self.actions.right_arm.class_type = OncePerStepDiffIKAction


def main():
    env = ManagerBasedRLEnv(cfg=TacoTeleopEnvCfg())
    env.reset()

    device_cfg = OpenXRDeviceCfg(
        xr_cfg=XrCfg(anchor_pos=tuple(args_cli.anchor_pos), anchor_rot=tuple(args_cli.anchor_rot)),
        retargeters=[
            FrankaDuoSharpaRetargeterCfg(
                left_hand_joint_names=LEFT_HAND_JOINTS_ORDERED,
                right_hand_joint_names=RIGHT_HAND_JOINTS_ORDERED,
                sim_device=env.device,
                enable_visualization=args_cli.visualize_hands,
            )
        ],
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

    callbacks = {"START": _start, "STOP": _stop, "RESET": _reset}
    teleop = create_teleop_device("handtracking", DevicesCfg(devices={"handtracking": device_cfg}).devices, callbacks)

    robot = env.scene["robot"]

    def to_root_frame(action: torch.Tensor) -> torch.Tensor:
        root_pos = robot.data.root_pos_w[0] - env.scene.env_origins[0]
        root_quat = robot.data.root_quat_w[0]
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

    prof = LoopProfiler(enabled=args_cli.profile)
    # "render_call" = CPU-blocked wall time of sim.render() (submission + any XR pacing
    # wait) — the async GPU render/CloudXR encode is NOT included. Split out of "step".
    prof.wrap_render(env.sim, "render_call")
    prof.wrap_method(env.sim, "step", "physx")  # raw physics call, separated from step's other work
    # decompose the rest of env.step: IK action apply, sim write, state readback, observations
    prof.wrap_method(env.action_manager, "apply_action", "action")
    prof.wrap_method(env.scene, "write_data_to_sim", "write")
    prof.wrap_method(env.scene, "update", "readback")
    prof.wrap_method(env.observation_manager, "compute", "obs")
    prof.wrap_method(env.action_manager, "process_action", "ik")

    print("[INFO] Starting teleop loop. AVP: Play=start, Stop=pause, Reset=reset scene.")
    with torch.inference_mode():
        while simulation_app.is_running():
            if reset_requested:
                env.reset()
                teleop.reset()
                reset_requested = False
            if not teleop_active:
                prof.begin()
                env.sim.render()  # wrapped: lands in the "render_call" bucket
                prof.end()
                continue
            prof.begin()
            action = teleop.advance()          # XR poll + wrist offsets + DexPilot QPs
            prof.lap("retarget")
            action = to_root_frame(action.to(env.device))
            prof.lap("frame")
            env.step(action.unsqueeze(0))      # 8 physics substeps + 4 renders
            prof.lap("step")
            prof.end()

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

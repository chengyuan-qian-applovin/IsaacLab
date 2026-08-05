# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal bimanual teleop scene: FR3 Duo + SharpaWave hands, a table, a banana, one light.

A self-contained alternative to the RoboLab benchmark environments: no RoboLab
registration, no task predicates, no cameras, no background dome, no recording —
just the rig in front of a table with a banana, teleoperated with AVP dual-hand
tracking. Useful as a sandbox for retargeting/IK tuning and as a template for
building custom scenes around the duo rig.

Assets and robot/action configs are read from the SharpaWave RoboLab fork on disk
(it must be pip-installed, same as the full teleop; see README).

    ./isaaclab.sh -p scripts/environments/teleoperation/robolab/minimal_sharpa_duo_teleop.py --headless

XR controls: Play = start, Stop = pause, Reset = reset the scene.
"""

# isort: skip_file
import argparse
import functools
import os

import cv2  # noqa: F401  Must import before isaaclab/omni modules (RoboLab requirement).
import pinocchio  # noqa: F401  Must import before AppLauncher (dex_retargeting builds in-kit).

print = functools.partial(print, flush=True)  # noqa: A001

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Minimal FR3 Duo + SharpaWave teleop scene.")
parser.add_argument(
    "--anchor_pos", type=float, nargs=3, default=(0.0, 0.0, -0.7), help="XR anchor position (x y z)."
)
parser.add_argument(
    "--anchor_rot", type=float, nargs=4, default=(1.0, 0.0, 0.0, 0.0), help="XR anchor rotation quat (w x y z)."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.xr = True

if not args_cli.headless and not os.environ.get("DISPLAY"):
    print("[WARNING] XR in GUI mode without a DISPLAY: the AR session will never start. Add --headless.")

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.envs.mdp import reset_scene_to_default, time_out
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from robolab.constants import ASSET_DIR, OBJECT_DIR
from robolab.robots.franka_duo_sharpa_wave import (
    FrankaDuoProprioceptionObservationCfg,
    FrankaDuoSharpaIKActionCfg,
    FrankaDuoSharpaWaveCfg,
    LEFT_HAND_JOINTS_ORDERED,
    RIGHT_HAND_JOINTS_ORDERED,
)

from sharpa_duo_retargeters import FrankaDuoSharpaRetargeterCfg


# -- Scene: table + banana + light + robot. Nothing else. ----------------------


@configclass
class MinimalDuoSceneCfg(InteractiveSceneCfg):
    """Table (RoboLab fixture), one YCB banana, a dome light, the duo rig."""

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(ASSET_DIR, "fixtures", "franka_table.usd")),
        # Same pose RoboLab's benchmark scenes use for this fixture (tabletop at z≈0,
        # 180° yaw — the fixture is asymmetric).
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.087, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
    )

    banana = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/banana",
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(OBJECT_DIR, "ycb", "banana.usd")),
        # Dropped slightly above the tabletop between the two ready-pose wrists;
        # it settles onto the table during the first steps after reset.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, 0.0, 0.10)),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.92)),
    )

    # Same placement as the RoboLab duo registration: torso pedestal behind the table.
    robot = FrankaDuoSharpaWaveCfg().robot


@configclass
class ObservationsCfg:
    policy = FrankaDuoProprioceptionObservationCfg()


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
class MinimalDuoEnvCfg(ManagerBasedRLEnvCfg):
    scene: MinimalDuoSceneCfg = MinimalDuoSceneCfg(num_envs=1, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: FrankaDuoSharpaIKActionCfg = FrankaDuoSharpaIKActionCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        self.episode_length_s = 120.0
        self.decimation = 8
        self.sim.dt = 1 / 120
        self.sim.render_interval = 2  # 60 Hz render for the headset
        # The duo articulation requests 64 solver iterations; clamp lower for a
        # snappy minimal scene (one banana's worth of contact needs far less).
        self.sim.physx.max_position_iteration_count = 16


# -- Teleop loop ----------------------------------------------------------------


def main():
    env = ManagerBasedRLEnv(cfg=MinimalDuoEnvCfg())
    env.reset()

    device_cfg = OpenXRDeviceCfg(
        xr_cfg=XrCfg(anchor_pos=tuple(args_cli.anchor_pos), anchor_rot=tuple(args_cli.anchor_rot)),
        retargeters=[
            FrankaDuoSharpaRetargeterCfg(
                left_hand_joint_names=LEFT_HAND_JOINTS_ORDERED,
                right_hand_joint_names=RIGHT_HAND_JOINTS_ORDERED,
                sim_device=env.device,
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
        """XR/world-frame wrist poses -> robot-root frame (what the IK action expects)."""
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

    print("[INFO] Starting teleop loop. AVP: Play=start, Stop=pause, Reset=reset scene.")
    with torch.inference_mode():
        while simulation_app.is_running():
            if reset_requested:
                env.reset()
                teleop.reset()
                reset_requested = False
            if not teleop_active:
                env.sim.render()
                continue
            action = teleop.advance()
            action = to_root_frame(action.to(env.device))
            env.step(action.unsqueeze(0))  # time_out auto-resets the scene

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

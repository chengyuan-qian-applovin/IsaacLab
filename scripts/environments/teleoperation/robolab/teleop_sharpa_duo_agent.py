# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleoperate the FR3 Duo + SharpaWave rig in RoboLab scenes with AVP dual-hand tracking.

GR1T2-equivalent teleop for the bimanual rig: both hands drive the two arms via
absolute wrist-pose differential IK (RoboLab's ``FrankaDuoSharpaIKActionCfg``), and
all ten fingers are dex-retargeted onto the two 22-DoF SharpaWave hands. Episodes
record through RoboLab's native pipeline (``run_<N>.hdf5``), convertible to a
robomimic dataset with ``convert_robolab_to_robomimic.py``.

Requires the SharpaWave RoboLab fork (with ``robolab.registrations.sharpa_wave``)
installed — see README "SharpaWave duo rig" section.

    ./isaaclab.sh -p scripts/environments/teleoperation/robolab/teleop_sharpa_duo_agent.py \\
        --task BananaInBowlTask --num_demos 5 --headless --device cuda:0

XR controls: Play = start, Stop = pause, Reset = discard demo.
"""

# isort: skip_file
import argparse
import functools
import os

import cv2  # noqa: F401  Must import before isaaclab/omni modules (RoboLab requirement).

# Import pinocchio BEFORE AppLauncher so Isaac Lab's version wins over Isaac Sim's and
# AppLauncher applies its pxr.Gf.Matrix4d compatibility patch (same as teleop_se3_agent's
# --enable_pinocchio). dex_retargeting imports pinocchio at hand-retargeter build time;
# without this pre-import, kit dies silently during that build.
import pinocchio  # noqa: F401

print = functools.partial(print, flush=True)  # noqa: A001  kit may hard-exit before buffers flush

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Teleoperate the SharpaWave duo rig via AVP hand tracking.")
parser.add_argument("--task", type=str, default="BananaInBowlTask", help="RoboLab task class name.")
parser.add_argument("--num_demos", type=int, default=0, help="Stop after N recorded demos (0 = unlimited).")
parser.add_argument(
    "--anchor_pos", type=float, nargs=3, default=(0.0, 0.0, -0.7),
    help="XR anchor position (x y z): shifts where the sim world appears relative to you.",
)
parser.add_argument(
    "--anchor_rot", type=float, nargs=4, default=(1.0, 0.0, 0.0, 0.0), help="XR anchor rotation quat (w x y z)."
)
parser.add_argument(
    "--episode_length_s", type=float, default=None, help="Override the task's episode time limit in seconds."
)
parser.add_argument(
    "--render_interval", type=int, default=2, help="Sim render interval under XR (2 -> 60 Hz)."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# This rig's teleop is XR-only (dual-hand tracking has no keyboard analogue).
args_cli.xr = True
args_cli.enable_cameras = False  # cameras are stripped under XR (see robolab_teleop_common)

if not args_cli.headless and not os.environ.get("DISPLAY"):
    print(
        "[WARNING] XR in GUI mode without a DISPLAY: the AR session will never start. Add --headless.",
    )

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import robolab.constants
from robolab.core.environments.config import parse_env_cfg as robolab_parse_env_cfg
from robolab.core.environments.factory import get_envs
from robolab.core.environments.runtime import create_env, end_episode
from robolab.registrations.sharpa_wave.auto_env_registrations_duo import auto_register_franka_duo_envs
from robolab.robots.franka_duo_sharpa_wave import LEFT_HAND_JOINTS_ORDERED, RIGHT_HAND_JOINTS_ORDERED

from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.utils.math import subtract_frame_transforms

from robolab_teleop_common import strip_cameras_for_xr
from sharpa_duo_retargeters import FrankaDuoSharpaRetargeterCfg

robolab.constants.VERBOSE = False
robolab.constants.RECORD_IMAGE_DATA = False


def main():
    auto_register_franka_duo_envs(task=[args_cli.task], ik_actions=True)
    env_names = get_envs(task=[args_cli.task])
    if not env_names:
        raise ValueError(f"No SharpaWave duo environments registered for task '{args_cli.task}'.")
    env_name = env_names[0]
    print(f"[INFO] Teleoperating SharpaWave duo environment: {env_name}")

    env_cfg = robolab_parse_env_cfg(env_name, device=args_cli.device or "cuda:0", num_envs=1, use_fabric=True)
    env_cfg.env_name = env_name
    removed = strip_cameras_for_xr(env_cfg)
    if removed:
        print(f"[INFO] XR mode: removed scene cameras {removed} and their image observations.")
    env, env_cfg = create_env(env_cfg, num_envs=1, use_fabric=True)

    env.cfg.sim.render_interval = args_cli.render_interval
    if args_cli.episode_length_s is not None:
        env.cfg.episode_length_s = args_cli.episode_length_s

    # -- teleop device: OpenXR dual-hand tracking -> 58-D duo IK action --
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

    teleop_active = False  # XR starts paused until the client sends "start"
    discard_requested = False

    def _start():
        nonlocal teleop_active
        teleop_active = True

    def _stop():
        nonlocal teleop_active
        teleop_active = False

    def _reset():
        nonlocal discard_requested
        discard_requested = True

    callbacks = {"START": _start, "STOP": _stop, "RESET": _reset}
    teleop = create_teleop_device("handtracking", DevicesCfg(devices={"handtracking": device_cfg}).devices, callbacks)

    robot = env.scene["robot"]

    def to_root_frame(action: torch.Tensor) -> torch.Tensor:
        """Convert the two wrist-pose slices from XR/world frame to the robot root frame.

        FrankaDuoSharpaIKActionCfg expects flange poses in the ROBOT ROOT frame; the
        retargeter emits them in the XR anchor (= env/world) frame.
        """
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

    recorder = env.recorder_manager
    demo_idx = 0
    successful = 0
    print("[INFO] Starting teleop loop. AVP: Play=start, Stop=pause, Reset=discard demo.")

    while simulation_app.is_running():
        env.reset_eval_state()
        env.reset()
        env.reset()
        teleop.reset()
        discard_requested = False

        if recorder is not None and hasattr(recorder, "set_hdf5_file"):
            recorder.set_hdf5_file(f"run_{demo_idx}.hdf5")
            recorder.set_episode_index(0, env_ids=[0])

        episode_done = False
        with torch.inference_mode():
            while simulation_app.is_running() and not episode_done:
                if discard_requested:
                    break
                if not teleop_active:
                    env.sim.render()
                    continue
                action = teleop.advance()
                action = to_root_frame(action.to(env.device))
                env.step(action.unsqueeze(0))
                if env.all_terminated:
                    episode_done = True

        if not simulation_app.is_running():
            break

        if discard_requested:
            print(f"[INFO] Demo {demo_idx}: discarded by user.")
            if recorder is not None:
                recorder.clear()
            continue

        end_episode(env)
        results = env.get_env_results()
        success = bool(results[0]["success"]) if results and results[0]["success"] is not None else False
        successful += int(success)
        print(
            f"[INFO] Demo {demo_idx}: {'SUCCESS' if success else 'failure/timeout'} "
            f"(step {results[0]['step']}). Total successes: {successful}."
        )
        demo_idx += 1
        if args_cli.num_demos > 0 and demo_idx >= args_cli.num_demos:
            print(f"[INFO] Collected {demo_idx} demos ({successful} successful). Done.")
            break

    output_dir = env.output_dir
    env.close()
    print(f"[INFO] RoboLab episode data written under: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

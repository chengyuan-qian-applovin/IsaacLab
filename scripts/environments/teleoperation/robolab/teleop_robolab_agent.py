# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleoperate RoboLab benchmark tasks with Isaac Lab XR hand tracking and record demonstrations.

This script drives NVIDIA RoboLab environments (https://github.com/NVLabs/RoboLab) with the
Isaac Lab OpenXR teleoperation stack (e.g. Apple Vision Pro hand tracking via CloudXR).
Each episode is recorded through RoboLab's native streaming recorder, producing
``run_<N>.hdf5`` files that RoboLab's replay/analysis/dashboard tooling understands.
The companion script ``convert_robolab_to_robomimic.py`` turns those files into a single
robomimic-style dataset for the Isaac Lab imitation-learning pipeline.

Requirements:
    RoboLab must be importable (see install_robolab.sh) and its assets available.

Example (inside the Isaac Lab container, with the CloudXR runtime up):

    ./isaaclab.sh -p scripts/environments/teleoperation/robolab/teleop_robolab_agent.py \\
        --task BananaInBowlTask --teleop_device handtracking

    # keyboard smoke test without a headset
    ./isaaclab.sh -p scripts/environments/teleoperation/robolab/teleop_robolab_agent.py \\
        --task BananaInBowlTask --teleop_device keyboard

Teleop controls (XR client): Play = start teleoperation, Stop = pause,
Reset = discard the current demo and restart the episode.
Keyboard: R = discard current demo and reset.
"""

# isort: skip_file
import argparse
import contextlib
import sys

import cv2  # noqa: F401  Must import before isaaclab/omni modules (RoboLab requirement).

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Teleoperate RoboLab tasks via Isaac Lab XR devices.")
parser.add_argument("--task", type=str, default="BananaInBowlTask", help="RoboLab task class name.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="handtracking",
    help="Teleop device: 'handtracking' (XR, right hand), 'handtracking_left' (XR, left hand) or 'keyboard'.",
)
parser.add_argument("--num_demos", type=int, default=0, help="Stop after N recorded demos (0 = unlimited).")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Keyboard sensitivity factor.")
parser.add_argument(
    "--anchor_pos", type=float, nargs=3, default=(-0.4, 0.0, -0.9), help="XR anchor position (x y z)."
)
parser.add_argument(
    "--anchor_rot", type=float, nargs=4, default=(1.0, 0.0, 0.0, 0.0), help="XR anchor rotation quat (w x y z)."
)
parser.add_argument("--record_images", action="store_true", help="Also record per-step image observations to HDF5.")
parser.add_argument(
    "--render_interval", type=int, default=None,
    help="Override sim render interval (default: 2 under XR, RoboLab default otherwise).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Hand-tracking teleop requires the OpenXR kit experience; AppLauncher resolves
# the experience file from the xr flag (same pattern as teleop_se3_agent.py).
if "handtracking" in args_cli.teleop_device.lower():
    args_cli.xr = True

# RoboLab scenes carry a wrist camera on the robot; cameras must be enabled.
args_cli.enable_cameras = True

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import robolab.constants
from robolab.core.environments.factory import get_envs
from robolab.core.environments.runtime import create_env, end_episode
from robolab.registrations.droid.auto_env_registrations_abs_ik import auto_register_droid_abs_ik_envs
from robolab.robots.droid import EEF_OFFSET_ROT

from isaaclab.devices import Se3Keyboard
from isaaclab.devices.device_base import DeviceBase, DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.utils.math import quat_inv, quat_mul

# Local module (script directory is on sys.path when run as __main__).
from robolab_retargeters import RobolabAbsIKRetargeterCfg, RobolabGripperRetargeterCfg

robolab.constants.VERBOSE = False
robolab.constants.RECORD_IMAGE_DATA = bool(args_cli.record_images)

_EEF_OFFSET_ROT_INV = quat_inv(torch.tensor(EEF_OFFSET_ROT, dtype=torch.float32).unsqueeze(0))


class KeyboardAbsIKAdapter:
    """Integrates Se3Keyboard 6-D deltas into an absolute base_link pose target.

    Lets the RoboLab abs-IK action space be driven without a headset, for smoke
    tests and non-XR use. The target pose is (re)initialized from the robot's
    current ``eef_frame`` pose on every reset.
    """

    def __init__(self, env, sensitivity: float):
        self._env = env
        self._device = Se3Keyboard(
            pos_sensitivity=0.005 * sensitivity, rot_sensitivity=0.01 * sensitivity
        )
        self._target_pos: torch.Tensor | None = None
        self._target_quat: torch.Tensor | None = None

    def add_callback(self, key: str, func):
        self._device.add_callback(key, func)

    def reset(self):
        self._device.reset()
        self._target_pos = None
        self._target_quat = None

    def _init_target_from_robot(self):
        frames = self._env.scene["frames"]
        eef_idx = frames.data.target_frame_names.index("eef_frame")
        self._target_pos = frames.data.target_pos_w[0, eef_idx, :].cpu().clone()
        self._target_quat = frames.data.target_quat_w[0, eef_idx, :].cpu().clone()

    def advance(self) -> torch.Tensor:
        if self._target_pos is None:
            self._init_target_from_robot()
        delta_pose, gripper_close = self._device.advance()
        delta_pose = torch.as_tensor(delta_pose, dtype=torch.float32)
        self._target_pos += delta_pose[:3]
        if torch.any(delta_pose[3:] != 0.0):
            # Compose small world-frame rotation onto the target orientation.
            angle = torch.linalg.norm(delta_pose[3:])
            axis = delta_pose[3:] / angle
            half = angle / 2.0
            dquat = torch.cat([torch.cos(half).reshape(1), torch.sin(half) * axis])
            self._target_quat = quat_mul(dquat.unsqueeze(0), self._target_quat.unsqueeze(0)).squeeze(0)
        # eef_frame target -> base_link target (positions equal, orientation un-offset).
        base_quat = quat_mul(self._target_quat.unsqueeze(0), _EEF_OFFSET_ROT_INV).squeeze(0)
        gripper = torch.tensor([1.0 if gripper_close else 0.0], dtype=torch.float32)
        return torch.cat([self._target_pos, base_quat, gripper])


def build_xr_teleop_device(device_name: str, sim_device: str, callbacks: dict):
    """Construct the OpenXR hand-tracking device with RoboLab retargeters."""
    bound_hand = (
        DeviceBase.TrackingTarget.HAND_LEFT
        if device_name.endswith("_left")
        else DeviceBase.TrackingTarget.HAND_RIGHT
    )
    device_cfg = OpenXRDeviceCfg(
        xr_cfg=XrCfg(anchor_pos=tuple(args_cli.anchor_pos), anchor_rot=tuple(args_cli.anchor_rot)),
        retargeters=[
            RobolabAbsIKRetargeterCfg(
                bound_hand=bound_hand,
                zero_out_xy_rotation=False,
                use_wrist_rotation=False,
                use_wrist_position=False,  # pinch midpoint gives finer control than the wrist
                eef_offset_rot=EEF_OFFSET_ROT,
                sim_device=sim_device,
            ),
            RobolabGripperRetargeterCfg(bound_hand=bound_hand, sim_device=sim_device),
        ],
    )
    devices_cfg = DevicesCfg(devices={device_name: device_cfg})
    return create_teleop_device(device_name, devices_cfg.devices, callbacks)


def main():
    # Register the abs-IK variant of the requested task and resolve the env name.
    auto_register_droid_abs_ik_envs(task=args_cli.task)
    env_names = get_envs(task=args_cli.task)
    if not env_names:
        raise ValueError(f"No RoboLab environments found for task '{args_cli.task}'.")
    env_name = env_names[0]
    print(f"[INFO] Teleoperating RoboLab environment: {env_name}")

    env, env_cfg = create_env(env_name, device=args_cli.device or "cuda:0", num_envs=1, use_fabric=True)

    # XR-friendly rendering: RoboLab's default renders once per control step (15 Hz),
    # which is too choppy for a headset.
    render_interval = args_cli.render_interval
    if render_interval is None and args_cli.xr:
        render_interval = 2
    if render_interval is not None:
        # ManagerBasedEnv.step() reads cfg.sim.render_interval live on every physics
        # substep, so mutating the cfg after env creation takes effect immediately.
        env.cfg.sim.render_interval = render_interval

    # Teleoperation state, toggled by the XR client's Play/Stop/Reset buttons.
    teleop_active = not args_cli.xr  # XR starts paused until the client sends "start"
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

    callbacks = {"START": _start, "STOP": _stop, "RESET": _reset, "R": _reset}

    if "handtracking" in args_cli.teleop_device.lower():
        teleop = build_xr_teleop_device(args_cli.teleop_device, env.device, callbacks)
        keyboard_mode = False
    elif args_cli.teleop_device.lower() == "keyboard":
        teleop = KeyboardAbsIKAdapter(env, args_cli.sensitivity)
        for key, cb in callbacks.items():
            with contextlib.suppress(ValueError, KeyError):
                teleop.add_callback(key, cb)
        keyboard_mode = True
    else:
        raise ValueError(f"Unsupported teleop device: {args_cli.teleop_device}")

    recorder = env.recorder_manager
    demo_idx = 0
    successful_demos = 0

    print("[INFO] Starting teleop loop. XR: Play=start, Stop=pause, Reset=discard demo. Keyboard: R=discard.")

    while simulation_app.is_running():
        # -- begin a new episode/demo --
        env.reset_eval_state()
        env.reset()
        env.reset()  # double reset per RoboLab eval convention (settles randomization)
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
                if keyboard_mode:
                    # Keyboard adapter reads robot state before first step; XR path is stateless.
                    action = action.to(env.device)
                env.step(action.unsqueeze(0))
                if env.all_terminated:
                    episode_done = True

        if not simulation_app.is_running():
            break

        if discard_requested:
            print(f"[INFO] Demo {demo_idx}: discarded by user.")
            if recorder is not None:
                recorder.clear()
            continue  # reuse the same demo index / overwrite run file

        # Episode terminated: RobolabEnv already exported the frozen env's episode.
        end_episode(env)
        results = env.get_env_results()
        success = bool(results[0]["success"]) if results and results[0]["success"] is not None else False
        successful_demos += int(success)
        print(
            f"[INFO] Demo {demo_idx}: {'SUCCESS' if success else 'failure/timeout'} "
            f"(step {results[0]['step']}). Total successes: {successful_demos}."
        )
        demo_idx += 1

        if args_cli.num_demos > 0 and demo_idx >= args_cli.num_demos:
            print(f"[INFO] Collected {demo_idx} demos ({successful_demos} successful). Done.")
            break

    output_dir = env.output_dir
    env.close()
    print(f"[INFO] RoboLab episode data written under: {output_dir}")
    print(
        "[INFO] To build a robomimic-style dataset for Isaac Lab imitation learning, run:\n"
        f"    python scripts/environments/teleoperation/robolab/convert_robolab_to_robomimic.py \\\n"
        f"        --input_dir {output_dir} --output {output_dir}/robomimic_dataset.hdf5"
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

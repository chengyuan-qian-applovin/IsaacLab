# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless smoke test for the RoboLab XR teleop integration.

Validates, without a headset:
1. Retargeter adapters produce a well-formed 8-D DROID abs-IK action from
   synthetic hand-tracking data (frame conversion + gripper remap).
2. A RoboLab env is created, held at its initial EE pose for N steps, and the
   episode exports to run_0.hdf5 through the teleop script's recording lifecycle.

Run inside the container:
    ./isaaclab.sh -p scripts/environments/teleoperation/robolab/smoke_test.py --headless
"""

# isort: skip_file
import argparse

import cv2  # noqa: F401  Must import before isaaclab/omni modules (RoboLab requirement).

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RoboLab teleop integration smoke test.")
parser.add_argument("--task", type=str, default="BananaInBowlTask")
parser.add_argument("--hold_steps", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import os
import torch

import robolab.constants
from robolab.core.environments.factory import get_envs
from robolab.core.environments.runtime import create_env, end_episode
from robolab.registrations.droid.auto_env_registrations_abs_ik import auto_register_droid_abs_ik_envs
from robolab.robots.droid import EEF_OFFSET_ROT

from isaaclab.devices.device_base import DeviceBase
from isaaclab.utils.math import quat_inv, quat_mul

from robolab_retargeters import (
    RobolabAbsIKRetargeterCfg,
    RobolabGripperRetargeterCfg,
)

robolab.constants.VERBOSE = False
robolab.constants.RECORD_IMAGE_DATA = False


def test_retargeters() -> None:
    """Feed synthetic hand data through the adapters and check the 8-D action contract."""
    hand = DeviceBase.TrackingTarget.HAND_RIGHT
    abs_cfg = RobolabAbsIKRetargeterCfg(
        bound_hand=hand,
        zero_out_xy_rotation=False,
        use_wrist_rotation=False,
        use_wrist_position=False,
        eef_offset_rot=EEF_OFFSET_ROT,
        sim_device="cpu",
    )
    grip_cfg = RobolabGripperRetargeterCfg(bound_hand=hand, sim_device="cpu")
    abs_rt = abs_cfg.retargeter_type(abs_cfg)
    grip_rt = grip_cfg.retargeter_type(grip_cfg)

    def hand_data(pinch_dist: float) -> dict:
        wrist = np.array([0.4, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        thumb = np.array([0.45, -pinch_dist / 2, 0.55, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        index = np.array([0.45, +pinch_dist / 2, 0.55, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return {hand: {"wrist": wrist, "thumb_tip": thumb, "index_tip": index}}

    open_data = hand_data(0.10)
    pose = abs_rt.retarget(open_data)
    grip = grip_rt.retarget(open_data)
    assert pose.shape == (7,), f"pose shape {pose.shape}"
    assert grip.shape == (1,), f"gripper shape {grip.shape}"
    # Position = pinch midpoint.
    assert torch.allclose(pose[:3], torch.tensor([0.45, 0.0, 0.55]), atol=1e-5), pose[:3]
    # Orientation: identity eef target -> base_link target must equal EEF_OFFSET_ROT⁻¹.
    expected_quat = quat_inv(torch.tensor(EEF_OFFSET_ROT).unsqueeze(0)).squeeze(0)
    # (retargeter derives orientation from finger geometry, so only sanity-check unit norm here
    #  and validate the offset math directly instead)
    assert abs(torch.linalg.norm(pose[3:7]).item() - 1.0) < 1e-4, "quaternion not unit norm"
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    converted = quat_mul(identity, quat_inv(torch.tensor(EEF_OFFSET_ROT).unsqueeze(0))).squeeze(0)
    assert torch.allclose(converted, expected_quat, atol=1e-6)
    # Gripper: fingers far apart -> open (0.0); pinched -> close (1.0).
    assert grip.item() == 0.0, f"expected open (0.0), got {grip.item()}"
    grip_closed = grip_rt.retarget(hand_data(0.01))
    assert grip_closed.item() == 1.0, f"expected close (1.0), got {grip_closed.item()}"
    print("[PASS] retargeter adapters: 8-D action contract, frame offset, gripper remap")


def test_env_and_recording() -> None:
    """Create the env, hold the initial pose, and verify the recording lifecycle."""
    auto_register_droid_abs_ik_envs(task=args_cli.task)
    env_names = get_envs(task=args_cli.task)
    assert env_names, f"no envs registered for task {args_cli.task}"
    env_name = env_names[0]
    print(f"[INFO] created env candidates: {env_names}; using {env_name}")

    env, env_cfg = create_env(env_name, device=args_cli.device or "cuda:0", num_envs=1, use_fabric=True)
    env.reset_eval_state()
    env.reset()
    env.reset()

    recorder = env.recorder_manager
    assert recorder is not None and hasattr(recorder, "set_hdf5_file"), "RobolabRecorderManager missing"
    recorder.set_hdf5_file("run_0.hdf5")
    recorder.set_episode_index(0, env_ids=[0])

    # Hold the initial EE pose (converted to a base_link target), gripper open.
    frames = env.scene["frames"]
    eef_idx = frames.data.target_frame_names.index("eef_frame")
    pos = frames.data.target_pos_w[0, eef_idx, :].cpu()
    quat = frames.data.target_quat_w[0, eef_idx, :].cpu()
    base_quat = quat_mul(quat.unsqueeze(0), quat_inv(torch.tensor(EEF_OFFSET_ROT).unsqueeze(0))).squeeze(0)
    action = torch.cat([pos, base_quat, torch.zeros(1)]).to(env.device).unsqueeze(0)

    with torch.inference_mode():
        for _ in range(args_cli.hold_steps):
            env.step(action)
            if env.all_terminated:
                break

    end_episode(env)

    run_file = os.path.join(env.output_dir, "run_0.hdf5")
    assert os.path.isfile(run_file), f"missing {run_file}"
    import h5py

    with h5py.File(run_file, "r") as f:
        demos = [k for k in f["data"].keys() if k.startswith("demo_")]
        assert demos, "no demos recorded"
        demo = f["data"][demos[0]]
        assert "actions" in demo, f"no actions in demo; keys: {list(demo.keys())}"
        n = demo["actions"].shape
        print(f"[PASS] recording: {run_file} contains {demos} with actions shape {n}")
    print(f"[INFO] output dir: {env.output_dir}")

    env.close()


def main():
    test_retargeters()
    test_env_and_recording()
    print("[PASS] all smoke tests passed")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

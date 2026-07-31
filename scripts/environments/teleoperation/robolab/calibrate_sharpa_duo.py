# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Derive the OpenXR-wrist -> panda_link8 rotation offsets for the SharpaWave duo rig.

Creates the rig at its IK-solved ready pose (fingers forward +x, palms down),
reads each flange's orientation, and combines it with the analytic OpenXR wrist
orientation for the same human pose (per XR_EXT_hand_tracking: -Z along the bone
toward the fingertips, +Y away from the palm/dorsal):

    q_offset = q_xr_ready^-1 ⊗ q_link8_ready

Prints the wxyz constants to bake into ``FrankaDuoSharpaRetargeterCfg``.

    ./isaaclab.sh -p scripts/environments/teleoperation/robolab/calibrate_sharpa_duo.py --headless
"""

# isort: skip_file
import argparse
import functools

import cv2  # noqa: F401  Must import before isaaclab/omni modules (RoboLab requirement).

print = functools.partial(print, flush=True)  # noqa: A001

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="BananaInBowlTask")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

import robolab.constants  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.registrations.sharpa_wave.auto_env_registrations_duo import auto_register_franka_duo_envs  # noqa: E402

from isaaclab.utils.math import quat_inv, quat_mul  # noqa: E402

robolab.constants.VERBOSE = False
robolab.constants.RECORD_IMAGE_DATA = False


def analytic_xr_wrist_ready() -> torch.Tensor:
    """OpenXR wrist orientation (w,x,y,z) for palms-down, fingers-forward(+x)."""
    # Columns are the joint axes in world: X=-y_w, Y=+z_w, Z=-x_w.
    m = np.array([
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    q = R.from_matrix(m).as_quat()  # x,y,z,w
    return torch.tensor([q[3], q[0], q[1], q[2]], dtype=torch.float32)


def main():
    auto_register_franka_duo_envs(task=[args_cli.task], ik_actions=True)
    env_name = get_envs(task=[args_cli.task])[0]
    env, _ = create_env(env_name, num_envs=1, use_fabric=True)
    env.reset()

    robot = env.scene["robot"]
    q_xr = analytic_xr_wrist_ready().unsqueeze(0)
    root_pos = robot.data.root_pos_w[0] - env.scene.env_origins[0]
    root_quat = robot.data.root_quat_w[0]
    print(f"\nrobot root (env frame): pos={root_pos.cpu().tolist()} quat={root_quat.cpu().tolist()}")

    for side in ("left", "right"):
        idx = robot.data.body_names.index(f"{side}_panda_link8")
        q_link8 = robot.data.body_quat_w[0, idx].cpu().unsqueeze(0)
        p_link8 = (robot.data.body_pos_w[0, idx] - env.scene.env_origins[0]).cpu()
        q_offset = quat_mul(quat_inv(q_xr), q_link8).squeeze(0)
        print(f"{side}: link8 pos(env)={[round(v, 4) for v in p_link8.tolist()]}")
        print(f"{side}: link8 quat={[round(v, 5) for v in q_link8.squeeze(0).tolist()]}")
        print(f"{side}_wrist_rot_offset = ({', '.join(f'{v:.6f}' for v in q_offset.tolist())})  # w,x,y,z")

    # Distance flange -> hand wrist body, for wrist_pos_offset tuning.
    for side in ("left", "right"):
        i8 = robot.data.body_names.index(f"{side}_panda_link8")
        try:
            iw = robot.data.body_names.index(f"{side}_hand_wrist")
        except ValueError:
            continue
        d = torch.linalg.norm(robot.data.body_pos_w[0, iw] - robot.data.body_pos_w[0, i8]).item()
        print(f"{side}: |flange -> hand_wrist| = {d:.4f} m  (candidate wrist_pos_offset z)")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

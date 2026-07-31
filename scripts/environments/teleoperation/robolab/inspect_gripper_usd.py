# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Dump the DROID robot's per-joint drive gains, limits and tendon count from a live RoboLab env.

Answers "what holds the passive Robotiq linkage joints in place?" with the values
PhysX actually uses (USD-baked drives for joints not covered by Isaac Lab actuators).

    ./isaaclab.sh -p scripts/environments/teleoperation/robolab/inspect_gripper_usd.py --headless
"""

# isort: skip_file
import argparse

import cv2  # noqa: F401  Must import before isaaclab/omni modules (RoboLab requirement).

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="BananaInBowlTask")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_abs_ik import auto_register_droid_abs_ik_envs  # noqa: E402

robolab.constants.VERBOSE = False
robolab.constants.RECORD_IMAGE_DATA = False

auto_register_droid_abs_ik_envs(task=args_cli.task)
env, _ = create_env(get_envs(task=args_cli.task)[0], num_envs=1, use_fabric=True)
env.reset()

robot = env.scene["robot"]
data = robot.data
print(f"\nJOINT PROPERTY DUMP  (num_joints={robot.num_joints}, fixed_tendons={robot.num_fixed_tendons})")
print(f"{'joint':34s} {'stiffness':>12s} {'damping':>10s} {'limit_lo':>9s} {'limit_hi':>9s} {'friction':>9s}")
limits = data.joint_pos_limits[0]
friction = getattr(data, "joint_friction_coeff", getattr(data, "joint_friction", None))
for i, name in enumerate(robot.joint_names):
    fr = f"{friction[0, i].item():9.3f}" if friction is not None else "      n/a"
    print(
        f"{name:34s} {data.default_joint_stiffness[0, i].item():12.2f}"
        f" {data.default_joint_damping[0, i].item():10.3f}"
        f" {limits[i, 0].item():9.3f} {limits[i, 1].item():9.3f} {fr}"
    )
print(f"\nactuated by Isaac Lab actuators: {[n for a in robot.actuators.values() for n in a.joint_names]}")

env.close()
simulation_app.close()

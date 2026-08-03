# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Measure RoboLab env.step() wall cost vs device and solver iterations.

Boots kit once, then creates the env at several solver-iteration settings and
times stepping with a fixed hold-pose action. Separates physics cost from
iteration count to reveal whether GPU stepping is solver-bound or
overhead-bound (kernel launch / CPU-GPU sync).

    ./isaaclab.sh -p .../benchmark_step_cost.py --headless --device cuda:0
    ./isaaclab.sh -p .../benchmark_step_cost.py --headless --device cpu
"""

# isort: skip_file
import argparse
import functools
import time

import cv2  # noqa: F401

print = functools.partial(print, flush=True)  # noqa: A001

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="BananaInBowlTask")
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--iters", type=int, nargs="*", default=[32, 8], help="solver iteration settings to test")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import robolab.constants
from robolab.core.environments.config import parse_env_cfg as robolab_parse_env_cfg
from robolab.core.environments.factory import get_envs
from robolab.core.environments.runtime import create_env
from robolab.robots.droid import EEF_OFFSET_ROT

from isaaclab.utils.math import quat_inv, quat_mul

from robolab_teleop_common import strip_cameras_for_xr

robolab.constants.VERBOSE = False
robolab.constants.RECORD_IMAGE_DATA = False

from robolab.registrations.droid.auto_env_registrations_abs_ik import auto_register_droid_abs_ik_envs

auto_register_droid_abs_ik_envs(task=args_cli.task)
env_name = get_envs(task=args_cli.task)[0]

results = []
for iters in args_cli.iters:
    env_cfg = robolab_parse_env_cfg(env_name, device=args_cli.device or "cuda:0", num_envs=1, use_fabric=True)
    strip_cameras_for_xr(env_cfg)  # isolate physics: no camera sensors
    env_cfg.sim.physx.max_position_iteration_count = iters
    env_cfg.sim.render_interval = 10_000  # effectively never render inside step
    env, _ = create_env(env_cfg, num_envs=1, use_fabric=True)
    env.reset()

    frames = env.scene["frames"]
    eef_idx = frames.data.target_frame_names.index("eef_frame")
    pos = frames.data.target_pos_w[0, eef_idx, :].cpu()
    quat = frames.data.target_quat_w[0, eef_idx, :].cpu()
    base_quat = quat_mul(quat.unsqueeze(0), quat_inv(torch.tensor(EEF_OFFSET_ROT).unsqueeze(0))).squeeze(0)
    # Press slightly INTO the table plane to create sustained contact load.
    contact_pos = pos.clone()
    contact_pos[2] = max(0.02, pos[2].item() - 0.22)
    hold = torch.cat([pos, base_quat, torch.zeros(1)]).to(env.device).unsqueeze(0)
    press = torch.cat([contact_pos, base_quat, torch.ones(1)]).to(env.device).unsqueeze(0)

    with torch.inference_mode():
        for phase, action in (("free", hold), ("contact", press)):
            for _ in range(20):  # warmup / move into phase
                env.step(action)
            t0 = time.perf_counter()
            for _ in range(args_cli.steps):
                env.step(action)
            dt_ms = (time.perf_counter() - t0) / args_cli.steps * 1000.0
            results.append((args_cli.device, iters, phase, dt_ms))
            print(f"[RESULT] device={args_cli.device} iters={iters} phase={phase} "
                  f"step={dt_ms:.2f} ms  ({1000.0/dt_ms:.1f} Hz control)")
    env.close()

print("[SUMMARY]")
for dev, iters, phase, ms in results:
    print(f"  {dev:8s} iters={iters:<3d} {phase:8s} {ms:7.2f} ms/step")
simulation_app.close()

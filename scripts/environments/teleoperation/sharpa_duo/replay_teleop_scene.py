# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kinematic replay of recorded duo teleop episodes: same USDA scene, one camera, MP4 out.

Loads the same scene USDA the episodes were recorded in, places the rig the same
way, and plays back the recorded joint states + object poses with **no physics at
all**: every frame is a prescribed state (``write_joint_position_to_sim_index`` /
``write_root_pose_to_sim_index``) pushed through ``sim.forward()`` (forward
kinematics + Fabric) — ``env.step()`` is never called, so nothing is solved,
integrated, or collided, and every PhysX solver knob is floored on top. A single
third-person RGB camera renders each demo to ``<output_dir>/<demo>/video.mp4``
at 30 fps (recordings are 60 Hz; frames are strided 2:1).

No domain randomization — each selected demo is replayed once, as recorded.

Example:

    ./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/replay_teleop_scene.py \\
        --scene_usda ~/sim_benchmark/scene/taco_hoi_178_023.usda --headless
"""

# isort: skip_file
import argparse
import functools
import json
import math
import os

print = functools.partial(print, flush=True)  # noqa: A001

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Kinematic replay of duo teleop recordings.")
parser.add_argument("--scene_usda", type=str, required=True, help="The scene USD/USDA the episodes were recorded in.")
parser.add_argument(
    "--embodiment",
    type=str,
    choices=("franka_duo", "yam_duo"),
    default=None,
    help=(
        "Robot embodiment the demos were recorded with (see duo_robot.py). Defaults to the"
        " 'embodiment' attribute stamped on the first selected demo (franka_duo for datasets"
        " recorded before the attribute existed)."
    ),
)
parser.add_argument(
    "--robot_pos",
    type=float,
    nargs=3,
    default=None,
    help=(
        "Rig root position in the scene frame [m]; must match the recording session."
        " Defaults to the embodiment's standard placement."
    ),
)
parser.add_argument(
    "--robot_rot",
    type=float,
    nargs=4,
    default=None,
    help="Rig root orientation quaternion (x y z w); must match the recording session.",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="./datasets/duo_teleop",
    help="Recorded HDF5 dataset from make_teleop_scene.py — a file, or a directory "
    "(the newest .hdf5 inside that contains demos is used).",
)
parser.add_argument(
    "--episodes",
    type=str,
    default="all",
    help="Which demos to replay: 'all', 'success', 'failure', or comma-separated indices (e.g. '0,3,7').",
)
parser.add_argument(
    "--output_dir", type=str, default="./replays", help="Output root; videos land in <output_dir>/<demo>/video.mp4."
)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument(
    "--cam_eye",
    type=float,
    nargs=3,
    default=(1.9, -2.0, 1.8),
    help="Camera position [m]. Default frames rig and tabletop from the operator's right.",
)
parser.add_argument("--cam_lookat", type=float, nargs=3, default=(0.0, -0.2, 0.8), help="Camera look-at point [m].")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # camera sensors need offscreen rendering

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

"""Rest everything follows."""

import h5py
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_physx.physics import PhysxCfg

from duo_env import DuoEnvCfg, DuoSceneCfg
from duo_robot import EMBODIMENTS
from usda_scene import add_usda_scene

# 45° vertical FOV at the output aspect ratio, on the standard 20.955 mm aperture:
# focal = aperture / (2 * tan(vfov/2) * width/height).
_APERTURE_MM = 20.955
_FOCAL_45V = _APERTURE_MM / (2.0 * math.tan(math.radians(45.0) / 2.0) * (args_cli.width / args_cli.height))


@configclass
class DuoReplaySceneCfg(DuoSceneCfg):
    """Duo scene + one third-person camera (spawned at identity, posed at runtime)."""

    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/replay_cam",
        update_period=0.0,
        height=args_cli.height,
        width=args_cli.width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_FOCAL_45V,
            horizontal_aperture=_APERTURE_MM,
            clipping_range=(0.02, 20.0),
        ),
    )


@configclass
class ReplayActionsCfg:
    """No action terms: the replay never steps the env, so no controllers are built."""


@configclass
class DuoReplayEnvCfg(DuoEnvCfg):
    """Pure kinematics. ``env.step()`` is never called; the solver floors are
    belt-and-suspenders for anything that does touch PhysX."""

    scene: DuoReplaySceneCfg = DuoReplaySceneCfg(num_envs=1, env_spacing=3.0)
    actions: ReplayActionsCfg = ReplayActionsCfg()

    def __post_init__(self):
        self.episode_length_s = 10000.0
        self.decimation = 1
        self.sim.dt = 1 / 30
        self.sim.render_interval = 1
        self.sim.physics = PhysxCfg(
            enable_ccd=False,
            min_position_iteration_count=1,
            max_position_iteration_count=1,
            min_velocity_iteration_count=0,
            max_velocity_iteration_count=0,
        )


# ---------------------------------------------------------------------------
# Dataset access
# ---------------------------------------------------------------------------


def resolve_dataset_path(path: str) -> str:
    """Accept a file or a directory; in a directory, pick the newest .hdf5 that
    actually contains demos (every teleop start leaves a file, but sessions that
    exported nothing leave an empty shell)."""
    path = os.path.abspath(path)
    if os.path.isdir(path):
        candidates = [os.path.join(path, n) for n in os.listdir(path) if n.endswith(".hdf5")]
        if not candidates:
            raise SystemExit(f"No .hdf5 files in {path}")
        non_empty = []
        for candidate in candidates:
            try:
                with h5py.File(candidate, "r") as f:
                    if "data" in f and len(f["data"]) > 0:
                        non_empty.append(candidate)
            except OSError:
                continue  # unreadable / still being written
        if not non_empty:
            raise SystemExit(f"No .hdf5 file in {path} contains demos (all empty shells).")
        path = max(non_empty, key=os.path.getmtime)
    return path


def list_demos(f: h5py.File) -> list[tuple[str, bool, int]]:
    """(demo name, success, num steps), sorted by demo index."""
    demos = []
    for name, group in f["data"].items():
        success = bool(group.attrs.get("success", False))
        steps = int(group.attrs.get("num_samples", len(group["actions"]) if "actions" in group else 0))
        demos.append((name, success, steps))
    demos.sort(key=lambda d: int(d[0].split("_")[-1]))
    return demos


def select_demos(demos: list[tuple[str, bool, int]], selector: str) -> list[str]:
    if selector == "all":
        return [d[0] for d in demos]
    if selector == "success":
        return [d[0] for d in demos if d[1]]
    if selector == "failure":
        return [d[0] for d in demos if not d[1]]
    wanted = {int(tok) for tok in selector.split(",") if tok.strip() != ""}
    return [d[0] for d in demos if int(d[0].split("_")[-1]) in wanted]


def load_episode(f: h5py.File, demo: str) -> dict:
    g = f[f"data/{demo}"]
    states = g["states"]
    out = {
        "joint_position": np.asarray(states["articulation"]["robot"]["joint_position"]),
        "objects": {
            name: np.asarray(states["rigid_object"][name]["root_pose"])
            for name in states.get("rigid_object", {}).keys()
        },
        "success": bool(g.attrs.get("success", False)),
    }
    # Recorded step dt (control rate) from env_args, default teleop's 1/60.
    step_dt = 1.0 / 60.0
    env_args = f["data"].attrs.get("env_args")
    if env_args is not None:
        try:
            sim_args = json.loads(env_args).get("sim_args", {})
            step_dt = float(sim_args["dt"]) * int(sim_args["decimation"])
        except Exception:
            pass
    out["step_dt"] = step_dt
    return out


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def render_frame(env: ManagerBasedRLEnv) -> None:
    """Forward kinematics + render + sensor refresh for the current prescribed state."""
    env.sim.forward()
    env.sim.render()
    env.scene.update(dt=env.cfg.sim.dt)


def replay_episode(env: ManagerBasedRLEnv, episode: dict, writer) -> int:
    """Write each (strided) recorded state and capture the camera. Returns frame count."""
    robot = env.scene["robot"]
    stride = max(1, round((1.0 / 30.0) / episode["step_dt"]))
    origin = env.scene.env_origins[0]

    q = torch.tensor(episode["joint_position"], dtype=torch.float32, device=env.device)
    objs = {
        name: torch.tensor(poses, dtype=torch.float32, device=env.device) for name, poses in episode["objects"].items()
    }
    zeros_j = torch.zeros(1, q.shape[1], device=env.device)
    zeros_6 = torch.zeros(1, 6, device=env.device)

    frames = 0
    for t in range(0, q.shape[0], stride):
        robot.write_joint_position_to_sim_index(position=q[t].unsqueeze(0))
        robot.write_joint_velocity_to_sim_index(velocity=zeros_j)
        for name, poses in objs.items():
            pose = poses[t].clone().unsqueeze(0)
            pose[:, :3] += origin  # states were recorded env-origin-relative
            env.scene[name].write_root_pose_to_sim_index(root_pose=pose)
            env.scene[name].write_root_velocity_to_sim_index(root_velocity=zeros_6)
        render_frame(env)
        rgb = env.scene["camera"].data.output["rgb"][0].cpu().numpy()
        writer.append_data(np.ascontiguousarray(rgb[..., :3].astype(np.uint8)))
        frames += 1
    return frames


def main():
    import imageio.v2 as imageio

    dataset_path = resolve_dataset_path(args_cli.dataset)
    print(f"[INFO] Dataset: {dataset_path}")
    with h5py.File(dataset_path, "r") as f:
        demos = list_demos(f)
        if not demos:
            raise SystemExit(f"No demos in {dataset_path}")
        print("[INFO] Demos in dataset:")
        for name, success, steps in demos:
            print(f"    {name}: success={success}, steps={steps}")
        selected = select_demos(demos, args_cli.episodes)
        if not selected:
            raise SystemExit(f"--episodes {args_cli.episodes!r} selected nothing.")
        episodes = {demo: load_episode(f, demo) for demo in selected}
        recorded_embodiment = str(f[f"data/{selected[0]}"].attrs.get("embodiment", "franka_duo"))
    print(f"[INFO] Replaying {len(selected)} demo(s): {selected}")

    embodiment = args_cli.embodiment or recorded_embodiment
    if embodiment not in EMBODIMENTS:
        raise SystemExit(f"Unknown embodiment {embodiment!r} (recorded attribute?); pass --embodiment.")
    spec = EMBODIMENTS[embodiment]
    print(f"[INFO] Embodiment: {embodiment}")
    robot_pos = tuple(args_cli.robot_pos) if args_cli.robot_pos is not None else spec.default_robot_pos
    robot_rot = tuple(args_cli.robot_rot) if args_cli.robot_rot is not None else spec.default_robot_rot

    env_cfg = DuoReplayEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.robot = spec.robot_cfg(pos=robot_pos, rot=robot_rot)
    add_usda_scene(env_cfg.scene, args_cli.scene_usda)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()

    eye = torch.tensor([args_cli.cam_eye], dtype=torch.float32, device=env.device)
    lookat = torch.tensor([args_cli.cam_lookat], dtype=torch.float32, device=env.device)
    env.scene["camera"].set_world_poses_from_view(eye, lookat)
    # Let the RTX renderer converge (shader/texture loads) before the first capture.
    for _ in range(5):
        render_frame(env)

    for demo in selected:
        episode = episodes[demo]
        out_dir = os.path.join(os.path.abspath(args_cli.output_dir), demo)
        os.makedirs(out_dir, exist_ok=True)
        writer = imageio.get_writer(os.path.join(out_dir, "video.mp4"), fps=30, codec="libx264", quality=7)
        env.reset()
        frames = replay_episode(env, episode, writer)
        writer.close()
        with open(os.path.join(out_dir, "meta.json"), "w") as meta:
            json.dump(
                {
                    "dataset": dataset_path,
                    "demo": demo,
                    "success": episode["success"],
                    "frames": frames,
                    "fps": 30,
                },
                meta,
                indent=2,
            )
        print(f"[INFO] {demo}: {frames} frames (success={episode['success']}) -> {out_dir}")

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

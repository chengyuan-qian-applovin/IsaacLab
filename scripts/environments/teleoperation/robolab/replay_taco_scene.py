# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kinematic replay of recorded TACO teleop episodes, with cameras + domain randomization.

Inherits the teleop scene/env cfg (``taco_scene_common.py``) and plays back the
recorded joint states + object poses with **no physics at all**: every frame is a
prescribed state (``write_joint_state_to_sim`` / ``write_root_pose_to_sim``) pushed
through ``sim.forward()`` (forward kinematics + Fabric) — ``env.step()`` is never
called, so nothing is solved, integrated, or collided. ``sim.dt = 1/30``.

Four RGB cameras render each episode to MP4: a fixed operator-eye view
(sim_benchmark's calibrated MuJoCo reference viewpoint, 45° vFOV), a third-person
side view, and two wrist cameras on the hand flanges.

Domain randomization runs as a deterministic grid over (background HDRI, lighting
preset, table material) — see ``taco_variations.py``; assets come from a RoboLab
checkout. ``--dr off`` replays each episode once, unrandomized.

Example:

    ./isaaclab.sh -p scripts/environments/teleoperation/robolab/replay_taco_scene.py \\
        --headless --dataset ./datasets/taco_teleop/dataset.hdf5 --episodes success
"""

# isort: skip_file
import argparse
import functools
import json
import math
import os

import cv2  # noqa: F401  Must import before isaaclab/omni modules.

print = functools.partial(print, flush=True)  # noqa: A001

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Kinematic replay of TACO teleop recordings.")
parser.add_argument("--dataset", type=str, default="./datasets/taco_teleop",
                    help="Recorded HDF5 dataset from teleop_taco_scene.py — a file, or a directory "
                         "(the newest .hdf5 inside is used; teleop writes one timestamped file per run).")
parser.add_argument("--episodes", type=str, default="success",
                    help="Which demos to replay: 'success', 'all', or comma-separated indices (e.g. '0,3,7').")
parser.add_argument("--output_dir", type=str, default="./replays",
                    help="Output root; videos land in <output_dir>/<combo>/demo_<N>/<camera>.mp4.")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--dr", choices=("grid", "off"), default="grid",
                    help="Domain randomization: deterministic grid (default) or off.")
parser.add_argument("--backgrounds", type=str, nargs="*", default=None,
                    help="Background HDRI names (file stems under <robolab>/assets/backgrounds). Default: curated 3.")
parser.add_argument("--lightings", type=str, nargs="*", default=None,
                    help="Lighting preset names (default/dim/bright/warm/cool). Default: default dim warm.")
parser.add_argument("--table_materials", type=str, nargs="*", default=None,
                    help="Table material names (Oak/Walnut_Planks/Bamboo/Carpaint_Solid). Default: all 4.")
parser.add_argument("--robolab_dir", type=str, default=None,
                    help="RoboLab checkout providing HDRI/MDL assets "
                         "(default: /workspace/robolab if mounted, else ~/RoboLab).")
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

import taco_variations as variations
from taco_scene_common import TacoTeleopEnvCfg, TacoTeleopSceneCfg
from sim_benchmark.taco_hoi import REFERENCE_EYE, REFERENCE_LOOKAT  # noqa: E402  (sys.path set by taco_scene_common)

# 45° vertical FOV at the output aspect ratio, on the standard 20.955 mm aperture:
# focal = aperture / (2 * tan(vfov/2) * width/height).
_APERTURE_MM = 20.955
_FOCAL_45V = _APERTURE_MM / (2.0 * math.tan(math.radians(45.0) / 2.0) * (args_cli.width / args_cli.height))

_THIRD_PERSON_EYE = (1.15, 0.35, 1.2)
_TABLE_CENTER = (0.0, 0.0, 0.55)


def _camera(prim_path: str, offset: CameraCfg.OffsetCfg | None = None) -> CameraCfg:
    return CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        height=args_cli.height,
        width=args_cli.width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_FOCAL_45V,
            horizontal_aperture=_APERTURE_MM,
            clipping_range=(0.02, 20.0),
        ),
        offset=offset if offset is not None else CameraCfg.OffsetCfg(),
    )


@configclass
class TacoReplaySceneCfg(TacoTeleopSceneCfg):
    """Teleop scene + the four replay cameras (teleop stays camera-free: XR deadlock)."""

    # Fixed world cameras: spawned at identity, posed via set_world_poses_from_view.
    operator_cam = _camera("{ENV_REGEX_NS}/operator_cam")
    third_person_cam = _camera("{ENV_REGEX_NS}/third_person_cam")
    # Wrist cameras ride the hand flanges. ROS convention: +Z forward — identity
    # looks along the flange axis (into the hand / at the fingers). First-guess
    # extrinsics: tune pos/rot by eye after one replay.
    wrist_cam_left = _camera(
        "{ENV_REGEX_NS}/robot/left_hand/left_hand_flange/wrist_cam",
        CameraCfg.OffsetCfg(pos=(0.0, -0.05, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
    )
    wrist_cam_right = _camera(
        "{ENV_REGEX_NS}/robot/right_hand/right_hand_flange/wrist_cam",
        CameraCfg.OffsetCfg(pos=(0.0, -0.05, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
    )


@configclass
class TacoReplayEnvCfg(TacoTeleopEnvCfg):
    """Point 7: pure kinematics. env.step() is never called; these floors are
    belt-and-suspenders for anything that does touch PhysX."""

    scene: TacoReplaySceneCfg = TacoReplaySceneCfg(num_envs=1, env_spacing=3.0)

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 1 / 30
        self.decimation = 1
        self.sim.render_interval = 1
        self.sim.physx.enable_ccd = False
        self.sim.physx.min_position_iteration_count = 1
        self.sim.physx.max_position_iteration_count = 1
        self.sim.physx.min_velocity_iteration_count = 0
        self.sim.physx.max_velocity_iteration_count = 0


# ---------------------------------------------------------------------------
# Dataset access
# ---------------------------------------------------------------------------


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
    wanted = {int(tok) for tok in selector.split(",") if tok.strip() != ""}
    return [d[0] for d in demos if int(d[0].split("_")[-1]) in wanted]


def load_episode(f: h5py.File, demo: str) -> dict:
    g = f[f"data/{demo}"]
    states = g["states"]
    out = {
        "joint_position": np.asarray(states["articulation"]["robot"]["joint_position"]),
        "objects": {
            name: np.asarray(states["rigid_object"][name]["root_pose"])
            for name in states["rigid_object"].keys()
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

CAMERA_NAMES = ("operator_cam", "third_person_cam", "wrist_cam_left", "wrist_cam_right")


def render_frame(env) -> None:
    """Forward kinematics + render + sensor refresh for the current prescribed state."""
    env.sim.forward()
    env.sim.render()
    env.scene.update(dt=env.cfg.sim.dt)


def replay_episode(env, episode: dict, writers: dict) -> int:
    """Write each (strided) recorded state and capture all cameras. Returns frame count."""
    robot = env.scene["robot"]
    stride = max(1, round((1.0 / 30.0) / episode["step_dt"]))
    origin = env.scene.env_origins[0]

    q = torch.tensor(episode["joint_position"], dtype=torch.float32, device=env.device)
    objs = {
        name: torch.tensor(poses, dtype=torch.float32, device=env.device)
        for name, poses in episode["objects"].items()
    }
    zeros_j = torch.zeros(1, q.shape[1], device=env.device)
    zeros_6 = torch.zeros(1, 6, device=env.device)

    frames = 0
    for t in range(0, q.shape[0], stride):
        robot.write_joint_state_to_sim(q[t].unsqueeze(0), zeros_j)
        for name, poses in objs.items():
            pose = poses[t].clone().unsqueeze(0)
            pose[:, :3] += origin  # states were recorded env-origin-relative
            env.scene[name].write_root_pose_to_sim(pose)
            env.scene[name].write_root_velocity_to_sim(zeros_6)
        render_frame(env)
        for cam_name, writer in writers.items():
            rgb = env.scene[cam_name].data.output["rgb"][0].cpu().numpy()
            writer.append_data(np.ascontiguousarray(rgb[..., :3].astype(np.uint8)))
        frames += 1
    return frames


def resolve_dataset_path(path: str) -> str:
    """Accept a file or a directory; in a directory, pick the newest .hdf5."""
    path = os.path.abspath(path)
    if os.path.isdir(path):
        candidates = [os.path.join(path, n) for n in os.listdir(path) if n.endswith(".hdf5")]
        if not candidates:
            raise SystemExit(f"No .hdf5 files in {path}")
        path = max(candidates, key=os.path.getmtime)
    return path


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
    print(f"[INFO] Replaying {len(selected)} demo(s): {selected}")

    env = ManagerBasedRLEnv(cfg=TacoReplayEnvCfg())
    env.reset()

    # Pose the fixed cameras (spawned at identity).
    eyes = lambda e: torch.tensor([e], dtype=torch.float32, device=env.device)  # noqa: E731
    env.scene["operator_cam"].set_world_poses_from_view(eyes(REFERENCE_EYE), eyes(REFERENCE_LOOKAT))
    env.scene["third_person_cam"].set_world_poses_from_view(eyes(_THIRD_PERSON_EYE), eyes(_TABLE_CENTER))

    # Domain randomization grid.
    if args_cli.dr == "grid":
        robolab_dir = args_cli.robolab_dir or variations.DEFAULT_ROBOLAB_DIR
        print(f"[INFO] DR assets from {robolab_dir}")
        combos = variations.build_grid(
            args_cli.backgrounds or variations.DEFAULT_BACKGROUNDS,
            args_cli.lightings or variations.DEFAULT_LIGHTINGS,
            args_cli.table_materials or variations.DEFAULT_TABLE_MATERIALS,
            robolab_dir=robolab_dir,
        )
        material_prims = variations.spawn_table_materials(
            sorted({c["table_material"] for c in combos}), robolab_dir=robolab_dir
        )
        combos = [c for c in combos if c["table_material"] in material_prims]
        print(f"[INFO] DR grid: {len(combos)} combos x {len(selected)} episodes.")
    else:
        combos = [{"label": "no_dr", "background": None, "lighting": None, "table_material": None}]
        material_prims = {}

    for combo in combos:
        if combo["background"] is not None:
            variations.apply_lighting(combo["lighting"])  # sets dome + key intensities/color
            variations.apply_background(combo["background_path"])
            variations.apply_table_material(material_prims[combo["table_material"]])
        # settle renderer on the new look (shader/texture loads) before capturing
        for _ in range(5):
            render_frame(env)

        for demo in selected:
            episode = episodes[demo]
            out_dir = os.path.join(os.path.abspath(args_cli.output_dir), combo["label"], demo)
            os.makedirs(out_dir, exist_ok=True)
            writers = {
                name: imageio.get_writer(os.path.join(out_dir, f"{name}.mp4"), fps=30, codec="libx264", quality=7)
                for name in CAMERA_NAMES
            }
            env.reset()
            frames = replay_episode(env, episode, writers)
            for writer in writers.values():
                writer.close()
            with open(os.path.join(out_dir, "meta.json"), "w") as meta:
                json.dump({
                    "dataset": dataset_path,
                    "demo": demo,
                    "success": episode["success"],
                    "frames": frames,
                    "fps": 30,
                    "background": combo["background"],
                    "lighting": combo["lighting"],
                    "table_material": combo["table_material"],
                    "cameras": list(CAMERA_NAMES),
                }, meta, indent=2)
            print(f"[INFO] {combo['label']} / {demo}: {frames} frames -> {out_dir}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

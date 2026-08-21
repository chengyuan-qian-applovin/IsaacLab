# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleoperate the sim_benchmark TACO scene (brush + bowl on a table) with the SharpaWave duo.

Scene/env configs live in ``taco_scene_common.py`` (shared with the replay script).
See RECORD_REPLAY_GUIDE.md for the record→replay pipeline this script anchors:

- Records robot joint states + object poses per control step into a robomimic-style
  HDF5 (``--record_dir``), one demo per episode, with a success flag.
- Episode flow: AVP Play starts, the cross-hand stop gesture (all five fingertip
  pairs touching for 0.5 s) ends the episode and pops a Success/Failure dialog on
  the headset; Reset discards the in-flight episode.
- ``--arm_visual`` renders the arms 5% transparent (or hides them) during teleop.
- ``--self_collision`` enables the duo articulation's self collisions.
- The AVP Align button re-anchors the session so the table is straight in front.

Run (sim_benchmark is expected at <IsaacLab>/sim_benchmark):

    ./isaaclab.sh -p scripts/environments/teleoperation/robolab/teleop_taco_scene.py --headless
"""

# isort: skip_file
import argparse
import functools
import os
import time

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
    help=(
        "Print rolling per-stage loop timings once per second: retarget/frame/step, plus "
        "step sub-buckets (render_call/physx/action/write/readback/obs/ik)."
    ),
)
parser.add_argument(
    "--visualize_hands", action="store_true",
    help="Draw red spheres on the tracked OpenXR hand joints (like the default GR1T2 teleop).",
)
parser.add_argument(
    "--render_frequency", type=float, default=30.0,
    help="Target render rate in Hz; the physics-substep render interval is computed "
         "from the sim dt (default 30 = every 16th substep at dt=1/480).",
)
parser.add_argument(
    "--record_dir", type=str, default="./datasets/taco_teleop",
    help="Directory for the recorded HDF5 dataset (one file, one demo per episode).",
)
parser.add_argument(
    "--no_record", action="store_true",
    help="Disable episode recording entirely.",
)
parser.add_argument(
    "--arm_visual", choices=("transparent", "hidden", "normal"), default="transparent",
    help="Arm rendering during teleop: 5%% transparent (default), hidden (render only; physics untouched), or normal.",
)
parser.add_argument(
    "--self_collision", action="store_true",
    help="Enable self collisions on the duo articulation (default off, matching sim_benchmark).",
)
parser.add_argument(
    "--gesture_touch_cm", type=float, default=2.0,
    help="Stop gesture: max same-finger tip distance (cm) counted as touching (default 2).",
)
parser.add_argument(
    "--gesture_hold_s", type=float, default=0.5,
    help="Stop gesture: seconds all five pairs must stay touching to trigger (default 0.5).",
)
parser.add_argument(
    "--align_head_xy", type=float, nargs=2, default=(0.0, -1.0),
    help="Align button: world xy the head is moved to. Default (0, -1.0) stands you "
         "~40 cm from the table's near edge (tabletop spans y in [-0.6, 0.6]).",
)
parser.add_argument(
    "--client_msg_dispatch", action="store_true",
    help="Send server->client messages with dispatch() instead of push() (try this if the S/F dialog never appears).",
)
AppLauncher.add_app_launcher_args(parser)
# Keep AppLauncher's "balanced" rendering default, made explicit here (set_defaults is
# otherwise a no-op). Pass --rendering_mode performance for the cheap preset: profiling
# on the old dt=1/120 config showed ~13 ms/render vs ~24 ms on "balanced".
parser.set_defaults(rendering_mode="balanced")
args_cli = parser.parse_args()
args_cli.xr = True

if not args_cli.headless and not os.environ.get("DISPLAY"):
    print("[WARNING] XR in GUI mode without a DISPLAY: the AR session will never start. Add --headless.")

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from robolab.robots.franka_duo_sharpa_wave import (  # noqa: E402
    LEFT_HAND_JOINTS_ORDERED,
    RIGHT_HAND_JOINTS_ORDERED,
)

import isaaclab.sim as sim_utils
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import (
    InitialStateRecorderCfg,
    PostStepStatesRecorderCfg,
    PreStepActionsRecorderCfg,
)
from isaaclab.managers.recorder_manager import DatasetExportMode, RecorderManagerBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab.managers import EventTermCfg

from robolab_teleop_common import LoopProfiler, filter_self_collision_except_fingertips
from sharpa_duo_retargeters import FrankaDuoSharpaRetargeterCfg
from taco_scene_common import TacoTeleopEnvCfg
from xr_session_tools import (
    AnchorAligner,
    CrossHandStopGesture,
    RawXrCapture,
    RawXrCaptureCfg,
    TeleopCommandBridge,
    current_head_pose,
)


@configclass
class TacoRecorderManagerCfg(RecorderManagerBaseCfg):
    """initial_state + per-step states (joints + object poses) + raw actions.

    All demos land in one file with a per-demo ``success`` attr (EXPORT_ALL);
    exports happen only through this script's explicit calls, never on env.reset
    (``export_in_record_pre_reset=False`` — Reset means discard, not export).
    """

    record_initial_state = InitialStateRecorderCfg()
    record_post_step_states = PostStepStatesRecorderCfg()
    record_pre_step_actions = PreStepActionsRecorderCfg()

    dataset_export_mode = DatasetExportMode.EXPORT_ALL
    export_in_record_pre_reset = False


def apply_arm_visual(mode: str) -> None:
    """Make the two arm subtrees 5% transparent or invisible (render-only)."""
    arm_paths = sim_utils.find_matching_prim_paths("/World/envs/env_.*/robot/(left|right)_arm")
    if not arm_paths:
        print("[WARNING] --arm_visual: no arm prims matched; skipping.")
        return
    if mode == "hidden":
        import isaacsim.core.utils.stage as stage_utils

        stage = stage_utils.get_current_stage()
        for path in arm_paths:
            sim_utils.set_prim_visibility(stage.GetPrimAtPath(path), False)
        print(f"[INFO] Arms hidden (render only): {arm_paths}")
    elif mode == "transparent":
        material_path = "/World/Looks/ArmGhostMaterial"
        sim_utils.spawn_preview_surface(
            material_path,
            sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.78, 0.85), opacity=0.05, roughness=0.0),
        )
        for path in arm_paths:
            sim_utils.bind_visual_material(path, material_path, stronger_than_descendants=True)
        print(f"[INFO] Arms 5% transparent: {arm_paths}")


def main():
    env_cfg = TacoTeleopEnvCfg()
    env_cfg.sim.device = args_cli.device  # honor --device (SimulationCfg defaults to cuda:0 otherwise)
    # Renders happen every N physics substeps; N follows from the requested rate.
    physics_hz = 1.0 / env_cfg.sim.dt
    env_cfg.sim.render_interval = max(1, round(physics_hz / args_cli.render_frequency))
    actual_hz = physics_hz / env_cfg.sim.render_interval
    print(f"[INFO] Render frequency {args_cli.render_frequency:g} Hz -> interval "
          f"{env_cfg.sim.render_interval} ({actual_hz:g} Hz actual at {physics_hz:g} Hz physics)")
    env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = args_cli.self_collision
    if args_cli.self_collision:
        print("[INFO] Self collision: fingertips-only (DP/elastomer/fingertip links of different "
              "fingers collide; everything else within the robot is filtered).")
        # Unfiltered self-collision jams the knuckles: the Sharpa MCP gimbals route
        # through zero-length virtual links with their own convex hulls, and the
        # resulting rest-pose contacts overpower the tiny finger drives. See
        # filter_self_collision_except_fingertips.
        env_cfg.events.filter_self_collision = EventTermCfg(func=filter_self_collision_except_fingertips, mode="startup")
    if args_cli.arm_visual == "transparent":
        # opacity needs translucency support in the renderer; must be set pre-sim-context
        env_cfg.sim.render.enable_translucency = True
    recording = not args_cli.no_record
    if recording:
        env_cfg.recorders = TacoRecorderManagerCfg()
        env_cfg.recorders.dataset_export_dir_path = os.path.abspath(args_cli.record_dir)
        # Unique per run: the HDF5 handler opens its file in "w" (truncate) mode at
        # env creation, so a fixed name would wipe earlier sessions on every start.
        env_cfg.recorders.dataset_filename = time.strftime("dataset_%Y%m%d_%H%M%S")

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()
    apply_arm_visual(args_cli.arm_visual)

    capture_cfg = RawXrCaptureCfg(retargeter_type=RawXrCapture)
    device_cfg = OpenXRDeviceCfg(
        xr_cfg=XrCfg(anchor_pos=tuple(args_cli.anchor_pos), anchor_rot=tuple(args_cli.anchor_rot)),
        retargeters=[
            FrankaDuoSharpaRetargeterCfg(
                left_hand_joint_names=LEFT_HAND_JOINTS_ORDERED,
                right_hand_joint_names=RIGHT_HAND_JOINTS_ORDERED,
                sim_device=env.device,
                enable_visualization=args_cli.visualize_hands,
            ),
            capture_cfg,  # zero-dim: raw hands for the stop gesture, head for align
        ],
    )

    teleop_active = False
    reset_requested = False
    align_requested = False
    # None = not awaiting; float = time the S/F request was last (re)sent to the client.
    awaiting_result_since: float | None = None
    record_result: bool | None = None
    demo_count = 0

    def _start():
        nonlocal teleop_active
        if awaiting_result_since is not None:
            print("[INFO] Ignoring Play: waiting for the Success/Failure dialog (or press Reset to discard).")
            return
        teleop_active = True

    def _stop():
        nonlocal teleop_active
        teleop_active = False

    def _reset():
        nonlocal reset_requested
        reset_requested = True

    def _on_record_result(success: bool):
        nonlocal record_result
        record_result = success

    def _on_align():
        nonlocal align_requested
        align_requested = True

    callbacks = {"START": _start, "STOP": _stop, "RESET": _reset}
    teleop = create_teleop_device("handtracking", DevicesCfg(devices={"handtracking": device_cfg}).devices, callbacks)
    capture = next(r for r in teleop._retargeters if isinstance(r, RawXrCapture))
    bridge = TeleopCommandBridge(_on_record_result, _on_align, use_dispatch=args_cli.client_msg_dispatch)
    gesture = CrossHandStopGesture(touch_dist=args_cli.gesture_touch_cm / 100.0, hold_s=args_cli.gesture_hold_s)
    aligner = AnchorAligner(args_cli.anchor_pos, args_cli.anchor_rot, target_head_xy=tuple(args_cli.align_head_xy))

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

    def episode_has_data() -> bool:
        return recording and not env.recorder_manager.get_episode(0).is_empty()

    def finish_episode():
        """Stop gesture fired: close the buffer and ask the AVP for Success/Failure."""
        nonlocal teleop_active, awaiting_result_since, record_result

        teleop_active = False
        if not episode_has_data():
            print("[INFO] Stop gesture: no recorded steps, nothing to save.")
            return
        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
        record_result = None
        awaiting_result_since = time.monotonic()
        bridge.request_record_result(demo_count)
        # EpisodeData stores each key as a python list of per-step tensors.
        actions = env.recorder_manager.get_episode(0).data.get("actions")
        n_steps = len(actions) if actions is not None else 0
        print(f"[INFO] Episode ended by gesture ({n_steps} steps). "
              "Choose Success/Failure on the headset (Reset discards).")

    def export_episode(success: bool):
        nonlocal awaiting_result_since, demo_count
        env.recorder_manager.set_success_to_episodes([0], torch.tensor([[success]], dtype=torch.bool, device=env.device))
        env.recorder_manager.export_episodes([0])
        env.recorder_manager.reset()
        demo_count += 1
        awaiting_result_since = None
        print(f"[INFO] Episode exported as demo_{demo_count - 1} (success={success}). Press Play for the next one.")

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

    if recording:
        print(f"[INFO] Recording to {env_cfg.recorders.dataset_export_dir_path}/{env_cfg.recorders.dataset_filename}.hdf5")
    print("[INFO] Starting teleop loop. AVP: Play=start, Stop=pause, Reset=reset scene (discards episode), "
          "Align=re-anchor, stop gesture=end episode.")
    with torch.inference_mode():
        while simulation_app.is_running():
            # An exception escaping this loop means simulation_app.close() under a
            # live XR session, which deadlocks kit's shutdown (frozen stream, dead
            # UI; '[XR] Render thread failed to render frame N' repeating in the
            # kit log). Catch, report, pause teleop, and keep the session alive.
            try:
                # Export BEFORE reset: the client auto-sends stop+reset right after
                # the Success/Failure choice, and all three messages can arrive in
                # one pump — reset first would discard the episode before export.
                if record_result is not None and awaiting_result_since is not None:
                    export_episode(record_result)
                    record_result = None
                if reset_requested:
                    if awaiting_result_since is not None:
                        print("[INFO] Reset while awaiting Success/Failure: episode discarded.")
                    if recording:
                        env.recorder_manager.reset()
                    env.reset()
                    teleop.reset()
                    gesture.reset()
                    awaiting_result_since = None
                    reset_requested = False
                if align_requested:
                    align_requested = False
                    # Query XRCore directly: the capture retargeter only refreshes inside
                    # teleop.advance(), which doesn't run while teleop is paused.
                    head = current_head_pose()
                    if head is None:
                        head = capture.head_pose
                    if head is None:
                        print("[WARNING] Align: XR reports no head pose (is the headset session live and the "
                              "visor on?). Try again in a moment.")
                    else:
                        aligner.align(head)
                if awaiting_result_since is not None and time.monotonic() - awaiting_result_since > 5.0:
                    awaiting_result_since = time.monotonic()  # re-send in case the push was lost
                    bridge.request_record_result(demo_count)
                if not teleop_active:
                    prof.begin()
                    env.sim.render()  # wrapped: lands in the "render_call" bucket
                    prof.end()
                    continue
                prof.begin()
                action = teleop.advance()          # XR poll + wrist offsets + DexPilot QPs
                prof.lap("retarget")
                if gesture.update(capture.latest):
                    finish_episode()
                    prof.end()
                    continue
                action = to_root_frame(action.to(env.device))
                prof.lap("frame")
                env.step(action.unsqueeze(0))      # 8 physics substeps; renders per --render_frequency
                prof.lap("step")
                prof.end()
            except Exception:
                import traceback

                traceback.print_exc()
                print("[ERROR] Teleop loop iteration failed (see traceback above). "
                      "Teleop paused; the XR session stays alive. Press Play to retry or Reset to reset.")
                teleop_active = False

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

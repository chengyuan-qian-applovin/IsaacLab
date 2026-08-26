# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load a scene USDA, add the FR3 Duo + SharpaWave rig, and teleoperate it via XR.

The pipeline in one line: your USDA becomes the environment, the duo rig is
placed into it at ``--robot_pos``/``--robot_rot``, and your tracked hands drive
it — wrists through per-arm differential IK, all fingers through DexPilot
retargeting. Episodes are recorded to a robomimic-style HDF5 by default and
labeled hands-free: the cross-hand stop gesture ends an episode, and saying
"success" or "failure" (transcribed locally with OpenAI Whisper) labels and
exports it — see README.md for the full flow.

Example (headless is required for XR without a desktop display):

    ./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/make_teleop_scene.py \\
        --scene_usda ~/sim_benchmark/scene/taco_hoi_178_023.usda --headless

Defaults (robot pose, XR anchor) are the placement validated on the TACO
tabletop scenes: the rig's torso stands south of the table facing +y, and the
anchor puts you at the torso with the tabletop at a comfortable height. For
other scenes, override:

- ``--robot_pos/--robot_rot``: where the rig's torso goes, in the scene's frame.
- ``--anchor_pos/--anchor_rot``: where the XR session's origin goes; standing at
  the rig's torso with the same yaw makes the robot arms line up with yours.

Validation without a headset:

    # env + IK smoke test (holds the ready pose, then tracks a +3 cm target)
    ./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/make_teleop_scene.py \\
        --scene_usda <scene.usda> --smoke 120 --headless

    # microphone + Whisper check (prints levels, transcripts, and label events)
    ./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/make_teleop_scene.py \\
        --scene_usda unused --voice_test 20
"""

# isort: skip_file
import argparse
import functools
import os

print = functools.partial(print, flush=True)  # noqa: A001

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Teleoperate the FR3 Duo + SharpaWave rig in a USDA scene.")
parser.add_argument("--scene_usda", type=str, required=True, help="Path to the scene USD/USDA file to load.")
parser.add_argument(
    "--robot_pos",
    type=float,
    nargs=3,
    default=(0.0, -0.7, 1.0),
    help="Rig torso position in the scene frame [m]. Default matches the TACO tabletop placement.",
)
parser.add_argument(
    "--robot_rot",
    type=float,
    nargs=4,
    default=(0.7071068, 0.0, 0.0, 0.7071068),
    help="Rig torso orientation quaternion (w x y z). Default: +90 deg yaw (facing +y).",
)
parser.add_argument(
    "--anchor_pos",
    type=float,
    nargs=3,
    default=(0.0, -0.7, -0.21),
    help="XR anchor position: default stands you at the rig torso with the TACO tabletop at ~0.75 m.",
)
parser.add_argument(
    "--anchor_rot",
    type=float,
    nargs=4,
    default=(0.7071068, 0.0, 0.0, 0.7071068),
    help="XR anchor rotation (w x y z): should match the robot yaw so the arms line up with yours.",
)
parser.add_argument(
    "--no_track_objects",
    action="store_true",
    help="Do not register the scene's rigid bodies with the env (their poses then survive resets).",
)
parser.add_argument(
    "--render_frequency",
    type=float,
    default=30.0,
    help="Target render rate in Hz; the physics-substep render interval is computed from the sim dt.",
)
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=300.0,
    help="Episode length before the scene auto-resets [s].",
)
parser.add_argument(
    "--cloudxr_env",
    type=str,
    default="cloudxrjs",
    help=(
        "CloudXR .env file path, or a shorthand: 'cloudxrjs' (Quest/Pico, default) or 'avp'"
        " (Apple Vision Pro). 'none' disables the CloudXR auto-launch."
    ),
)
parser.add_argument(
    "--record_dir",
    type=str,
    default="./datasets/duo_teleop",
    help="Directory for the recorded HDF5 dataset (one timestamped file per session, one demo per episode).",
)
parser.add_argument("--no_record", action="store_true", help="Disable episode recording entirely.")
parser.add_argument(
    "--arm_visual",
    choices=("transparent", "hidden", "normal"),
    default="transparent",
    help="Arm rendering during teleop: 5%% transparent (default), hidden (render only), or normal.",
)
parser.add_argument("--visualize_hands", action="store_true", help="Draw red spheres on the tracked XR hand joints.")
parser.add_argument(
    "--gesture_touch_cm",
    type=float,
    default=2.0,
    help="Stop gesture: max same-finger cross-hand tip distance (cm) counted as touching.",
)
parser.add_argument(
    "--gesture_hold_s",
    type=float,
    default=0.5,
    help="Stop gesture: seconds all five pairs must stay touching to trigger.",
)
parser.add_argument("--no_voice", action="store_true", help="Disable the Whisper success/failure voice labeling.")
parser.add_argument(
    "--whisper_model", type=str, default="base.en", help="Whisper model for voice labels (e.g. base.en, small.en)."
)
parser.add_argument(
    "--whisper_device",
    type=str,
    default="cpu",
    help="Torch device for Whisper. Default cpu so transcription never competes with the sim for the GPU.",
)
parser.add_argument("--mic_device", type=str, default="default", help="ALSA capture device (arecord -D).")
parser.add_argument(
    "--voice_test",
    type=int,
    default=None,
    metavar="SECONDS",
    help="Mic/Whisper check without the simulator: listen for N seconds, print transcripts and labels, exit.",
)
parser.add_argument(
    "--smoke",
    type=int,
    default=None,
    metavar="N",
    help="No-XR validation: hold the ready pose for N control steps, report flange drift, exit.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.voice_test is not None:
    # Standalone mic + Whisper check; the simulator never starts.
    import time

    from voice_labeler import VoiceLabeler

    labeler = VoiceLabeler(
        model_name=args_cli.whisper_model, device=args_cli.whisper_device, mic_device=args_cli.mic_device
    )
    print(f"[VOICE] Test mode: say 'success' or 'failure' within the next {args_cli.voice_test} s ...")
    deadline = time.time() + args_cli.voice_test
    next_meter = time.time() + 2.0
    while time.time() < deadline:
        label = labeler.poll()
        if label is not None:
            print(f"[VOICE] EVENT: {label}")
        if time.time() >= next_meter:
            next_meter += 2.0
            peak = labeler.take_peak()
            if labeler.threshold is not None:
                bar = "#" * min(40, int(40 * peak / (4 * labeler.threshold)))
                print(f"[VOICE] level {peak:.4f} / gate {labeler.threshold:.4f} |{bar}", flush=True)
        time.sleep(0.05)
    labeler.close()
    raise SystemExit(0)

if args_cli.smoke is None:
    args_cli.xr = True
    if not args_cli.headless and not os.environ.get("DISPLAY"):
        print("[WARNING] XR in GUI mode without a DISPLAY: the AR session will never start. Add --headless.")

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

"""Rest everything follows."""

import time
import traceback

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import joint_pos_rel, reset_scene_to_default, time_out
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_physx.physics import PhysxCfg

from duo_robot import DuoIKActionsCfg, duo_robot_cfg
from usda_scene import add_usda_scene


# -- Environment config -----------------------------------------------------------


@configclass
class DuoTeleopSceneCfg(InteractiveSceneCfg):
    """The rig plus a dome light; the USDA scene and its objects are added at runtime."""

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.95, 0.95, 0.92)),
    )
    robot = None  # placed from CLI args in main()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=joint_pos_rel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


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
class DuoTeleopEnvCfg(ManagerBasedRLEnvCfg):
    scene: DuoTeleopSceneCfg = DuoTeleopSceneCfg(num_envs=1, env_spacing=3.0)
    actions: DuoIKActionsCfg = DuoIKActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        # Teleop timing: 240 Hz physics, 60 Hz control (decimation 4).
        self.decimation = 4
        self.sim.dt = 1 / 240
        self.sim.physics = PhysxCfg(
            solver_type=1,  # TGS
            max_position_iteration_count=32,
            bounce_threshold_velocity=0.2,
            # Guards fast hand motions against tunneling objects through tables
            # (CPU pipeline only; silently ignored on GPU).
            enable_ccd=True,
        )


def build_env_cfg() -> ManagerBasedRLEnvCfg:
    """Assemble the env config from the CLI arguments."""
    env_cfg = DuoTeleopEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.episode_length_s = args_cli.episode_length_s
    env_cfg.scene.robot = duo_robot_cfg(pos=tuple(args_cli.robot_pos), rot=tuple(args_cli.robot_rot))
    add_usda_scene(env_cfg.scene, args_cli.scene_usda, track_objects=not args_cli.no_track_objects)
    # Render every Nth physics substep, as close to the requested rate as the dt allows.
    interval = max(1, round(1.0 / (env_cfg.sim.dt * args_cli.render_frequency)))
    env_cfg.sim.render_interval = interval
    print(f"[INFO] render interval {interval} substeps -> {1.0 / (env_cfg.sim.dt * interval):.1f} Hz")
    if args_cli.arm_visual == "transparent":
        # Opacity needs translucency support in the renderer; must be set pre-sim-context.
        env_cfg.sim.render.enable_translucency = True
    if args_cli.smoke is None and not args_cli.no_record:
        from recording import DuoRecorderManagerCfg

        env_cfg.recorders = DuoRecorderManagerCfg()
        env_cfg.recorders.dataset_export_dir_path = os.path.abspath(args_cli.record_dir)
        # Unique per session: the HDF5 handler opens its file in "w" (truncate) mode
        # at env creation, so a fixed name would wipe earlier sessions on every start.
        env_cfg.recorders.dataset_filename = time.strftime("dataset_%Y%m%d_%H%M%S")
    return env_cfg


# -- Teleop -----------------------------------------------------------------------


def to_root_frame(env: ManagerBasedRLEnv, action: torch.Tensor) -> torch.Tensor:
    """Convert the two world-frame wrist poses to the robot-root frame.

    The differential-IK action terms expect root-frame commands and do no frame
    conversion themselves; the finger slices (14:58) pass through untouched.
    """
    robot = env.scene["robot"]
    root_pos = robot.data.root_pos_w.torch[0] - env.scene.env_origins[0]
    root_quat = robot.data.root_quat_w.torch[0]
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


def run_teleop(env: ManagerBasedRLEnv) -> None:
    """Drive the env from XR hand tracking until the app closes.

    Episode lifecycle: Play starts teleop (and, when recording, the episode
    buffer). The cross-hand stop gesture ends an episode and waits for a voice
    label; saying "success"/"failure" at any time labels AND ends the current
    episode; either way the labeled demo is exported and the scene resets for
    the next one. Reset (headset button) discards the in-flight episode.
    """
    from isaaclab_teleop import CLOUDXR_AVP_ENV, CLOUDXR_JS_ENV, create_isaac_teleop_device, poll_control_events
    from isaaclab_teleop.isaac_teleop_cfg import IsaacTeleopCfg
    from isaaclab_teleop.xr_cfg import XrCfg

    from duo_teleop_pipeline import build_duo_pipeline
    from recording import XrHandsRecorder
    from xr_extras import XR_EXTRAS_DIM, CrossHandStopGesture, HandJointMarkers, apply_arm_visual

    recording = not args_cli.no_record

    apply_arm_visual(args_cli.arm_visual)
    markers = HandJointMarkers() if args_cli.visualize_hands else None
    gesture = CrossHandStopGesture(touch_dist=args_cli.gesture_touch_cm / 100.0, hold_s=args_cli.gesture_hold_s)

    labeler = None
    if not args_cli.no_voice:
        from voice_labeler import VoiceLabeler

        labeler = VoiceLabeler(
            model_name=args_cli.whisper_model, device=args_cli.whisper_device, mic_device=args_cli.mic_device
        )

    pipeline, retargeters = build_duo_pipeline(include_xr_hands=True)
    teleop_cfg = IsaacTeleopCfg(
        xr_cfg=XrCfg(anchor_pos=tuple(args_cli.anchor_pos), anchor_rot=tuple(args_cli.anchor_rot)),
        pipeline_builder=lambda: pipeline,
        retargeters_to_tune=lambda: retargeters,
        sim_device=env.device,
    )

    cloudxr_env = {"cloudxrjs": CLOUDXR_JS_ENV, "avp": CLOUDXR_AVP_ENV, "none": None}.get(
        args_cli.cloudxr_env, args_cli.cloudxr_env
    )

    teleop_active = False
    reset_requested = False
    awaiting_label = False  # gesture ended the episode; waiting for the voice label
    demo_count = 0

    def _start():
        nonlocal teleop_active
        if awaiting_label:
            print("[INFO] Play ignored: say 'success' or 'failure' first (or press Reset to discard).")
            return
        teleop_active = True

    def _stop():
        nonlocal teleop_active
        teleop_active = False

    def _reset():
        nonlocal reset_requested
        reset_requested = True

    teleop = create_isaac_teleop_device(
        teleop_cfg,
        sim_device=env.device,
        callbacks={"START": _start, "STOP": _stop, "RESET": _reset},
        cloudxr_env_file=cloudxr_env,
        auto_launch_cloudxr=cloudxr_env is not None,
    )

    def episode_has_data() -> bool:
        return recording and not env.recorder_manager.get_episode(0).is_empty()

    def close_episode() -> None:
        """Freeze the episode buffer and start waiting for the voice label."""
        nonlocal teleop_active, awaiting_label
        teleop_active = False
        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
        awaiting_label = True
        actions = env.recorder_manager.get_episode(0).data.get("actions")
        n_steps = len(actions) if actions is not None else 0
        print(f"[INFO] Episode ended ({n_steps} steps). Say 'success' or 'failure' (Reset discards).")

    def export_episode(success: bool) -> None:
        nonlocal awaiting_label, demo_count, reset_requested
        env.recorder_manager.set_success_to_episodes(
            [0], torch.tensor([[success]], dtype=torch.bool, device=env.device)
        )
        env.recorder_manager.export_episodes([0])
        env.recorder_manager.reset()
        demo_count += 1
        awaiting_label = False
        reset_requested = True  # hands-free: fresh scene for the next episode
        print(f"[INFO] Episode exported as demo_{demo_count - 1} (success={success}). Scene resets; press Play.")

    if recording:
        print(
            f"[INFO] Recording to {env.cfg.recorders.dataset_export_dir_path}/{env.cfg.recorders.dataset_filename}.hdf5"
        )
    print(
        "[INFO] Teleop loop started. Headset: Play = start, Stop = pause, Reset = reset (discards episode). "
        "Cross-hand stop gesture ends an episode; say 'success'/'failure' to label it (also ends it mid-run). "
        "Episode timeout discards without export."
    )
    with teleop, torch.inference_mode():
        env.reset()
        teleop.reset()
        while simulation_app.is_running():
            # An exception escaping this loop means simulation_app.close() under a
            # live XR session, which can deadlock kit's shutdown — catch, report,
            # pause teleop, and keep the session alive.
            try:
                # Voice labels first, and exports BEFORE any reset is processed,
                # so a label + reset burst cannot discard the episode.
                label = labeler.poll() if labeler is not None else None
                if label is not None and recording:
                    if not awaiting_label and episode_has_data():
                        # Label spoken mid-episode: it ends AND labels in one utterance.
                        teleop_active = False
                        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                        awaiting_label = True
                    if awaiting_label:
                        export_episode(label == "success")
                    else:
                        print(f"[INFO] Voice label '{label}' ignored: no recorded steps yet.")
                if reset_requested:
                    if awaiting_label:
                        print("[INFO] Reset while awaiting the voice label: episode discarded.")
                    if recording:
                        env.recorder_manager.reset()
                    env.reset()
                    teleop.reset()
                    gesture.reset()
                    awaiting_label = False
                    reset_requested = False

                action = teleop.advance()
                ctrl = poll_control_events(teleop)
                if ctrl.is_active is not None:
                    if ctrl.is_active and awaiting_label:
                        print("[INFO] Play ignored: say 'success' or 'failure' first (or press Reset to discard).")
                    else:
                        teleop_active = ctrl.is_active
                if ctrl.should_reset:
                    reset_requested = True

                # action is None until the XR session has started.
                if action is None:
                    env.sim.render()
                    continue

                xr_hands = action[-XR_EXTRAS_DIM:].reshape(2, 26, 7)
                action = action[:-XR_EXTRAS_DIM]
                XrHandsRecorder.latest = xr_hands
                if markers is not None:
                    markers.update(xr_hands)

                if teleop_active and gesture.update(xr_hands):
                    if episode_has_data():
                        close_episode()
                    else:
                        teleop_active = False
                        print("[INFO] Stop gesture: no recorded steps, teleop paused.")
                    continue
                if not teleop_active:
                    env.sim.render()
                    continue
                action = to_root_frame(env, action.to(env.device))
                env.step(action.unsqueeze(0))  # time_out auto-resets the scene
            except Exception:
                traceback.print_exc()
                print(
                    "[ERROR] Teleop loop iteration failed (see traceback above). Teleop paused; "
                    "the XR session stays alive. Press Play to retry or Reset to reset."
                )
                teleop_active = False
    if labeler is not None:
        labeler.close()


def run_smoke(env: ManagerBasedRLEnv, num_steps: int) -> None:
    """Hold the ready pose for ``num_steps`` control steps and report flange drift."""
    from isaaclab.utils.math import quat_error_magnitude

    env.reset()
    robot = env.scene["robot"]
    flange_ids = [robot.body_names.index(f"{side}_panda_link8") for side in ("left", "right")]

    def flange_poses_root() -> torch.Tensor:
        """Both flange poses in the root frame, flattened to (14,)."""
        chunks = []
        for body_id in flange_ids:
            pos, quat = subtract_frame_transforms(
                robot.data.root_pos_w.torch,
                robot.data.root_quat_w.torch,
                robot.data.body_pos_w.torch[:, body_id],
                robot.data.body_quat_w.torch[:, body_id],
            )
            chunks += [pos[0], quat[0]]
        return torch.cat(chunks)

    action_dim = env.action_manager.total_action_dim
    assert action_dim == 58, f"expected the 58-D duo action space, got {action_dim}"

    with torch.inference_mode():
        # Phase 1: hold the ready pose.
        start = flange_poses_root()
        action = torch.zeros(action_dim, device=env.device)
        action[:14] = start  # fingers stay at zero
        for _ in range(num_steps):
            env.step(action.unsqueeze(0))
        end = flange_poses_root()
        # Phase 2: command both flanges 3 cm up (in the root frame) and check they follow.
        target = start.clone()
        target[2] += 0.03
        target[9] += 0.03
        action[:14] = target
        for _ in range(num_steps):
            env.step(action.unsqueeze(0))
        moved = flange_poses_root()

    ok = True
    for i, side in enumerate(("left", "right")):
        pos_drift = (end[7 * i : 7 * i + 3] - start[7 * i : 7 * i + 3]).norm().item()
        rot_drift = quat_error_magnitude(
            start[7 * i + 3 : 7 * i + 7].unsqueeze(0), end[7 * i + 3 : 7 * i + 7].unsqueeze(0)
        ).item()
        track_err = (moved[7 * i : 7 * i + 3] - target[7 * i : 7 * i + 3]).norm().item()
        print(
            f"[SMOKE] {side} flange: hold drift {pos_drift * 1000:.2f} mm / {rot_drift:.4f} rad,"
            f" +3 cm tracking error {track_err * 1000:.2f} mm"
        )
        ok = ok and pos_drift < 0.005 and track_err < 0.01
    print("[SMOKE] OK" if ok else "[SMOKE] FAILED")
    if not ok:
        raise SystemExit(1)


def main():
    env = ManagerBasedRLEnv(cfg=build_env_cfg())
    try:
        if args_cli.smoke is not None:
            run_smoke(env, args_cli.smoke)
        else:
            run_teleop(env)
    finally:
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

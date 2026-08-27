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
parser.add_argument(
    "--scene_usda",
    type=str,
    default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "scenes", "taco", "scene", "taco_hoi_178_023.usda"
    ),
    help=(
        "Path to the scene USD/USDA file to load. Defaults to the vendored TACO brush-and-bowl scene;"
        " more examples live under scenes/ (scenegen scenes in scenes/scenegen/04_episode_scenegen/runs/scenes)."
    ),
)
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
    default=(0.0, 0.0, 0.7071068, 0.7071068),
    help="Rig torso orientation quaternion (x y z w). Default: +90 deg yaw (facing +y).",
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
    default=(0.0, 0.0, 0.7071068, 0.7071068),
    help="XR anchor rotation (x y z w): should match the robot yaw so the arms line up with yours.",
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
parser.add_argument(
    "--hand_calibration",
    type=str,
    default="hand_calibration.yml",
    help=(
        "Operator hand-shape calibration yml (resolved against assets/dex_retargeting)."
        " Pass '' to retarget uncalibrated."
    ),
)
parser.add_argument(
    "--no_auto_start",
    action="store_true",
    help="Disable auto-start (teleop engaging by itself when your wrists match the robot's hand poses).",
)
parser.add_argument(
    "--auto_start_pos_tol",
    type=float,
    default=0.10,
    help="Auto-start: max wrist-to-flange position error [m] counted as matching (both hands).",
)
parser.add_argument(
    "--auto_start_rot_tol",
    type=float,
    default=25.0,
    help="Auto-start: max wrist-to-flange orientation error [deg] counted as matching (both hands).",
)
parser.add_argument(
    "--debug_auto_start",
    action="store_true",
    help=(
        "Auto-start debugging: draw coordinate frames on both panda_link8 flanges (large) and on both"
        " calibrated wrist targets (small), and print the per-hand position/rotation errors once per second."
    ),
)
parser.add_argument(
    "--align_head_xy",
    type=float,
    nargs=2,
    default=(0.0, -0.6),
    help=(
        "Voice 'align' command: world xy the head is moved to (facing the robot's forward axis)."
        " Default (0, -0.6) stands you at the TACO table's near edge."
    ),
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
parser.add_argument(
    "--mic_device",
    type=str,
    default="default",
    help=(
        "Voice-command audio source: an ALSA capture device (arecord -D), or 'quest' (optionally"
        " 'quest:<port>', default 8444) to stream the headset microphone from a page opened in the"
        " Quest browser — see quest_mic.py. Open the printed URL on the headset before teleoperating."
    ),
)
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

import math
import time
import traceback

import torch

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_physx.physics import PhysxCfg

from duo_env import DuoEnvCfg
from duo_robot import duo_robot_cfg
from usda_scene import add_usda_scene


# -- Environment config -----------------------------------------------------------


@configclass
class DuoTeleopEnvCfg(DuoEnvCfg):
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


class AutoStartMatcher:
    """Start teleop automatically when the operator's wrists match the robot's hands.

    While teleop is stopped, compares the pipeline's wrist TARGETS (world frame,
    calibration and flange offsets already applied — i.e. where the flanges
    would be commanded to go) against the robot flanges' actual poses. When both
    hands are within ``pos_tol`` [m] and ``rot_tol`` [rad] for ``hold_s``
    seconds, a host-initiated Play fires — so teleop always engages with zero
    initial IK error instead of the robot snapping to wherever the hands are.

    Hysteresis: after any active period (or a failed match), re-arming requires
    the match to be clearly broken once (any hand beyond 1.5x tolerance).
    Without this, stopping via the cross-hand gesture — hands still at the pose
    the robot holds — would instantly restart teleop.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        flow: "EpisodeFlow",
        pos_tol: float,
        rot_tol: float,
        hold_s: float,
        debug: bool = False,
    ):
        from isaaclab.utils.math import quat_error_magnitude

        self._quat_err = quat_error_magnitude
        self._env = env
        self._flow = flow
        self._pos_tol = pos_tol
        self._rot_tol = rot_tol
        self._hold_s = hold_s
        self._flange_ids = [env.scene["robot"].body_names.index(f"{s}_panda_link8") for s in ("left", "right")]
        self._armed = True
        self._match_since: float | None = None
        self._debug_markers = None
        self._last_debug_print = 0.0
        if debug:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
            from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

            frame_usd = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd"
            self._debug_markers = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/auto_start_frames",
                    markers={
                        "flange": sim_utils.UsdFileCfg(usd_path=frame_usd, scale=(0.15, 0.15, 0.15)),
                        "target": sim_utils.UsdFileCfg(usd_path=frame_usd, scale=(0.08, 0.08, 0.08)),
                    },
                )
            )

    def _debug_show(self, targets: torch.Tensor, flange_poses: torch.Tensor, pos_err, rot_err) -> None:
        """Frames on the flanges (large) and the wrist targets (small); errors once per second."""
        device = flange_poses.device
        poses = torch.cat([flange_poses, targets.to(device)], dim=0)  # (4, 7): L/R flange, L/R target
        self._debug_markers.visualize(
            translations=poses[:, :3], orientations=poses[:, 3:7], marker_indices=[0, 0, 1, 1]
        )
        now = time.monotonic()
        if now - self._last_debug_print >= 1.0:
            self._last_debug_print = now
            state = "ACTIVE" if self._flow.teleop_active else ("armed" if self._armed else "disarmed")
            print(
                f"[AUTO-START] L {100 * pos_err[0]:.1f} cm / {torch.rad2deg(rot_err[0]):.0f} deg,"
                f" R {100 * pos_err[1]:.1f} cm / {torch.rad2deg(rot_err[1]):.0f} deg"
                f" (tol {100 * self._pos_tol:.0f} cm / {math.degrees(self._rot_tol):.0f} deg, {state})"
            )

    def update(self, action_world: torch.Tensor) -> None:
        """Feed the 58-D world-frame action of one frame; may fire Play."""
        flow = self._flow
        if (flow.teleop_active or flow.awaiting_label) and self._debug_markers is None:
            # No error computation (two pose readbacks) during teleop unless debugging.
            self._armed = False
            self._match_since = None
            return
        targets = action_world[:14].view(2, 7)
        if float(targets[:, :3].norm(dim=-1).min()) < 1e-6:
            self._match_since = None
            return  # a hand is untracked
        robot = self._env.scene["robot"]
        origin = self._env.scene.env_origins[0]
        pos_err = torch.empty(2)
        rot_err = torch.empty(2)
        flange_poses = torch.empty(2, 7, device=origin.device)
        for i, body_id in enumerate(self._flange_ids):
            flange_pos = robot.data.body_pos_w.torch[0, body_id] - origin
            flange_quat = robot.data.body_quat_w.torch[0, body_id]
            flange_poses[i, :3] = flange_pos
            flange_poses[i, 3:7] = flange_quat
            pos_err[i] = (targets[i, :3].to(flange_pos.device) - flange_pos).norm()
            rot_err[i] = self._quat_err(targets[i, 3:7].to(flange_quat.device).unsqueeze(0), flange_quat.unsqueeze(0))[
                0
            ]
        if self._debug_markers is not None:
            self._debug_show(targets, flange_poses, pos_err, rot_err)

        if flow.teleop_active or flow.awaiting_label:
            self._armed = False
            self._match_since = None
            return
        matched = bool((pos_err < self._pos_tol).all() and (rot_err < self._rot_tol).all())
        far = bool((pos_err > 1.5 * self._pos_tol).any() or (rot_err > 1.5 * self._rot_tol).any())
        if not self._armed:
            if far:
                self._armed = True
            return
        if not matched:
            self._match_since = None
            return
        now = time.monotonic()
        if self._match_since is None:
            self._match_since = now
        elif now - self._match_since >= self._hold_s:
            self._match_since = None
            print(
                f"[AUTO-START] Wrists matched the robot hands (pos {100 * pos_err.max():.1f} cm,"
                f" rot {torch.rad2deg(rot_err.max()):.0f} deg) — starting teleop."
            )
            flow.request_client_start()


class EpisodeFlow:
    """Episode lifecycle state for the teleop loop.

    Play starts teleop (and, when recording, the episode buffer). The
    cross-hand stop gesture ends an episode and waits for a voice label;
    saying "success"/"failure" at any time labels AND ends the current
    episode; either way the labeled demo is exported, the scene resets, and
    teleop ends stopped — the operator presses Play for the next episode.
    Reset (headset button) discards the in-flight episode.
    """

    def __init__(self, env: ManagerBasedRLEnv, gesture, labeler, recording: bool):
        self.env = env
        self.gesture = gesture
        self.labeler = labeler
        self.recording = recording
        self.teleop = None  # bound after the device is created
        self.teleop_active = False
        self.reset_requested = False
        self.awaiting_label = False  # gesture ended the episode; waiting for the voice label
        self.align_requested = False  # voice "align" heard; served by the loop
        self.suppress_active_frames = 0  # ignore the client's stale "playing" state after a host stop
        self.demo_count = 0
        self.success_count = 0

    # -- device callbacks ---------------------------------------------------

    def on_start(self) -> None:
        if self.awaiting_label:
            print("[INFO] Play ignored: say 'success' or 'failure' first (or press Reset to discard).")
            return
        self.teleop_active = True

    def on_stop(self) -> None:
        self.teleop_active = False

    def on_reset(self) -> None:
        self.reset_requested = True

    # -- episode bookkeeping ------------------------------------------------

    def episode_has_data(self) -> bool:
        return self.recording and not self.env.recorder_manager.get_episode(0).is_empty()

    def _request_client_toggle(self, action: str) -> None:
        """Drive the teleop state machine, as if the operator pressed Play/Stop.

        The device reports the client's CURRENT play state each poll, so a local
        flag alone would be overwritten a frame later. There is no public
        host-initiated start/stop (only ``inject_reset``), so this enqueues the
        same run-toggle sequence a client "start"/"stop" message would produce.
        """
        try:
            from isaaclab_teleop.teleop_message_processor import _START_TOGGLE_SEQUENCES, _STOP_TOGGLE_SEQUENCES

            sequences = _START_TOGGLE_SEQUENCES if action == "start" else _STOP_TOGGLE_SEQUENCES
            proc = self.teleop._session_lifecycle._message_processor
            if proc is not None and not proc._run_toggle_queue:
                proc._run_toggle_queue = proc._make_toggle_sequence(sequences[proc._shadow_state])
        except Exception as exc:
            print(f"[WARNING] Could not push {action} into the teleop state machine: {exc}")

    def request_client_stop(self) -> None:
        self._request_client_toggle("stop")

    def request_client_start(self) -> None:
        if self.awaiting_label:
            print("[INFO] Play ignored: say 'success' or 'failure' first (or press Reset to discard).")
            return
        self._request_client_toggle("start")

    def stop_teleop(self) -> None:
        self.teleop_active = False
        self.request_client_stop()
        self.suppress_active_frames = 5  # the injected Stop lands within a frame or two

    def close_episode(self, prompt_label: bool = True) -> None:
        """Stop teleop and freeze the episode buffer."""
        self.stop_teleop()
        self.env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
        self.awaiting_label = True
        actions = self.env.recorder_manager.get_episode(0).data.get("actions")
        n_steps = len(actions) if actions is not None else 0
        prompt = " Say 'success' or 'failure' (Reset discards)." if prompt_label else ""
        print(f"[INFO] Episode ended ({n_steps} steps).{prompt}")

    def export_episode(self, success: bool) -> None:
        rm = self.env.recorder_manager
        actions = rm.get_episode(0).data.get("actions")
        n_steps = len(actions) if actions is not None else 0
        rm.set_success_to_episodes([0], torch.tensor([[success]], dtype=torch.bool, device=self.env.device))
        rm.export_episodes([0])
        rm.reset()
        self.demo_count += 1
        self.awaiting_label = False
        self.reset_requested = True  # hands-free: fresh scene for the next episode
        if success:
            self.success_count += 1
        cfg = self.env.cfg.recorders
        dataset_path = os.path.join(cfg.dataset_export_dir_path, cfg.dataset_filename + ".hdf5")
        try:
            size_mb = os.path.getsize(dataset_path) / 1e6
            size_str = f", {size_mb:.1f} MB on disk"
        except OSError:
            size_str = ""
        print(
            f"[SAVED] demo_{self.demo_count - 1}: success={success}, {n_steps} steps"
            f" ({n_steps / 60.0:.1f} s) -> {dataset_path}\n"
            f"[SAVED] Session total: {self.demo_count} demos"
            f" ({self.success_count} success / {self.demo_count - self.success_count} failure){size_str}."
            " Scene reset; press Play or match the start pose."
        )

    # -- per-iteration handlers ----------------------------------------------

    def handle_voice_label(self) -> None:
        """Route a voice command; labels end (if needed) and export the current episode."""
        label = self.labeler.poll() if self.labeler is not None else None
        if label is None:
            return
        if label == "align":
            self.align_requested = True
            return
        if label == "play":
            self.request_client_start()  # teleop_active follows via the state poll
            return
        if label == "reset":
            self.reset_requested = True  # same as the client button: discards the episode
            return
        if not self.recording:
            return
        if not self.awaiting_label and self.episode_has_data():
            # Label spoken mid-episode: it ends AND labels in one utterance.
            self.close_episode(prompt_label=False)
        if self.awaiting_label:
            self.export_episode(label == "success")
        else:
            print(f"[INFO] Voice label '{label}' ignored: no recorded steps yet.")

    def handle_reset(self) -> None:
        if not self.reset_requested:
            return
        if self.awaiting_label:
            print("[INFO] Reset while awaiting the voice label: episode discarded.")
        if self.recording:
            self.env.recorder_manager.reset()
        self.env.reset()
        self.teleop.reset()
        self.gesture.reset()
        self.awaiting_label = False
        self.reset_requested = False

    def handle_control_events(self, poll_control_events) -> None:
        ctrl = poll_control_events(self.teleop)
        if ctrl.is_active is not None:
            if ctrl.is_active and self.awaiting_label:
                print("[INFO] Play ignored: say 'success' or 'failure' first (or press Reset to discard).")
            elif ctrl.is_active and self.suppress_active_frames > 0:
                pass  # stale "playing" state; the host-initiated Stop has not landed yet
            else:
                self.teleop_active = ctrl.is_active
        if self.suppress_active_frames > 0:
            self.suppress_active_frames -= 1
        if ctrl.should_reset:
            self.reset_requested = True

    def handle_gesture(self, xr_hands: torch.Tensor) -> bool:
        """Returns True if the stop gesture fired this frame."""
        if not (self.teleop_active and self.gesture.update(xr_hands)):
            return False
        if self.episode_has_data():
            self.close_episode()
        else:
            self.stop_teleop()
            print("[INFO] Stop gesture: no recorded steps, teleop paused.")
        return True


def run_teleop(env: ManagerBasedRLEnv) -> None:
    """Drive the env from XR hand tracking until the app closes (see :class:`EpisodeFlow`)."""
    from isaaclab_teleop import CLOUDXR_AVP_ENV, CLOUDXR_JS_ENV, create_isaac_teleop_device, poll_control_events
    from isaaclab_teleop.isaac_teleop_cfg import IsaacTeleopCfg
    from isaaclab_teleop.xr_cfg import XrCfg

    from scipy.spatial.transform import Rotation as R

    from duo_teleop_pipeline import build_duo_pipeline
    from recording import XrHandsRecorder
    from xr_extras import (
        XR_EXTRAS_DIM,
        AnchorAligner,
        CrossHandStopGesture,
        HandJointMarkers,
        apply_arm_visual,
        current_head_pose,
    )

    # Debugging aid: the session lifecycle reports step failures with only
    # str(exception); surface the full traceback so failures are diagnosable.
    import isaaclab_teleop.session_lifecycle as _session_lifecycle_mod

    if not getattr(_session_lifecycle_mod, "_traceback_shim", False):
        _session_lifecycle_mod._traceback_shim = True
        _orig_warning = _session_lifecycle_mod.logger.warning

        def _warning_with_traceback(msg, *w_args, **w_kwargs):
            _orig_warning(msg, *w_args, **w_kwargs)
            if "session step failed" in str(msg):
                traceback.print_exc()

        _session_lifecycle_mod.logger.warning = _warning_with_traceback

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

    flow = EpisodeFlow(env, gesture, labeler, recording)

    pipeline, retargeters = build_duo_pipeline(include_xr_hands=True, hand_calibration=args_cli.hand_calibration)
    teleop_cfg = IsaacTeleopCfg(
        xr_cfg=XrCfg(anchor_pos=tuple(args_cli.anchor_pos), anchor_rot=tuple(args_cli.anchor_rot)),
        pipeline_builder=lambda: pipeline,
        retargeters_to_tune=lambda: retargeters,
        sim_device=env.device,
    )
    cloudxr_env = {"cloudxrjs": CLOUDXR_JS_ENV, "avp": CLOUDXR_AVP_ENV, "none": None}.get(
        args_cli.cloudxr_env, args_cli.cloudxr_env
    )
    flow.teleop = teleop = create_isaac_teleop_device(
        teleop_cfg,
        sim_device=env.device,
        callbacks={"START": flow.on_start, "STOP": flow.on_stop, "RESET": flow.on_reset},
        cloudxr_env_file=cloudxr_env,
        auto_launch_cloudxr=cloudxr_env is not None,
    )
    # Voice "align": re-anchor so the user faces along the robot's forward axis.
    # The rig faces +x at identity, so the facing angle IS the root quat's yaw
    # (the default +90 deg yaw faces +y, reproducing the source branch's target).
    robot_yaw = float(R.from_quat(args_cli.robot_rot).as_euler("ZYX")[0])
    aligner = AnchorAligner(teleop, tuple(args_cli.align_head_xy), robot_yaw)
    auto_start = None
    if not args_cli.no_auto_start:
        auto_start = AutoStartMatcher(
            env, flow,
            pos_tol=args_cli.auto_start_pos_tol,
            rot_tol=math.radians(args_cli.auto_start_rot_tol),
            hold_s=0.5,
            debug=args_cli.debug_auto_start,
        )  # fmt: skip
        print(
            "[INFO] Auto-start armed: hold your wrists at the robot's hand poses"
            f" (< {100 * args_cli.auto_start_pos_tol:.0f} cm, < {args_cli.auto_start_rot_tol:.0f} deg,"
            " both hands, 0.5 s) to start teleop without saying 'play'."
        )

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
                flow.handle_voice_label()
                flow.handle_reset()
                action = teleop.advance()
                flow.handle_control_events(poll_control_events)

                if flow.align_requested:
                    flow.align_requested = False
                    if flow.teleop_active:
                        print("[ALIGN] Ignored while teleop is running: press Stop (or use the gesture) first.")
                    else:
                        head = current_head_pose()
                        if head is None:
                            print("[ALIGN] No head pose available (is the visor on?); say 'align' again in a moment.")
                        else:
                            aligner.align(head)

                # action is None until the XR session has started.
                if action is None:
                    env.sim.render()
                    continue

                xr_hands = action[-XR_EXTRAS_DIM:].reshape(2, 26, 7)
                action = action[:-XR_EXTRAS_DIM]
                XrHandsRecorder.latest = xr_hands
                if markers is not None:
                    markers.update(xr_hands)

                if flow.handle_gesture(xr_hands):
                    continue
                if auto_start is not None:
                    auto_start.update(action)
                if not flow.teleop_active:
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
                flow.teleop_active = False
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

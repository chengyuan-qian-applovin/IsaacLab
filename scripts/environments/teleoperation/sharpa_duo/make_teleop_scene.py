# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load a scene USDA, add a bimanual SharpaWave rig, and teleoperate it via XR.

The pipeline in one line: your USDA becomes the environment, the rig selected
by ``--embodiment`` (FR3 Duo torso by default, or two table-edge-mounted I2RT
YAM Ultra arms) is placed into it at ``--robot_pos``/``--robot_rot``, and your
tracked hands drive it — wrists through per-arm differential IK, all fingers
through DexPilot retargeting. Episodes are recorded to a robomimic-style HDF5 by default and
labeled hands-free: saying "success" or "failure" (transcribed locally with
OpenAI Whisper) ends, labels, and exports the episode — see README.md for the
full flow.

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

parser = argparse.ArgumentParser(description="Teleoperate a bimanual SharpaWave rig in a USDA scene.")
parser.add_argument(
    "--embodiment",
    type=str,
    choices=("franka_duo", "yam_duo"),
    default="franka_duo",
    help=(
        "Robot embodiment: 'franka_duo' (fixed torso + two 7-DoF Panda arms, default) or 'yam_duo'"
        " (two 6-DoF I2RT YAM Ultra arms on a table-edge rail, bases 0.565 m apart). Both carry"
        " SharpaWave hands. See duo_robot.py."
    ),
)
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
    default=None,
    help=(
        "Rig root position in the scene frame [m]. The default is per-embodiment: the franka_duo"
        " torso at (0, -0.8, 1.3) clearing the raised (1 m) TACO tabletop, the yam_duo rail ON the"
        " tabletop at its near edge (0, -0.55, 1.0)."
    ),
)
parser.add_argument(
    "--robot_rot",
    type=float,
    nargs=4,
    default=None,
    help="Rig root orientation quaternion (x y z w). Default: +90 deg yaw (facing +y).",
)
parser.add_argument(
    "--anchor_pos",
    type=float,
    nargs=3,
    default=(0.0, -0.8, -0.21),
    help="XR anchor position: default stands you at the rig torso; the raised TACO tabletop lands at ~1.21 m.",
)
parser.add_argument(
    "--anchor_rot",
    type=float,
    nargs=4,
    default=(0.0, 0.0, 0.7071068, 0.7071068),
    help="XR anchor rotation (x y z w): should match the robot yaw so the arms line up with yours.",
)
parser.add_argument(
    "--scene_list",
    type=str,
    default=None,
    help=(
        'JSON file with the scenes to teleop: a list of USDA paths (or {"scenes": [...]}), relative'
        " paths resolved against the JSON's directory. Starts at the first; say 'next' to advance"
        " (wraps around). Overrides --scene_usda. Example: scenes/scene_list.json."
    ),
)
parser.add_argument(
    "--no_track_objects",
    action="store_true",
    help="Do not register the scene's rigid bodies with the env (their poses then survive resets).",
)
parser.add_argument("--no_dr", action="store_true", help="Disable domain randomization on episode resets.")
parser.add_argument(
    "--dr_arm_jitter",
    type=float,
    default=0.08,
    help="DR: uniform per-joint offset range [rad] added to the arms' ready pose on each reset.",
)
parser.add_argument(
    "--dr_object_xy",
    type=float,
    default=0.05,
    help="DR: uniform xy offset range [m] around each tracked object's authored pose on each reset.",
)
parser.add_argument(
    "--dr_object_yaw",
    type=float,
    default=180.0,
    help="DR: uniform yaw offset range [deg] around each tracked object's authored orientation.",
)
parser.add_argument(
    "--dr_object_bias",
    type=float,
    default=0.3,
    help=(
        "DR: fixed shift [m] moving every object's randomization center horizontally toward the robot"
        " base (never past it), bringing objects within easier reach; 0 disables."
    ),
)
parser.add_argument(
    "--settle_time",
    type=float,
    default=1.0,
    help=(
        "Seconds of physics run after every scene reset (robot held still) so the randomized objects"
        " settle onto the table before teleop; 0 disables."
    ),
)
parser.add_argument(
    "--no_task_display",
    action="store_true",
    help="Do not show the scene's task description (from the instructions JSON next to it) in the headset.",
)
parser.add_argument(
    "--task_display_pos",
    type=float,
    nargs=3,
    default=(0.0, 0.9, 1.6),
    help="World position [m] of the floating task-description panel (default: beyond the table, head height).",
)
parser.add_argument(
    "--no_voice_display",
    action="store_true",
    help="Do not echo what the voice labeler heard (and did) on a floating panel in the headset.",
)
parser.add_argument(
    "--voice_display_pos",
    type=float,
    nargs=3,
    default=(0.0, 0.9, 1.25),
    help="World position [m] of the floating voice-feedback panel (default: just below the task panel).",
)
parser.add_argument(
    "--voice_display_seconds",
    type=float,
    default=4.0,
    help="How long a voice message stays on the headset panel before it hides.",
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
parser.add_argument(
    "--dataset_file",
    type=str,
    default=None,
    help=(
        "Append every demo (across all scenes and sessions) to this one HDF5 file instead of creating"
        " a timestamped file per scene; created if missing. Demos carry a 'scene' attribute, so one"
        " shared file stays scene-attributable. Used by the teleop launcher UI."
    ),
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
    "--arm_kp",
    type=float,
    default=400.0,
    help="Arm joint drive stiffness kp [N·m/rad] (all arm joints). Stamped on every recorded demo.",
)
parser.add_argument(
    "--arm_kd",
    type=float,
    default=80.0,
    help="Arm joint drive damping kd [N·m·s/rad] (all arm joints). Stamped on every recorded demo.",
)
parser.add_argument(
    "--hand_kp",
    type=float,
    default=400.0,
    help="Finger joint drive stiffness kp [N·m/rad] (all 44 SharpaWave joints). Stamped on every recorded demo.",
)
parser.add_argument(
    "--hand_kd",
    type=float,
    default=4.0,
    help="Finger joint drive damping kd [N·m·s/rad] (all 44 SharpaWave joints). Stamped on every recorded demo.",
)
parser.add_argument(
    "--user",
    type=str,
    default=None,
    help=(
        "User name: loads the per-user hand calibration written by calibrate_hand_shape.py"
        " (assets/dex_retargeting/hand_calibration_<user>.yml). An explicit --hand_calibration wins."
    ),
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
        "Auto-start debugging: keep the alignment axis frames visible during teleop too (they always show"
        " while auto-start is waiting), and print the per-hand position/rotation errors once per second."
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
parser.add_argument(
    "--align_head_z",
    type=float,
    default=1.5,
    help=(
        "Voice 'align' command: world height [m] the head is moved to, putting the scene floor (z=0)"
        " that far below your eyes. Pass 0 or negative to keep the headset's own floor calibration."
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
        "Voice-command audio source: an ALSA capture device (arecord -D), or 'quest' / 'avp'"
        " (optionally ':<port>', default 8444) to stream the headset microphone — see headset_mic.py."
        " Quest: open the printed URL in the headset browser before teleoperating. AVP: the Isaac XR"
        " Teleop client streams by itself once connected (feature/avp-voice-mic build)."
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
        event = labeler.poll()
        if event is not None and event.command is not None:
            print(f"[VOICE] EVENT: {event.command}")
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
from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_physx.physics import PhysxCfg

from duo_env import DuoEnvCfg
from duo_robot import EMBODIMENTS
from usda_scene import add_usda_scene

# The selected robot embodiment; per-embodiment placement defaults resolve here.
SPEC = EMBODIMENTS[args_cli.embodiment]
if args_cli.robot_pos is None:
    args_cli.robot_pos = list(SPEC.default_robot_pos)
if args_cli.robot_rot is None:
    args_cli.robot_rot = list(SPEC.default_robot_rot)


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


def build_env_cfg(scene_usda: str) -> ManagerBasedRLEnvCfg:
    """Assemble the env config for one scene from the CLI arguments."""
    env_cfg = DuoTeleopEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.episode_length_s = args_cli.episode_length_s
    env_cfg.actions = SPEC.actions_cfg()
    env_cfg.scene.robot = SPEC.robot_cfg(
        pos=tuple(args_cli.robot_pos),
        rot=tuple(args_cli.robot_rot),
        arm_stiffness=args_cli.arm_kp,
        arm_damping=args_cli.arm_kd,
        hand_stiffness=args_cli.hand_kp,
        hand_damping=args_cli.hand_kd,
    )
    add_usda_scene(env_cfg.scene, scene_usda, track_objects=not args_cli.no_track_objects)
    # Render every Nth physics substep, as close to the requested rate as the dt allows.
    interval = max(1, round(1.0 / (env_cfg.sim.dt * args_cli.render_frequency)))
    env_cfg.sim.render_interval = interval
    print(f"[INFO] render interval {interval} substeps -> {1.0 / (env_cfg.sim.dt * interval):.1f} Hz")
    if args_cli.arm_visual == "transparent":
        # Opacity needs translucency support in the renderer; must be set pre-sim-context.
        env_cfg.sim.render.enable_translucency = True
    if not args_cli.no_dr:
        # Domain randomization on every episode reset: jitter the arms' start
        # pose, and shuffle the tracked objects with a collision-checked draw
        # (bounding circles from the USD footprints; see usda_scene).
        from isaaclab.envs.mdp import reset_joints_by_offset
        from isaaclab.managers import SceneEntityCfg

        from usda_scene import randomize_tracked_objects

        env_cfg.events.dr_arm = EventTermCfg(
            func=reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-args_cli.dr_arm_jitter, args_cli.dr_arm_jitter),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=[SPEC.arm_joint_regex]),
            },
        )
        env_cfg.events.dr_objects = EventTermCfg(
            func=randomize_tracked_objects,
            mode="reset",
            params={
                "xy_range": args_cli.dr_object_xy,
                "yaw_range": math.radians(args_cli.dr_object_yaw),
                "margin": 0.01,
                "bias_toward": (args_cli.robot_pos[0], args_cli.robot_pos[1]),
                "bias_dist": args_cli.dr_object_bias,
            },
        )
    if args_cli.smoke is None and not args_cli.no_record:
        from recording import AppendableHDF5DatasetFileHandler, DuoRecorderManagerCfg

        env_cfg.recorders = DuoRecorderManagerCfg()
        if args_cli.dataset_file:
            # One shared file for all scenes/sessions, opened in append mode.
            dataset_file = os.path.abspath(args_cli.dataset_file)
            env_cfg.recorders.dataset_file_handler_class_type = AppendableHDF5DatasetFileHandler
            env_cfg.recorders.dataset_export_dir_path = os.path.dirname(dataset_file)
            env_cfg.recorders.dataset_filename = os.path.splitext(os.path.basename(dataset_file))[0]
        else:
            env_cfg.recorders.dataset_export_dir_path = os.path.abspath(args_cli.record_dir)
            # Unique per session AND per scene: the stock HDF5 handler opens its file
            # in "w" (truncate) mode at env creation, so a fixed name would wipe
            # earlier sessions on every start.
            scene_stem = os.path.splitext(os.path.basename(scene_usda))[0][:60]
            env_cfg.recorders.dataset_filename = time.strftime("dataset_%Y%m%d_%H%M%S") + f"_{scene_stem}"
    return env_cfg


# -- Teleop -----------------------------------------------------------------------


def settle_scene(env: ManagerBasedRLEnv) -> None:
    """Run physics for ``--settle_time`` seconds right after a reset.

    Domain randomization can leave objects hovering a hair above (or leaning
    into) the table; this lets them drop and come to rest before the operator
    sees the scene. The robot is pinned by targeting its just-reset joint
    positions — reset writes joint STATE, but the PD targets may still hold
    stale values that would drag the arms away while the sim steps. When
    recording, the episode buffer is restarted afterwards so the demo's
    ``initial_state`` is the settled scene, not the mid-air draw.

    The env's own episode-timeout auto-reset (inside ``env.step``) bypasses
    this; timeouts discard the episode anyway.
    """
    if args_cli.settle_time <= 0.0:
        return
    robot = env.scene["robot"]
    robot.set_joint_position_target_index(target=robot.data.joint_pos.torch.clone())
    steps = max(1, round(args_cli.settle_time / env.physics_dt))
    render_interval = max(1, int(env.cfg.sim.render_interval))
    for i in range(steps):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        if (i + 1) % render_interval == 0:
            env.sim.render()
        env.scene.update(env.physics_dt)
    if env.cfg.recorders is not None:
        env.recorder_manager.reset()
        env.recorder_manager.record_post_reset([0])


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

    While teleop is stopped, compares the pipeline's wrist TARGETS against the
    SharpaWave hands' actual wrist poses (``*_hand_wrist`` — where the
    operator's wrist maps onto the robot, more natural to line up with than the
    arm flange). The pipeline commands the flanges, so each flange target is
    pushed through the fixed flange-to-wrist transform before comparing. When
    both hands are within ``pos_tol`` [m] and ``rot_tol`` [rad] for ``hold_s``
    seconds, a host-initiated Play fires — so teleop always engages with zero
    initial IK error instead of the robot snapping to wherever the hands are.

    Hysteresis: after any active period (or a failed match), re-arming requires
    the match to be clearly broken once (any hand beyond 1.5x tolerance).
    Without this, any stop that leaves the hands at the pose the robot holds
    would instantly restart teleop.

    While auto-start is waiting, axis frames are drawn on both hand wrists
    (large) and both calibrated wrist targets (small) so the operator can see
    the poses to match; they disappear once teleop engages. ``debug`` keeps the
    frames up during teleop too and prints the per-hand errors once per second.
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
        self._debug = debug
        body_names = env.scene["robot"].body_names
        self._wrist_ids = [body_names.index(f"{s}_hand_wrist") for s in ("left", "right")]
        # The IK-commanded bodies (Panda flanges; the wrist bodies themselves on
        # the YAM rig, making the transform below the identity).
        self._flange_ids = [body_names.index(SPEC.ik_body(s)) for s in ("left", "right")]
        self._flange_to_wrist: tuple[torch.Tensor, torch.Tensor] | None = None  # fixed, computed on first update
        self._armed = True
        self._match_since: float | None = None
        self._frame_markers = None
        self._frames_visible = False
        self._last_debug_print = 0.0
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
            from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

            frame_usd = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd"
            self._frame_markers = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/auto_start_frames",
                    markers={
                        "flange": sim_utils.UsdFileCfg(usd_path=frame_usd, scale=(0.15, 0.15, 0.15)),
                        "target": sim_utils.UsdFileCfg(usd_path=frame_usd, scale=(0.08, 0.08, 0.08)),
                    },
                )
            )
        except Exception as exc:
            print(f"[WARNING] Auto-start axis frames unavailable ({exc}); matching still works.")

    def _show_frames(self, wrist_poses: torch.Tensor | None, targets: torch.Tensor | None) -> None:
        """Draw frames on the hand wrists (large) and wrist targets (small); None-None hides them."""
        if self._frame_markers is None:
            return
        if wrist_poses is None:
            if self._frames_visible:
                self._frame_markers.set_visibility(False)
                self._frames_visible = False
            return
        if targets is None:  # hands untracked: robot wrist frames only
            poses, indices = wrist_poses, [0, 0]
        else:
            poses = torch.cat([wrist_poses, targets.to(wrist_poses.device)], dim=0)
            indices = [0, 0, 1, 1]
        self._frame_markers.visualize(translations=poses[:, :3], orientations=poses[:, 3:7], marker_indices=indices)
        if not self._frames_visible:
            self._frame_markers.set_visibility(True)
            self._frames_visible = True

    def _debug_print(self, pos_err: torch.Tensor, rot_err: torch.Tensor) -> None:
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
        busy = flow.teleop_active or flow.awaiting_label
        if busy and not self._debug:
            # No error computation (two pose readbacks) during teleop, and the
            # alignment frames get out of the operator's way.
            self._armed = False
            self._match_since = None
            self._show_frames(None, None)
            return
        from isaaclab.utils.math import combine_frame_transforms

        robot = self._env.scene["robot"]
        origin = self._env.scene.env_origins[0]
        wrist_poses = torch.empty(2, 7, device=origin.device)
        for i, body_id in enumerate(self._wrist_ids):
            wrist_poses[i, :3] = robot.data.body_pos_w.torch[0, body_id] - origin
            wrist_poses[i, 3:7] = robot.data.body_quat_w.torch[0, body_id]
        if self._flange_to_wrist is None:
            # The hand wrist is mounted to the flange by fixed joints, so this
            # offset is configuration-independent: compute it once from FK.
            flange_poses = torch.empty(2, 7, device=origin.device)
            for i, body_id in enumerate(self._flange_ids):
                flange_poses[i, :3] = robot.data.body_pos_w.torch[0, body_id] - origin
                flange_poses[i, 3:7] = robot.data.body_quat_w.torch[0, body_id]
            self._flange_to_wrist = subtract_frame_transforms(
                flange_poses[:, :3], flange_poses[:, 3:7], wrist_poses[:, :3], wrist_poses[:, 3:7]
            )
        flange_targets = action_world[:14].view(2, 7)
        if float(flange_targets[:, :3].norm(dim=-1).min()) < 1e-6:
            # A hand is untracked: show where the robot wrists are, nothing to match yet.
            self._show_frames(wrist_poses, None)
            self._match_since = None
            return
        # Where the hand wrists would be with the flanges at their targets.
        target_pos, target_quat = combine_frame_transforms(
            flange_targets[:, :3].to(origin.device),
            flange_targets[:, 3:7].to(origin.device),
            self._flange_to_wrist[0],
            self._flange_to_wrist[1],
        )
        targets = torch.cat([target_pos, target_quat], dim=-1)
        pos_err = torch.empty(2)
        rot_err = torch.empty(2)
        for i in range(2):
            pos_err[i] = (targets[i, :3] - wrist_poses[i, :3]).norm()
            rot_err[i] = self._quat_err(targets[i, 3:7].unsqueeze(0), wrist_poses[i, 3:7].unsqueeze(0))[0]
        self._show_frames(wrist_poses, targets)
        if self._debug:
            self._debug_print(pos_err, rot_err)

        if busy:
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

    Play starts teleop (and, when recording, the episode buffer). Saying
    "success"/"failure" at any time ends AND labels the current episode; the
    labeled demo is exported, the scene resets, and teleop ends stopped — the
    operator presses Play (or matches the start pose) for the next episode.
    Reset (headset button or voice) discards the in-flight episode and also
    leaves teleop stopped.
    """

    def __init__(self, env: ManagerBasedRLEnv, labeler, recording: bool, scene_name: str = ""):
        self.env = env
        self.labeler = labeler
        self.scene_name = scene_name
        self.next_requested = False  # voice "next": advance to the next scene in the list
        self.recording = recording
        self.teleop = None  # bound after the device is created
        self.voice_display = None  # bound after the headset panel is spawned
        self.teleop_active = False
        self.reset_requested = False
        self.awaiting_label = False  # an episode was closed; waiting for the voice label
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
        # File-level demo id (differs from the session count when appending to a
        # shared dataset); also tag the demo with the scene it was recorded in
        # (the handler writes only its fixed attrs, so set ours on the group).
        try:
            handler = rm._dataset_file_handler
            demo_id = handler.get_num_episodes() - 1
            demo_group = handler._hdf5_data_group[f"demo_{demo_id}"]
            if self.scene_name:
                demo_group.attrs["scene"] = self.scene_name
            # The robot embodiment and drive gains, for training/replay.
            demo_group.attrs["embodiment"] = SPEC.name
            demo_group.attrs["arm_kp"] = args_cli.arm_kp
            demo_group.attrs["arm_kd"] = args_cli.arm_kd
            demo_group.attrs["hand_kp"] = args_cli.hand_kp
            demo_group.attrs["hand_kd"] = args_cli.hand_kd
            handler.flush()
        except Exception as exc:
            demo_id = self.demo_count
            print(f"[WARNING] Could not tag the demo with its scene name and gains: {exc}")
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
            f"[SAVED] demo_{demo_id}: success={success}, {n_steps} steps"
            f" ({n_steps / 60.0:.1f} s) -> {dataset_path}\n"
            f"[SAVED] Session total: {self.demo_count} demos"
            f" ({self.success_count} success / {self.demo_count - self.success_count} failure){size_str}."
            " Scene reset; press Play or match the start pose."
        )

    # -- per-iteration handlers ----------------------------------------------

    def handle_voice_label(self) -> None:
        """Route a voice command, echoing what was heard and done in the headset."""
        event = self.labeler.poll() if self.labeler is not None else None
        if event is None:
            return
        if event.command is None:
            # Mis-hearings are shown too: without them a command that silently
            # failed to parse is indistinguishable from a dead microphone.
            self.show_voice_message(f'Heard: "{event.text}"' if event.text else "Heard a sound, but no speech")
            return
        outcome = self._run_voice_command(event.command)
        self.show_voice_message(f'Detected "{event.text}" - executed: {outcome}')

    def show_voice_message(self, message: str) -> None:
        """Put ``message`` on the headset feedback panel, when one is spawned."""
        if self.voice_display is not None:
            self.voice_display.show(message)

    def _run_voice_command(self, label: str) -> str:
        """Act on a recognized command and describe the effect for the operator.

        The returned text is what the headset panel shows after "executed:", so
        it names what happened — or why the command was ignored.
        """
        if label == "align":
            self.align_requested = True
            return "align (re-anchoring)"
        if label == "play":
            blocked = self.awaiting_label
            self.request_client_start()  # no-ops (and explains) while awaiting a label
            return "play ignored - label the episode first" if blocked else "play - starting teleop"
        if label == "stop":
            # Hands-free pause (the client Stop button's voice twin): the episode
            # buffer is kept, so Play/auto-start resumes recording where it left off.
            if not self.teleop_active:
                return "stop ignored - teleop was not running"
            self.stop_teleop()
            print("[VOICE] Teleop paused. Resume with 'play', the client button, or the start pose.")
            return "stop - teleop paused"
        if label == "reset":
            self.reset_requested = True  # same as the client button: discards the episode
            return "reset - discarding the episode"
        if label == "next":
            if self.awaiting_label:
                print("[SCENE] 'next' ignored: say 'success' or 'failure' first (or press Reset to discard).")
                return "next ignored - label the episode first"
            if self.recording and self.episode_has_data():
                print("[SCENE] Switching scenes: the unlabeled in-flight episode is discarded.")
            self.next_requested = True
            return "next - switching scene"
        if not self.recording:
            return f"{label} ignored - recording is disabled"
        if not self.awaiting_label and self.episode_has_data():
            # Label spoken mid-episode: it ends AND labels in one utterance.
            self.close_episode(prompt_label=False)
        if not self.awaiting_label:
            print(f"[INFO] Voice label '{label}' ignored: no recorded steps yet.")
            return f"{label} ignored - no recorded steps yet"
        self.export_episode(label == "success")
        return f"{label} - episode exported"

    def handle_reset(self) -> None:
        if not self.reset_requested:
            return
        if self.awaiting_label:
            print("[INFO] Reset while awaiting the voice label: episode discarded.")
        # A reset always leaves teleop STOPPED: the client's play state is
        # level-polled, so without a host-initiated Stop a voice "reset" would
        # resume teleop the instant the scene is back — the robot would chase
        # the hands with no auto-start/Play phase in between.
        self.stop_teleop()
        if self.recording:
            self.env.recorder_manager.reset()
        self.env.reset()
        settle_scene(self.env)
        self.teleop.reset()
        self.awaiting_label = False
        self.reset_requested = False
        print("[INFO] Scene reset; teleop stopped — press Play, say 'play', or match the start pose.")

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


def serve_align(flow: EpisodeFlow, aligner, head_pose_fn) -> bool:
    """Serve a pending voice "align" request; returns True if the re-anchor ran.

    Align NEVER runs while teleop is active — re-anchoring shifts every
    world-frame hand target at once, which would jerk the robot under a live
    session. The request is consumed either way; while a head pose is merely
    unavailable the operator is told to try again.
    """
    if not flow.align_requested:
        return False
    flow.align_requested = False
    if flow.teleop_active:
        print("[ALIGN] Ignored while teleop is running: say 'stop' (or press Stop), align, then resume.")
        return False
    head = head_pose_fn()
    if head is None:
        print("[ALIGN] No head pose available (is the visor on?); say 'align' again in a moment.")
        return False
    return aligner.align(head)


def run_teleop(env: ManagerBasedRLEnv, labeler, scene_name: str, anchor: tuple) -> tuple[str, tuple]:
    """Drive the env from XR hand tracking for one scene (see :class:`EpisodeFlow`).

    Args:
        env: The environment (one scene of the list).
        labeler: The shared :class:`~voice_labeler.VoiceLabeler` (or None).
        scene_name: Stamped on every exported demo as its ``scene`` HDF5 attr.
        anchor: ``(anchor_pos, anchor_rot)`` xyzw to start from, so "align"
            adjustments carry across scene switches.

    Returns:
        ``(reason, anchor)`` where reason is ``"next"`` (voice command: advance
        to the next scene) or ``"quit"`` (app closed), and anchor is the
        possibly-realigned pose to seed the next scene with.
    """
    from isaaclab_teleop import create_isaac_teleop_device, poll_control_events
    from isaaclab_teleop.isaac_teleop_cfg import IsaacTeleopCfg
    from isaaclab_teleop.xr_cfg import XrCfg

    from scipy.spatial.transform import Rotation as R

    from duo_teleop_pipeline import build_duo_pipeline
    from recording import XrHandsRecorder
    from xr_extras import (
        XR_EXTRAS_DIM,
        AnchorAligner,
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

    flow = EpisodeFlow(env, labeler, recording, scene_name=scene_name)
    if labeler is not None and not args_cli.no_voice_display:
        from task_display import MessageDisplay

        # Same convention as the task panel: at yaw 0 it faces -y, so it
        # counter-rotates with the rig to stay square to the operator.
        voice_panel_yaw = float(R.from_quat(args_cli.robot_rot).as_euler("ZYX", degrees=True)[0]) - 90.0
        flow.voice_display = MessageDisplay(
            tuple(args_cli.voice_display_pos), voice_panel_yaw, seconds=args_cli.voice_display_seconds
        )

    # --user selects the per-user calibration from calibrate_hand_shape.py; an
    # explicitly overridden --hand_calibration wins over it.
    hand_calibration = args_cli.hand_calibration
    if args_cli.user and hand_calibration == parser.get_default("hand_calibration"):
        hand_calibration = f"hand_calibration_{args_cli.user}.yml"
        print(f"[INFO] Using user '{args_cli.user}' hand calibration: {hand_calibration}")
    pipeline, retargeters = build_duo_pipeline(
        include_xr_hands=True, hand_calibration=hand_calibration, wrist_offsets_xyzw=SPEC.wrist_offsets_xyzw
    )
    teleop_cfg = IsaacTeleopCfg(
        xr_cfg=XrCfg(anchor_pos=tuple(anchor[0]), anchor_rot=tuple(anchor[1])),
        pipeline_builder=lambda: pipeline,
        retargeters_to_tune=lambda: retargeters,
        sim_device=env.device,
    )
    # The CloudXR runtime is owned by main() and survives scene switches, so
    # the device must not launch (or stop) a runtime of its own.
    flow.teleop = teleop = create_isaac_teleop_device(
        teleop_cfg,
        sim_device=env.device,
        callbacks={"START": flow.on_start, "STOP": flow.on_stop, "RESET": flow.on_reset},
        cloudxr_env_file=None,
        auto_launch_cloudxr=False,
    )
    # Voice "align": re-anchor so the user faces along the robot's forward axis.
    # The rig faces +x at identity, so the facing angle IS the root quat's yaw
    # (the default +90 deg yaw faces +y, reproducing the source branch's target).
    robot_yaw = float(R.from_quat(args_cli.robot_rot).as_euler("ZYX")[0])
    align_head_z = args_cli.align_head_z if args_cli.align_head_z > 0 else None
    aligner = AnchorAligner(teleop, tuple(args_cli.align_head_xy), robot_yaw, target_head_z=align_head_z)
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
        "Say 'success'/'failure' to end, label, and export the episode. "
        "Episode timeout discards without export."
    )
    with teleop, torch.inference_mode():
        env.reset()
        settle_scene(env)
        teleop.reset()
        while simulation_app.is_running():
            # An exception escaping this loop means simulation_app.close() under a
            # live XR session, which can deadlock kit's shutdown — catch, report,
            # pause teleop, and keep the session alive.
            try:
                # Voice labels first, and exports BEFORE any reset is processed,
                # so a label + reset burst cannot discard the episode.
                flow.handle_voice_label()
                if flow.voice_display is not None:
                    flow.voice_display.update()  # hides the panel once its message ages out
                if flow.next_requested:
                    flow.request_client_stop()
                    break
                flow.handle_reset()
                action = teleop.advance()
                flow.handle_control_events(poll_control_events)

                serve_align(flow, aligner, current_head_pose)

                # action is None until the XR session has started.
                if action is None:
                    env.sim.render()
                    continue

                xr_hands = action[-XR_EXTRAS_DIM:].reshape(2, 26, 7)
                action = action[:-XR_EXTRAS_DIM]
                XrHandsRecorder.latest = xr_hands
                if markers is not None:
                    markers.update(xr_hands)

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
    # Carry any "align" adjustment into the next scene's device.
    xr_cfg = teleop._anchor_manager._xr_cfg
    anchor_out = (tuple(xr_cfg.anchor_pos), tuple(xr_cfg.anchor_rot))
    return ("next" if flow.next_requested else "quit", anchor_out)


def run_smoke(env: ManagerBasedRLEnv, num_steps: int) -> None:
    """Hold the ready pose for ``num_steps`` control steps and report flange drift."""
    from isaaclab.utils.math import quat_error_magnitude

    env.reset()
    settle_scene(env)
    robot = env.scene["robot"]
    flange_ids = [robot.body_names.index(SPEC.ik_body(side)) for side in ("left", "right")]

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


def load_scene_list() -> list[tuple[str, str | None]]:
    """The ``(usda_path, task_description)`` pairs to teleop.

    From the --scene_list JSON if given (else just --scene_usda). Entries may be
    plain paths or scene-generation dicts (``{"scene": ..., "task_description":
    ...}``, as in a run's instructions JSON — so that file doubles as a scene
    list). Relative paths resolve against the JSON's directory; absolute paths
    that don't exist locally (authored on another machine) fall back to their
    basename next to the JSON. Descriptions not in the list itself are looked up
    in any instructions JSON sitting next to the scene file.
    """
    from task_display import find_task_description

    if args_cli.scene_list is None:
        return [(args_cli.scene_usda, find_task_description(args_cli.scene_usda))]
    import json

    list_path = os.path.abspath(args_cli.scene_list)
    with open(list_path) as f:
        data = json.load(f)
    entries = data["scenes"] if isinstance(data, dict) else data
    base = os.path.dirname(list_path)
    scenes: list[tuple[str, str | None]] = []
    missing: list[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            path, description = entry.get("scene", ""), entry.get("task_description")
            description = str(description).strip().strip("'\"") if description else None
        else:
            path, description = entry, None
        if not os.path.isabs(path):
            path = os.path.join(base, path)
        elif not os.path.exists(path):
            path = os.path.join(base, os.path.basename(path))
        if not os.path.exists(path):
            missing.append(path)
            continue
        scenes.append((path, description if description is not None else find_task_description(path)))
    if not scenes or missing:
        raise SystemExit(f"--scene_list {args_cli.scene_list}: empty or missing scenes {missing}")
    return scenes


def main():
    scenes = load_scene_list()
    if args_cli.smoke is not None:
        env = ManagerBasedRLEnv(cfg=build_env_cfg(scenes[0][0]))
        try:
            run_smoke(env, args_cli.smoke)
        finally:
            env.close()
        return

    # Everything that must SURVIVE scene switches lives here: the voice labeler
    # (Whisper model + microphone stream) and the CloudXR runtime (stopping it
    # would disconnect the headset between scenes).
    labeler = None
    if not args_cli.no_voice:
        from voice_labeler import VoiceLabeler

        labeler = VoiceLabeler(
            model_name=args_cli.whisper_model, device=args_cli.whisper_device, mic_device=args_cli.mic_device
        )
    launcher = None
    from isaaclab_teleop import CLOUDXR_AVP_ENV, CLOUDXR_JS_ENV

    cloudxr_env = {"cloudxrjs": CLOUDXR_JS_ENV, "avp": CLOUDXR_AVP_ENV, "none": None}.get(
        args_cli.cloudxr_env, args_cli.cloudxr_env
    )
    if cloudxr_env is not None:
        from pathlib import Path

        from isaacteleop.cloudxr import CloudXRLauncher

        launcher = CloudXRLauncher(install_dir=str(Path.home() / ".cloudxr"), env_config=cloudxr_env, accept_eula=False)
        print("[INFO] CloudXR runtime launched (kept alive across scene switches).")

    anchor = (tuple(args_cli.anchor_pos), tuple(args_cli.anchor_rot))
    index = 0
    try:
        while True:
            scene, task_description = scenes[index]
            print(f"[SCENE] {index + 1}/{len(scenes)}: {scene}")
            if task_description:
                print(f"[TASK] {task_description}")
            env = ManagerBasedRLEnv(cfg=build_env_cfg(scene))
            if task_description and not args_cli.no_task_display:
                from scipy.spatial.transform import Rotation as R

                from task_display import spawn_task_display

                # At yaw 0 the panel faces -y; the rig faces +y at its default
                # +90 deg yaw, so the panel counter-rotates with the robot.
                panel_yaw = float(R.from_quat(args_cli.robot_rot).as_euler("ZYX", degrees=True)[0]) - 90.0
                spawn_task_display(task_description, tuple(args_cli.task_display_pos), panel_yaw)
            try:
                reason, anchor = run_teleop(env, labeler, os.path.basename(scene), anchor)
            finally:
                env.close()
            if reason != "next":
                break
            index = (index + 1) % len(scenes)
            if index == 0 and len(scenes) > 1:
                print("[SCENE] End of the list; wrapping around to the first scene.")
    finally:
        if labeler is not None:
            labeler.close()
        if launcher is not None:
            launcher.stop()


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

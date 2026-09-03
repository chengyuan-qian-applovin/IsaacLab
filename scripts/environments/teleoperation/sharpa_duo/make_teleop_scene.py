# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load a scene USDA, add a bimanual SharpaWave rig, and teleoperate it via XR.

The pipeline in one line: your USDA becomes the environment, the rig selected
by ``--embodiment`` (FR3 Duo torso by default, or two table-edge-mounted I2RT
YAM Ultra arms) is placed into it at ``--robot_pos``/``--robot_rot``, and your
tracked hands drive it — wrists through per-arm differential IK, all fingers
through DexPilot retargeting. Episodes are recorded by default — each labeled
episode becomes its own robomimic-style HDF5 file — and labeled hands-free:
saying "success" or "failure" (transcribed locally with OpenAI Whisper, on the
CPU) ends, labels, and exports the episode — see README.md for the full flow.
With ``--fleet_server``, this collector coordinates with the central fleet
server: status syncs at startup, the scenes to work on are downloaded, and
every labeled episode uploads immediately (see the fleet section in README.md).

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

    # microphone + ASR check (prints levels, transcripts, and label events)
    ./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/make_teleop_scene.py \\
        --scene_usda unused --voice_test 20
"""

# isort: skip_file
import argparse
import functools
import os
import sys

print = functools.partial(print, flush=True)  # noqa: A001

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Teleoperate a bimanual SharpaWave rig in a USDA scene.")
parser.add_argument(
    "--embodiment",
    type=str,
    choices=("franka_duo", "yam_duo"),
    default="yam_duo",
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
        "Path to the scene USD/USDA/USDZ file to load. Defaults to the vendored TACO brush-and-bowl scene;"
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
    default=0.0,
    help=(
        "DR: fixed shift [m] moving every object's randomization center horizontally toward the robot"
        " base (never past it), bringing objects within easier reach. Off by default, so a scene's"
        " authored layout (including one saved by the 'initial' editor) is used as the center as-is;"
        " note that a nonzero shift compounds if you then re-save the shifted layout."
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
    "--range_panel_pos",
    type=float,
    nargs=3,
    default=None,
    help=(
        "World position [m] of the adjust-mode range panel. Defaults to within arm's reach at the"
        " operator's station, above the tabletop (derived from --align_head_xy and the scene)."
    ),
)
parser.add_argument(
    "--no_adjust",
    action="store_true",
    help=(
        "Disable the 'initial' pose editor (adjust mode) entirely: no panels, no controller block in the"
        " pipeline. Normal teleop runs exactly as before the feature existed."
    ),
)
parser.add_argument(
    "--debug_adjust_buttons",
    action="store_true",
    help=(
        "Draw a small sphere at every adjust-panel key's world-space hit center while the panels are"
        " shown, so a mismatch between where a key LOOKS and where its pinch target IS becomes visible."
    ),
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
    help=(
        "Directory for the recorded episodes: each labeled episode becomes its own HDF5 file"
        " (<scene>_<timestamp>_<uuid8>.hdf5, one demo_0 inside). Also holds the fleet outbox"
        " and scene cache when --fleet_server is used."
    ),
)
parser.add_argument(
    "--fleet_server",
    type=str,
    default=None,
    help=(
        "URL of the duo fleet server (e.g. http://fleet-host:8080). When set, this collector checks"
        " in at startup, declares which scene it is working on, and uploads every labeled episode"
        " immediately (queued locally and retried if the server is unreachable). Without an explicit"
        " --scene_list/--scene_usda, the scenes to work on are fetched from the server and downloaded."
    ),
)
parser.add_argument(
    "--collector_id",
    type=str,
    default=None,
    help="Collector name reported to the fleet server (default: this machine's hostname).",
)
parser.add_argument(
    "--fleet_token",
    type=str,
    default=None,
    help="Fleet server auth token (default: the FLEET_TOKEN environment variable).",
)
parser.add_argument(
    "--fleet_scenes",
    type=int,
    default=8,
    help=(
        "Fleet mode: how many server-suggested scenes to download and cycle through with 'next'"
        " (only when neither --fleet_scene_ids nor a local scene selection is given)."
    ),
)
parser.add_argument(
    "--fleet_scene_ids",
    type=str,
    nargs="+",
    default=None,
    help=(
        "Fleet mode: collect exactly these server scenes (ids are the scene file basenames known to"
        " the server, e.g. foo.usdz); they are downloaded from --fleet_server (sha256-verified) and"
        " cycled with 'next'."
        " Requires --fleet_server; mutually exclusive with --scene_list. This is what the launcher's"
        " 'Fleet server' scene source passes."
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
parser.add_argument("--no_voice", action="store_true", help="Disable voice commands (success/failure labels etc.).")
parser.add_argument(
    "--whisper_model",
    type=str,
    default="base.en",
    help="Whisper model for voice commands (base.en default; small.en is more accurate, ~3x slower on a CPU).",
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
        "Voice-command audio source: an ALSA capture device (arecord -D); 'quest' / 'avp'"
        " (optionally ':<port>', default 8444) to stream the headset microphone — see headset_mic.py"
        " (Quest: open the printed URL in the headset browser before teleoperating; AVP: the Isaac XR"
        " Teleop client streams by itself once connected, feature/avp-voice-mic build); or 'hub'"
        " (optionally 'hub:<port>' / 'hub:<host>:<port>', default 127.0.0.1:8500) to take audio from"
        " the always-on teleop app — see teleop_app.py, which keeps the microphone alive across runs."
    ),
)
parser.add_argument(
    "--voice_test",
    type=int,
    default=None,
    metavar="SECONDS",
    help="Mic/ASR check without the simulator: listen for N seconds, print transcripts and labels, exit.",
)
parser.add_argument(
    "--smoke",
    type=int,
    default=None,
    metavar="N",
    help="No-XR validation: hold the ready pose for N control steps, report flange drift, exit.",
)
parser.add_argument(
    "--smoke_adjust",
    type=int,
    default=None,
    metavar="N",
    help=(
        "No-XR validation of the adjust-mode pose editor: grab an object with a synthetic pinch,"
        " drag it through a full 6-DoF motion over N control steps, check the final pose, that no"
        " other object moved, and the exact robot restore; exit."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# The two scene sources are exclusive: a scene list names local files, scene
# ids name server scenes — mixing them would leave the actual source ambiguous.
if args_cli.fleet_scene_ids is not None:
    if not args_cli.fleet_server:
        parser.error("--fleet_scene_ids requires --fleet_server")
    if args_cli.scene_list is not None:
        parser.error("--fleet_scene_ids and --scene_list are mutually exclusive (pick one scene source)")

if args_cli.voice_test is not None:
    # Standalone mic + ASR check; the simulator never starts.
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

if args_cli.smoke is None and args_cli.smoke_adjust is None:
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
print(
    f"[INFO] Embodiment: {SPEC.name} (--embodiment) at pos {tuple(args_cli.robot_pos)}, rot {tuple(args_cli.robot_rot)}"
)


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
        # (The DR-region overlay renders solid without it — cosmetic only, so it does
        # not force the flag on; normal teleop's renderer setup stays exactly as-is.)
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
        # Ranges: the scene's sidecar (written by the "initial" editor's "done")
        # wins over the CLI defaults, so tuned ranges follow the scene across
        # sessions like the poses do; a --dr_object_* value explicitly changed
        # from its default still wins over the sidecar.
        from usda_scene import load_randomization_overrides

        xy_range, yaw_range = args_cli.dr_object_xy, math.radians(args_cli.dr_object_yaw)
        saved = load_randomization_overrides(scene_usda)
        if saved:
            if "xy_range" in saved and args_cli.dr_object_xy == parser.get_default("dr_object_xy"):
                xy_range = saved["xy_range"]
            if "yaw_range" in saved and args_cli.dr_object_yaw == parser.get_default("dr_object_yaw"):
                yaw_range = saved["yaw_range"]
            print(
                f"[INFO] Randomization ranges from the scene sidecar: xy ±{xy_range:.3f} m,"
                f" yaw ±{math.degrees(yaw_range):.1f} deg"
            )
        env_cfg.events.dr_objects = EventTermCfg(
            func=randomize_tracked_objects,
            mode="reset",
            params={
                "xy_range": xy_range,
                "yaw_range": yaw_range,
                "margin": 0.01,
                "bias_toward": (args_cli.robot_pos[0], args_cli.robot_pos[1]),
                "bias_dist": args_cli.dr_object_bias,
            },
        )
    if args_cli.smoke is None and args_cli.smoke_adjust is None and not args_cli.no_record:
        from recording import DuoRecorderManagerCfg, PerEpisodeHDF5DatasetFileHandler

        # Every labeled episode becomes its own HDF5 file under record_dir; the
        # dataset_filename is only the fallback filename prefix (the flow names
        # each file <scene>_<timestamp>_<uuid8> at export time).
        env_cfg.recorders = DuoRecorderManagerCfg()
        env_cfg.recorders.dataset_file_handler_class_type = PerEpisodeHDF5DatasetFileHandler
        env_cfg.recorders.dataset_export_dir_path = os.path.abspath(args_cli.record_dir)
        env_cfg.recorders.dataset_filename = os.path.splitext(os.path.basename(scene_usda))[0][:60]
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


_FRAME_USD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "frame_prim.usd")
"""Vendored Isaac Sim axis-frame prop (``Props/UIElements/frame_prim.usd``) for the auto-start markers."""


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

            # The vendored copy of Isaac Sim's frame_prim.usd, not the one on the
            # cloud asset server: that download is cached under /tmp/Assets, a
            # path shared by every user of the workstation, and a copy another
            # account cached with owner-only permissions loads as an empty
            # prototype — the frames then silently draw nothing.
            self._frame_markers = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/auto_start_frames",
                    markers={
                        "flange": sim_utils.UsdFileCfg(usd_path=_FRAME_USD, scale=(0.15, 0.15, 0.15)),
                        "target": sim_utils.UsdFileCfg(usd_path=_FRAME_USD, scale=(0.08, 0.08, 0.08)),
                    },
                )
            )
        except Exception as exc:
            print(f"[WARNING] Auto-start axis frames unavailable ({exc}); matching still works.")

    def suspend_frames(self) -> None:
        """Hide the alignment axis frames immediately.

        Adjust mode calls this on entry: :meth:`update` stops running there, so
        without it the frames would stay frozen mid-air at their last drawn
        spot — floating arrows over the hands and the table.
        """
        self._show_frames(None, None)
        self._match_since = None

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

    def __init__(self, env: ManagerBasedRLEnv, labeler, recording: bool, scene_name: str = "", fleet=None):
        self.env = env
        self.labeler = labeler
        self.scene_name = scene_name
        self.fleet = fleet  # FleetClient or None; episode reports go through its outbox
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
        # --- "initial" (adjust) mode state (see :mod:`adjust_mode`). --------------
        # ``adjuster`` and ``range_panel`` are bound after they are spawned in
        # :func:`run_teleop`; ``dr_object_params`` is the live event-term param
        # dict for :func:`randomize_tracked_objects`, mutated by the stdin
        # reader while adjust mode is active (None when DR is disabled).
        self.adjust_requested = False
        self.exit_adjust_requested = False
        self.adjust_reset_requested = False
        self.adjust_mode = False
        self.adjuster = None
        self.range_panel = None
        self.dr_object_params = None
        # The DR-region overlay, bound in run_teleop when the scene has
        # tracked objects and DR is enabled.
        self.region_overlay = None
        self.button_markers = None  # --debug_adjust_buttons spheres

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

    def discard_episode(self) -> None:
        """Drop whatever the recorder has buffered, without exporting it.

        Adjust mode runs the full teleop stack, so the recorder fills up as
        usual; authoring object poses must not leave a demo behind.
        """
        if self.recording:
            self.env.recorder_manager.reset()
        self.awaiting_label = False

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
        """Export the frozen episode to its own HDF5 file and report it to the fleet.

        The episode UUID minted here is the idempotency key everywhere: it names
        the file, is stamped as an HDF5 attribute, and keys the fleet upload —
        so a retried upload can only ever overwrite itself.
        """
        import uuid

        rm = self.env.recorder_manager
        actions = rm.get_episode(0).data.get("actions")
        n_steps = len(actions) if actions is not None else 0
        episode_uuid = uuid.uuid4().hex
        handler = rm._dataset_file_handler
        scene_stem = os.path.splitext(self.scene_name)[0][:60] or "episode"
        handler.next_episode_stem = f"{scene_stem}_{time.strftime('%Y%m%d_%H%M%S')}_{episode_uuid[:8]}"
        rm.set_success_to_episodes([0], torch.tensor([[success]], dtype=torch.bool, device=self.env.device))
        rm.export_episodes([0])
        # Tag the freshly written (still open) file with the episode identity and
        # tuning (the handler writes only its fixed attrs, so set ours on demo_0),
        # then close it so it is complete on disk before the uploader touches it.
        episode_path = None
        try:
            demo_group = handler._hdf5_data_group["demo_0"]
            if self.scene_name:
                demo_group.attrs["scene"] = self.scene_name
            demo_group.attrs["episode_uuid"] = episode_uuid
            # The robot embodiment and drive gains, for training/replay.
            demo_group.attrs["embodiment"] = SPEC.name
            demo_group.attrs["arm_kp"] = args_cli.arm_kp
            demo_group.attrs["arm_kd"] = args_cli.arm_kd
            demo_group.attrs["hand_kp"] = args_cli.hand_kp
            demo_group.attrs["hand_kd"] = args_cli.hand_kd
            if self.fleet is not None:
                demo_group.attrs["collector_id"] = self.fleet.collector_id
            handler.flush()
            episode_path = handler.last_file_path
        except Exception as exc:
            print(f"[WARNING] Could not tag the demo with its scene name and gains: {exc}")
        handler.close()
        rm.reset()
        self.demo_count += 1
        self.awaiting_label = False
        self.reset_requested = True  # hands-free: fresh scene for the next episode
        if success:
            self.success_count += 1
        if self.fleet is not None and episode_path is not None:
            self.fleet.report_episode(
                episode_uuid,
                self.scene_name,
                success,
                episode_path,
                meta={
                    "embodiment": SPEC.name,
                    "num_steps": n_steps,
                    "arm_kp": args_cli.arm_kp,
                    "arm_kd": args_cli.arm_kd,
                    "hand_kp": args_cli.hand_kp,
                    "hand_kd": args_cli.hand_kd,
                },
            )
        try:
            size_str = f", {os.path.getsize(episode_path) / 1e6:.1f} MB" if episode_path else ""
        except OSError:
            size_str = ""
        print(
            f"[SAVED] Episode {episode_uuid[:8]}: success={success}, {n_steps} steps"
            f" ({n_steps / 60.0:.1f} s){size_str} -> {episode_path}\n"
            f"[SAVED] Session total: {self.demo_count} demos"
            f" ({self.success_count} success / {self.demo_count - self.success_count} failure)."
            " Scene reset; press Play or match the start pose."
        )

    # -- per-iteration handlers ----------------------------------------------

    def handle_voice_label(self) -> None:
        """Route a voice command, echoing what was heard and done in the headset."""
        event = self.labeler.poll() if self.labeler is not None else None
        if event is None:
            return
        text = event.text if len(event.text) <= 60 else event.text[:57] + "..."  # the panel is one line
        if event.command is None:
            # Mis-hearings are shown too: without them a command that silently
            # failed to parse is indistinguishable from a dead microphone.
            self.show_voice_message(f'Heard: "{text}"' if text else "Heard a sound, but no speech")
            return
        outcome = self._run_voice_command(event.command)
        self.show_voice_message(f'Detected "{text}" - executed: {outcome}')

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
            if self.adjust_mode:
                # While adjusting, "reset" undoes the edits rather than discarding
                # an episode. The episode reset is skipped in this mode anyway, so
                # setting that flag here would only fire it on the way out.
                self.adjust_reset_requested = True
                return "reset - reverting objects to their poses at entry"
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
        if label == "adjust":
            # Pose-authoring mode: a kinematic editor — the rig is parked,
            # nothing is recorded or simulated, and the edited object poses
            # become the scene's authored ones.
            if self.adjuster is None:
                return "adjust ignored - no tracked objects in this scene"
            if self.adjust_mode:
                return "adjust ignored - already adjusting"
            self.adjust_requested = True
            return "adjust - pinch an object to grab it"
        if label == "done":
            if not self.adjust_mode:
                return "finish ignored - not in adjust mode"
            self.exit_adjust_requested = True
            return "finish - saving poses and randomization ranges"
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


def _live_dr_params(env: ManagerBasedRLEnv) -> dict | None:
    """The ``dr_objects`` param dict the event manager ACTUALLY reads, or None.

    The obvious handle — ``env.cfg.events.dr_objects.params`` — is a dead end:
    every manager deep-copies its config at construction (``ManagerBase.cfg =
    copy.deepcopy(cfg)``), so the event manager fires
    ``randomize_tracked_objects`` with ITS copy's params, and in-place edits of
    the env-cfg dict silently never apply. Only the term cfg returned by
    :meth:`EventManager.get_term_cfg` is live.
    """
    if args_cli.no_dr:
        return None
    try:
        return env.event_manager.get_term_cfg("dr_objects").params
    except ValueError:
        return None


def _adjust_panel_layout() -> tuple[tuple[float, float, float], float]:
    """Placement of the adjust-mode range panel: ``(range_pos, yaw_deg)``.

    An interaction panel must be REACHABLE — pinch-taps are hit-tested at the
    keys' world positions — unlike the read-only task/voice billboards, which
    sit beyond the table. The panel goes within arm's reach of the operator's
    station (``--align_head_xy``), just above the tabletop (from the scene's
    tracked objects, ``usda_scene.SUPPORT_SURFACE_Z``), slightly to the
    operator's left so the workspace stays clear. ``--range_panel_pos``
    overrides it.
    """
    from scipy.spatial.transform import Rotation as R

    import usda_scene

    tabletop = usda_scene.SUPPORT_SURFACE_Z
    if tabletop is None:
        tabletop = float(args_cli.robot_pos[2])
    robot_yaw_deg = float(R.from_quat(args_cli.robot_rot).as_euler("ZYX", degrees=True)[0])
    yaw_rad = math.radians(robot_yaw_deg)
    fwd = (math.cos(yaw_rad), math.sin(yaw_rad))
    right = (math.sin(yaw_rad), -math.cos(yaw_rad))
    head = tuple(args_cli.align_head_xy)
    lateral = -0.35
    range_pos = (
        head[0] + 0.55 * fwd[0] + lateral * right[0],
        head[1] + 0.55 * fwd[1] + lateral * right[1],
        tabletop + 0.55,
    )
    if args_cli.range_panel_pos is not None:
        range_pos = tuple(args_cli.range_panel_pos)
    # Same convention as the other billboards: at yaw 0 a panel faces -y, so
    # this keeps it square to an operator facing along the robot's forward.
    return range_pos, robot_yaw_deg - 90.0


def _current_ranges(flow) -> tuple[float, float]:
    """(xy_range [m], yaw_range [deg]) — from the live DR params if present, else CLI defaults."""
    params = flow.dr_object_params
    xy_m = float(params["xy_range"]) if params is not None else float(args_cli.dr_object_xy)
    yaw_rad = float(params["yaw_range"]) if params is not None else math.radians(args_cli.dr_object_yaw)
    return xy_m, math.degrees(yaw_rad)


def _show_range_panel(flow) -> None:
    """Refresh the RangePanel with the current ranges and show it."""
    if flow.range_panel is None:
        return
    xy_m, yaw_deg = _current_ranges(flow)
    flow.range_panel.set_ranges(xy_m, yaw_deg)
    flow.range_panel.show()
    if flow.button_markers is not None:
        # --debug_adjust_buttons: mark every key's actual pinch-hit center.
        positions = [entry[1] for entry in flow.range_panel.button_positions_world()]
        flow.button_markers.set_visibility(True)
        flow.button_markers.visualize(translations=torch.tensor(positions, dtype=torch.float32))


def _hide_range_panel(flow) -> None:
    """Hide the RangePanel."""
    if flow.range_panel is not None:
        flow.range_panel.hide()
    if flow.button_markers is not None:
        flow.button_markers.set_visibility(False)


#: In-VR pinch-tap step sizes and clamps for the RangePanel buttons.
_XY_STEP_M = 0.01
_XY_CLAMP_M = (0.0, 0.5)
_YAW_STEP_RAD = math.radians(5.0)
_YAW_CLAMP_RAD = (0.0, math.pi)


def _on_range_button(flow, kind: str) -> None:
    """Handle a pinch-tap on one of the four RangePanel buttons.

    Mutates :attr:`EpisodeFlow.dr_object_params` in place so the next reset
    picks up the new range; refreshes the panel texture so the operator sees
    the update.
    """
    params = flow.dr_object_params
    if params is None:
        print("[ADJUST] Panel button ignored: DR is disabled (--no_dr).")
        return
    if kind == "xy_dec":
        params["xy_range"] = max(_XY_CLAMP_M[0], float(params["xy_range"]) - _XY_STEP_M)
    elif kind == "xy_inc":
        params["xy_range"] = min(_XY_CLAMP_M[1], float(params["xy_range"]) + _XY_STEP_M)
    elif kind == "yaw_dec":
        params["yaw_range"] = max(_YAW_CLAMP_RAD[0], float(params["yaw_range"]) - _YAW_STEP_RAD)
    elif kind == "yaw_inc":
        params["yaw_range"] = min(_YAW_CLAMP_RAD[1], float(params["yaw_range"]) + _YAW_STEP_RAD)
    else:
        return
    xy_m, yaw_deg = _current_ranges(flow)
    print(f"[ADJUST] {kind}: xy_range={xy_m:.3f} m, yaw_range={yaw_deg:.1f} deg")
    if flow.range_panel is not None:
        # ``pressed`` lights the key up, so panel taps AND controller-thumbstick
        # steps (which dispatch the same kinds) both give visible feedback.
        flow.range_panel.set_ranges(xy_m, yaw_deg, pressed=kind)


class _AdjustStdinReader:
    """Daemon thread that consumes ``xy_range=<m>`` / ``yaw_range=<deg>`` from stdin.

    Only reads while adjust mode is active — starts on ``enter``, stops on
    ``exit`` — and mutates the reset-time DR params dict in place so subsequent
    resets pick the new values up. Also refreshes the in-headset RangePanel.
    """

    def __init__(self, flow):
        import threading

        self._flow = flow
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        import threading

        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="adjust-stdin-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        import select

        while not self._stop.is_set():
            # Poll so the loop can exit promptly when adjust mode ends.
            r, _, _ = select.select([sys.stdin], [], [], 0.25)
            if not r:
                continue
            line = sys.stdin.readline()
            if not line:
                return  # stdin closed
            self._apply(line.strip())

    def _apply(self, line: str) -> None:
        if "=" not in line:
            return
        key, _, raw = line.partition("=")
        key = key.strip().lower()
        try:
            value = float(raw.strip())
        except ValueError:
            print(f"[ADJUST] Ignored input '{line}': value is not a number")
            return
        params = self._flow.dr_object_params
        if params is None:
            print("[ADJUST] Ignored input: DR is disabled (--no_dr), so ranges have no effect.")
            return
        if key == "xy_range":
            params["xy_range"] = value
        elif key == "yaw_range":
            params["yaw_range"] = math.radians(value)  # accept degrees, store radians
        else:
            print(f"[ADJUST] Unknown key '{key}'; expected 'xy_range' or 'yaw_range'.")
            return
        xy_m, yaw_deg = _current_ranges(self._flow)
        print(f"[ADJUST] xy_range={xy_m:.4f} m, yaw_range={yaw_deg:.2f} deg")
        if self._flow.range_panel is not None:
            self._flow.range_panel.set_ranges(xy_m, yaw_deg)


def _serve_adjust_transitions(flow: "EpisodeFlow", stdin_reader, auto_start) -> None:
    """Serve pending adjust-mode enter / undo / exit requests (voice-driven flags)."""
    if flow.adjust_requested and flow.adjuster is not None:
        flow.adjust_requested = False
        flow.adjust_mode = True
        flow.adjuster.enter()  # snapshot, pin, park the rig, show ghosts + cursor, ghost the desk
        if auto_start is not None:
            auto_start.suspend_frames()  # no rig to match; drop the frozen arrows
        flow.discard_episode()  # nothing authored should become a demo
        # The editor needs no Play: teleop.advance() streams hand data while
        # stopped, and there is no robot to drive.
        flow.stop_teleop()
        _show_range_panel(flow)
        if flow.region_overlay is not None:
            flow.region_overlay.show()
        if stdin_reader is not None:
            stdin_reader.start()
        flow.show_voice_message("Adjust: pinch an object to grab it. 'reset' undoes, 'finish' saves.")
        print(
            "[ADJUST] Entered adjust mode — a kinematic pose editor; the rig is parked and"
            " nothing is recorded or simulated. Pinch directly on an object to grab it: it is"
            " welded to your hand 1:1 (the grabbed spot stays under your fingers; the"
            " tabletop is a hard floor) and stays exactly where you release it. Pinch with"
            " BOTH hands in empty air to grab the WORLD: move to pan, turn to rotate,"
            " spread/close to zoom ('align' resets the view). Ghosts mark the entry poses."
            " Retune xy_range and yaw_range (degrees) with a controller thumbstick"
            " (left/right = xy, up/down = yaw), by pinching the panel's '-' / '+' keys, or"
            " by typing 'xy_range=0.08' / 'yaw_range=45' at the terminal. Say 'reset' to"
            " undo every change since entering, 'finish' to save."
        )
    if flow.adjust_reset_requested and flow.adjust_mode:
        flow.adjust_reset_requested = False
        restored = flow.adjuster.reset()
        print(f"[ADJUST] Reverted {restored} object(s) onto their ghosts (the poses at mode entry).")
    if flow.exit_adjust_requested and flow.adjust_mode:
        flow.exit_adjust_requested = False
        flow.stop_teleop()  # belt and braces: leave the mode with teleop stopped
        flow.adjuster.exit(flow.dr_object_params)  # saves poses + ranges, restores the rig, clears the aids
        flow.discard_episode()
        flow.adjust_mode = False
        _hide_range_panel(flow)
        if flow.region_overlay is not None:
            flow.region_overlay.hide()
        if stdin_reader is not None:
            stdin_reader.stop()
        print("[ADJUST] Exited adjust mode; teleop stopped. Press Play or say 'play' to resume.")


def run_teleop(
    env: ManagerBasedRLEnv, labeler, scene_name: str, anchor: tuple, scene_usda: str, fleet=None
) -> tuple[str, tuple]:
    """Drive the env from XR hand tracking for one scene (see :class:`EpisodeFlow`).

    Args:
        env: The environment (one scene of the list).
        labeler: The shared :class:`~voice_labeler.VoiceLabeler` (or None).
        scene_name: Stamped on every exported demo as its ``scene`` HDF5 attr
            (and used as the fleet scene id).
        anchor: ``(anchor_pos, anchor_rot)`` xyzw to start from, so "align"
            adjustments carry across scene switches.
        scene_usda: Absolute path to the scene USDA — the "initial" (adjust) mode
            writes edited poses to ``<scene_usda>.poses.json`` next to it.
        fleet: The shared :class:`~fleet_client.FleetClient` (or None); every
            labeled episode is queued for immediate upload through it.

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
        XR_CONTROLLERS_DIM,
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

    flow = EpisodeFlow(env, labeler, recording, scene_name=scene_name, fleet=fleet)
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

    # "initial" (adjust) mode: park the rig and edit tracked object poses
    # kinematically by pinch-grabbing them. Spawn the panels + wire the
    # ObjectAdjuster only if the scene actually has tracked objects; otherwise
    # leave flow.adjuster None so the voice dispatcher explains the no-op.
    tracked_names = [n.removeprefix("object_") for n in env.scene.rigid_objects.keys() if n.startswith("object_")]
    if tracked_names and not args_cli.no_adjust:
        from adjust_mode import ObjectAdjuster
        from task_display import RangePanel

        # The editor reuses the --visualize_hands markers as its cursor when
        # present (the loop updates them every frame); else it makes its own.
        flow.adjuster = ObjectAdjuster(env, tracked_names, scene_usda, shared_hand_markers=markers)
        range_pos, range_panel_yaw = _adjust_panel_layout()
        flow.range_panel = RangePanel(range_pos, range_panel_yaw)
        print(f"[ADJUST] RangePanel at {tuple(round(v, 2) for v in range_pos)} (--range_panel_pos to move)")
        # Live handle to the reset-time DR params — pinch-tap buttons on the
        # panel, the controller thumbstick, and the stdin reader all mutate
        # this dict in place — plus the in-headset overlay that draws what the
        # xy range means on the table. MUST be the event manager's own dict
        # (see _live_dr_params); the env-cfg dict is a deep-copied dead end.
        flow.dr_object_params = _live_dr_params(env)
        if flow.dr_object_params is not None:
            from region_overlay import RegionOverlay

            flow.region_overlay = RegionOverlay(env, tracked_names)
        # Wire pinch-tap dispatch onto the panel: hit-test every key each
        # frame, fire :func:`_on_range_button` on the rising edge (see
        # :meth:`ObjectAdjuster.step`). The panel reports per-key radii and
        # its plane normal, so the hit test is tight on the panel and slack
        # in depth; the dispatch radius below is only the legacy fallback.
        flow.adjuster.set_button_dispatch(
            buttons_fn=lambda _flow=flow: _flow.range_panel.button_positions_world(),
            on_press=lambda kind, _flow=flow: _on_range_button(_flow, kind),
            hit_radius=RangePanel.HIT_RADIUS_M,
        )
        if args_cli.debug_adjust_buttons:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            flow.button_markers = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/adjust_button_spheres",
                    markers={
                        "hit": sim_utils.SphereCfg(
                            radius=0.02,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.8), opacity=0.5),
                        )
                    },
                )
            )
            flow.button_markers.set_visibility(False)

    pipeline, retargeters = build_duo_pipeline(
        include_xr_hands=True,
        include_xr_controllers=not args_cli.no_adjust,
        hand_calibration=hand_calibration,
        wrist_offsets_xyzw=SPEC.wrist_offsets_xyzw,
    )
    # The tuning UI is a GLFW (X11) window, not a Kit one: without a display
    # its thread dies with "Failed to initialize GLFW", so don't request it.
    has_display = bool(os.environ.get("DISPLAY"))
    if not has_display:
        print("[INFO] No DISPLAY: skipping the retargeter tuning UI.")
    teleop_cfg = IsaacTeleopCfg(
        xr_cfg=XrCfg(anchor_pos=tuple(anchor[0]), anchor_rot=tuple(anchor[1])),
        pipeline_builder=lambda: pipeline,
        retargeters_to_tune=(lambda: retargeters) if has_display else None,
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
    if flow.adjuster is not None:
        # World-grab view navigation drives the same live anchor cfg the voice
        # "align" command mutates; exit() restores the entry view exactly.
        flow.adjuster.set_view_control(teleop._anchor_manager._xr_cfg)
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
        print(f"[INFO] Recording to {env.cfg.recorders.dataset_export_dir_path} (one HDF5 file per labeled episode).")
    print(
        "[INFO] Teleop loop started. Headset: Play = start, Stop = pause, Reset = reset (discards episode). "
        "Say 'success'/'failure' to end, label, and export the episode. "
        "Episode timeout discards without export."
    )
    stdin_reader = _AdjustStdinReader(flow) if flow.adjuster is not None else None
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
                # Adjust-mode transitions: served BEFORE handle_reset so a stray
                # reset request doesn't fire while the operator is repositioning.
                _serve_adjust_transitions(flow, stdin_reader, auto_start)
                if flow.next_requested:
                    flow.request_client_stop()
                    break
                if not flow.adjust_mode:
                    flow.handle_reset()
                action = teleop.advance()
                flow.handle_control_events(poll_control_events)

                serve_align(flow, aligner, current_head_pose)

                # action is None until the XR session has started.
                if action is None:
                    env.sim.render()
                    continue

                # Slice the ride-along blocks off the 58-D robot action, in
                # reverse order of how the pipeline appended them (the
                # controllers block exists only when adjust mode is available).
                xr_controllers = None
                if not args_cli.no_adjust:
                    xr_controllers = action[-XR_CONTROLLERS_DIM:].reshape(2, 11)
                    action = action[:-XR_CONTROLLERS_DIM]
                xr_hands = action[-XR_EXTRAS_DIM:].reshape(2, 26, 7)
                action = action[:-XR_EXTRAS_DIM]
                XrHandsRecorder.latest = xr_hands
                if markers is not None:
                    markers.update(xr_hands)

                if flow.adjust_mode:
                    # The kinematic editor owns its own stepping: pinches (panel
                    # taps first, else grabs), controller input, the DR-region
                    # overlay, and the pin-everything substep loop. env.step
                    # never runs here, so the recorder stays empty and the
                    # episode timeout cannot fire mid-adjust.
                    flow.adjuster.step(xr_hands)
                    flow.adjuster.step_controllers(xr_controllers)
                    flow.adjuster.update_cursor(xr_hands)
                    if flow.region_overlay is not None and flow.dr_object_params is not None:
                        flow.region_overlay.update(flow.dr_object_params)
                    flow.adjuster.step_sim()
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
        if stdin_reader is not None:
            stdin_reader.stop()
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


def run_adjust_smoke(env: ManagerBasedRLEnv, num_steps: int, scene_usda: str) -> None:
    """No-XR check of the kinematic pose editor.

    Opens the editor, grabs the first tracked object with a synthetic pinch,
    drags it +10 cm in x and +5 cm up while yawing the wrist 90° and rolling it
    30° (full 6-DoF), releases, and asserts that (a) the object landed exactly
    there, (b) no other object moved (physics is inert), (b2) a shove below the
    table clamps the mesh onto the tabletop, (c) exiting restores the robot
    exactly, and (d) the range panel lands reachable and above the tabletop.
    Also asserts that a range edit reaches the event manager, and backs the
    scene's pose sidecar up so a smoke never re-authors it.
    """
    from adjust_mode import _ROBOT_PARK_LIFT_M, ObjectAdjuster
    from scipy.spatial.transform import Rotation as R
    from task_display import RangePanel

    import usda_scene

    from isaaclab.utils.math import quat_error_magnitude, quat_mul

    env.reset()
    settle_scene(env)
    tracked = [n.removeprefix("object_") for n in env.scene.rigid_objects.keys() if n.startswith("object_")]
    if not tracked:
        print("[SMOKE-ADJUST] Scene has no tracked objects; nothing to validate.")
        return
    robot = env.scene["robot"]
    before_root = torch.cat([robot.data.root_pos_w.torch[0], robot.data.root_quat_w.torch[0]]).clone()
    before_joints = robot.data.joint_pos.torch.clone()
    entry_poses = {name: env.scene[f"object_{name}"].data.root_pos_w.torch[0].clone().cpu() for name in tracked}
    tabletop = usda_scene.SUPPORT_SURFACE_Z
    if tabletop is None:
        tabletop = float(args_cli.robot_pos[2])
    ok = True

    adjuster = ObjectAdjuster(env, tracked, scene_usda)
    overlay = None
    dr_params = _live_dr_params(env)
    if dr_params is not None:
        from region_overlay import RegionOverlay

        overlay = RegionOverlay(env, tracked)
        overlay.show()
        # Regression: a range edit through this handle must reach the dict the
        # event manager unpacks into randomize_tracked_objects. The env-cfg
        # dict does NOT (managers deep-copy their cfg), which once made every
        # in-headset range edit a silent no-op.
        dr_params["xy_range"] = 0.123
        applied = float(env.event_manager.get_term_cfg("dr_objects").params["xy_range"])
        print(f"[SMOKE-ADJUST] range edit reaches the event manager: xy_range={applied}")
        ok = ok and applied == 0.123

    # Synthetic pinch: thumb (5) and index (10) tips straddle a midpoint that
    # starts on the first object; the wrist (1) supplies the full rotation.
    grabbed = tracked[0]
    start = env.scene[f"object_{grabbed}"].data.root_pos_w.torch[0].clone().cpu()
    start_quat = env.scene[f"object_{grabbed}"].data.root_quat_w.torch[0].clone().cpu()
    xr_hands = torch.zeros(2, 26, 7)
    # Full 6-DoF drag target: +10 cm x, +8 cm up, yaw 90 deg + roll 30 deg.
    drag_offset = torch.tensor([0.10, 0.0, 0.08])
    _t, _i = 5, 10  # OpenXR thumb/index tips
    _IDENT = torch.tensor([0.0, 0.0, 0.0, 1.0])
    zero3 = torch.zeros(3)

    def wrist_rot(frac: float) -> torch.Tensor:
        return torch.tensor(R.from_euler("zx", [90.0 * frac, 30.0 * frac], degrees=True).as_quat(), dtype=torch.float32)

    def set_hand(offset: torch.Tensor, gap: float, frac: float) -> None:
        mid = torch.tensor([float(start[0]), float(start[1]), float(start[2]) + 0.05]) + offset

        # A rigid synthetic hand rotating as one piece: thumb/index tip+distal
        # (the grip frame the editor welds to) plus wrist + two knuckles (its
        # twist reference) — all read as POSITIONS by _grip_rotation.
        rot = R.from_euler("zx", [90.0 * frac, 30.0 * frac], degrees=True)

        def place(local) -> torch.Tensor:
            return mid + torch.tensor(rot.apply(local), dtype=torch.float32)

        xr_hands[1, _t] = torch.cat([place([0.0, gap / 2, 0.0]), _IDENT])
        xr_hands[1, 4] = torch.cat([place([-0.02, gap / 2, 0.0]), _IDENT])
        xr_hands[1, _i] = torch.cat([place([0.0, -gap / 2, 0.0]), _IDENT])
        xr_hands[1, 9] = torch.cat([place([-0.02, -gap / 2, 0.0]), _IDENT])
        wrist = place([-0.08, 0.0, 0.0])
        xr_hands[1, 1] = torch.cat([wrist, wrist_rot(frac)])
        xr_hands[1, 7] = torch.cat([wrist + torch.tensor(rot.apply([0.07, 0.02, 0.0]), dtype=torch.float32), _IDENT])
        xr_hands[1, 22] = torch.cat([wrist + torch.tensor(rot.apply([0.06, -0.03, 0.0]), dtype=torch.float32), _IDENT])

    with torch.inference_mode():
        adjuster.enter()
        parked_lift = float(robot.data.root_pos_w.torch[0, 2]) - float(before_root[2])
        ok = ok and abs(parked_lift - _ROBOT_PARK_LIFT_M) < 1e-3
        # Approach open, then close the pinch on the object (rising edge = grab).
        for _ in range(5):
            set_hand(zero3, 0.10, 0.0)
            adjuster.step(xr_hands)
            adjuster.step_sim()
        set_hand(zero3, 0.005, 0.0)
        adjuster.step(xr_hands)
        adjuster.step_sim()
        held = adjuster.held_objects()
        print(f"[SMOKE-ADJUST] grabbed {held} (expected ['{grabbed}'])")
        ok = ok and held == [grabbed]
        # Drag through the full 6-DoF motion.
        for step in range(num_steps):
            frac = (step + 1) / num_steps
            set_hand(drag_offset * frac, 0.005, frac)
            adjuster.step(xr_hands)
            adjuster.step_sim()
            if overlay is not None:
                overlay.update(dr_params)
        # Release, then keep stepping: pinning must hold every pose (inert physics).
        set_hand(drag_offset, 0.10, 1.0)
        adjuster.step(xr_hands)
        for _ in range(20):
            adjuster.step_sim()

        # (a) the rigid weld: T_obj = H(t) ∘ O. The pinch mid sat 5 cm above
        # the object at grab, so that offset swings with the hand's rotation.
        final = env.scene[f"object_{grabbed}"].data.root_pos_w.torch[0].cpu()
        final_quat = env.scene[f"object_{grabbed}"].data.root_quat_w.torch[0].cpu()
        hand_off = torch.tensor([0.0, 0.0, 0.05])
        swung = torch.tensor(
            R.from_euler("zx", [90.0, 30.0], degrees=True).apply(-hand_off.numpy()), dtype=torch.float32
        )
        expected_pos = start + hand_off + drag_offset + swung
        pos_err = float((final - expected_pos).norm())
        # The hand frame started at identity, so the weld's rotation delta IS
        # the synthetic hand's final rotation.
        expected_quat = quat_mul(wrist_rot(1.0).unsqueeze(0), start_quat.unsqueeze(0))
        rot_err = float(quat_error_magnitude(final_quat.unsqueeze(0), expected_quat)[0])
        print(f"[SMOKE-ADJUST] '{grabbed}': pos err {pos_err * 1000:.2f} mm, rot err {math.degrees(rot_err):.2f} deg")
        ok = ok and pos_err < 0.005 and rot_err < math.radians(3.0)

        # (b) physics is inert: every other object stayed put.
        for name in tracked[1:]:
            moved = float((env.scene[f"object_{name}"].data.root_pos_w.torch[0].cpu() - entry_poses[name]).norm())
            print(f"[SMOKE-ADJUST] bystander '{name}' moved {moved * 1000:.3f} mm")
            ok = ok and moved < 1e-3
        for name in tracked:
            ok = ok and bool(torch.isfinite(env.scene[f"object_{name}"].data.root_pos_w.torch[0]).all())

        # (b2) the tabletop is a hard floor: re-grab (still tilted) and shove
        # the object 30 cm down — the clamp must land its lowest mesh point
        # exactly ON the support surface, never below it.
        set_hand(drag_offset, 0.005, 1.0)
        adjuster.step(xr_hands)
        adjuster.step_sim()
        ok = ok and adjuster.held_objects() == [grabbed]
        push = drag_offset + torch.tensor([0.0, 0.0, -0.30])
        for step in range(20):
            frac = (step + 1) / 20
            set_hand(drag_offset + (push - drag_offset) * frac, 0.005, 1.0)
            adjuster.step(xr_hands)
            adjuster.step_sim()
        set_hand(push, 0.10, 1.0)  # release at the floor
        adjuster.step(xr_hands)
        adjuster.step_sim()
        floor = usda_scene.SUPPORT_SURFACE_Z if usda_scene.SUPPORT_SURFACE_Z is not None else tabletop
        pos_f = env.scene[f"object_{grabbed}"].data.root_pos_w.torch[0].cpu().numpy()
        quat_f = env.scene[f"object_{grabbed}"].data.root_quat_w.torch[0].cpu().numpy()
        support = usda_scene.OBJECT_SUPPORT_POINTS[grabbed]
        lowest = float(pos_f[2] + R.from_quat(quat_f).apply(support)[:, 2].min())
        print(f"[SMOKE-ADJUST] floor clamp: lowest mesh point {lowest:.4f} m vs tabletop {floor:.4f} m")
        ok = ok and lowest > floor - 1e-3 and lowest < floor + 3e-3

        # (c) exit restores the robot exactly. Back the sidecar up first: exit()
        # saves the edited pose, and a smoke run must not re-author the scene.
        sidecar = usda_scene.sidecar_path(scene_usda)
        backup = None
        if os.path.exists(sidecar):
            with open(sidecar) as f:
                backup = f.read()
        adjuster.exit(dr_params)
        if backup is not None:
            with open(sidecar, "w") as f:
                f.write(backup)
        elif os.path.exists(sidecar):
            os.remove(sidecar)
        after_root = torch.cat([robot.data.root_pos_w.torch[0], robot.data.root_quat_w.torch[0]])
        root_err = float((after_root[:3] - before_root[:3]).norm())
        joint_err = float((robot.data.joint_pos.torch - before_joints).abs().max())
        print(f"[SMOKE-ADJUST] restore: root err {root_err * 1000:.3f} mm, joint err {joint_err:.5f} rad")
        ok = ok and root_err < 1e-4 and joint_err < 1e-3

    # (d) panel placement: every key above the tabletop and within reach.
    range_pos, panel_yaw = _adjust_panel_layout()
    panel = RangePanel(range_pos, panel_yaw)
    head = tuple(args_cli.align_head_xy)
    worst_reach, lowest_key = 0.0, float("inf")
    for kind, pos, radius, normal in panel.button_positions_world():
        worst_reach = max(worst_reach, math.hypot(pos[0] - head[0], pos[1] - head[1]))
        lowest_key = min(lowest_key, pos[2])
    print(
        f"[SMOKE-ADJUST] panels: lowest key {lowest_key:.2f} m (tabletop {tabletop:.2f}),"
        f" farthest key {worst_reach:.2f} m from the operator"
    )
    ok = ok and lowest_key > tabletop + 0.02 and worst_reach < 0.95

    print("[SMOKE-ADJUST] OK" if ok else "[SMOKE-ADJUST] FAILED")
    if not ok:
        raise SystemExit(1)


def load_scene_list(fleet=None) -> list[tuple[str, str | None]]:
    """The ``(usda_path, task_description)`` pairs to teleop.

    From the --scene_list JSON if given (else just --scene_usda). Entries may be
    plain paths or scene-generation dicts (``{"scene": ..., "task_description":
    ...}``, as in a run's instructions JSON — so that file doubles as a scene
    list). Relative paths resolve against the JSON's directory; absolute paths
    that don't exist locally (authored on another machine) fall back to their
    basename next to the JSON. Descriptions not in the list itself are looked up
    in any instructions JSON sitting next to the scene file.

    Fleet scene sources (exclusive with a local scene list): with
    ``--fleet_scene_ids`` exactly those server scenes are used; otherwise, with
    no explicit local selection, the server picks the ``--fleet_scenes``
    most-needed ones. Either way each scene — one self-contained ``.usdz``
    package, nothing else to fetch — is downloaded sha256-verified into
    ``<record_dir>/fleet_cache/scenes/`` and cycled exactly like a local
    scene list.
    """
    from task_display import find_task_description

    def fetch_fleet_scene(row: dict, verb: str) -> tuple[str, str | None]:
        """Bring one server scene local: the single self-contained package, current."""
        scene_id = row["scene_id"]
        path = fleet.download_scene(scene_id, row.get("sha256") or None)
        workers = ", ".join(w for w in row["active_workers"] if w != fleet.collector_id) or "nobody else"
        print(
            f"[FLEET] {verb} {scene_id}: {row['successes']}/{row['target_successes']} successes,"
            f" {row.get('size_bytes', 0) / 1e6:.1f} MB package, working: {workers}"
        )
        return path, row.get("task_description") or find_task_description(path)

    if fleet is not None and args_cli.fleet_scene_ids:
        rows = {row["scene_id"]: row for row in fleet.list_scenes()}
        missing = [sid for sid in args_cli.fleet_scene_ids if sid not in rows]
        if missing:
            raise SystemExit(f"[FLEET] Scenes not on the server: {missing} (push them with fleet_push_scenes.py)")
        return [fetch_fleet_scene(rows[sid], "Selected") for sid in args_cli.fleet_scene_ids]

    if fleet is not None and args_cli.scene_list is None and args_cli.scene_usda == parser.get_default("scene_usda"):
        entries = fleet.suggest(args_cli.fleet_scenes)
        if not entries:
            raise SystemExit("[FLEET] Nothing to collect: every scene on the server is at its target (or retired).")
        return [fetch_fleet_scene(entry, "Assigned") for entry in entries]

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
    # Fleet mode: sync the collection status from the server before anything else.
    fleet = None
    if args_cli.fleet_server:
        from fleet_client import FleetClient

        fleet = FleetClient(
            args_cli.fleet_server,
            collector_id=args_cli.collector_id,
            token=args_cli.fleet_token,
            state_dir=os.path.abspath(args_cli.record_dir),
        )
        snapshot = fleet.check_in()  # fail fast here if the server is misconfigured/unreachable
        totals = snapshot["totals"]
        online = [c["collector_id"] for c in snapshot["collectors"] if c["online"]]
        print(
            f"[FLEET] Checked in as '{fleet.collector_id}' at {fleet.server_url}:"
            f" {totals['successes_toward_target']}/{totals['target_successes']} successes"
            f" across {totals['scenes']} scenes; online collectors: {', '.join(online) or 'just you'}."
        )
        # The loose scene-doc JSONs (task descriptions) are always re-fetched so
        # the local copies, sitting next to the cached scene files, are current.
        docs = fleet.sync_docs()
        if docs:
            print(f"[FLEET] Synced {len(docs)} scene doc(s): {', '.join(docs)}")
        fleet.start()

    scenes = load_scene_list(fleet)
    if args_cli.smoke is not None:
        env = ManagerBasedRLEnv(cfg=build_env_cfg(scenes[0][0]))
        try:
            run_smoke(env, args_cli.smoke)
        finally:
            env.close()
            if fleet is not None:
                fleet.close()
        return
    if args_cli.smoke_adjust is not None:
        env = ManagerBasedRLEnv(cfg=build_env_cfg(scenes[0][0]))
        try:
            run_adjust_smoke(env, args_cli.smoke_adjust, scenes[0][0])
        finally:
            env.close()
        return

    # Everything that must SURVIVE scene switches lives here: the voice labeler
    # (ASR model + microphone stream) and the CloudXR runtime (stopping it
    # would disconnect the headset between scenes).
    labeler = None
    if not args_cli.no_voice:
        from voice_labeler import VoiceLabeler

        labeler = VoiceLabeler(
            model_name=args_cli.whisper_model, device=args_cli.whisper_device, mic_device=args_cli.mic_device
        )
    launcher = None
    from isaaclab_teleop import CLOUDXR_AVP_ENV, CLOUDXR_JS_ENV, patch_cloudxr_wss_backend_port

    cloudxr_env = {"cloudxrjs": CLOUDXR_JS_ENV, "avp": CLOUDXR_AVP_ENV, "none": None}.get(
        args_cli.cloudxr_env, args_cli.cloudxr_env
    )
    if cloudxr_env is not None:
        from pathlib import Path

        from isaacteleop.cloudxr import CloudXRLauncher

        # Ports are taken from the environment (NV_CXR_SERVER_PORT, NV_CXR_MEDIA_PORT,
        # PROXY_PORT — what the launcher UI's "Network ports" group sets). The WSS
        # proxy has to be told about a moved signaling port explicitly.
        patch_cloudxr_wss_backend_port()
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
            if fleet is not None:
                # Presence only (never a lock); heartbeats keep it fresh, and a
                # server outage here must not stop the operator from collecting.
                try:
                    row = fleet.declare_scene(os.path.basename(scene))
                    others = [w for w in row["active_workers"] if w != fleet.collector_id]
                    also = f" (also being collected by {', '.join(others)})" if others else ""
                    print(f"[FLEET] Working on {row['scene_id']}: {row['successes']}/{row['target_successes']}{also}")
                except Exception as exc:
                    print(f"[FLEET] Could not declare the scene ({exc}); collecting anyway.")
            env = ManagerBasedRLEnv(cfg=build_env_cfg(scene))
            if task_description and not args_cli.no_task_display:
                from scipy.spatial.transform import Rotation as R

                from task_display import spawn_task_display

                # At yaw 0 the panel faces -y; the rig faces +y at its default
                # +90 deg yaw, so the panel counter-rotates with the robot.
                panel_yaw = float(R.from_quat(args_cli.robot_rot).as_euler("ZYX", degrees=True)[0]) - 90.0
                spawn_task_display(task_description, tuple(args_cli.task_display_pos), panel_yaw)
            try:
                reason, anchor = run_teleop(env, labeler, os.path.basename(scene), anchor, scene, fleet=fleet)
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
        if fleet is not None:
            fleet.close()  # drains the upload queue (bounded); leftovers sync next session


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

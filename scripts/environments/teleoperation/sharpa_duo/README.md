# SharpaWave Duo USDA Teleop

Load any scene USDA, drop the FR3 Duo + SharpaWave rig into it, and teleoperate
the rig with XR hand tracking:

```bash
./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/make_teleop_scene.py \
    --scene_usda ~/sim_benchmark/scene/taco_hoi_178_023.usda --headless
```

Put the headset on, open the CloudXR client (Quest/Pico: the CloudXR.js web
client, e.g. `https://nvidia.github.io/IsaacTeleop/client/release-1.3.x`; Apple
Vision Pro: pass `--cloudxr_env avp` and use the Isaac XR Teleop Sample Client),
connect, and press **Play**. Your wrists drive the two Panda arms through
differential IK and all ten fingers are retargeted onto the SharpaWave hands.
The arms render 5% transparent by default so they don't block your view
(`--arm_visual normal|hidden` to change; `--visualize_hands` draws markers on
the tracked joints).

## Recording episodes, hands-free

Recording is on by default (`--no_record` disables it). Every episode becomes
one demo in a timestamped robomimic-style HDF5 under `--record_dir`
(default `./datasets/duo_teleop`):

1. **Play** starts teleop and the episode buffer — by the client button, the
   voice command, or **auto-start**: hold both wrists at the robot's hand
   poses (within 5 cm / 20° for 0.5 s, `--auto_start_*_tol` to tune,
   `--no_auto_start` to disable) and teleop engages by itself with zero
   initial IK error, so the robot never snaps to distant hands. After a stop
   it re-arms only once you move your hands clearly away.
2. End the episode either way:
   - **Cross-hand stop gesture** — touch all five fingertip pairs of the two
     hands together for 0.5 s. The episode closes and waits for your label.
   - **Just say the label** — speaking a label mid-episode ends AND labels it
     in one utterance.
3. **Say "success" or "failure"** into the machine's microphone. The demo is
   exported with that label, the scene resets, and teleop ends in the stopped
   state — press Play to start the next episode.
4. **Reset** (headset button) discards the in-flight episode instead; an
   episode timeout also discards.

Each demo carries per-step robot joint states, tracked object poses, the 58-D
actions, the raw XR hand poses (`obs/xr_hands`, (T, 2, 26, 7), the retargeter
input — enough to re-tune retargeting offline), the PD drive setpoints
(`obs/joint_setpoints`, (T, 58), the differential-IK output), and a boolean
`success` attribute.

## Replaying episodes

```bash
./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/replay_teleop_scene.py \
    --scene_usda ~/sim_benchmark/scene/taco_hoi_178_023.usda --headless
```

Loads the same USDA, replays the newest recorded dataset (or `--dataset
<file-or-dir>`) **kinematically** — every frame is a prescribed recorded state
pushed through forward kinematics; `env.step()` is never called and every PhysX
solver knob is floored, so nothing is solved or collided. One third-person
camera (`--cam_eye`/`--cam_lookat`) writes `<output_dir>/<demo>/video.mp4` at
30 fps plus a `meta.json` with the success label. `--episodes
all|success|failure|0,3,7` selects demos. No domain randomization.

### Voice labels (OpenAI Whisper)

Voice commands are transcribed locally by `openai-whisper` (`pip install
openai-whisper` into the env; done via `--whisper_model`, default `base.en`,
running on `--whisper_device`, default `cpu` so it never competes with the sim
and CloudXR for the GPU). Besides the labels, saying **"align"** (while teleop
is stopped) re-anchors the XR session: it rotates the world about your head
until you face the robot's forward axis and moves you to `--align_head_xy`
(default: the TACO table's near edge, z untouched) — the port of the source
branch's AVP Align button, with voice replacing the button. The head pose is
queried from XRCore on demand; do NOT put a head tracker in the retargeting
pipeline (it makes every session step fail on this stack).
Every transcription is printed to the console, labels and mis-hearings alike.
The full voice vocabulary: **"success"** / **"failure"** (label + export the
episode), **"align"** (re-anchor, teleop stopped only), **"play"** (or
"start" — starts teleop, driven through the same state machine as the client
button), **"reset"** (discards the in-flight episode and resets the scene),
and **"next"** (or "skip" — advance to the next scene in the `--scene_list`,
wrapping at the end; an unlabeled in-flight episode is discarded, a
label-pending one must be labeled first). An utterance matching more than one
command is ignored.

## Launcher UI

```bash
./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/teleop_launcher.py
```

A two-page launcher (plain tkinter; Isaac Sim only starts when you press
Start): page 1 groups the teleop parameters by concern (operator & voice,
session start, domain randomization, stop gesture, visuals, advanced); page 2
picks a scene directory and a dataset HDF5 file and shows a table of every
scene with the success/failure trajectory counts already collected for it in
that dataset — tick the scenes to collect this session. Start writes the
selection to a scene-list JSON and runs the teleop with `--dataset_file`:
demos from **all** scenes and sessions append into the chosen file (each
tagged with its scene), so the table's counts accumulate across sessions.
Cycle the selected scenes with the "next" voice command; when the run exits,
the launcher returns to the table with refreshed counts.

## Multi-scene sessions

`--scene_list scenes/scene_list.json` teleops through a list of scenes (JSON:
a list of USDA paths or `{"scenes": [...]}`, relative to the JSON's
directory); the session starts at the first and **"next"** advances. Each
scene switch rebuilds the environment (the CloudXR runtime, headset
connection, Whisper model, and any "align" adjustment all survive it) and
opens a fresh dataset file named `dataset_<time>_<scene>.hdf5`; every demo
also carries a `scene` HDF5 attribute naming the scene it was recorded in.

## Domain randomization

On by default (`--no_dr` disables), applied at every episode reset:

- **Arm start pose**: each arm joint gets a uniform offset within
  `--dr_arm_jitter` (0.08 rad) around the ready pose. Auto-start adapts —
  you match the robot wherever it actually is.
- **Object placement**: each tracked object gets a uniform xy offset within
  `--dr_object_xy` (5 cm) and a yaw within `--dr_object_yaw` (180°) around
  its authored pose. Draws are rejection-sampled against bounding-circle
  overlap (footprints from the USD bounds + 1 cm margin — the collision model
  of sim_benchmark's scenegen solvers), never demanding more clearance than
  the authored layout had; after 50 failed draws the authored poses are kept.
  Stacked arrangements (xy-coincident objects, e.g. the ARCTIC box lid on its
  base) move as one group and skip yaw so they are never knocked apart.

Audio can come from two places:

- **Workstation microphone** (default): captured via `arecord`
  (`--mic_device` selects the ALSA device); stay within speaking range.
- **Headset microphone** (`--mic_device quest`): nothing in the CloudXR stack
  streams the headset mic to the server, so `quest_mic.py` provides the path —
  the script serves a small HTTPS page and prints its URL; open it in the
  Quest browser *before* connecting the CloudXR client, accept the certificate
  (it reuses the CloudXR proxy's), tap **Start microphone**, grant the mic
  permission, then connect the CloudXR client as usual. The page streams
  16 kHz PCM over WSS from the background tab, putting the mic at your mouth
  instead of across the room. Open the port first
  (`sudo ufw allow 8444/tcp`; `quest:<port>` changes it). Stay quiet for the
  first ~2 s after tapping Start — the energy gate calibrates on that ambient.

Test the mic + Whisper chain without starting the simulator:

```bash
./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/make_teleop_scene.py \
    --scene_usda unused --voice_test 20
```

Microphone notes for this machine (Legion, ALC287 codec): the capture channel
carries near-full-scale infrasonic wander, so **do not max the input volume** —
at 100% the wander clips and drowns speech. Keep the source around 10–25%
(`wpctl set-volume @DEFAULT_AUDIO_SOURCE@ 0.15`); the labeler high-passes the
stream at 80 Hz and calibrates its energy gate on ambient noise at startup (do
that in a quiet moment). If it warns about a saturated microphone, lower the
capture volume. `--mic_device` selects another ALSA capture device.

Sanity-check an installation or a new scene without a headset:

```bash
./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/make_teleop_scene.py \
    --scene_usda <scene.usda> --smoke 120 --headless
```

This creates the environment, holds the rig's ready pose for 120 control steps,
then commands both flanges 3 cm up and verifies the IK follows (it fails loudly
if the action space or IK wiring is broken).

## What the robot is

One PhysX articulation (58 actuated joints): a fixed torso, two 7-DoF Franka
Panda arms, and two 22-DoF SharpaWave dexterous hands on the arm flanges. The
robot USD and the SharpaWave URDF/DexPilot configs are **vendored** under
`assets/` (~46 MB) so this directory is self-contained — they originally lived
in the untracked `sim_benchmark` checkout and were copied verbatim.

## How a hand pose becomes robot motion

```
headset (CloudXR) → OpenXR hand tracking (26 joints per hand)
  │  IsaacTeleop retargeting pipeline (duo_teleop_pipeline.py)
  ├─ wrists  → Se3AbsRetargeter × 2 → absolute flange pose targets, world frame
  ├─ fingers → DexHandRetargeter × 2 → DexPilot QP against the vendored
  │            SharpaWave URDFs → 22 joint angles per hand
  └─ TensorReorderer → 58-D action: [L wrist 7 | R wrist 7 | L fingers 22 | R fingers 22]
  │  make_teleop_scene.py teleop loop
  ├─ to_root_frame(): world-frame wrists → robot-root frame
  └─ env.step()
  │  action terms (duo_robot.py)
  ├─ left_arm / right_arm: damped-least-squares differential IK → 7+7 arm joint targets
  │  (solved once per control step, not per physics substep — OncePerStepDiffIKAction)
  └─ left_hand / right_hand: direct joint-position targets → 22+22 finger targets
  ▼ PhysX: 240 Hz physics, 60 Hz control, 30 Hz render (configurable)
```

Two conventions worth knowing:

- **Quaternions are xyzw everywhere** — Isaac Lab 3.0's data and math stack
  uses the Warp layout, and `Se3AbsRetargeter` happens to emit the same, so
  poses flow through without reordering.
- **The IK action terms expect root-frame commands** and do no frame conversion
  themselves; the teleop loop owns the world→root conversion.

The wrist rotation offsets (left roll −180°/yaw 45°, right roll 180°/yaw 135°)
map the OpenXR wrist frame onto the `panda_link8` flange frame. They were
calibrated on the original branch at the rig's ready pose; the clean ±45°
structure is the rig's flange-mount rotations. If the hands ever track with a
constant twist, these offsets in `duo_teleop_pipeline.py` are the knob.

### Operator hand-shape calibration

Finger retargeting uses the operator's hand-shape calibration from the source
branch (`assets/dex_retargeting/hand_calibration.yml`, measured with flat
hands; loaded by default, `--hand_calibration ''` disables). Per hand it holds
a global rotation + scale (~1.18 — wrist-pinned Procrustes on
index/middle/ring tips) and thumb/pinky length ratios + tip-direction
rotations. Recalibrate per operator with the ported calibration scene:

```bash
./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/calibrate_hand_shape.py \
    --user alice --headless
```

Connect the headset, press Play, hold both hands flat with fingers straight;
5 s later the frame is captured, solved, and written to
`assets/dex_retargeting/hand_calibration_alice.yml`. Teleop then loads it with
`--user alice` (an explicit `--hand_calibration` wins over `--user`).

**Fingertip convention**: Quest/OpenXR `*_TIP` joints sit at the center of the
fingertip capsule — about one tip-radius inside the skin — while the MANO
keypoints the DexPilot configs expect are on the skin surface, so raw Quest
fingers read ~1 cm short and touching fingertips still read ~2 cm apart. New
calibrations therefore extend each tip along its distal-bone direction by the
runtime-reported joint radius (`--no_tip_extension` opts out); the choice is
stamped into the yml and the teleop retargeter mirrors it automatically, so
calibration and runtime always use the same convention. Old calibration files
without the stamp keep the original center-tip behavior.

Applied exactly as on the source branch (`sharpa_retargeting.py`):
the rotation composes wrist-side into the flange offset (the arm tilts the
robot hand until its fingers align with yours), the scale multiplies the
DexPilot keypoints (the yml `scaling_factor` is overridden to 1.0), thumb and
pinky get their per-finger corrections afterwards, and DexPilot's pinch
project/escape hysteresis runs on your RAW fingertip distances so calibration
never shifts pinch timing.

## Files

| File | Role |
|---|---|
| `make_teleop_scene.py` | CLI entrypoint: builds the env from a USDA + CLI args, runs the XR teleop loop (or `--smoke`/`--voice_test`). |
| `replay_teleop_scene.py` | Kinematic replay of recorded demos in the same USDA scene, one camera to MP4. |
| `duo_env.py` | Env config shared by teleop and replay (scene skeleton + managers). |
| `duo_robot.py` | The rig: articulation config (actuators, ready pose) and the 58-D action space, including the once-per-step IK optimization. |
| `duo_teleop_pipeline.py` | The IsaacTeleop retargeting pipeline (hand tracking → 58-D action). |
| `sharpa_retargeting.py` | Calibrated DexPilot finger retargeting (hand-shape calibration, raw-distance pinch hysteresis). |
| `usda_scene.py` | References the scene USDA into the env and registers its rigid bodies so resets restore their poses. |
| `assets/robots/` | Vendored robot USD (torso + arms + hands + skin material). |
| `assets/dex_retargeting/` | Vendored SharpaWave URDFs + DexPilot YAMLs for the finger retargeting. |

## Vendored example scenes

`scenes/` ships ready-to-use scenes (~13 MB, git-lfs), so nothing outside the
repo is needed:

- `scenes/taco/scene/taco_hoi_178_023.usda` — the TACO brush-and-bowl tabletop
  (the default `--scene_usda`), with its object USDs alongside.
- `scenes/scenegen/04_episode_scenegen/runs/scenes/*.usda` — six scenegen
  scenes (ARCTIC box, HOI4D toy car / trash can, OakInk USB hub + stick, two
  more TACO tasks) copied from
  `gs://foundational-research/yjw/example_usda/`, with only the eleven object
  payloads they reference mirrored under
  `scenes/scenegen/02_mesh/06_usd_conversion/runs/usd/`. The GCS directory
  layout is preserved because the scene files reference their payloads by
  relative path — keep it intact when adding more scenes.

## Scene requirements and placement

- The USDA must have a **default prim**; everything it authors (geometry,
  lights, physics, materials) is referenced in unmodified.
- Its rigid bodies are auto-discovered and re-posed on every env reset
  (disable with `--no_track_objects`).
- The robot's pose in the scene is `--robot_pos` / `--robot_rot`; the XR
  anchor (where *you* stand) is `--anchor_pos` / `--anchor_rot`. Defaults are
  the validated TACO-tabletop placement: the torso south of the table facing
  +y, you at the torso. Keep the anchor yaw equal to the robot yaw so the
  robot's arms line up with yours.

## Provenance and scope

This is a port of the `teleop_taco_scene.py` pipeline from the
`feature/robolab-xr-teleop` branch (which targets Isaac Lab 2.x) onto
`release/3.0.0-beta2`, restructured to be scene-agnostic and self-contained.
Deliberately **not** ported (yet): the Success/Failure client dialog and the
client-side Align button (both replaced by voice commands), the
`calibrate_hand_shape.py` capture script (its output yml is vendored and
used), self-collision contact filtering, and the domain-randomized
four-camera replay.

What changed in the port, beyond reorganization:

- 2.x `OpenXRDevice` + custom `RetargeterBase` (deprecated on this branch) →
  the native IsaacTeleop pipeline + `IsaacTeleopDevice`, matching how the
  GR1T2 teleop tasks work here. The SharpaWave DexPilot retargeter with the
  operator calibration is ported as a custom pipeline node
  (`sharpa_retargeting.py`) driving the same vendored URDFs/configs.
- 2.x `sim.physx.*` settings → `sim.physics = PhysxCfg(...)` (multi-backend
  split); asset data reads go through `.torch` views; quaternions wxyz → xyzw.

Known caveats:

- CCD is requested but PhysX disables it under GPU dynamics (warning at
  startup); run with `--device cpu` if fast-motion tunneling matters more
  than simulation speed.
- The wrist offsets were carried over on the argument that the GR1T2 offsets
  are identical between the 2.x and IsaacTeleop stacks; if the very first live
  session shows twisted wrists, recalibrate (see above) before blaming IK.

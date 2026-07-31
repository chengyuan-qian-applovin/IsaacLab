# RoboLab × Isaac Lab XR Teleoperation

Teleoperate [NVIDIA RoboLab](https://github.com/NVLabs/RoboLab) benchmark tasks with
Isaac Lab's XR hand-tracking stack (Apple Vision Pro via CloudXR), and record every
episode for **two** data pipelines at once:

1. **RoboLab-native** — `run_<N>.hdf5` per demo in RoboLab's streaming layout, with
   task success verdicts from RoboLab's predicates. Directly usable by RoboLab's
   `replay`, `analysis`, and `dashboard` tooling.
2. **robomimic / Isaac Lab imitation learning** — `convert_robolab_to_robomimic.py`
   merges the runs into a single robomimic-style dataset consumable by Isaac Lab's
   `replay_demos.py`, Isaac Lab Mimic, and robomimic training configs.

The AVP client app (Isaac XR Teleop Sample Client) is unchanged — Play/Stop/Reset
buttons work exactly as with the stock Isaac Lab teleop tasks.

> Design rationale, upstream findings, validation record, and bring-up bugs are
> documented in [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md).

## How it works

RoboLab's DROID robot exposes an 8-D absolute-IK action `[pos(3), quat wxyz(4), gripper(1)]`
targeting the gripper flange (`base_link`). Two thin retargeter adapters
(`robolab_retargeters.py`) map Isaac Lab's stock hand-tracking retargeters onto it:

| Adapter | Wraps | Change |
|---|---|---|
| `RobolabAbsIKRetargeter` | `Se3AbsRetargeter` (pinch-point 7-D absolute pose) | post-multiplies orientation by `EEF_OFFSET_ROT⁻¹` (eef_frame → base_link, per RoboLab's own `run_abs_ik_demo.py`) |
| `RobolabGripperRetargeter` | `GripperRetargeter` (pinch gesture, hysteresis) | remaps −1/+1 → 1/0 (RoboLab closes at >0.5) |

`teleop_robolab_agent.py` registers the abs-IK variant of the chosen task via
RoboLab's factory, creates the env, builds the OpenXR device with these retargeters,
and runs RoboLab's canonical episode lifecycle (`run_<N>.hdf5` + per-env demo index +
auto-export on termination), so recordings are indistinguishable from policy-eval runs.

## Setup

```bash
# 1) Host: clone RoboLab next to IsaacLab (or set ROBOLAB_PATH)
git clone https://github.com/NVLabs/RoboLab.git ~/RoboLab

# 2) Start containers with the CloudXR + RoboLab patches.
#    NOTE: both files go after ONE --files flag — container.py's --files uses
#    nargs="*", so a repeated --files flag silently overwrites the previous one
#    (dropping the CloudXR runtime).
cd ~/IsaacLab
./docker/container.py start \
    --files docker-compose.cloudxr-runtime.patch.yaml docker-compose.robolab.patch.yaml \
    --env-file .env.cloudxr-runtime

# 3) Once per container: install RoboLab into the Isaac Sim python
./docker/container.py enter base
bash scripts/environments/teleoperation/robolab/install_robolab.sh
```

The install script also applies the torch/CUDA `ldconfig` fix the Isaac Sim 5.1
container image needs (mismatched CUDA-13 nvidia packages vs torch cu128).

## Collecting demos

```bash
# Inside the container, with the AVP connected via CloudXR:
./isaaclab.sh -p scripts/environments/teleoperation/robolab/teleop_robolab_agent.py \
    --task BananaInBowlTask --teleop_device handtracking --num_demos 10
```

- **No desktop display / X11 forwarding disabled? Add `--headless`.** In GUI
  mode the AR session starts only when you click "Start AR" in the desktop
  window; the headless XR experience auto-starts it (`xr.profile.ar.enabled`).
  Without a display and without `--headless`, the headset connects but sees
  nothing (or a frozen frame from a previous session).
- Press **Play** in the AVP client to start teleoperating; **Stop** pauses;
  **Reset** discards the current demo and restarts the episode.
- Pinch (thumb–index) to close the gripper; move your hand to drive the arm.
  `--teleop_device handtracking_left` binds the left hand instead.
- An episode ends when RoboLab's success/failure predicates fire (or time out);
  the demo auto-exports and the next episode starts.
- `--anchor_pos X Y Z` shifts where the sim world appears relative to you
  (default places the tabletop workspace at a comfortable standing position —
  tune to taste).
- `--record_images` additionally records per-step camera observations
  (heavier; states-only recordings can regenerate imagery via RoboLab replay).
- No headset handy? `--teleop_device keyboard` drives the same action space
  (see `Se3Keyboard` bindings; `R` discards the demo).

Output lands in RoboLab's output directory (printed at exit), one
`run_<N>.hdf5` per demo plus `env_cfg.json` metadata.

## Building the robomimic dataset

```bash
# Pure h5py — runs anywhere, no Isaac Sim needed:
python scripts/environments/teleoperation/robolab/convert_robolab_to_robomimic.py \
    --input_dir /workspace/robolab/output \
    --output ./datasets/robolab_teleop.hdf5 \
    --env_name <task env name printed by the teleop script>
```

Successful demos only by default (`--include_failed` to keep everything). The
output follows the `data/demo_N/{initial_state, actions, states, ...}` layout with
robomimic `env_args` attrs — the same shape `record_demos.py` produces.

## SharpaWave duo rig (bimanual, dexterous hands)

`teleop_sharpa_duo_agent.py` teleoperates the FR3 Duo + dual SharpaWave rig
(58 DoF: 2×7 arms + 2×22 finger joints) with **both hands and full finger
retargeting** — the GR1T2 teleop experience on RoboLab benchmark scenes.

> Design record — which IK was chosen and why, frame calibration, integration
> architecture, validation — in [SHARPA_DUO_NOTES.md](SHARPA_DUO_NOTES.md).

Requires the SharpaWave RoboLab fork (adds `robolab.registrations.sharpa_wave`,
the `franka_duo_sharpa_wave` robot, and vendored Sharpa hand assets) installed in
place of stock RoboLab: point `ROBOLAB_PATH` at that checkout (or install it with
`pip install --no-deps -e <path>` inside the container).

```bash
./isaaclab.sh -p scripts/environments/teleoperation/robolab/teleop_sharpa_duo_agent.py \
    --task BananaInBowlTask --num_demos 5 --headless --device cuda:0
```

How it maps (GR1T2-pattern, per component):

- **Arms**: your two wrist poses drive the rig's existing 58-D
  `FrankaDuoSharpaIKActionCfg` — absolute differential IK per arm on
  `panda_link8`, commands in the robot root frame (the script converts from the
  XR world frame each step).
- **Fingers**: all ten fingers dex-retargeted (DexPilot, `dex_retargeting`)
  against the vendored SharpaWave URDFs (`sharpa_dex_retargeting/`), scattered
  by joint name into RoboLab's `HAND_JOINTS_ORDERED`.
- **Frame calibration**: the OpenXR-wrist → flange rotation offsets are baked
  constants derived at the rig's ready pose by `calibrate_sharpa_duo.py`
  (they encode the rig's ∓45° flange mounts). Re-run that script if the rig's
  mounts change; if the hands track with a constant twist, re-derive.
- **Recording**: identical dual pipeline to the single-arm flow (`run_<N>.hdf5`
  per demo + the robomimic converter).
- `--anchor_pos` (default `0 0 -0.7`) raises the scene so the tabletop sits at a
  comfortable standing height; tune to your body.

## Known limitations / notes

- Stock RoboLab ships single-arm DROID (Franka + Robotiq) only; the bimanual
  SharpaWave flow above needs the fork. A GR1T2 humanoid port inside RoboLab
  remains out of scope.
- Teleop is single-env (`num_envs=1`), matching Isaac Lab's teleop scripts.
- RoboLab renders at 15 Hz by default; under XR this script raises it
  (`--render_interval`, default 2 → 60 Hz) for headset comfort. Control rate stays
  at RoboLab's 15 Hz (`decimation=8`), same as its policy evals — so teleop data
  matches the benchmark's control cadence.
- The wrist/scene cameras stay in the scene (RoboLab requires `robot` + sensors);
  if XR streaming performance suffers, try lowering `--rendering_mode performance`.

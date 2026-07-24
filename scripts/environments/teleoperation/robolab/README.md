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

# 2) Start containers with the CloudXR + RoboLab patches
cd ~/IsaacLab
./docker/container.py start \
    --files docker-compose.cloudxr-runtime.patch.yaml \
    --files docker-compose.robolab.patch.yaml \
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

## Known limitations / notes

- Single-arm DROID (Franka + Robotiq) only. A GR1T2 humanoid port requires
  registering the robot inside RoboLab (robot/action/observation/contact configs)
  and is out of scope for this integration.
- Teleop is single-env (`num_envs=1`), matching Isaac Lab's teleop scripts.
- RoboLab renders at 15 Hz by default; under XR this script raises it
  (`--render_interval`, default 2 → 60 Hz) for headset comfort. Control rate stays
  at RoboLab's 15 Hz (`decimation=8`), same as its policy evals — so teleop data
  matches the benchmark's control cadence.
- The wrist/scene cameras stay in the scene (RoboLab requires `robot` + sensors);
  if XR streaming performance suffers, try lowering `--rendering_mode performance`.

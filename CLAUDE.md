# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of [Isaac Lab](https://github.com/isaac-sim/IsaacLab) (upstream `main` is the PR target). The actual work of this fork lives on `feature/robolab-xr-teleop` in two places:

- `scripts/environments/teleoperation/robolab/` — XR (Apple Vision Pro) teleoperation of RoboLab benchmark tasks and the TACO bimanual scene, with episode recording. All fork-specific code and docs are here.
- `sim_benchmark/` — untracked sibling project (TACO scene assets/configs, `franka_duo` robot placement) imported via `sys.path` by the TACO scripts. Expected at `<IsaacLab>/sim_benchmark`.

Everything under `source/` is upstream Isaac Lab (`isaaclab`, `isaaclab_assets`, `isaaclab_tasks`, `isaaclab_rl`, `isaaclab_mimic` extensions). Avoid editing upstream code unless intentionally diverging; prefer wrappers/adapters in the `robolab/` scripts dir (see `OncePerStepDiffIKAction`, retargeter adapters).

## Commands

All Python must run through Isaac Sim's interpreter:

```bash
./isaaclab.sh -p <script.py> [args]      # run a script
./isaaclab.sh -f                          # pre-commit format + lint
./isaaclab.sh -t                          # all pytest tests
./isaaclab.sh -p -m pytest source/isaaclab/test/<file>.py::<test>   # single test
./isaaclab.sh -i [LIB]                    # install extensions (+rl frameworks)
```

### Container lifecycle for teleop (the usual dev loop)

```bash
# Host: start base + CloudXR runtime + RoboLab mounts.
# BOTH files after ONE --files flag — container.py uses nargs="*", a second
# --files silently overwrites the first (dropping the CloudXR runtime).
./docker/container.py start \
    --files docker-compose.cloudxr-runtime.patch.yaml docker-compose.robolab.patch.yaml \
    --env-file .env.cloudxr-runtime
./docker/container.py enter base

# Once per fresh container: installs RoboLab into the kit python and applies
# the torch/CUDA ldconfig fix the Isaac Sim 5.1 image needs.
bash scripts/environments/teleoperation/robolab/install_robolab.sh
```

### Teleop entry points (run inside the container)

```bash
# Single-arm DROID (Franka + gripper) on RoboLab tasks:
./isaaclab.sh -p scripts/environments/teleoperation/robolab/teleop_robolab_agent.py \
    --task BananaInBowlTask --teleop_device handtracking --num_demos 10 --headless

# Bimanual FR3 duo + SharpaWave hands (58 DoF) on RoboLab scenes:
./isaaclab.sh -p scripts/environments/teleoperation/robolab/teleop_sharpa_duo_agent.py \
    --task BananaInBowlTask --num_demos 5 --headless

# TACO scene (sim_benchmark brush+bowl) with the duo rig + HDF5 recording:
./isaaclab.sh -p scripts/environments/teleoperation/robolab/teleop_taco_scene.py --headless

# Replay a TACO recording (kinematic, with domain-randomization variations):
./isaaclab.sh -p scripts/environments/teleoperation/robolab/replay_taco_scene.py

# Convert RoboLab runs to a robomimic dataset (pure h5py, no Isaac Sim):
python scripts/environments/teleoperation/robolab/convert_robolab_to_robomimic.py \
    --input_dir /workspace/robolab/output --output ./datasets/robolab_teleop.hdf5 --env_name <name>
```

Always pass `--headless` when there is no desktop display: in GUI mode the AR session only starts via the "Start AR" button; headless auto-starts it. Leave `--device` unset for XR teleop — CPU physics is faster than GPU for one env (~22 vs ~90 ms/step measured).

## The CloudXR teleop pipeline (big picture)

Three components, three codebases:

```
Apple Vision Pro                  Workstation (this repo)
┌────────────────────┐   WiFi    ┌──────────────────┐     ┌─────────────────────────┐
│ Isaac XR Teleop    │ ────────► │ CloudXR Runtime  │ ──► │ Isaac Lab teleop script │
│ Sample Client      │  hands,   │ (docker sidecar, │ OpenXR  OpenXRDevice        │
│ (~/isaac-xr-teleop-│  head,    │ ports 47998-48012│     │   → retargeters         │
│ sample-client-apple)│ UI events│  udp, 48010 tcp) │ ◄── │   → env.step(action)    │
└────────────────────┘ ◄──────── └──────────────────┘stereo└─────────────────────────┘
                        video                        frames
```

1. **AVP client** (`~/isaac-xr-teleop-sample-client-apple`, Swift/Xcode; check out the tag matching the Isaac Lab version, e.g. `v2.3.0`). Connects to the workstation IP, streams 26 tracked joints per hand + head pose upstream, renders the CloudXR stereo stream, and exposes Play/Stop/Reset buttons that arrive as `START`/`STOP`/`RESET` callbacks. The client is deliberately unmodified — all custom behavior lives server-side.
2. **CloudXR Runtime** — the docker sidecar started by the compose patch file. It is the machine's OpenXR runtime: kit discovers it via `XDG_RUNTIME_DIR`/`XR_RUNTIME_JSON` pointing into the shared `openxr/` mount (the compose patch wires this).
3. **Isaac Lab side** — `isaaclab.devices.OpenXRDevice` polls hand/head poses each `teleop.advance()` and pushes them through a retargeter list to produce the env action. This fork adds:
   - `robolab_retargeters.py` — thin adapters mapping stock `Se3AbsRetargeter`/`GripperRetargeter` onto RoboLab DROID's 8-D abs-IK action (EEF-frame rotation offset, gripper sign remap).
   - `sharpa_duo_retargeters.py` — bimanual: wrist poses → per-arm differential IK targets; ten fingers → DexPilot (`dex_retargeting`) against vendored SharpaWave URDFs (`sharpa_dex_retargeting/`).
   - `xr_session_tools.py` — server→client message bridge (Success/Failure dialog), cross-hand stop gesture, Align re-anchoring, raw-XR capture retargeter.
   - Recording: RoboLab-native `run_<N>.hdf5` per demo (agent scripts) or robomimic-style HDF5 via Isaac Lab's `RecorderManager` (TACO script); `convert_robolab_to_robomimic.py` bridges the two formats.

Reference docs: upstream how-to ([cloudxr_teleoperation](https://isaac-sim.github.io/IsaacLab/main/source/how-to/cloudxr_teleoperation.html)); in-repo deep dives under `scripts/environments/teleoperation/robolab/`: `README.md` (setup + flows), `IMPLEMENTATION_NOTES.md`, `SHARPA_DUO_NOTES.md`, `SCENE_GUIDE.md`, `IK_GUIDE.md`, `RECORD_REPLAY_GUIDE.md`, `DEXVERSE_RETARGETING_GUIDE.md`, `PHYSX_SIMULATION_BASICS.md`, and `teleop_architecture.svg`.

## Gotchas that bite in this codebase

- **Import order is load-bearing.** In teleop scripts: `cv2` and `pinocchio` before any `isaaclab`/`omni` import; everything sim-related only after `AppLauncher(...)` has run. The `robolab/` common modules say "Import only after AppLauncher" — respect it.
- **Never let an exception escape the teleop loop** while an XR session is live: `simulation_app.close()` under a live session deadlocks kit shutdown. The loops catch, print, and pause teleop instead.
- **Recorder HDF5 opens in truncate mode** at env creation — dataset filenames are timestamped for a reason.
- **Self-collision on the SharpaWave hands** cannot be plain `enabled_self_collisions=True`: zero-length `*_MCP_VL` virtual links interpenetrate at rest and jam the fingers. `filter_self_collision_except_fingertips` (in `robolab_teleop_common.py`) filters every pair except tips of different fingers, authored per rigid link with `UsdPhysics.FilteredPairsAPI`.
- **The hand USDs use instanceable `visuals/`/`collisions/` subtrees** — collider meshes are instance proxies: a plain `Usd.PrimRange` never reaches them (use the `Usd.TraverseInstanceProxies` predicate) and no opinion can be authored on them (author on the link prims instead). The panda arm links are the opposite convention: one real mesh that is both render geometry and collider. `--show_collision_meshes` on the TACO script renders the hands' cooked convex hulls in-scene for debugging.
- **Arm IK cost**: stock `DifferentialInverseKinematicsAction` re-solves per physics substep; `OncePerStepDiffIKAction` solves once per control step (decimation× cheaper). TACO env uses it via `class_type` override.
- Office/institutional WiFi often blocks device-to-device traffic — the AVP can't find the workstation; use a dedicated router.

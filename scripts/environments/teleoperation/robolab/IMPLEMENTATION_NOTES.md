# Implementation Notes: RoboLab × Isaac Lab XR Teleoperation

Engineering record for the integration on branch `feature/robolab-xr-teleop`
(July 2026). For day-to-day usage, see [README.md](README.md); this document
explains **how the integration was built, what was discovered, and why the
design is the way it is** — so it can be maintained, extended (e.g. the GR1T2
port), or re-derived if either upstream changes.

## 1. Goal

Use the Apple Vision Pro hand-tracking teleop stack (Isaac Lab OpenXR devices +
CloudXR runtime) to drive [NVIDIA RoboLab](https://github.com/NVLabs/RoboLab)
benchmark tasks, and record every demonstration for **two** data pipelines:

1. **RoboLab-native** episodes (its replay / analysis / dashboard tooling).
2. **robomimic-style** datasets (Isaac Lab imitation learning: `replay_demos.py`,
   Isaac Lab Mimic, robomimic training).

Constraints: keep the AVP client app unmodified; run inside the existing
`isaac-lab-base` container with the CloudXR runtime wiring already in place.

## 2. What investigation established

### RoboLab side (read from source, `~/RoboLab`)

| Fact | Where | Consequence |
|---|---|---|
| Envs are plain Isaac Lab `ManagerBasedRLEnv`s built by a factory (`auto_discover_and_create_cfgs`); `create_env(name_or_cfg)` → `(env, env_cfg)` | `robolab/core/environments/{factory,runtime}.py` | Isaac Lab teleop devices can drive them directly |
| `RobolabEnv` **freezes** terminated envs instead of resetting, records per-env success, and **auto-exports** the episode on termination | `robolab/core/environments/env.py` | Teleop loop gets export-on-success for free; needs `reset_eval_state()` between demos |
| `RobolabRecorderManager` **subclasses Isaac Lab's `RecorderManager`**; its streaming handler writes the standard `data/demo_N` `EpisodeData` layout **plus robomimic-style `env_args` attrs** | `robolab/core/logging/{recorder_manager,streaming_hdf5_handler}.py` | RoboLab's HDF5 is a *superset* of `record_demos.py` output → the robomimic pipeline needs only a converter, **not** a second recorder |
| Default recorder terms: `initial_state`, post-step `states`, pre-step `actions`, `ee_pose`, `bbox` (+ `obs` when `RECORD_IMAGE_DATA`) | `robolab/core/environments/base.py` | Low-dim recording by default; imagery regenerable via RoboLab replay |
| DROID abs-IK action = 8-D `[pos(3), quat wxyz(4), gripper(1)]`, IK tracks `base_link` (gripper flange); commands in the natural EE frame must be converted: `target_base_quat = target_eef_quat ⊗ EEF_OFFSET_ROT⁻¹`, `EEF_OFFSET_ROT = (0.5,-0.5,0.5,-0.5)` | `robolab/robots/droid.py`, `examples/run_abs_ik_demo.py` | The only math the integration adds |
| Gripper term binarizes at `> 0.5` = close (`BinaryJointPositionZeroToOneAction`) | `robolab/robots/droid.py` | Isaac Lab's gripper retargeter (−1 close / +1 open) needs a remap |
| Canonical episode lifecycle used by policy evals: double `env.reset()`, `set_hdf5_file(f"run_{n}.hdf5")`, `set_episode_index`, step until `env.all_terminated`, then `end_episode(env)` | `robolab/eval/episode.py` | Teleop copies this exactly so recordings are indistinguishable from eval runs |
| Official install is a uv venv with **pip-installed Isaac Sim/Isaac Lab pins** | `pyproject.toml`, README | Inside the container we must install `--no-deps` or pip drags in a second, conflicting simulator stack |
| `cv2` must be imported before isaaclab/omni modules | `examples/*.py` | Entry scripts start with `import cv2` |

### Isaac Lab side (this repo)

- Teleop devices are **pure config**: `DevicesCfg` → `create_teleop_device()`
  instantiates `class_type` + retargeter `retargeter_type`s. No registry to touch.
- `Se3AbsRetargeter` (pinch-midpoint absolute 7-D pose, wxyz) +
  `GripperRetargeter` (pinch distance with hysteresis) are the proven pair for
  single-arm AVP teleop (used by `Isaac-Stack-Cube-Franka-IK-Abs-v0`).
  Retargeter outputs concatenate in cfg-list order → exactly the 8-D action.
- `"handtracking"` in `--teleop_device` + `AppLauncher(xr=True)` selects the
  OpenXR kit experience; the AVP client's Play/Stop/Reset arrive as
  `teleop_command` messages → `START`/`STOP`/`RESET` callbacks.
- `env.cfg.sim.render_interval` and `env.cfg.episode_length_s` are read **live**
  each step (`manager_based_env.py:488`, `manager_based_rl_env.py:96-103`), so
  post-creation overrides work.

## 3. Design decisions

1. **DROID/Franka first, GR1T2 deferred.** RoboLab ships no humanoid; a GR1T2
   port needs robot/action/observation/contact-gripper registration inside
   RoboLab's factory plus per-task workspace fit — a separate project. The
   single-arm path reuses proven retargeters 1:1. (The GR1T2 teleop machinery
   itself is config-driven and portable when that day comes.)
2. **One recorder, two pipelines.** Because RoboLab's recorder already writes a
   superset of Isaac Lab's `record_demos.py` layout, the robomimic pipeline is a
   **pure-h5py converter** (`convert_robolab_to_robomimic.py`) that merges
   `run_N.hdf5` files, filters by the per-demo `success` attr, renumbers demos,
   and stamps `env_args`. No second `RecorderManager`, no dual-write overhead,
   one source of truth.
3. **Adapters, not forks.** Two small subclasses in `robolab_retargeters.py`
   (`RobolabAbsIKRetargeter`, `RobolabGripperRetargeter`) wrap the stock
   retargeters with the frame offset / gripper remap. Neither upstream repo is
   modified; the AVP app is untouched.
4. **Copy RoboLab's eval lifecycle verbatim** (per-demo `run_N.hdf5`, episode
   index arming, freeze→auto-export, `end_episode`) so teleop demos are
   drop-in-compatible with RoboLab replay/dashboard, including the discard path
   (`RESET` → `recorder.clear()` → same run file gets overwritten).
5. **Container install via `--no-deps`** (`install_robolab.sh`): RoboLab
   editable-installed against the container's binary Isaac Sim + source Isaac
   Lab; only `opencv-python-headless` added. The script also applies the
   torch/CUDA `ldconfig` fix this container image needs (see §5).
6. **Keyboard fallback** (`KeyboardAbsIKAdapter`) integrates `Se3Keyboard`
   deltas into an absolute pose target initialized from the robot's `eef_frame`
   — enables headset-free smoke testing of the identical action path.

## 4. Files

```
scripts/environments/teleoperation/robolab/
├── teleop_robolab_agent.py           # main teleop + recording script
├── robolab_retargeters.py            # the two adapter retargeters
├── convert_robolab_to_robomimic.py   # pure-h5py dataset converter
├── install_robolab.sh                # container install (+ ldconfig fix)
├── README.md                         # usage guide
└── IMPLEMENTATION_NOTES.md           # this file
docker/docker-compose.robolab.patch.yaml  # mounts ~/RoboLab → /workspace/robolab
```

## 5. Environment prerequisites (hard-won, machine-level)

These predate the integration but are required for it to run (July 2026,
workstation `axon-1100`, 2× RTX 6000 Ada):

- **NVIDIA driver must be on the R580 branch.** R590/595.x segfaults Isaac Sim
  5.1's RTX renderer (`librtx.scenedb.plugin.so`) right after "app ready" —
  known upstream issue (IsaacSim discussions #648/#651). With Secure Boot,
  driver reinstalls need MOK re-enrollment.
- **Torch CUDA-library fix in the container image.** Isaac Sim 5.1's image
  ships torch 2.7.0+cu128 but mismatched CUDA-13 nvidia packages in
  site-packages → `libcusparseLt.so.0` / `libcufile.so.0` ImportErrors. Fix:
  register the version-matched prebundled libs with the loader (done by
  `install_robolab.sh`):
  `ls -d /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/*/lib \
        /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/cusparselt/lib \
        > /etc/ld.so.conf.d/isaac-torch-cuda.conf && ldconfig`
  Container-local: reapply (or rerun the install script) after every container
  recreation.

## 6. Validation performed

All headless inside the running `isaac-lab-base` container (via `smoke_test.py`,
removed after the live-headset validation superseded it — see git history):

1. **Retargeter contract** (`smoke_test.py`): synthetic hand data → 8-D action;
   pinch midpoint position, unit-norm quaternion, frame-offset math checked
   against `EEF_OFFSET_ROT⁻¹`; gripper 10 cm apart → 0.0 (open), 1 cm → 1.0 (close).
2. **Env + recording** (`smoke_test.py`): `BananaInBowlTask` abs-IK env created,
   initial pose held 30 steps, `run_0.hdf5` verified to contain
   `data/demo_0/{actions, states/…, initial_state/…, ee_pose, subtask}` with
   `num_samples`/`success` attrs and robomimic `env_args`.
3. **Full lifecycle** (`teleop_robolab_agent.py --teleop_device keyboard
   --num_demos 1 --episode_length_s 10`): reached the teleop loop, stepped at
   15 Hz, episode terminated at exactly step 150, RoboLab auto-exported,
   summary printed, clean exit:
   `Demo 0: failure/timeout (step 150) … Collected 1 demos … data written under /workspace/robolab/output`.
4. **Converter**: merged the recorded run into a robomimic-style file; demo
   renumbering, `total` samples, and `env_args` verified by re-opening it.
5. **Not yet validated**: the live XR loop with a headset (everything up to it is).

## 7. Bugs hit and fixed during bring-up

| Symptom | Cause | Fix |
|---|---|---|
| `'Se3Keyboard' object has no attribute '_input'` at construction | Current Isaac Lab devices take a `Cfg` object, not kwargs | `Se3Keyboard(Se3KeyboardCfg(...))` |
| Keyboard adapter unpack error | `advance()` now returns one 7-el tensor `[dx dy dz rx ry rz grip]`, not `(pose, bool)` | slice tensor; gripper = `cmd[6] < 0` |
| Final demo-summary prints missing from logs despite correct HDF5 output | stdout is block-buffered when redirected; kit hard-exits before Python flushes | module-wide `print = functools.partial(print, flush=True)` |
| Test episodes never ended in wall-clock budget | RoboLab task time limits are long; 15 Hz control at <1× real time | added `--episode_length_s` live override (also useful for real collection) |

## 8. Future work

- **GR1T2 humanoid in RoboLab**: register `GR1T2_HIGH_PD_CFG` +
  `PinkInverseKinematicsActionCfg` (36-D) + proprio obs + dex-hand contact
  regexes through `auto_discover_and_create_cfgs`; reuse the existing
  `GR1T2Retargeter` teleop device config unchanged. Expect per-task workspace
  and success-predicate fit-up.
- **In-headset UI**: Isaac Lab's `XRVisualization` / instruction widgets (used by
  `record_demos.py`) could show per-demo success counts in the AVP.
- **`--record_images`** path is wired (`RECORD_IMAGE_DATA`) but unprofiled under
  XR streaming; if needed, profile wrist-cam recording vs. framerate.

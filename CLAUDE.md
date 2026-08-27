# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

## Overview

Isaac Lab is a GPU-accelerated robot-learning framework (RL, imitation learning, teleoperation) built on NVIDIA Isaac Sim. This branch (3.0.0-beta2) targets Isaac Sim 6.0.0/6.0.1. Newton-backend ("kit-less") workflows run without Isaac Sim entirely.

## Commands

Everything goes through `./isaaclab.sh` (Windows: `isaaclab.bat`), a thin launcher for `isaaclab.cli` that picks the Python interpreter (active venv/conda → `./env_isaaclab` → bundled `_isaac_sim`). Testing and pre-commit workflows are covered in AGENTS.md above.

```bash
# Install (core is always installed; tokens select optional submodules/extras)
./isaaclab.sh -i                      # all: core + mimic/teleop + newton, rl, visualizer extras
./isaaclab.sh -i core                 # core only
./isaaclab.sh -i newton,'rl[rsl-rl]'  # selectors: rl[rsl-rl|skrl|sb3|rl-games],
                                      #   visualizer[kit|newton|rerun|viser], ov[ovrtx|ovphysx], contrib[rlinf]

# Train / play (unified entrypoints; dispatch to scripts/reinforcement_learning/<lib>/)
./isaaclab.sh train --rl_library rsl_rl --task Isaac-Cartpole-v0 --num_envs 64
./isaaclab.sh play --rl_library rsl_rl --task Isaac-Cartpole-v0

# Other
./isaaclab.sh -n     # scaffold a new external project or internal task from template
./isaaclab.sh -s     # launch the Isaac Sim GUI
./isaaclab.sh -o     # docker helper (docker/container.sh)
./isaaclab.sh -v     # generate VSCode settings
```

- RL libraries: `rl_games`, `rsl_rl`, `sb3`, `skrl`, `rlinf`. The old per-framework `scripts/reinforcement_learning/<lib>/train.py` scripts are deprecated shims.
- Runs are **headless by default** (`--headless` is deprecated); opt into a display with `--visualizer kit|newton|rerun|viser`.
- Hydra-style overrides go after the flags: `physics=newton_mjwarp`, `presets=inference,newton_mjwarp`, `env.decimation=10`. Presets REPLACE config sections rather than merge (see `isaaclab_tasks/utils/hydra.py` and `utils/presets.py`).

## Architecture

### Multi-backend physics (new in 3.0)

Core `isaaclab` classes (`Articulation`, `RigidObject`, sensors, renderers) are backend-dispatching factories — user code never imports backend packages directly. The backend is selected at sim construction via `SimulationCfg(physics=PhysxCfg() | NewtonCfg(...) | OvPhysxCfg())` or the `physics=...` CLI preset.

- `isaaclab_physx` — PhysX via Isaac Sim/Kit: stable, the parity reference; requires Isaac Sim.
- `isaaclab_newton` — Newton/MuJoCo-Warp GPU backend: beta, differentiable, runs kit-less (no Isaac Sim).
- `isaaclab_ovphysx` — kit-less PhysX (highly experimental); `isaaclab_ov` — Omniverse RTX renderers, decoupled from physics.

Supporting layers in core: `isaaclab/physics` (`PhysicsManager` lifecycle: MODEL_INIT → PHYSICS_READY → STOP), `isaaclab/sim/service_locator.py` (typed singleton registry), `isaaclab/scene_data` (Warp-struct bridge from physics backends to renderers/visualizers).

### Packages under `source/`

Each package is both a pip package and a Kit extension (`config/extension.toml`, `changelog.d/`, `test/`).

- `isaaclab` — core framework: `app` (AppLauncher), `assets`, `envs`, `managers`, `scene` (`InteractiveScene`), `sensors`, `sim` (`SimulationContext`, spawners, URDF/MJCF converters), `actuators`, `controllers`, `devices` (teleop input), `terrains`, `utils` (`configclass`, math), `cli`.
- `isaaclab_assets` — robot/sensor configuration instances.
- `isaaclab_tasks` — environment suite; **importing it registers all Gym task IDs** (auto-walks subpackages).
- `isaaclab_rl` — wrappers bridging envs to rl_games/rsl_rl/sb3/skrl.
- `isaaclab_mimic` — imitation-learning data generation (Apache-2.0 licensed, requires cuRobo).
- `isaaclab_teleop` — IsaacTeleop/XR device handling; `isaaclab_visualizers` — kit/newton/rerun/viser backends, lazily loaded; `isaaclab_ppisp` — physically plausible ISP post-processing; `isaaclab_contrib` / `isaaclab_experimental` / `isaaclab_tasks_experimental` — community and early-access staging.

### Environments: two workflows

- **Manager-based**: the env class is generic (`isaaclab.envs:ManagerBasedRLEnv`); all behavior comes from a config assembled out of manager terms (observations, rewards, terminations, events, actions, commands, curriculum). Task dir holds `*_env_cfg.py` + a local `mdp/`.
- **Direct**: the task subclasses `DirectRLEnv`/`DirectMARLEnv` and implements step/reward logic itself. Task dir holds `*_env.py` + `*_env_cfg.py`.

Task registration: each task package's `__init__.py` calls `gym.register(id, entry_point, kwargs)` mapping the task ID to an `env_cfg_entry_point` plus one `<framework>_cfg_entry_point` per RL library (`agents/` holds YAML for rl_games/skrl/sb3, `@configclass` Python for rsl_rl). Robot families specialize a shared base cfg under `config/<robot>/`. Cfg resolution lives in `isaaclab_tasks/utils/parse_cfg.py`.

Everything is configured through `@configclass` (`isaaclab/utils/configclass.py`) — dataclass-like, composable, CLI-overridable.

### `tools/`

`tools/changelog` compiles the `changelog.d/` fragments in nightly CI (never hand-edit `CHANGELOG.rst`); `tools/template` backs `./isaaclab.sh -n`; `tools/run_all_tests.py` and `tools/run_train_envs.py` drive CI-scale test/train sweeps.

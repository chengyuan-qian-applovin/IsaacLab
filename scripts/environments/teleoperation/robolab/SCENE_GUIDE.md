# Building a Scene and Teleoperating It with the AVP — Step by Step

How to create a simulation scene and drive it with the Apple Vision Pro
dual-hand teleop stack. Uses `minimal_sharpa_duo_teleop.py` as the worked
example (~210 lines, one file — keep it open beside this guide). Read
[PHYSX_SIMULATION_BASICS.md](PHYSX_SIMULATION_BASICS.md) first if terms like
"articulation" or "solver iterations" are new.

The recipe has five parts, and they always come in this order:

```
1. scene entities   →  2. robot + actions   →  3. env config   →
4. teleop device (retargeters + XR anchor)  →  5. the loop
```

---

## Part 1 — Scene entities: what's in the world

A scene is a `configclass` inheriting `InteractiveSceneCfg`. **Every attribute
you declare becomes one thing in the world.** Three kinds cover almost
everything:

| Kind | Config type | Use for | Physics |
|---|---|---|---|
| Static prop | `AssetBaseCfg` | tables, shelves, walls | collides, never moves |
| Dynamic object | `RigidObjectCfg` | banana, bowl, anything grabbable | full rigid-body physics, resettable |
| Light | `AssetBaseCfg` + a light spawn cfg | dome/sphere/distant lights | none |

From the minimal scene:

```python
@configclass
class MinimalDuoSceneCfg(InteractiveSceneCfg):
    table = AssetBaseCfg(                              # static prop
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.UsdFileCfg(usd_path=".../fixtures/franka_table.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.087, 0, 0), rot=(0, 0, 0, 1)),
    )
    banana = RigidObjectCfg(                           # dynamic object
        prim_path="{ENV_REGEX_NS}/banana",
        spawn=sim_utils.UsdFileCfg(usd_path=".../objects/ycb/banana.usd"),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.10)),
    )
    light = AssetBaseCfg(prim_path="/World/light",     # light
        spawn=sim_utils.DomeLightCfg(intensity=2500.0))
    robot = FrankaDuoSharpaWaveCfg().robot             # see Part 2
```

Details that matter:

- **`prim_path`** is the object's address in the USD scene tree. Use the
  `{ENV_REGEX_NS}` prefix for anything per-environment; plain `/World/...` for
  globals like lights.
- **`usd_path`** points at any USD file on disk. We reuse the RoboLab fork's
  local assets (no downloads); IsaacLab's Nucleus assets work too.
- **`init_state`** is the reset pose. Positions are meters in the env frame;
  rotations are quaternions in **(w, x, y, z)** order — Isaac Lab's universal
  convention. Drop dynamic objects a few cm above their resting surface and
  let physics settle them.
- **Physics comes with the asset.** A USD object file carries its own
  colliders and mass; you don't add physics here, you *place* things. If an
  object falls through the table, the asset lacks colliders — fix the asset,
  not the scene.
- **Placement is trial-and-error against real geometry.** A USD fixture's
  origin is wherever its author put it, not necessarily the visual center —
  our table needed `(-0.087, 0, 0)` + 180° yaw to put its top at z≈0, and the
  banana's reachable spot was found by looking in the headset. Budget one
  look-and-nudge cycle for every new asset.

## Part 2 — Robot + actions: who acts, and what a command means

Two declarations: the robot's **articulation config** (its USD, initial joint
pose, actuator gains) and its **action config** (how a command vector maps to
joints). For the duo rig both already exist in the RoboLab fork — reuse them:

```python
robot = FrankaDuoSharpaWaveCfg().robot          # 58-DoF articulation
actions = FrankaDuoSharpaIKActionCfg()          # 58-D action:
# [L wrist pose 7 | R wrist pose 7 | L fingers 22 | R fingers 22]
#   ↑ absolute IK per arm (poses in ROBOT ROOT frame)   ↑ joint targets
```

The action config is the **contract** everything else is built against: the
teleop retargeter must emit exactly this vector, in exactly this order, in the
frames each term expects. If you change the action space, the retargeter
changes with it — nothing else does.

## Part 3 — Env config: assembling the pieces

`ManagerBasedRLEnvCfg` bundles scene + actions + three small "manager"
configs, plus timing:

```python
@configclass
class MinimalDuoEnvCfg(ManagerBasedRLEnvCfg):
    scene = MinimalDuoSceneCfg(num_envs=1, env_spacing=3.0)
    actions = FrankaDuoSharpaIKActionCfg()
    observations = ObservationsCfg()     # proprioception group (reused from the fork)
    events = EventsCfg()                 # one term: reset_scene_to_default
    terminations = TerminationsCfg()     # one term: time_out
    rewards = RewardsCfg()               # empty — teleop needs no reward

    def __post_init__(self):
        self.episode_length_s = 120.0
        self.sim.dt = 1 / 120            # physics at 120 Hz
        self.decimation = 8              # one command per 8 steps → 15 Hz control
        self.sim.render_interval = 2     # render every 2 steps → 60 Hz for the headset
        self.sim.physx.max_position_iteration_count = 16   # solver budget (see PHYSX doc)
```

Then `env = ManagerBasedRLEnv(cfg=MinimalDuoEnvCfg())` — that's the whole
environment. (The RoboLab benchmark envs are this same class plus their
factory, recorders, and task predicates on top.)

## Part 4 — The teleop device: hands → action vector

Three pieces, all declarative:

```python
device_cfg = OpenXRDeviceCfg(
    xr_cfg=XrCfg(anchor_pos=(-0.35, 0, -0.7), anchor_rot=(1, 0, 0, 0)),
    retargeters=[FrankaDuoSharpaRetargeterCfg(
        left_hand_joint_names=LEFT_HAND_JOINTS_ORDERED,
        right_hand_joint_names=RIGHT_HAND_JOINTS_ORDERED,
        sim_device=env.device,
    )],
)
teleop = create_teleop_device("handtracking", {"handtracking": device_cfg},
                              callbacks={"START": ..., "STOP": ..., "RESET": ...})
```

- **`OpenXRDeviceCfg`** — receives the headset's tracking (26 joints per hand)
  via CloudXR. Nothing to configure beyond the anchor.
- **The retargeter** turns tracked hands into the Part-2 action vector:
  wrist poses (with calibrated wrist→flange rotation offsets) + DexPilot
  finger retargeting. It's per-robot; for a new action space you write a new
  one (see `sharpa_duo_retargeters.py` as the template, and
  `SHARPA_DUO_NOTES.md` §4 for how the offsets are derived).
- **The XR anchor** decides where the sim world sits relative to *your body*:
  `anchor_pos` is the sim-frame point your floor-level tracking origin maps
  to. For embodied bimanual control put yourself at the robot torso —
  `(-0.35, 0, -0.7)`: x/y = torso position, z = −(desired tabletop height
  above your floor). Get this wrong and you'll stand inside the table seeing
  nothing (near-plane clipping) — the #1 first-run surprise.
- **Callbacks** wire the AVP client's Play/Stop/Reset buttons to your loop.

## Part 5 — The loop

```python
while simulation_app.is_running():
    if not teleop_active:          # before Play / after Stop
        env.sim.render()           # keep frames flowing to the headset
        continue
    action = teleop.advance()                       # 58-D, wrists in WORLD frame
    action = to_root_frame(action.to(env.device))   # world → robot root (IK expects root frame)
    env.step(action.unsqueeze(0))
```

`to_root_frame` is the one frame conversion the loop owns (the duo's IK action
takes root-frame poses and does no conversion itself). Everything else is
plumbing you can copy verbatim.

**Boilerplate that must be at the very top of the file, in this order** (each
line prevents a real, silent failure we hit):

```python
import cv2         # before any isaaclab/omni import (ABI clash otherwise)
import pinocchio   # before AppLauncher (dex-retargeting builds hands in-kit;
                   # without this, kit dies silently mid-URDF-parse)
print = functools.partial(print, flush=True)  # kit hard-exits before buffers flush
...AppLauncher...  # only AFTER it may you import isaaclab/robolab modules
args_cli.xr = True # hand tracking needs the OpenXR kit experience
```

## Running it

```bash
./docker/container.py enter base
./isaaclab.sh -p scripts/environments/teleoperation/robolab/minimal_sharpa_duo_teleop.py --headless
# wait for "Starting teleop loop", connect the AVP client, press Play
```

`--headless` is required when X11 forwarding is off (the AR session
auto-starts only in the headless XR experience). Scripts are bind-mounted —
edits apply on next launch, no container restart.

## Checklist for a NEW scene

1. Copy `minimal_sharpa_duo_teleop.py`.
2. Edit the `InteractiveSceneCfg`: swap/add `AssetBaseCfg` props and
   `RigidObjectCfg` objects; set `init_state` poses.
3. Keep robot/actions/retargeter untouched (they're scene-independent).
4. Smoke test headless: reaches `Starting teleop loop` with no traceback.
5. Look at it in the headset; nudge object positions and `--anchor_pos`.
6. Optional: tune `max_position_iteration_count` to the scene's contact
   complexity (8–16 for a few objects; RoboLab uses 32 for cluttered scenes).

## Loading an existing scene USD instead of building one

Worked example: `teleop_taco_scene.py` drives `sim_benchmark`'s TACO scene
(table + brush + bowl, physics-matched to a MuJoCo reference) with the duo rig.
Two patterns beyond Part 1, both visible in that file:

**1. One `AssetBaseCfg` can spawn a whole scene file** — sim_benchmark's
`TacoSceneCfg` loads the entire `scene/taco_hoi_178_023.usda` (lights, table,
both objects) as a single entity:

```python
scene = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/scene",
                     spawn=sim_utils.UsdFileCfg(usd_path=SCENE_USD))
```

Physics travels with the USD: the brush/bowl carry rigid-body + collider
schemas inside their asset files, so they're dynamic even though you never
declared a `RigidObjectCfg` to spawn them.

**2. Wrap existing prims to track/reset them individually** — a
`RigidObjectCfg` with `spawn=None` doesn't create anything; it *adopts* a prim
that the scene USD already created, giving you per-object state and reset:

```python
brush = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/scene/taco_178", spawn=None,
                       init_state=RigidObjectCfg.InitialStateCfg(pos=..., rot=...))
```

(`init_state` must be duplicated from the USD because `reset_scene_to_default`
restores from the config, not the file.)

**What still comes from you:** the robot (external scenes are robot-agnostic),
the action space (sim_benchmark only ships joint-position actions — the teleop
IK action config comes from the RoboLab fork; they compose because both repos
build the same articulation with the same joint names), and the XR anchor.

**Anchor for a differently-placed robot** — the TACO robot stands at
`(0, −0.7, 1.0)` *facing +y* (90° yaw), with the tabletop at z = 0.54. Two
consequences, both generalizable:

- `anchor_pos` = (torso x, torso y, tabletop_z − desired real tabletop height)
  → `(0, −0.7, 0.5421 − 0.75) ≈ (0, −0.7, −0.21)`.
- **`anchor_rot` must equal the robot root's yaw** — `(0.7071, 0, 0, 0.7071)`
  here — so that "your forward" is "the robot's forward". This also keeps the
  calibrated wrist rotation offsets valid: the anchor yaw and the root yaw
  cancel in the world→root conversion, which is exactly the frame the offsets
  were derived in. Robot yawed θ in the scene ⇒ anchor yawed θ. Miss this and
  the arms respond to your hands rotated 90° sideways.

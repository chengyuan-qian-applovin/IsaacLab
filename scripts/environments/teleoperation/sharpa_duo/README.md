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
**Stop** pauses, **Reset** restores the scene. There is no data recording in
this version.

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

## Files

| File | Role |
|---|---|
| `make_teleop_scene.py` | CLI entrypoint: builds the env from a USDA + CLI args, runs the XR teleop loop (or `--smoke`). |
| `duo_robot.py` | The rig: articulation config (actuators, ready pose) and the 58-D action space, including the once-per-step IK optimization. |
| `duo_teleop_pipeline.py` | The IsaacTeleop retargeting pipeline (hand tracking → 58-D action). |
| `usda_scene.py` | References the scene USDA into the env and registers its rigid bodies so resets restore their poses. |
| `assets/robots/` | Vendored robot USD (torso + arms + hands + skin material). |
| `assets/dex_retargeting/` | Vendored SharpaWave URDFs + DexPilot YAMLs for the finger retargeting. |

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
Deliberately **not** ported (yet): episode recording and the record-flow XR UX
(stop gesture, Align button, Success/Failure dialog), per-operator hand-shape
calibration, self-collision contact filtering, and the transparent-arm
rendering option.

What changed in the port, beyond reorganization:

- 2.x `OpenXRDevice` + custom `RetargeterBase` (deprecated on this branch) →
  the native IsaacTeleop pipeline + `IsaacTeleopDevice`, matching how the
  GR1T2 teleop tasks work here. The custom SharpaWave DexPilot retargeter was
  replaced by isaacteleop's generic `DexHandRetargeter` driving the same
  vendored URDFs/configs.
- 2.x `sim.physx.*` settings → `sim.physics = PhysxCfg(...)` (multi-backend
  split); asset data reads go through `.torch` views; quaternions wxyz → xyzw.

Known caveats:

- CCD is requested but PhysX disables it under GPU dynamics (warning at
  startup); run with `--device cpu` if fast-motion tunneling matters more
  than simulation speed.
- The wrist offsets were carried over on the argument that the GR1T2 offsets
  are identical between the 2.x and IsaacTeleop stacks; if the very first live
  session shows twisted wrists, recalibrate (see above) before blaming IK.

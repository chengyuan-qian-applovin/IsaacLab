# TACO Teleop Record → Replay Pipeline

How each requirement of the record/replay pipeline is implemented, across three
codebases:

- **This repo** — `scripts/environments/teleoperation/robolab/` (server-side Python)
- **AVP client** — `~/isaac-xr-teleop-sample-client-apple` (Swift, needs one Xcode rebuild)
- **RoboLab** — `~/RoboLab/assets/` (domain-randomization assets, referenced read-only)

File map after implementation:

| File | Role |
|---|---|
| `taco_scene_common.py` | **new** — scene/env cfgs shared by teleop and replay (point 6's "inherit") |
| `teleop_taco_scene.py` | teleop entry point: recording, gesture, transparency, self-collision, align |
| `xr_session_tools.py` | **new** — raw-XR capture retargeter, stop gesture, client messaging, align math |
| `replay_taco_scene.py` | **new** — kinematic replay + cameras + domain randomization |
| `taco_variations.py` | **new** — background/lighting/table-material grid (assets from `~/RoboLab`) |
| `RECORD_REPLAY_GUIDE.md` | this doc |
| AVP client Swift files | Align button, receive stream, success/failure sheet |

---

## Point 1 — Record arms, hands, and object poses

**What is recorded.** Isaac Lab's `RecorderManager` is attached to the teleop env
with three stock terms (`isaaclab.envs.mdp.recorders`):

- `initial_state` (`InitialStateRecorder`, on reset)
- `states` (`PostStepStatesRecorder`, every control step) — this is
  `env.scene.get_state(is_relative=True)`, which contains **everything needed for
  kinematic replay**: the robot articulation's `joint_position` (all 58 DoF = both
  arms + both SharpaWave hands), `root_pose`, and the `root_pose` of the tracked
  rigid objects `brush` (taco_178) and `bowl` (taco_023).
- `actions` (`PreStepActionsRecorder`) — the 58-D root-frame teleop action, kept for
  provenance/debugging (replay does not need it).

**Rate.** Recorder hooks fire inside `env.step()`, i.e. once per control step =
1/60 s of sim time (dt 1/480 × decimation 8). While teleop is paused the loop never
calls `env.step()`, so pauses are seamlessly absent from the data.

**Format.** One HDF5 file (`--record_dir`, default `./datasets/taco_teleop/`),
robomimic-compatible layout `data/demo_N/{initial_state, states, actions}` with a
`success` attribute per demo (set by the point-4 dialog) and `env_args` metadata
(sim dt, decimation). Export mode is `EXPORT_ALL`: failed demos are kept too, with
`success=False`.

**Episode lifecycle.**

| Event | Effect |
|---|---|
| AVP **Play** | teleop active; recording accumulates |
| AVP **Stop** | pause — buffer stays open, Play resumes the same episode |
| **Stop gesture** (point 4) | episode ends → success/failure dialog → export |
| AVP **Reset** | current buffer **discarded**, scene reset |

## Point 2 — Arm transparency during teleop

`--arm_visual {transparent,hidden,normal}`, default `transparent`.

- **transparent**: a `PreviewSurfaceCfg(opacity=0.5)` material is spawned once at
  `/World/Looks/ArmGhostMaterial` and bound (strongly) to the `left_arm` / `right_arm`
  subtrees of the robot. `sim.render.enable_translucency = True` is set in the env
  cfg (required for opacity to actually render under RaytracedLighting).
  Hands and torso stay opaque — you need to see the fingers.
- **hidden**: `UsdGeom.Imageable.MakeInvisible()` on the two arm subtrees — render
  visibility only; physics, collisions and articulation are untouched (the arms are
  still simulated, just not drawn).
- If transparency looks wrong on your renderer settings, fall back to `hidden`.

## Point 3 — SharpaWave self-collision toggle

`--self_collision` (default off, matching current behavior). It overrides
`enabled_self_collisions` in the robot spawn's `ArticulationRootPropertiesCfg`
(currently hardcoded `False` via sim_benchmark's `robot_spawn_props()`).

Caveat: the duo is **one articulation**, so this single flag governs *all*
intra-robot contacts at once — finger↔finger, arm↔hand, arm↔torso, and left↔right
hand. Expect some PhysX cost and possible new jamming behavior when enabled.

## Point 4 — Stop-recording hand gesture + AVP success/failure dialog

**Gesture options investigated** (short version — full analysis in the conversation):
neither visionOS nor Isaac Lab has a prebuilt gesture recognizer. All existing
"gestures" are fingertip-distance computations (thumb–index pinch in
`GripperRetargeter`; DexPilot's per-hand pinch hysteresis in our retargeter), and
all of the client-side system gestures are pinch-driven — which teleop already
consumes. The custom cross-hand pose is therefore the right mechanism.

**Gesture spec** (server-side, `CrossHandStopGesture` in `xr_session_tools.py`):

- Both hands tracked (sanity check: joint spread > 8 cm, wrist not at origin).
- All **5 same-finger tip pairs** (thumb…little) within **2 cm** simultaneously.
- Must **hold 0.5 s** continuously to trigger (rejects transient crossings during
  bimanual work near the bowl).
- Hysteresis: after triggering, re-arms only once any pair separates beyond 10 cm.

Raw joints come from a zero-cost `RawXrCapture` passthrough retargeter appended to
the device's retargeter list. It declares `HAND_TRACKING` + `HEAD_TRACKING`
(head is needed by point 5), stores the raw data dict, and returns a **0-length
tensor** so the 58-D action is unchanged by concatenation.

**Dialog flow** (uses the bidirectional CloudXR message channel — verified: the
Isaac Sim OpenXR plugin ships an outgoing relay listening for
`omni.kit.cloudxr.send_message` carb events and forwards them to the client's
`MessageChannel.receivedMessageStream`; the sample client simply never wired the
receive side):

1. Gesture triggers → teleop deactivates, episode buffer finalized (pre-reset
   records run, not yet exported).
2. Server pushes `{"type": "recording_result_request", "episode": N}` onto the XR
   message bus as an `omni.kit.cloudxr.send_message` event.
3. Client (new Swift code): a receive task on the teleop channel parses the message
   and presents a **Success / Failure sheet** on the control window.
4. The choice returns over the existing channel as teleop command
   `"record_success"` / `"record_failure"` (strings deliberately avoid the
   `start`/`stop`/`reset` substrings that `OpenXRDevice._on_teleop_command`
   dispatches on).
5. Server (own subscription to the `teleop_command` event type — the stock device
   handler silently drops unknown commands) sets the episode's success flag and
   exports it. Press Play to start the next episode.

Voice Control (visionOS Accessibility) can press the sheet's buttons by saying
"Success"/"Failure" — labels are single words on purpose.

## Point 5 — Align button

**Client**: a fourth button ("Align", `gearshape`-style icon) in
`TeleopControlView`, sending `"align scene"` (no forbidden substrings).

**Server** (`xr_session_tools.py::AnchorAligner`): on receipt, using the current
head pose `H_w` from the capture retargeter and the tracked anchor transform `A`
(the `/World/XRAnchor` prim):

1. Compute the head's world yaw from its forward axis (OpenXR head looks along −Z).
2. Build a world-frame correction `ΔT` = rotate about the head's position by
   `(π/2 − yaw_head)` (so you face **+y, straight at the table**), then translate
   the head's xy to `--align_head_xy` (default `(0, −0.9)`, just behind the torso
   at y=−0.7). **z is never touched** — your physical floor height stays calibrated.
3. New anchor `A′ = ΔT ∘ A`, written at runtime via
   `XRCore.set_world_transform_matrix("/World/XRAnchor", …)` (the same call
   `XrAnchorSynchronizer` uses every frame in the prim-follow mode, so mid-session
   writes are proven).

Because the correction is a rigid world-frame transform applied to the anchor, the
hand poses the retargeter sees rotate *with* your body — the calibrated wrist
offsets are preserved (they encode wrist→flange in the wrist's own frame). This is
flagged as an explicit test item anyway.

## Point 6 — Replay a selected record in an inheriting scene

The scene/env cfgs move out of `teleop_taco_scene.py` into `taco_scene_common.py`
(they were unimportable before — the script parses CLI args at module scope).
`replay_taco_scene.py` defines `TacoReplayEnvCfg(TacoTeleopEnvCfg)` — literal
inheritance — overriding only timing, physics minimization, and cameras.

Episode selection: `--dataset <file.hdf5>` + `--episodes all|success|0,3,7`
(default `success`). The script lists available demos with their success flags
before running.

## Point 7 — Pure kinematic replay

There is no global "physics off" mode in Isaac Lab, but there is something better:
**never step physics at all**. The replay loop (pattern proven in
`sim_benchmark/run.py --replay_mode kinematic`) does, per frame:

```python
robot.write_joint_state_to_sim(q, zeros)         # prescribe all 58 joints
brush.write_root_pose_to_sim(...); bowl...       # prescribe object poses (+ zero vel)
env.sim.forward()                                # FK + Fabric push — zero solver work
env.sim.render(); env.scene.update(dt)           # render + refresh camera sensors
```

`env.step()` is never called → no PhysX solve, no contacts, no gravity integration,
no actuators. Every frame is a complete prescribed state.

Timing: `sim.dt = 1/30`, `decimation = 1`, `render_interval = 1` as requested.
Recorded states arrive at 60 Hz (control rate), so replay strides **every 2nd
frame** — sim-time duration is preserved exactly. Residual physics params are also
floored (solver iterations 1/0, CCD off) purely as belt-and-suspenders; they are
never exercised.

## Point 8 — Replay cameras

Four `CameraCfg` sensors in the replay env cfg, all RGB, default 1280×720, written
as one MP4 per camera per episode (imageio/libx264, 30 fps):

| Camera | Mount | Pose |
|---|---|---|
| `operator_cam` (first person) | world, fixed | sim_benchmark's `REFERENCE_EYE` (0, −0.5625, 1.3281) looking at the tabletop, 45° vertical FOV — the calibrated MuJoCo reference viewpoint (per design decision: fixed, not headset-trajectory) |
| `third_person_cam` | world, fixed | side ¾ view, (1.15, 0.35, 1.2) → table center |
| `wrist_cam_left` | `robot/left_hand/left_hand_flange` | small offset, looking along the flange axis at the fingers |
| `wrist_cam_right` | `robot/right_hand/right_hand_flange` | mirrored |

Wrist camera orientations are first-guess constants, documented in the cfg as
tune-by-eye (verify with one replay and adjust the `OffsetCfg`).

Cameras and XR cannot coexist (headless-XR deadlock), which is exactly why cameras
live only in the replay scene — the teleop scene stays camera-free.

## Point 9 — Domain randomization (grid, RoboLab assets)

`taco_variations.py`, patterned on `~/RoboLab/robolab/variations/` and
`~/RoboLab/policies/pi0_family/run_table_variation.py`, but fixing RoboLab's known
sharp edges (name-traversal material lookup, first-match-only binding, per-env
rebuild per variation).

Per design decision: **deterministic grid** (`itertools.product`), one pass of each
selected episode per combo, and combo provenance written to a `meta.json` next to
the videos (RoboLab's `extra_fields` convention).

- **Background** — mutate the existing dome light
  (`/World/envs/env_0/scene/DomeLight`) in place per combo: `inputs:texture:file`
  (HDRI from `~/RoboLab/assets/backgrounds/`, `latlong` format), intensity. No env
  rebuild between combos (unlike RoboLab, which registers one env per background).
  Default set: `empty_warehouse.hdr`, `brown_photostudio.hdr`, `billiard_hall.hdr`;
  `--backgrounds` overrides (names or `all` = every HDRI found).
- **Lighting** — named presets mutating the scene's `KeyLight` (DistantLight) and
  dome intensity: `default`, `dim`, `bright`, `warm`, `cool`. `--lightings` selects.
- **Table material** — the TACO table is a bare `Cube` with no visual material, so
  materials are spawned once at startup via `sim_utils.spawn_from_mdl_file`
  (MDLs from `~/RoboLab/assets/materials/`: `Oak`, `Walnut_Planks`, `Bamboo`,
  `Black_Matte`) and rebound per combo with `sim_utils.bind_visual_material` —
  Isaac's tested code paths rather than hand-rolled `UsdShade` (RoboLab's approach
  needs pre-authored `Looks` libraries; we don't have one in the TACO asset).

Default grid: 3 backgrounds × 3 lightings (`default`, `dim`, `warm`) × 4 materials
= 36 combos per episode; `--dr off` renders each episode once, unrandomized.

Requires `~/RoboLab` present at replay time (`--robolab_dir` overrides the path).

---

## AVP client changes (one Xcode rebuild)

1. **`TeleopControlView.swift`** — Align button next to Reset (same row, so the
   fixed 100 pt stack height still fits); sends `"align scene"`.
2. **`AppModel.swift`** — `recordResultEpisode: Int?` published state; a receive
   `Task` iterating `teleopChannel.receivedMessageStream` (started right after
   channel discovery, cancelled in `resetChannelState()`); a hoisted
   `sendTeleopCommand(_:)` so the sheet doesn't depend on the conditionally-mounted
   teleop view.
3. **`TopConfigView.swift`** — `.sheet` bound to `recordResultEpisode != nil`
   presenting **Success** / **Failure**; the choice sends
   `record_success`/`record_failure` and clears the state. Bound to the config
   window (always present in mixed immersion; the model state survives the
   window-reopen cycle on headset removal).

Build: Xcode 26 on a Mac, your own signing team/bundle id, deploy to device
(README steps). All server-side Python iterates freely afterwards.

## Known risks / explicit test items

1. **`omni.kit.cloudxr.send_message` delivery** — the relay is verified to exist in
   the plugin binary, but whether it subscribes with pop or immediate semantics is
   not; the sender uses `push()` and falls back to `dispatch()` via
   `--client_msg_dispatch` if the dialog never appears.
2. **Align vs. wrist calibration** — verify hands still track without twist after
   aligning from a rotated stance.
3. **Transparency under `balanced` RaytracedLighting** — verify 50% looks right;
   otherwise use `--arm_visual hidden`.
4. **Wrist camera aim** — first-guess extrinsics; tune after one replay.
5. **Gesture false positives** — 2 cm × 5 fingers × 0.5 s should be safe, but if it
   fires during clapping-like bimanual motions, raise `--gesture_hold_s`.

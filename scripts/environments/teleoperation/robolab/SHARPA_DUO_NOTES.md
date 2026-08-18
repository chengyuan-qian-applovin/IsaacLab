# Implementation Notes: SharpaWave Duo Bimanual XR Teleoperation

Engineering record for the FR3 Duo + dual SharpaWave teleop added to
`feature/robolab-xr-teleop` (July 2026). Companion to
[IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) (single-arm DROID flow);
usage lives in [README.md](README.md). This document explains **which IK was
chosen, why, and how every layer integrates**, so the design can be maintained
or revisited.

## 1. Goal

Teleoperate the bimanual rig documented in the SharpaWave RoboLab fork
(`docs/sharpa_wave_duo_scene.md`: FR3 Duo torso, two Panda arms, two 22-DoF
SharpaWave dexterous hands — one 58-DoF PhysX articulation) on RoboLab
benchmark scenes via AVP hand tracking, with the **same teleop effect as the
default GR1T2 scene**: both hands drive the arms, and all fingers are
retargeted. Keep both upstream repos unmodified; record demos through the
existing dual pipeline (RoboLab HDF5 → robomimic converter).

## 2. The IK decision (the core of this change)

### What the GR1T2 reference does

GR1T2 teleop (`Isaac-PickPlace-GR1T2-Abs-v0`) uses
`PinkInverseKinematicsAction` (Isaac Lab): a Pink/Pinocchio **QP-based
whole-upper-body IK** with one `FrameTask` per wrist plus damping and
null-space posture tasks, solved per-env on the CPU against a URDF converted
from the robot USD at config-init time. Its 36-D action is
`[L wrist pose 7 | R wrist pose 7 | 22 hand joints]`, produced by
`GR1T2Retargeter` (wrist passthrough + `dex_retargeting` DexPilot fingers).
The action term converts world-frame wrist targets to the robot base frame
internally.

### What this rig uses instead — and why

The duo rig already ships `FrankaDuoSharpaIKActionCfg`
(`robolab/robots/franka_duo_sharpa_wave.py` in the fork): **per-arm absolute
wrist-pose differential IK** — one Isaac Lab
`DifferentialInverseKinematicsAction` (damped-least-squares) per arm tracking
`left_panda_link8` / `right_panda_link8`, commands in the **robot root
frame**, plus two 22-joint `JointPositionAction` terms for the hands. 58-D
action: `[L wrist 7 | R wrist 7 | L fingers 22 | R fingers 22]`.

I reused that differential IK rather than porting Pink:

| Consideration | Pink (GR1T2) | DiffIK (chosen) |
|---|---|---|
| Kinematic coupling | GR1T2's arms share a torso/waist chain — whole-body QP genuinely matters | Duo arms are two independent 7-DoF chains on a **fixed** torso; per-arm IK is structurally sufficient |
| Redundancy handling | Null-space posture task keeps elbows sane | 7-DoF arm has 1 redundant DoF; DLS damping leaves it near the previous configuration — acceptable, revisit if elbows drift |
| Runtime | CPU QP per env per step (pinocchio), needs USD→URDF conversion at init | GPU-friendly torch Jacobian DLS, no conversion step |
| Validation status | — | The rig's ready pose was itself solved with this IK by the fork's author; my calibration run reproduced the documented wrist poses `(0.35, ±0.15, 0.25)` exactly |
| Action-space equivalence | 36-D wrists+hand | 58-D wrists+hands — same shape of contract, so the GR1T2 retargeting pattern transplants directly |

**"Similar effect" is satisfied at the action contract level**: in both stacks
the teleop layer emits absolute wrist poses + retargeted finger joints, and an
IK action term turns wrist poses into arm joint targets. What changed is only
*which* solver does that last step. If bimanual coordination problems appear
(elbow drift, joint-limit saturation during long sessions), the escalation
path is porting `PinkInverseKinematicsActionCfg` to this robot — two
`FrameTask`s on the flanges + `NullSpacePostureTask` on the ready pose — which
the retargeter side would not need to change for.

### One integration difference that matters

GR1T2's Pink action converts world→base **inside the action term**. The duo's
DiffIK action expects **root-frame commands and does no conversion** (it was
built for GR00T, which outputs root-frame poses). So the teleop script owns
that conversion: `to_root_frame()` in `teleop_sharpa_duo_agent.py` applies
`subtract_frame_transforms(root_pose, wrist_target)` to both 7-D wrist slices
every step, reading the live root pose from the articulation. XR anchor frame
= env world frame, so no other frames are involved.

## 3. Integration architecture

```
AVP (unmodified Isaac XR Teleop client)
  ▼ CloudXR → OpenXR → omni.kit.xr
OpenXRDevice._get_raw_data()          {HAND_LEFT: {26 joints}, HAND_RIGHT: {26 joints}}
  ▼
FrankaDuoSharpaRetargeter.retarget()               (sharpa_duo_retargeters.py)
  ├─ wrists: pos passthrough; quat ⊗ per-side calibrated offset      → 2 × 7-D (world frame)
  └─ fingers: SharpaWaveDexRetargeting (DexPilot, vendored URDFs)
       26 XR joints → 21 MANO points → QP per hand → 22 joints,
       scattered by NAME into HAND_JOINTS_ORDERED                     → 2 × 22-D
  ▼ concat → 58-D action (declaration order: L arm, R arm, L hand, R hand)
teleop loop: to_root_frame() on slices [0:7] and [7:14]
  ▼ env.step()
FrankaDuoSharpaIKActionCfg
  ├─ left_arm / right_arm: DLS differential IK → 7+7 arm joint targets
  └─ left_hand / right_hand: JointPositionAction → 22+22 finger targets
  ▼ PhysX (120 Hz physics, decimation 8 → 15 Hz control)
RobolabRecorderManager → run_<N>.hdf5 (same lifecycle as the single-arm flow)
```

The retargeter follows the `RetargeterBase` contract, so it plugs into the
standard `OpenXRDeviceCfg`/`create_teleop_device` machinery — no Isaac Lab or
RoboLab source changes anywhere; the whole integration lives in this scripts
directory.

## 4. Wrist frame calibration

The Fourier GR1T2 hands were modeled so OpenXR wrist frames map to the robot's
wrist links near-identity. `panda_link8` is not, so a constant per-side
rotation offset is required:

```
q_flange_target = q_xr_wrist ⊗ q_offset,   q_offset = q_xr_ready⁻¹ ⊗ q_link8_ready
```

`calibrate_sharpa_duo.py` computes this by creating the env at the rig's
IK-solved ready pose (fingers forward +x, palms down) and reading each
flange's orientation; `q_xr_ready` is the analytic OpenXR wrist orientation
for the same human pose (XR_EXT_hand_tracking: −Z along the bone toward the
fingertips, +Y dorsal). Measured results, baked into
`FrankaDuoSharpaRetargeterCfg`:

- `left_wrist_rot_offset  = (0, −0.9238795, +0.3826834, 0)`  (w,x,y,z)
- `right_wrist_rot_offset = (0, +0.3826834, −0.9238795, 0)`
- The cos/sin(22.5°) structure is exactly the rig's ∓45° flange-mount
  rotations surfacing — strong evidence the derivation is correct, not noise.
- Positional offset: **zero.** Calibration measured the Sharpa `hand_wrist`
  body 0.5 mm from the flange — human wrist ↦ flange directly. (A
  `wrist_pos_offset` cfg field exists for future rigs where they don't
  coincide.)

If the hands track with a constant twist in a live session, rerun the
calibration script and update the two constants; if the rig's mounts change,
same procedure.

## 5. Finger retargeting

Byte-for-byte the GR1T2 pattern with new data:

- `sharpa_dex_retargeting/sharpa_wave_{left,right}_dexpilot.yml` — DexPilot
  configs: 5 `*_fingertip` task links, all 22 actuated joints as targets,
  `scaling_factor 1.1`, `low_pass_alpha 0.2` (GR1T2's smoothing).
- URDFs vendored unmodified from `sharpa-robotics/sharpa-urdf-usd-xml`
  (`wave_01/…_with_flange.urdf`) — the same source and vintage as the fork's
  USD assets, so joint names match `HAND_JOINTS_ORDERED` exactly (verified: 22
  revolute joints, no mimics). Kinematics only; unresolvable `package://` mesh
  warnings are expected and harmless.
- Improvements over the GR1T2 utils: configs are loaded via
  `RetargetingConfig.from_dict` with the URDF path resolved in memory (GR1T2
  rewrites its ymls on disk at every init — the source of the perpetually
  dirty `fourier_hand_*.yml` files in git status), and dex outputs are
  scattered into the action by **joint name**, so the optimizer's DoF order
  never needs to match the action order.
- The DexPilot QP needs gradients: calls run inside
  `torch.enable_grad() + torch.inference_mode(False)` because the teleop loop
  wraps everything in `inference_mode` (same escape hatch GR1T2 uses).

## 6. Files

```
scripts/environments/teleoperation/robolab/
├── teleop_sharpa_duo_agent.py        # bimanual teleop + recording loop
├── sharpa_duo_retargeters.py         # FrankaDuoSharpaRetargeter + SharpaWaveDexRetargeting
├── calibrate_sharpa_duo.py           # derives the wrist rot offsets from the ready pose
├── calibrate_hand_shape.py           # operator hand-shape calibration scene (§10)
├── robolab_teleop_common.py          # shared strip_cameras_for_xr (also used by single-arm script)
└── sharpa_dex_retargeting/           # DexPilot ymls + vendored Sharpa URDFs + provenance README
    └── hand_calibration.yml          # per-operator hand-shape calibration (auto-loaded, §10)
```

Prerequisite: the **SharpaWave RoboLab fork** installed as `robolab`
(it provides `robolab.registrations.sharpa_wave` and the rig assets):
`ROBOLAB_PATH=<fork path>` with the compose patch, then
`pip install --no-deps -e /workspace/robolab` in the container (the
`--no-deps` rationale is in IMPLEMENTATION_NOTES.md §3).

## 7. Validation performed (headless, in-container, Isaac Sim 5.1)

1. **Dex configs standalone**: both hands build against the vendored URDFs;
   22 DoFs each; DexPilot retarget returns finite 22-D outputs.
2. **Rig on Isaac Sim 5.1** (fork was developed on 5.0): env creation OK;
   ready-pose flange positions match the fork's documentation to 0.1 mm.
3. **Calibration**: offsets derived; clean flange-constant structure (§4).
4. **Full teleop script under headless XR**: registration → env (4 action
   terms, cameras stripped) → device + both dex hands built in-kit →
   `Starting teleop loop` reached.
5. **Retargeter contract** (synthetic dual-hand data in-kit): 58-D output,
   wrist positions track inputs, unit quaternions, left/right finger blocks
   independent and within joint ranges.
6. **Not yet validated**: live headset session — anchor height comfort and
   absence of constant wrist twist are the two things only that can confirm.

## 8. Bugs hit during bring-up

| Symptom | Cause | Fix |
|---|---|---|
| Kit dies **silently** (clean exit, no traceback) mid-way through the left hand's URDF parse during device construction | `pinocchio` first imported *after* kit boot (by `dex_retargeting`) clashes with Isaac Sim's copy / `pxr.Gf` | `import pinocchio` before `AppLauncher` — the same requirement behind `teleop_se3_agent.py --enable_pinocchio`; AppLauncher then applies its `Gf.Matrix4d` patch |
| (inherited) headless-XR camera deadlock, stdout buffering, XR-forces-CPU device | documented in IMPLEMENTATION_NOTES.md §§5-7 | shared `strip_cameras_for_xr`, flushed prints, explicit `--device cuda:0` |

## 9. Revisit triggers

- **Elbow drift / joint-limit saturation** during long bimanual sessions →
  port Pink IK for this rig (two flange `FrameTask`s + `NullSpacePostureTask`
  anchored on the ready pose); the retargeter and recording layers are
  solver-agnostic and stay as-is.
- **Wrist twist** in the headset → rerun `calibrate_sharpa_duo.py` (§4).
- **Fingers "shuffled"** → the fork's own caveat about `HAND_JOINTS_ORDERED`
  vs vendor URDF order; our name-based scatter makes this unlikely, but check
  the yml `target_joint_names` against the fork first.

## 10. Operator hand-shape calibration (Aug 2026)

The blanket DexPilot `scaling_factor: 1.1` under-corrected the operator↔Sharpa
shape mismatch (raw fingertip gaps 25–65 mm in the wrist frame; the Sharpa is a
large hand — measured global scale came out ≈ 1.18). A per-operator calibration
now measures and removes it.

### Measuring — `calibrate_hand_shape.py`

Empty XR scene (dome light + red joint markers, no robot). Connect the AVP,
click Play, hold **both hands flat with fingers straight**; `--delay` s later
(default 5) the first frame with both hands validly tracked is captured. Per
hand, with human keypoints in the human wrist frame and Sharpa fingertips
(pinocchio FK at q = 0, its straight-finger pose) in the Sharpa wrist-link
frame — the two wrist frames identified, wrist pinned at the origin:

1. **Global fit** on index/middle/ring tips: wrist-pinned similarity Procrustes
   (closed-form SVD), `min over (s, R) of Σ‖s·R·h_i − t_i‖²`, no translation.
2. **Per-finger corrections** for thumb and pinky (excluded from the fit —
   their geometry differs most): a length ratio `|t|/|s·R·h|` plus the minimal
   Rodrigues rotation swinging the globally-calibrated tip direction onto the
   Sharpa's. Exact tip placement at the capture pose by construction.

Results go to `sharpa_dex_retargeting/hand_calibration.yml` (transform +
diagnostics + all ten captured fingertips, so corrections can be re-derived
offline without re-capturing). Current operator: s ≈ 1.181/1.178 (L/R),
residual wrist rotation ≈ 14.4°/14.3° (a MANO↔Sharpa axis convention — nearly
identical across hands and sessions, so not tracking noise), fit rms 4.8/4.1 mm,
thumb ratio ≈ 1.09 (rot 2.5°/7.6°), pinky ratio ≈ 1.15 (rot ≈ 0.5°).

### Applying — `sharpa_duo_retargeters.py`

`FrankaDuoSharpaRetargeterCfg.hand_calibration` (default
`"hand_calibration.yml"`) auto-loads the file when present — all Sharpa teleop
scenes pick it up with no flags; set the field to `""` to run uncalibrated.
On load, per side:

- **Rotation, applied WRIST-side**: `R_corr = M·R_calᵀ·Mᵀ` (M = OPERATOR2MANO)
  right-multiplies the tracked wrist rotation. Finger keypoints derived from
  the rotated wrist equal `R_cal·p` exactly (rotating the frame ≡ rotating the
  points), and the same correction composes into the arm's wrist→flange offset
  (`q_flange = q_wrist ⊗ q_corr ⊗ q_offset`) — the robot's **fingers**, not its
  wrist axes, align with the operator's in the world (~14° flange tilt vs the
  uncalibrated rig).
- **Scale** multiplies the keypoints; the DexPilot optimizer `scaling` is
  overridden to 1.0 (the yml `scaling_factor` no longer applies).
- **Thumb/pinky corrections** (`{finger}_ratio`, `{finger}_rotation`) apply
  after the global transform to that finger's MANO rows only (rows 1–4 / 17–20),
  about the wrist. Hand-tunable in the yml; identity entries cost nothing.
- **Pinch detection stays on the operator's real fingers**
  (`raw_pinch_detection`, default on with calibration): the calibrated scale
  would otherwise inflate every pair distance ~18%, shifting DexPilot's
  3 cm/5 cm project/escape hysteresis. The same hysteresis now runs on the
  *uncalibrated* thumb-pair distances and is written into
  `optimizer.projected[:4]`; the internal update is neutralized with ∓inf
  thresholds. (Known residue: the library's S2 finger–finger rule keeps its
  hardcoded ≤ 3 cm gate on calibrated distances — multi-finger-pinch spacing
  only.)

Related knob from the same bring-up: `pinch_separation` (cfg) patches
`optimizer.projected_dist[:4]` — the commanded thumb–finger gap once a pinch
projects; negative overshoots past contact so the PD drives squeeze the object.

### Revisit triggers

- **New operator** → re-run `calibrate_hand_shape.py` (≈10 s); the yml is
  per-person. Thumb comfortably extended in the natural flat pose — its
  captured direction sets `thumb_rotation`.
- **Thumb over/under-reaches mid-curl** → tune `thumb_ratio` in the yml
  (corrections are exact at the flat pose, first-order elsewhere).
- **Fit rms > ~1 cm** in the calibration output → fingers weren't straight;
  re-capture.
- **Success scoring** on this rig is unvalidated upstream (single-fingertip
  `contact_gripper`, predicates written for parallel grippers) — treat
  recorded `success` attrs with care until the fork validates them.

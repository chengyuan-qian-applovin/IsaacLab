# DexVerse Retargeting, Step by Step

A guided tour of how DexVerse turns tracked human hand poses into robot commands:
the **wrist ("arm") retargeting**, the **finger ("hand") retargeting**, and how the
two are **assembled into one action vector** and applied to the simulated robot.

All file paths are relative to the DexVerse repo root
(https://github.com/ycyao216/DexVerse). The two files that matter most:

- `source/dexverse/dexverse/devices/retargeters/simple_relative_retargeting.py` — the default (relative) retargeter, and all shared finger logic
- `source/dexverse/dexverse/devices/retargeters/simple_absolute_retargeting.py` — the absolute-mode wrist variant (fingers inherited unchanged)

Supporting files:

- `source/dexverse/dexverse/devices/wrist_origin.py` — derives the "home wrist pose" needed by absolute mode
- `source/dexverse/dexverse/robot_agents/shadow/floating.py` — the Shadow Hand action layouts, actuator gains, and dex-retargeting spec
- `source/dexverse/dexverse/robot_agents/shadow/retarget/{right,left}_{dexpilot,vector}.yml` — finger-optimizer configs
- `scripts/teleop_agent.py` — the teleop loop that calls all of this

Throughout, "the retargeter" means `SimpleRelativeRetargeter` unless stated otherwise.

---

## 0. The mental model

One frame of teleop does this:

```
Apple Vision Pro hand tracking (via CloudXR → OpenXR → Isaac Lab OpenXRDevice)
        │  raw data: 26 joints per hand, each a 7-vector [x y z | qw qx qy qz]
        │  in the simulator world frame (= the XR anchor frame)
        ▼
┌───────────────────────────────────────────────────────────────┐
│ SimpleRelativeRetargeter.retarget(data)                        │
│                                                                │
│   A. WRIST  → 6 numbers per hand                               │
│      (3 translation + 3 rotation, for the virtual wrist joints)│
│                                                                │
│   B. FINGERS → 22 numbers per hand                             │
│      (absolute joint angles, from the dex-retargeting QP)      │
│                                                                │
│   C. ASSEMBLY → one flat action vector                         │
│      28-D single hand / 56-D bimanual                          │
└───────────────────────────────────────────────────────────────┘
        │  torch.Tensor on the sim device
        ▼
env.step(action)  →  Isaac Lab JointPositionAction terms  →  PD actuators (PhysX)
```

Two properties to hold onto:

1. **Wrist and fingers are computed independently.** The finger solve is made
   *wrist-invariant* by re-expressing all finger keypoints in a frame attached to
   the wrist (Section 3.2). Waving your arm around does not change the finger
   command at all; curling your fingers does not change the wrist command.
   The only thing they share is the same wrist pose measurement.
2. **There is no arm IK in the released pipeline.** The Shadow Hand "floats": its
   wrist is a chain of 3 prismatic + 3 revolute *virtual joints*
   (`x/y/z_translation_joint`, `x/y/z_rotation_joint`), so the wrist command is
   written **directly as joint targets** — a 3-vector of meters and a 3-vector of
   Euler angles. The code contains hooks for real arms with
   `DifferentialIKController` (Section 2.4), but no shipped robot uses them.

---

## 1. Conventions you must know before reading the code

These trip up everyone the first time.

**Quaternion element order.** Isaac Lab / OpenXR data uses **wxyz** (scalar
first). SciPy's `Rotation.from_quat` expects **xyzw** (scalar last). The file
defines converters `wxyz_to_xyzw` / `xyzw_to_wxyz` at
`simple_relative_retargeting.py:55-60` and converts at every boundary. When you
see a bare 7-vector pose in this codebase it is `[x y z | qw qx qy qz]`.

**Rotation composition direction.** With SciPy `Rotation` objects, `A * B` means
"apply B first, then A" (matrix product `R_A R_B` acting on column vectors).
So `R_rel = R_now * R_calib.inv()` is the rotation that maps the calibrated
orientation to the current one, *expressed in the world frame* (a "left delta").

**Row-vector matrix convention.** The finger code stores keypoints as an
`(N, 3)` array and writes `points @ M`. For a rotation matrix `M`, `p_row @ M`
equals `Mᵀ p_col` — i.e. it applies the **inverse** of `M` to the points, which
is the same as *re-expressing the points in the frame whose orientation is `M`*.
This is used twice (canonicalization and heading compensation) and both times the
intent is "change of basis", not "rotate the object".

**Euler angles.** `R.as_euler("XYZ")` with **uppercase** letters means
**intrinsic** rotations: `R = Rx(a) · Ry(b) · Rz(c)`, each about the *new* axis.
This matters because it exactly matches how a chain of revolute joints
`x_rotation → y_rotation → z_rotation` composes (parent to child), which is why
the wrist rotation command can be dropped straight into those three joints.

**The world frame is the XR anchor frame.** Isaac Lab places the headset's
tracking origin at a configured pose in the simulator world (`XrCfg.anchor_pos` /
`anchor_rot`; DexVerse uses e.g. `(-0.5, 0, 0.1)` in
`tasks/config/floating_teleop.py`). After that, hand poses arriving in
`retarget()` are ordinary world-frame poses. No further "human frame vs robot
frame" bookkeeping happens at the device level.

**Action offset semantics (critical!).** Isaac Lab's `JointPositionActionCfg`
defaults to `use_default_offset=True`, meaning the simulator applies

```
joint_target = action * scale + default_joint_pos
```

where `default_joint_pos` comes from the articulation's `init_state`. So an
action of **zero holds the robot at its home pose**, and the wrist command only
has to be a *displacement*. This single fact explains why the relative
retargeter can subtract a calibration pose and be done (Section 2.2), and why
the absolute retargeter must subtract the *home wrist world pose* rather than
the raw joint origin (Section 2.3). It is stated explicitly in the docstring of
`wrist_origin.py:14-25`.

---

## 2. Stage A — wrist retargeting (the "arm")

### 2.0 Input and output of this stage

| | |
|---|---|
| **Input** | one 7-vector wrist pose per hand, world frame: `p_hand ∈ ℝ³`, `q_hand` (wxyz) — extracted from the OpenXR `"wrist"` joint (`_extract_wrist_pose`, `simple_relative_retargeting.py:335`) |
| **State** | the calibration pose `(p₀, q₀)` snapshotted at START (relative mode); or a constant "wrist joint origin" (absolute mode) |
| **Output** | 6 floats per hand: `[tx ty tz]` in meters + `[yaw pitch roll]` in radians, written into the action vector at the layout's `wrist_trans_indices` / `wrist_rot_indices` |

### 2.1 The canonical hand frame (used by everything downstream)

Raw OpenXR wrist orientation follows the `XR_EXT_hand_tracking` convention,
which is awkward for robotics (roughly: −Z runs along the forearm toward the
fingertips, +Y comes out of the back of the hand). Every wrist computation first
passes through `_get_normalized_wrist_rotation()`
(`simple_relative_retargeting.py:400`):

```python
R_canon = R_raw * ( Ry(+90°) · Rx(−90°) )
```

(Heads-up: the local variable names are misleading — `x_plus_90` actually holds
`Rx(−90°)` and `y_minus_90` holds `Ry(+90°)`. Trust the arguments, not the names.)

Right-multiplying by a fixed rotation is a **change of body-axis convention**: it
relabels which directions count as the hand's x/y/z without moving the hand.
Multiplying out `M = Ry(90°)Rx(−90°)` and reading its columns (each column = a
new axis expressed in old axes):

```
new x = −z_old   → points along the fingers ("forward")
new y = −x_old   → completes the right-handed frame
new z = +y_old   → out of the back of the hand ("up")
```

So in the **canonical frame, a flat hand pointing forward, palm down, has
orientation ≈ identity**, and its intrinsic-XYZ Euler angles read naturally as
(roll, pitch, yaw). That is the entire point of the normalization: it makes the
Euler decomposition in the next step correspond to intuitive wrist motions.

### 2.2 Relative mode (the default): displacement from a calibration pose

Class: `SimpleRelativeRetargeter`. Entry point per hand:
`_assign_hand_wrist_command()` (`simple_relative_retargeting.py:422`).

**Calibration.** When you make the START gesture on the headset, the teleop loop
calls `calibrate_wrist_pose()` (`scripts/teleop_agent.py:410-416` →
`simple_relative_retargeting.py:191`), which copies the most recent wrist pose
of every tracked hand into `retarget_base_wrist_poses[hand]`. Call these
`(p₀, q₀)`. Until the first calibration the base is the identity pose, so
**always START with your hand held where you want "zero" to be.**

**Translation** (`:445-447`):

```
Δp = p_hand − p₀                       # world-frame displacement, meters
cmd_trans = R_base⁻¹ · Δp              # optional re-expression in the layout's frame
```

`R_base` comes from an optional `wrist_base_rot` quaternion in the robot layout
(`_layout_rotation`, `:346`). The Shadow layouts don't define it, so it is the
identity and `cmd_trans = Δp` — move your hand 10 cm along world +x and the
command is `(0.1, 0, 0)`. The hook exists for robots whose base is rotated
relative to the world.

**Rotation** (`_compute_relative_wrist_euler`, `:367`):

```
R_rel   = R_canon(q_hand) · R_canon(q₀)⁻¹     # world-frame delta rotation
R_rel'  = R_base⁻¹ · R_rel · R_base           # conjugation into the layout frame (identity for Shadow)
(a,b,c) = R_rel'.as_euler("XYZ")              # intrinsic: R = Rx(a)·Ry(b)·Rz(c)
cmd_rot = reorder per layout, e.g. yaw_pitch_roll → (c, b, a), then × wrist_rot_signs
```

Why this works end-to-end: the robot's virtual rotation chain is
`x_rotation_joint → y_rotation_joint → z_rotation_joint`, whose composed
rotation is exactly `Rx(qx)·Ry(qy)·Rz(qz)` — the intrinsic-XYZ form. The home
rotation joints are all zero, so with `use_default_offset` the commanded wrist
orientation becomes `R_base_robot · Rx(a)Ry(b)Rz(c) = R_rel` (robot base is
unrotated): **the robot wrist reproduces, as an absolute orientation, the
rotation your hand has made since calibration.** Note the action cfg orders the
rotation joints `[z, y, x]` with `preserve_order=True`
(`shadow/floating.py:102-107`) to match the retargeter's `yaw_pitch_roll`
output — a classic place for silent bugs if either side changes alone.

Two properties of this formulation worth appreciating:

- It is **not incremental**. Every frame recomputes the total displacement from
  the calibration pose, so there is no drift accumulation and dropped frames are
  harmless. (Contrast with delta/velocity teleop schemes.)
- It inherits Euler-angle limitations: at pitch ≈ ±90° the decomposition
  degenerates (gimbal lock), so extreme wrist orientations can command jumps.
  The 60 Hz rate and PD damping mask small ones in practice.

**Stationary-hand semantics:** hold your hand still at the calibration pose and
the command is all zeros → `joint_target = default_joint_pos` → the robot holds
its home pose. This is the "position-control semantics" the docstrings advertise.

### 2.3 Absolute mode (`--teleop_retargeter absolute`)

Class: `SimpleAbsoluteRetargeter` (`simple_absolute_retargeting.py:79`). It
overrides *only* `_assign_hand_wrist_command`; fingers are inherited unchanged.
Here the robot wrist is driven to the hand's **absolute world pose** — no START
calibration involved (START still gates whether stepping happens, but the
mapping doesn't depend on it).

The catch: joint commands are still offsets from the home pose (Section 1's
`use_default_offset`). So the code needs the world pose the wrist occupies when
all six virtual joints are at their defaults — the **wrist joint origin**
`(p_origin, R_origin)`. It is derived, not hand-tuned, by
`compute_wrist_joint_origin()` (`wrist_origin.py:73`):

```
p_origin = p_base + R_base · (mount_offset + home_translation_joint_values)
R_origin = R_base · Rx(qx⁰)Ry(qy⁰)Rz(qz⁰)          # home rotation-joint values
```

For the single right Shadow hand (`shadow/floating.py:236-253`): base at
`(-0.75, 0, 0.5)` plus home translation joints `(0.5, 0, 0.3)` gives
`p_origin = (-0.25, 0, 0.8)`, identity rotation. For the bimanual rig each
hand's chain hangs off a mount link offset `(0, ∓0.3, 0)` inside the USD — the
`mount_offset` argument handles that.

Then (`simple_absolute_retargeting.py:142-190`):

```
cmd_trans = R_origin⁻¹ · (p_hand − p_origin)          # hand pose in the origin frame
R_local   = R_origin⁻¹ · R_canon(q_hand)
cmd_rot   = Euler-XYZ of R_local, reordered/signed as in relative mode
```

Same Euler machinery, different reference: **relative mode measures against
where *your hand* was at START; absolute mode measures against where *the
robot's wrist* is at home.** Absolute mode is what makes the robot hand coincide
with the red-dot visualization of your tracked hand; the per-mode XR anchor
offset (`DEFAULT_RETARGETER_ANCHOR_POS_OFFSET` in
`tasks/config/floating_teleop.py`, −0.3 m in z) exists to keep the workspace
comfortable in this mode.

### 2.4 `quat_absolute`: the dormant arm-IK path

Both classes also support a third wrist representation, `wrist_rot_repr =
"quat_absolute"`, meant for real arms driven by Isaac Lab's
`DifferentialIKController(use_relative_mode=False)` — the action carries a
7-vector `[pos, quat wxyz]` end-effector target in the robot base frame instead
of 6 joint values:

- Relative variant (`_assign_absolute_wrist_command`,
  `simple_relative_retargeting.py:464`): `target = home EE pose ⊕ displacement
  since calibration`, i.e. `p* = p_ee_home + (p_hand − p₀)`,
  `R* = R_rel · R_ee_home`.
- Absolute variant (`_assign_absolute_wrist_command_world`,
  `simple_absolute_retargeting.py:192`): the hand pose transformed straight
  into the robot base frame: `p* = R_origin⁻¹(p_hand − p_origin)`,
  `R* = R_origin⁻¹ · R_canon`.

No shipped robot uses these — they are the seam where arm embodiments would
plug in. (This is essentially the design that the IsaacLab
`feature/robolab-xr-teleop` branch ships for real arms.)

---

## 3. Stage B — finger retargeting (the "hand")

### 3.0 Input and output of this stage

| | |
|---|---|
| **Input** | 21 keypoint positions per hand (selected from the 26 OpenXR joints), world frame, plus the wrist pose for canonicalization |
| **Output** | 22 **absolute joint angles** for the Shadow finger joints (`FFJ1..4, MFJ1..4, RFJ1..4, LFJ1..5, THJ1..5`), in the layout's `finger_joint_names` order |

Unlike the wrist, fingers have **no calibration and no relative mode**: your
current hand *shape* maps to robot joint angles every frame. Entry point:
`_retarget_hand_fingers()` (`simple_relative_retargeting.py:526`).

### 3.1 Keypoint selection: 26 → 21

OpenXR reports 26 joints per hand (palm, wrist, and 5 joints per digit
including metacarpals). `DEX_RETARGETING_HAND_JOINT_INDICES`
(`simple_relative_retargeting.py:80`) keeps 21 of them — it drops the palm and
the four finger metacarpals (the thumb's metacarpal is kept, since the thumb's
mobility lives there). The resulting array layout, which the YAML configs index
into, is:

```
0 = wrist | 1–4 thumb | 5–8 index | 9–12 middle | 13–16 ring | 17–20 little
          (each digit ordered proximal → tip)
```

This is the de-facto 21-point MANO-style convention the `dex_retargeting`
library expects.

### 3.2 Canonicalization + heading compensation

`_convert_hand_to_canonical_joint_positions()`
(`simple_relative_retargeting.py:568`) makes the keypoints wrist-invariant:

```
q_i        = p_i − p_wrist                 # translate: wrist at origin
q_i_canon  = q_i @ M_canon                 # rotate into the canonical wrist frame
                                           # (M_canon = R_canon as matrix; row-vector
                                           #  convention ⇒ this applies R_canon⁻¹)
q_i_final  = q_i_canon @ Rz(θ_hand)        # heading compensation, θ = −15° right / +15° left
```

After the first two lines the finger cloud is expressed in the frame of
Section 2.1 (x = forward along fingers, z = out of the back of the hand),
**independent of where your arm is or which way your wrist points**. That is
what decouples Stage B from Stage A.

The third line is the hard-coded `FINGER_Z_ROTATION_DEG` yaw
(`simple_relative_retargeting.py:89-92`): an empirically tuned ~15° twist about
the wrist axis compensating for the residual heading mismatch between where
human fingers point in this frame and where the Shadow URDF's fingers point at
neutral. Signs are mirrored between hands because hands are mirror images. It
affects **only** the finger solve — the wrist command never sees it.

You can watch both stages live: the retargeter draws raw keypoints as red
spheres and the canonicalized cloud (translated back to the wrist for display)
as cyan spheres (`_visualize_hand_keypoints` / `_visualize_canonical_hand_keypoints`).

### 3.3 Building the optimization target ("ref value")

`_compute_dex_ref_value()` (`simple_relative_retargeting.py:610`) converts the
21 canonical points into whatever the chosen optimizer consumes. For the two
shipped schemes (both are "vector-type" optimizers), the ref value is a stack of
**difference vectors** `ref_k = q_task(k) − q_origin(k)`:

- **`vector` scheme** (`shadow/retarget/right_vector.yml`): 5 vectors, palm/wrist →
  each fingertip. Human indices `[0,0,0,0,0] → [4,8,12,16,20]` (wrist to the
  five tips); robot side `palm → {thtip, fftip, mftip, rftip, lftip}`.
  `scaling_factor: 1.125`.
- **`dexpilot` scheme** (`right_dexpilot.yml`, the default): the DexPilot
  formulation. From `finger_tip_link_names` + `wrist_link_name` the library
  builds the vector set itself: all **inter-fingertip vectors** (C(5,2) = 10
  pairs) plus **wrist → fingertip vectors** (5), i.e. 15 vectors. The
  inter-tip vectors are what make precision grasps work (next section).
  `scaling_factor: 1.15`.

Before handing the ref value to the optimizer, `_apply_finger_scales()`
(`:628`) can multiply individual rows by per-finger factors from
`cfg.finger_scales`, matched by longest prefix of the target link name (e.g.
`{"th": 0.9, "ff": 1.1, ...}` — compress the thumb, stretch the fingers). This
exists because a single global `scaling_factor` can't fix all digits at once
when robot finger proportions differ from yours; the YAML factor multiplies on
top afterwards, inside the library.

### 3.4 The optimization itself (inside the `dex_retargeting` library)

Setup: `_initialize_dex_retargeters()` (`:248`) loads the per-hand YAML,
patches in the resolved URDF path (into a *temp copy* — the original YAML is
never modified, `:304`), and builds a `SeqRetargeting` object whose optimizer
holds a Pinocchio model of the hand built from
`shadow/retarget/floating_shadow_{right,left}.urdf`.

Each frame the library solves, over the 22 target joints `q`:

```
min_q  Σ_k  w_k · huber( ‖ α · s_k · ref_k_human  −  v_k_robot(q) ‖ )
s.t.   joint limits
```

where `v_k_robot(q)` is the corresponding link-to-link vector computed by
forward kinematics at `q`, `α` is the YAML `scaling_factor`, and `s_k` the
per-finger scales. Practical details that matter:

- **DexPilot's trick** (why it's the default): the weights `w_k` on the
  inter-fingertip vectors are *state-dependent*. When two of your fingertips
  come within a small threshold of each other, those pairs get strongly
  up-weighted and their target lengths collapsed toward contact — so a human
  pinch reliably produces a robot pinch that actually closes, instead of two
  fingertips hovering 1 cm apart because the least-squares fit spread the error
  around. That behavior is the core idea of the DexPilot paper (Handa et al.,
  2020), implemented in `dex_retargeting`'s `DexPilotOptimizer`.
- **Solver**: sequential quadratic programming via NLopt (SLSQP), with the
  objective's gradients obtained through PyTorch autograd — which is why the
  call sites wrap it in `torch.enable_grad()` + `torch.inference_mode(False)`
  (`:547-549`): the surrounding teleop loop runs under `torch.inference_mode()`,
  which would otherwise kill autograd.
- **Warm start + smoothing**: each solve starts from the previous frame's
  solution, and the output passes through a first-order low-pass filter
  `q_out = α·q_new + (1−α)·q_prev` with `low_pass_alpha: 0.8` (light smoothing;
  lower alpha = heavier). Both make the output temporally coherent, but the
  mapping remains **absolute**: freeze your hand and the solution converges to
  a fixed pose determined by your hand shape alone.
- **Non-target joints**: the retarget URDFs include the 6 floating wrist joints
  so the kinematics are complete, but they are *not* optimized — they're pinned
  to zero via the `fixed_qpos` argument (`:537-546`). The optimizer solves the
  fingers in the wrist frame, consistent with the canonicalization of §3.2.

### 3.5 Output ordering — the by-name remap

`SeqRetargeting.retarget()` returns the full qpos in **Pinocchio DOF order**
(URDF traversal order), *not* the YAML `target_joint_names` order — an easy
off-by-permutation trap. DexVerse never assumes an order: at init it records the
optimizer's DOF names (`_get_dex_output_joint_names`, `:321`) and builds an
index map to the layout's `finger_joint_names` (`_build_finger_name_mapping`,
`:552`); at runtime `_assign_hand_fingers` (`:496`) scatters values through
that map into action indices 6–27 (single hand). Joints the optimizer doesn't
produce fall back to 0. A positional `finger_permutation` path exists as a
fallback for layouts without names.

---

## 4. Stage C — assembly and application

### 4.1 One `retarget()` call, one action vector

`retarget()` (`simple_relative_retargeting.py:164`) per frame:

1. Pull each tracked hand's payload out of the device dict
   (`{TrackingTarget.HAND_LEFT: {...}, HAND_RIGHT: {...}}`) and cache the wrist
   poses (this is also what calibration snapshots later).
2. Update the three debug visualizations.
3. Allocate the action: `cfg.default_command` if configured, else zeros of
   `output_dim`.
4. For each hand: write the wrist command (Stage A), then run the finger solve
   and scatter its output (Stage B).
5. Return `torch.tensor(action, device=sim_device)`.

The action layout is data, not code — a dict on the robot's module, resolved by
`robot_type` string (`_get_action_layout`, `:218`). Concretely for
`floating_shadow_right` (`shadow/floating.py:116-129`):

```
index   0  1  2 | 3    4     5    | 6 ............ 27
        tx ty tz| yaw  pitch roll | FFJ1..4, MFJ1..4, RFJ1..4, LFJ1..5, THJ1..5
```

and for `floating_shadow_bimanual` (`:282-304`) the same blocks packed as
`[R wrist 0–5 | L wrist 6–11 | R fingers 12–33 | L fingers 34–55]` → 56-D. One
retargeter instance handles both hands; adding a robot means adding a layout
dict + dex spec to its module, no retargeter changes.

### 4.2 What the environment does with it

The Isaac Lab action manager splits the vector into three `JointPositionAction`
terms (`FloatingShadowRightAbsJointPosActionCfg`, `shadow/floating.py:92-113`):
translation joints, rotation joints (declared `[z, y, x]` with
`preserve_order=True` to match `yaw_pitch_roll`), and the 22 finger joints
(`preserve_order=True` to match the layout tuple). Each applies
`target = action + default_joint_pos` and hands targets to implicit PD
actuators — wrist: stiffness 2000 / damping 400; fingers: stiffness 10 /
damping 0.1 (`:171-228`). So the "control law" of the whole pipeline is simply
PD tracking of retargeted targets; compliance and smoothness come from the
gains and the 60 Hz update (`sim.dt = 1/120`, `decimation = 2`).

### 4.3 The session loop

`scripts/teleop_agent.py` (same skeleton in `record_demos.py`):

```
while running:
    action = teleop_interface.advance()     # OpenXRDevice: raw data → retarget()
    if teleop_active:
        env.step(action)                    # 60 Hz
    else:
        env.sim.render()                    # keep the XR stream alive while paused
```

Gestures/buttons on the headset fire callbacks: **START** activates stepping
*and* calls `calibrate_wrist_pose()` (`teleop_agent.py:410-417`) — so the
relative wrist zero is re-taken every time you resume; **STOP** pauses;
**RESET** (in the recording script) discards the current episode.

---

## 5. Gotchas checklist

- **wxyz vs xyzw**: every SciPy boundary converts. If you extend this code and
  see a hand rotated 180°-ish in a weird axis, check quaternion order first.
- **Misleading variable names** in `_get_normalized_wrist_rotation`
  (`x_plus_90` holds a −90° rotation). Trust `R.from_euler("x", -90)`.
- **Row-vector `@ M` applies `M⁻¹`** to points. Both uses in
  `_convert_hand_to_canonical_joint_positions` are basis changes.
- **`use_default_offset=True`** makes all wrist commands *displacements from
  home*. If you switch an action cfg to `use_default_offset=False`, absolute
  mode's origin math and relative mode's zero-at-home semantics both break.
- **Joint-order contracts**: retargeter `yaw_pitch_roll` ↔ action cfg
  `["z_rotation_joint","y_rotation_joint","x_rotation_joint"] + preserve_order`;
  dex output is Pinocchio DOF order and must go through the name map. Never
  reorder one side alone.
- **Euler gimbal lock** at pitch ±90° in both wrist modes.
- **`FINGER_Z_ROTATION_DEG` is empirical**, per-hand, and only touches fingers.
  New hand embodiment → expect to retune it (or eliminate it by fixing the URDF
  base orientation).
- **`torch.inference_mode`**: any code path that calls the dex optimizer must
  re-enable grad exactly as `:547-549` does, or NLopt gets zero gradients.
- **Calibrate before moving** (relative mode): the base pose defaults to
  identity until the first START, producing a large initial command if you skip it.

## 6. Where this differs from our RoboLab/SharpaWave pipeline (one paragraph)

Same substrate (AVP → CloudXR → Isaac Lab `OpenXRDevice` → `RetargeterBase` →
`env.step`), same finger machinery (dex-retargeting DexPilot, name-based
scatter, grad-mode escape hatch). The divergence is the wrist: DexVerse defaults
to *calibrated relative displacement written directly into virtual wrist joints*
(no IK, floating hand), while our branch does *absolute wrist pose with
analytically derived rig offsets, solved by differential IK on real arm chains*
(`scripts/environments/teleoperation/robolab/sharpa_duo_retargeters.py`,
`SHARPA_DUO_NOTES.md`). DexVerse's `quat_absolute` hooks are the unshipped
equivalent of what our branch ships.

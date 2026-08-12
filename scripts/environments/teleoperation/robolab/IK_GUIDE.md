# IK in the TACO Teleop Scene — A Step-by-Step Guide

This document explains every inverse-kinematics (IK) computation that happens when you
teleoperate the TACO scene (`teleop_taco_scene.py`) with the Apple Vision Pro. There are
**two different IK problems** solved on every control step, plus one non-IK mapping that
glues them together:

1. **Arm IK** — *differential IK with damped least squares*: given a desired 6-DoF pose for
   each Panda flange, find the 7 arm joint angles. Solved by Isaac Lab's
   `DifferentialIKController`, closed-loop, one linearized step per control tick.
2. **Hand IK ("dex retargeting")** — *nonlinear optimization*: given 21 tracked human hand
   points, find the 22 SharpaWave finger joint angles whose fingertips *geometrically imitate*
   the human hand. Solved by the `dex_retargeting` library's DexPilot optimizer (SLSQP).
3. **Wrist mapping** — *not IK at all*: the human wrist pose is passed through with a constant
   calibrated rotation offset. It becomes the *target* that arm IK then solves for.

They are **deliberately decoupled**: the arm IK never sees finger joints, and the hand
retargeting is computed entirely in a wrist-local frame so it doesn't care where the arm is.
The only coupling is physical — the hand rides on the flange the arm IK placed.

Throughout, quaternions are **(w, x, y, z)** and poses are 7-vectors `[x y z | qw qx qy qz]`
unless stated otherwise.

---

## 0. Background you need (2-minute primer)

**Forward kinematics (FK).** For a serial chain with joint vector $q \in \mathbb{R}^n$, FK is a
function $x = f(q)$ giving the pose $x \in SE(3)$ of some link. FK is easy (multiply
transforms down the chain); IK — inverting $f$ — is the hard direction (nonlinear, multiple or
no solutions).

**The geometric Jacobian.** $J(q) = \partial f / \partial q \in \mathbb{R}^{6 \times n}$ maps joint
velocities to the link's spatial velocity (twist):

$$\begin{bmatrix} v \\ \omega \end{bmatrix} = J(q)\,\dot q$$

Rows 0–2 are linear velocity, rows 3–5 angular. Crucially, for *small* displacements the same
linear map approximates pose changes: $\Delta x \approx J(q)\,\Delta q$. All of differential IK
lives inside this approximation.

**The two ways to use it.** You can either (a) iterate $\Delta q = J^{\dagger} \Delta x$ to
convergence for a one-shot IK solve, or (b) take **one** step per control tick and let the
feedback loop (you re-measure the real robot state next tick) absorb the linearization error.
This scene does (b) for the arms. The hand side instead solves a full nonlinear program each
tick, because finger retargeting is not a "reach this exact pose" problem.

**Frames used in this pipeline.**

| Frame | What it is | Who uses it |
|---|---|---|
| *world* (`_w`) | Isaac stage origin | XR tracking data arrives here (anchor already applied) |
| *root* (`_b`) | the robot torso (`init_state`: pos (0, −0.7, 1.0), +90° yaw) | arm IK commands, error, Jacobian |
| *wrist-local MANO* | human wrist at origin, MANO axis convention | hand retargeting input |
| *flange* (`panda_link8`) | last arm link, the hand bolts onto it (∓45° mounts) | arm IK target body |

---

## 1. The data path of one control step (bird's-eye view)

```
 Vision Pro hand tracking (26 joints/hand, world frame, 7D poses)
        │  teleop.advance()                          teleop_taco_scene.py:331
        ▼
 FrankaDuoSharpaRetargeter.retarget()          sharpa_duo_retargeters.py:159
        │
        ├─ wrist passthrough + constant offsets  → left/right flange pose, WORLD frame (§4)
        └─ DexPilot NLP per hand                 → 22 finger angles per hand (§3)
        ▼
 58-D action  [Lwrist 7 | Rwrist 7 | Lfingers 22 | Rfingers 22]
        │  to_root_frame()                           teleop_taco_scene.py:254
        ▼
 same 58-D action, wrist poses now in ROBOT ROOT frame
        │  env.step(action)                          teleop_taco_scene.py:335
        ▼
 ActionManager splits by declaration order     franka_duo_sharpa_wave.py:270 (FrankaDuoSharpaIKActionCfg)
        ├─ action[0:7]   → left_arm  DiffIK  → 7 left arm joint targets   (§2)
        ├─ action[7:14]  → right_arm DiffIK  → 7 right arm joint targets  (§2)
        ├─ action[14:36] → left_hand  JointPositionAction → direct targets
        └─ action[36:58] → right_hand JointPositionAction → direct targets
        ▼
 PD drives (stiffness 400 / damping 80 arms; USD gains fingers)
        ▼
 4 PhysX substeps at dt = 1/120  (decimation = 4 → 30 Hz control loop)
```

Key timing fact (teleop_taco_scene.py:203–205): the **control step** is 30 Hz. Inside one
`env.step()`, `process_actions` runs **once**, then `apply_actions` + one physics substep run
**4 times** (`decimation = 4`).

---

## 2. Arm IK — differential IK with damped least squares

### 2.1 Problem statement

**Input:** desired flange pose in the robot root frame,
$x_{des} = (p_{des}, \hat q_{des}) \in \mathbb{R}^3 \times S^3$ — this is `action[0:7]` (left) /
`action[7:14]` (right) after `to_root_frame`.

**Output:** a joint-position *target* $q_{des} \in \mathbb{R}^7$ for the arm's PD drives.

Configured in `FrankaDuoSharpaIKActionCfg` (franka_duo_sharpa_wave.py:278–294): per arm, a
`DifferentialInverseKinematicsActionCfg` with `body_name="{left,right}_panda_link8"`,
`command_type="pose"`, `use_relative_mode=False` (targets are **absolute** poses, right for
teleop — your hand position *is* the target, not a delta), and `ik_method="dls"`.

### 2.2 The math

Differential IK linearizes FK around the current configuration. Per solve:

**Step 1 — measure the current pose.** `_compute_frame_pose`
(task_space_actions.py:224–243) reads the flange's world pose from the sim and converts it to
the root frame:

$$p_b = R(\hat q_{root})^{-1}(p_w - p_{root}), \qquad \hat q_b = \hat q_{root}^{-1} \otimes \hat q_w$$

(that's `subtract_frame_transforms`; the same math your script's `to_root_frame` applies to the
*command*, so both sides of the error live in the same frame).

**Step 2 — pose error.** `DifferentialIKController.compute`
(differential_ik.py:169–172) builds a 6-vector error:

$$e = \begin{bmatrix} p_{des} - p_b \\ \mathrm{axisangle}\!\left(\hat q_{des} \otimes \hat q_b^{-1}\right) \end{bmatrix} \in \mathbb{R}^6$$

The rotation part converts the *error quaternion* to an axis–angle vector (direction = rotation
axis, magnitude = rotation angle in radians). This is the standard way to make orientation error
compatible with the angular-velocity rows of the geometric Jacobian.

**Step 3 — the Jacobian.** Not computed by Isaac Lab — **read back from PhysX**, which
maintains it analytically. `jacobian_w` (task_space_actions.py:144–145) slices the full
articulation Jacobian down to (6 × 7): rows for the flange body, columns for that arm's 7
joints only. Two subtleties worth understanding:

- *Fixed-base index shift* (task_space_actions.py:77–79): PhysX doesn't produce a Jacobian row
  block for the root link of a fixed-base articulation, so body index `i` in Isaac Lab maps to
  Jacobian body `i − 1`.
- *Frame rotation* (`jacobian_b`, task_space_actions.py:148–154): PhysX gives $J$ in the world
  frame; both 3-row blocks are rotated by $R(\hat q_{root})^{-1}$ so $J$ matches the root-frame
  error: $J_b = \mathrm{diag}(R^{-1}, R^{-1}) J_w$.
- The finger joints are **not in the columns** — the arm IK is blind to the hand. (The fingers
  do shift the arm's *dynamics*, but the PD loop handles that; IK is purely kinematic.)

**Step 4 — damped least squares (DLS).** `_compute_delta_joint_pos`
(differential_ik.py:228–237) computes:

$$\Delta q = J^\top \left(J J^\top + \lambda^2 I_6\right)^{-1} e, \qquad \lambda = 0.01$$

($\lambda$ is the default `lambda_val`, differential_ik_cfg.py:64.)

Why not the plain pseudo-inverse $J^\dagger e$? Near a **singularity** (e.g. elbow fully
extended), $JJ^\top$ becomes ill-conditioned and $J^\dagger$ commands enormous joint velocities.
DLS is the closed-form solution of the *regularized* problem

$$\min_{\Delta q}\; \|J \Delta q - e\|^2 + \lambda^2 \|\Delta q\|^2,$$

which trades a little tracking accuracy for bounded joint motion everywhere. This is also called
Levenberg–Marquardt; $\lambda$ is the damping.
With 7 joints and a 6-D task there's a 1-D **null space** (redundancy); DLS just picks the
minimum-norm $\Delta q$ — nothing in this pipeline actively steers the elbow.

**Step 5 — target.** $q_{des} = q + \Delta q$ (differential_ik.py:175), then
`set_joint_position_target` hands it to the implicit PD actuators
(τ = 400·(q_des − q) − 80·q̇, from the `shoulders`/`forearms` actuator groups). Note the arm
does **not** teleport: one tick moves it a *fraction* of the way, and because next tick
re-measures everything, the loop converges like a first-order servo. That's what "differential"
buys you: robustness for the price of a small lag.

### 2.3 Where it runs: the `OncePerStepDiffIKAction` override

Stock Isaac Lab (task_space_actions.py:191–215) puts `set_command` in `process_actions` (once
per control step) but the *solve* in `apply_actions` — which runs **every physics substep**.
At decimation 4 with two arms that's 8 solves + 8 GPU Jacobian readbacks per control step, and
each solve differs only microscopically (the arm barely moves in 1/120 s).

`OncePerStepDiffIKAction` (robolab_teleop_common.py:15–40), swapped in by
`teleop_taco_scene.py`'s `__post_init__` (line 213), moves the solve into `process_actions`
(once, against the freshest state) and has `apply_actions` merely re-issue the cached
$q_{des}$ to the drives each substep. Same math, ~4× fewer solves (measured ~65 ms/step
saved on this scene at decimation 8).

---

## 3. Hand IK — DexPilot retargeting for the fingers

### 3.1 Why this is a different problem

You cannot differential-IK the fingers to "match the human hand": the SharpaWave has different
segment lengths, different joint axes, 22 DoF vs. the human hand's ~27, and there is no single
"end effector" — the *shape* of the whole hand is the target. **Retargeting** reframes it:
choose a set of geometric *features* (here: relative vectors between keypoints), and find robot
joint angles whose FK reproduces the human's features. That's a nonlinear least-squares problem
solved fresh each frame, warm-started from the last solution.

The implementation is the `dex_retargeting` library (inside the container at
`_isaac_sim/kit/python/lib/python3.11/site-packages/dex_retargeting/`), method **DexPilot**
(Handa et al. 2020, [arXiv:1910.03135](https://arxiv.org/abs/1910.03135)), configured per hand
by `sharpa_dex_retargeting/sharpa_wave_{left,right}_dexpilot.yml`, driven by
`SharpaWaveDexRetargeting` (sharpa_duo_retargeters.py:59).

### 3.2 Input preprocessing: 26 OpenXR joints → 21 wrist-local MANO points

`_convert_hand_joints` (sharpa_duo_retargeters.py:78–87) does three things:

1. **Subsample**: OpenXR reports 26 joints; `_HAND_JOINTS_INDEX` (line 50) keeps the 21
   matching the MANO hand-model ordering (wrist + 4 joints per finger), dropping the palm and
   the four `*_metacarpal` joints.
2. **Translate**: subtract the wrist position — all points become wrist-relative. This is what
   makes hand retargeting independent of where your arm is.
3. **Rotate**: `joint_position @ wrist_rot @ _OPERATOR2MANO` — right-multiplying row-vectors by
   $R_{wrist}$ expresses the points in the wrist's local axes (undoes your hand's orientation
   in the room), then `_OPERATOR2MANO` (line 52) permutes axes into MANO's convention, which is
   what the hand URDF (`right_sharpa_wave_with_flange.urdf`, rooted at `right_hand_wrist`) was
   aligned to.

**Output:** a (21 × 3) array; row 0 is the wrist at the origin; rows 4, 8, 12, 16, 20 are the
five fingertips.

### 3.3 The feature vectors

For a 5-fingered hand, `DexPilotOptimizer.generate_link_indices` (optimizer.py:363) builds
**15 vectors** between the keypoint set {wrist, thumb tip, index tip, middle tip, ring tip,
pinky tip}:

| Group | Count | Vectors | Role |
|---|---|---|---|
| **S1** | 4 | each finger tip → thumb tip | pinch detection & precision |
| **S2** | 6 | finger tip → finger tip (among the 4 non-thumb fingers) | inter-finger spacing |
| **base** | 5 | wrist → each fingertip | overall finger pose/curl |

Your wrapper computes the human versions in `_compute_one`
(sharpa_duo_retargeters.py:89–99): `ref_value = joint_pos[task] − joint_pos[origin]`, with the
index table coming from `optimizer.target_link_human_indices` (keypoint index × 4 = MANO row of
each fingertip). **Input to the optimizer: a (15 × 3) array of human feature vectors, meters,
wrist-local MANO frame.**

### 3.4 The DexPilot trick: projection with hysteresis

Plain vector-matching fails at the moment that matters most: a **pinch**. Tracking noise of
±5 mm is irrelevant when your fingers are 10 cm apart but fatal when you're trying to close a
2 mm gap on an object. DexPilot's fix (`get_objective_function`, optimizer.py:409–447): treat
"almost touching" as a discrete state and *snap* the target to contact.

Per frame, with $d_k = \|v_k^{human}\|$ for the 10 fingertip-pair vectors:

- **S1 switching (hysteresis)**: pair $k$ enters the *projected* state when $d_k < 0.03$ m
  (`project_dist`) and leaves it only when $d_k > 0.05$ m (`escape_dist`). The 2 cm dead band
  prevents flickering at the threshold (optimizer.py:419–420).
- **S2 switching**: a finger–finger pair is projected only if *both* fingers are already
  S1-projected against the thumb *and* $d_k \le 0.03$ (optimizer.py:421–426) — i.e. during a
  multi-finger pinch.
- **Reference vector** (what the robot should achieve), with scaling $s = 1.1$ from the yml
  (`scaling_factor`, compensating the SharpaWave being ~10% larger than a human hand):

$$r_k = \begin{cases} s \cdot v_k^{human} & \text{not projected — imitate the shape} \\ \eta \cdot \dfrac{v_k^{human}}{\|v_k^{human}\|} & \text{projected — snap: } \eta_1 = 10^{-4}\,\text{m (S1: touch)},\; \eta_2 = 3{\times}10^{-2}\,\text{m (S2: keep 3 cm apart)} \end{cases}$$

  So a projected thumb–finger pair is commanded to essentially **zero separation** (real
  contact, regardless of tracking noise), while projected finger–finger pairs are held 3 cm
  apart so the fingers don't crush into each other. The 5 wrist→fingertip vectors are never
  projected — they always track the scaled human shape.
- **Weights** $w_k$: 1 normally; **200** (S1) / **400** (S2) when projected; constant **15**
  for the five wrist→fingertip vectors (optimizer.py:428–437). Projection doesn't just change
  the target, it makes that target ~2 orders of magnitude more important than everything else.

### 3.5 The optimization problem

Putting it together (optimizer.py:450–509), each hand solves, at every control step:

$$\min_{q \in \mathbb{R}^{22}} \;\; \frac{1}{15}\sum_{k=1}^{15} w_k \, H_\beta\!\big(\|v_k^{robot}(q) - r_k\|\big) \;+\; \delta\,\|q - q_{last}\|^2 \quad \text{s.t.} \quad q_{lo} \le q \le q_{hi}$$

- $v_k^{robot}(q)$ = the same 15 vectors measured on the **robot** via Pinocchio FK of the hand
  URDF at candidate $q$ (fingertip link positions, `finger_tip_link_names` from the yml).
- $H_\beta$ = Huber loss (`torch.nn.SmoothL1Loss`, $\beta$ = `huber_delta` = 0.03): quadratic
  for errors < 3 cm, linear beyond — one badly-tracked finger can't hijack the whole solve.
- $\delta$ = `norm_delta` = 4×10⁻³: temporal smoothing toward last frame's solution. (In the
  code the quadratic term appears via its gradient, `grad += 2·norm_delta·(x − last_qpos)`,
  optimizer.py:505; the original DexPilot regularized toward the *open* hand instead.)
- **Solver**: NLopt **SLSQP** (sequential least-squares quadratic programming,
  optimizer.py:34), a gradient-based local NLP method, with box constraints = URDF joint limits
  (±10⁻³ slack) and `ftol_abs = 1e-6`.
- **Gradient**: analytic, not finite-differenced. Autograd differentiates the loss w.r.t. the
  12 keypoint positions; the chain rule closes through Pinocchio's per-link **position
  Jacobians** ($3 \times 22$ each, rotated into the frame the positions live in):
  $\partial L/\partial q = \sum_i (\partial L / \partial p_i) \, J_i(q)$ (optimizer.py:479–501).
  This is why `_compute_one` wraps the call in `torch.enable_grad()` — the teleop loop runs in
  `inference_mode`, which would silently break autograd.
- **Warm start**: `SeqRetargeting.retarget` (seq_retarget.py:108) starts SLSQP at
  $q_{last}$ (clipped to limits). Frame-to-frame hand motion is tiny, so the solver typically
  converges in a few iterations — this is what makes a 22-DoF NLP per hand per frame affordable
  (it's the `retarget` bucket in your `--profile` output).
- **Output filter**: an exponential moving average `LPFilter` (optimizer_utils.py:1),
  $y \leftarrow y + \alpha(x - y)$ with $\alpha$ = `low_pass_alpha` = 0.2 — heavy smoothing:
  each new solve contributes 20%, so a step change reaches ~63% in 5 frames (~0.36 s of lag at
  the observed ~14 Hz retarget rate). Raise $\alpha$ for snappier fingers, lower for less jitter.

**Output:** 22 joint angles in the URDF's DoF order (`left/right_dof_names`), which
`FrankaDuoSharpaRetargeter.retarget` scatters into RoboLab's `HAND_JOINTS_ORDERED` action
order by name (`_left_scatter`/`_right_scatter`, sharpa_duo_retargeters.py:119–120) — never
trust two systems to order joints identically; match by name.

These angles bypass IK entirely on the Isaac side: `left_hand`/`right_hand` are plain
`JointPositionActionCfg` terms (franka_duo_sharpa_wave.py:296–308) that feed them straight to
the finger PD drives as absolute targets (`use_default_offset=False`).

### 3.6 What the hand IK does *not* do

It does not see the scene (no collision avoidance with the table or objects — contact is left
to PhysX), does not enforce velocity limits (the LP filter is the only rate limiting), and does
not know about the arm. It is pure geometric imitation of the *hand shape*.

---

## 4. The wrist mapping — the glue (and why it isn't IK)

`FrankaDuoSharpaRetargeter._wrist_pose` (sharpa_duo_retargeters.py:146) turns each tracked
wrist pose into the arm-IK target with **constant-offset arithmetic** — no solving:

$$\hat q_{flange} = \hat q_{wrist}^{XR} \otimes \hat q_{offset}, \qquad p_{flange} = p_{wrist}^{XR} - R(\hat q_{flange})\, t_{offset}$$

- $\hat q_{offset}$ (per side, `left/right_wrist_rot_offset` in the cfg, sharpa_duo_retargeters.py:185)
  maps the OpenXR wrist axes onto the `panda_link8` flange axes. The values
  $(0, \mp0.924, \pm0.383, 0)$ are 180° flips composed with ±22.5°-structured terms — the
  algebraic fingerprint of the rig's ∓45° flange mounts; they were validated by
  `calibrate_sharpa_duo.py` at the ready pose.
- $t_{offset}$ (`wrist_pos_offset`) would pull the target back so the robot's *palm* (not its
  flange) lands on your palm; calibration measured flange ≈ hand-wrist (0.5 mm), so it's zero.

Right-multiplication is the key convention to internalize: $\hat q_{wrist} \otimes \hat q_{offset}$
applies the offset in the **wrist's local frame** — "the flange is rotated this way *relative
to my hand*" — which is exactly what a rigid mount is.

Then the script's `to_root_frame` (teleop_taco_scene.py:254) converts both wrist targets from
world to robot-root frame with `subtract_frame_transforms` (§2.2 Step 1's equation) — required
because the XR anchor stands you *in front of* the robot, and the DiffIK action interprets
commands in the root frame. This is also why the anchor yaw **must equal** the robot root yaw
(both +90°): the rotation offsets were calibrated in a frame where those coincide.

---

## 5. One control step, end to end (with shapes)

At 30 Hz, with you mid-pinch over the bowl:

| # | Stage | Code | Input → Output |
|---|---|---|---|
| 1 | XR poll | `teleop.advance()` → `OpenXRDevice` | — → 2 × dict of 26 joint poses (7,), world frame |
| 2 | Wrist map | `_wrist_pose` ×2 | wrist (7,) → flange target (7,), world, **O(1) math** |
| 3 | Hand preprocess | `_convert_hand_joints` ×2 | 26 poses → (21×3) wrist-local MANO points |
| 4 | Hand NLP | `SeqRetargeting.retarget` ×2 | (15×3) vectors → (22,) angles; SLSQP, warm-started, EMA-filtered |
| 5 | Assemble | `retarget` | → (58,) action |
| 6 | Reframe | `to_root_frame` | wrists world → root frame; fingers untouched |
| 7 | Arm IK ×2 | `OncePerStepDiffIKAction.process_actions` | target (7,) + $J$ (6×7) + $q$ (7,) → $q_{des}$ (7,); **one DLS step** |
| 8 | Apply ×4 substeps | `apply_actions` | cached $q_{des}$ (14,) + finger targets (44,) → PD drives |
| 9 | Physics ×4 | PhysX, dt = 1/120 | torques → motion; contacts happen here, not in any IK |

Steps 2–4 are "the retargeter" (your `retarget` profiler bucket, and where the red-ball
markers are drawn from raw step-1 data); step 7 is the `ik` bucket; steps 8–9 live in `step`.

The two IKs never exchange information within a step — yet the system tracks your hand because
*both* loops are closed at 30 Hz against ground truth: the arm servo re-measures the flange
every tick, and your own eyes close the outermost loop through the headset.

---

## 6. Knobs, failure modes, and where to look

| Symptom | Likely place | Knob |
|---|---|---|
| Arm lags / overshoots | §2.5 PD + one-step IK | actuator `stiffness`/`damping`; DLS `lambda_val` (in `DifferentialIKControllerCfg(ik_params={"lambda_val": ...})`) |
| Arm goes weird near reach limit | singularity | that's DLS damping working; keep targets inside the workspace |
| Hands twisted by a constant angle | §4 offsets | recalibrate `left/right_wrist_rot_offset` (`calibrate_sharpa_duo.py`) |
| Fingers jittery ↔ fingers laggy | §3.5 filter | `low_pass_alpha` in the ymls (up = snappy, down = smooth) |
| Pinch won't close / closes too eagerly | §3.4 | `project_dist` / `escape_dist` (DexPilotOptimizer args; would need plumbing through the yml) |
| Robot fingers curl too much/little | scale mismatch | `scaling_factor` in the ymls (1.1 now) |
| Everything slow | solve frequency | `--profile` buckets (retarget / ik / physx / render_call) |

**Primary sources:** Isaac Lab DLS controller `source/isaaclab/isaaclab/controllers/differential_ik.py`;
action term `source/isaaclab/isaaclab/envs/mdp/actions/task_space_actions.py`;
DexPilot `dex_retargeting/optimizer.py` (in-container site-packages) and the DexPilot paper
(arXiv:1910.03135); Buss, *Introduction to Inverse Kinematics* (the DLS reference the Isaac Lab
docstring cites).

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Free-floating SharpaWave hands for the teleop "adjust object" mode.

While adjust mode is open the full arm rig is parked away and the operator
manipulates objects with two hand-only articulations instead: each hand's
6-DoF base kinematically follows the tracked wrist (no arm kinematics limiting
wrist rotation), while the 22 finger joints run the same PD drives, the same
retargeted targets, and therefore the same force limits as ordinary teleop.
Objects still move only through compliant hand contact — the property that
makes teleop stable — just without an arm in the way.

The base is written once per physics substep, interpolated across the control
step, so contacts see 240 Hz-sized increments instead of one 60 Hz teleport.
The wrist mapping reuses the teleop retargeting stack: the calibrated
OpenXR-wrist correction composed with the SharpaWave wrist offset
(``q_target = q_wrist ⊗ q_corr ⊗ q_offset``), identical on both embodiments
because the offset is a property of the hand alone.

Outside adjust mode both hands sit parked at :data:`duo_robot.FLOATING_HAND_PARK`
(far from the workspace, above the floor half-space) and render-hidden.

Import only after AppLauncher.
"""

from __future__ import annotations

import torch

from duo_robot import _SHARPA_WRIST_OFFSETS, FINGER_JOINTS, FLOATING_HAND_PARK, sided
from scipy.spatial.transform import Rotation as R
from sharpa_retargeting import load_hand_calibration, wrist_correction

from isaaclab.utils.math import combine_frame_transforms, quat_mul, subtract_frame_transforms

# OpenXR wrist joint in the 26-joint layout of the ``xr_hands`` block.
_WRIST_JOINT = 1

_SIDES = ("left", "right")

_ROBOT_PARK_LIFT_M = 6.0
"""How far straight up the rig is parked while adjust mode is open [m].

Up rather than down: scene floors are PhysX half-space planes, so parking
below them would put every link in deep penetration.
"""

_MAX_LIN_SPEED_MPS = 1.5
"""Cap on the base's commanded translation speed in free space [m/s].

The base is kinematic — infinitely strong — so an uncapped swing rams objects
at full hand speed and launches them. The cap makes the hand lag a fast sweep
and catch up, while normal repositioning motion passes through unchanged.
"""

_MAX_ANG_SPEED_RADPS = 4.0
"""Cap on the base's commanded angular speed in free space [rad/s]."""

_NEAR_LIN_SPEED_MPS = 0.5
"""Tighter translation cap [m/s] while the hand is near a tracked object.

A kinematic palm meeting a light object transfers its full surface speed in
one substep, so contact speed IS the object's launch speed. Slowing down only
near the objects keeps travel fast and manipulation gentle.
"""

_NEAR_ANG_SPEED_RADPS = 2.5
"""Tighter angular cap [rad/s] near a tracked object."""

_NEAR_OBJECT_M = 0.20
"""Distance beyond an object's footprint radius counted as "near" for the caps."""

_ATTACH_CLEARANCE_M = 0.10
"""Extra clearance [m] beyond an object's footprint radius required around the
wrist target before a hand may (re)appear there. Appearing inside an object
would eject it at the depenetration cap — the classic fling after a voice
"reset" put the objects back under the operator's hovering hands."""


class FloatingHands:
    """Owns the adjust-mode embodiment swap: rig out, floating hands in.

    One instance per teleop session. :meth:`enter` parks the robot and arms the
    hands, :meth:`step` runs one control step of wrist tracking (including the
    physics substeps — the caller must NOT also step the env), :meth:`park_hands`
    stows the hands (used before object teleports), and :meth:`exit` restores
    the robot exactly where it stood.

    Args:
        env: The live :class:`~isaaclab.envs.ManagerBasedRLEnv` (with the two
            ``float_hand_*`` articulations in its scene).
        hand_calibration: Operator hand-shape calibration yml, as passed to the
            teleop pipeline; its rotation composes into the wrist mapping so the
            floating hands align with the operator's exactly like the rig does.
            None or "" runs uncalibrated.
        tracked_object_names: Names as returned by :func:`usda_scene.add_usda_scene`
            (without the ``object_`` prefix). Used only for the attach-clearance
            guard; None or empty disables it.
    """

    def __init__(self, env, hand_calibration: str | None = None, tracked_object_names: list[str] | None = None):
        self._env = env
        self._tracked = list(tracked_object_names or [])
        self._hands = {side: env.scene[f"float_hand_{side}"] for side in _SIDES}
        self._finger_ids = {
            side: self._hands[side].find_joints(sided(FINGER_JOINTS, side), preserve_order=True)[0] for side in _SIDES
        }
        self._wrist_body = {side: self._hands[side].body_names.index(f"{side}_hand_wrist") for side in _SIDES}
        # OpenXR wrist -> SharpaWave wrist rotation (xyzw), calibration folded in:
        # the same q_corr ⊗ q_offset the teleop pipeline composes wrist-side.
        calibration = load_hand_calibration(hand_calibration) if hand_calibration else None
        self._wrist_quat_offset: dict[str, torch.Tensor] = {}
        for side in _SIDES:
            offset = R.from_quat(_SHARPA_WRIST_OFFSETS[side])
            cal = (calibration or {}).get(side)
            if cal is not None:
                offset = R.from_matrix(wrist_correction(cal["rotation"])) * offset
            self._wrist_quat_offset[side] = torch.tensor(
                offset.as_quat(), dtype=torch.float32, device=env.device
            )
        # Constant wrist-body -> root-link transform per hand, from FK on first use.
        self._wrist_to_root: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None
        self._env0 = torch.tensor([0], device=env.device)
        self._active = False
        self._robot_state: dict[str, torch.Tensor] | None = None
        # Last written wrist-space target per hand (None = parked / tracking lost).
        self._last_target: dict[str, torch.Tensor | None] = {side: None for side in _SIDES}
        self._substep_count = 0
        # Render visibility handles (visuals only; physics is never toggled).
        from pxr import UsdGeom

        import isaaclab.sim as sim_utils

        stage = sim_utils.get_current_stage()
        self._hand_imageables = {
            side: UsdGeom.Imageable(stage.GetPrimAtPath(f"/World/envs/env_0/float_hand_{side}")) for side in _SIDES
        }
        self._robot_imageable = UsdGeom.Imageable(stage.GetPrimAtPath("/World/envs/env_0/robot"))
        for side in _SIDES:
            self._hand_imageables[side].MakeInvisible()

    def is_active(self) -> bool:
        """Whether the mode is currently open (robot parked, hands armed)."""
        return self._active

    # ------------------------------------------------------------------
    # Mode lifecycle
    # ------------------------------------------------------------------

    def enter(self) -> None:
        """Snapshot and park the robot; arm the hands (they appear on first track)."""
        if self._active:
            return
        self._ensure_wrist_to_root()
        robot = self._env.scene["robot"]
        self._robot_state = {
            "root_pose": torch.cat(
                [robot.data.root_pos_w.torch[0], robot.data.root_quat_w.torch[0]]
            ).clone(),
            "joint_pos": robot.data.joint_pos.torch.clone(),
            "joint_vel": robot.data.joint_vel.torch.clone(),
        }
        # Straight up and out of the way; joints held so the drives don't wander.
        park_pose = self._robot_state["root_pose"].clone()
        park_pose[2] += _ROBOT_PARK_LIFT_M
        robot.write_root_pose_to_sim_index(root_pose=park_pose.unsqueeze(0), env_ids=self._env0)
        if not robot.is_fixed_base:
            robot.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros(1, 6, device=self._env.device), env_ids=self._env0
            )
        robot.set_joint_position_target_index(target=self._robot_state["joint_pos"].clone())
        self._robot_imageable.MakeInvisible()
        self._last_target = {side: None for side in _SIDES}
        self._active = True

    def exit(self) -> None:
        """Park the hands and put the robot back exactly where :meth:`enter` found it."""
        if not self._active:
            return
        self.park_hands()
        robot = self._env.scene["robot"]
        state = self._robot_state
        if state is not None:
            robot.write_root_pose_to_sim_index(root_pose=state["root_pose"].unsqueeze(0), env_ids=self._env0)
            if not robot.is_fixed_base:
                robot.write_root_velocity_to_sim_index(
                    root_velocity=torch.zeros(1, 6, device=self._env.device), env_ids=self._env0
                )
            robot.write_joint_position_to_sim_index(position=state["joint_pos"])
            robot.write_joint_velocity_to_sim_index(velocity=state["joint_vel"])
            robot.set_joint_position_target_index(target=state["joint_pos"].clone())
        self._robot_imageable.MakeVisible()
        self._robot_state = None
        self._active = False

    def park_hands(self) -> None:
        """Stow both hands at their park spots, stopped dead, fingers open, hidden.

        Also called before adjust-mode object teleports, so a restored object is
        never dropped into the space a hand occupies.
        """
        device = self._env.device
        for side in _SIDES:
            hand = self._hands[side]
            pose = torch.tensor([*FLOATING_HAND_PARK[side], 0.0, 0.0, 0.0, 1.0], device=device)
            hand.write_root_pose_to_sim_index(root_pose=pose.unsqueeze(0), env_ids=self._env0)
            hand.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros(1, 6, device=device), env_ids=self._env0
            )
            zeros = torch.zeros(1, len(self._finger_ids[side]), device=device)
            hand.write_joint_velocity_to_sim_index(velocity=zeros, joint_ids=self._finger_ids[side])
            hand.set_joint_position_target_index(target=zeros, joint_ids=self._finger_ids[side])
            self._hand_imageables[side].MakeInvisible()
            self._last_target[side] = None

    # ------------------------------------------------------------------
    # Per-control-step update
    # ------------------------------------------------------------------

    def step(self, xr_hands: torch.Tensor, action: torch.Tensor, xr_controllers: torch.Tensor | None = None) -> None:
        """Track the wrists for one control step, including the physics substeps.

        The caller must not also call ``env.step`` for this frame — this method
        owns the decimation loop (mirroring ``_settle_adjust``), so no recorder
        fills and no episode timeout can fire while adjusting.

        Args:
            xr_hands: Shape ``(2, 26, 7)`` — ``[hand, joint, (px,py,pz,qx,qy,qz,qw)]``
                in the sim world frame. Untracked entries read as all zeros and
                freeze that hand in place (fingers hold their last targets too).
            action: The 58-D teleop action of this frame, world frame. Only the
                finger slices (14:36 left, 36:58 right) are read.
            xr_controllers: Shape ``(2, 11)`` controller block (see
                :mod:`xr_extras`), or None. A side whose controller is active is
                treated as untracked: holding a Quest controller replaces real
                hand tracking there, and the runtime's emulated hand data would
                otherwise drag the hand to a garbage pose (it "vanished" under
                the table).
        """
        env = self._env
        device = env.device
        hands_dev = xr_hands.to(device)
        action_dev = action.to(device)
        ctrl_dev = xr_controllers.to(device) if xr_controllers is not None else None
        decimation = max(1, int(env.cfg.decimation))
        step_dt = decimation * env.physics_dt

        # Wrist targets for this control step (None = hold in place).
        targets: dict[str, torch.Tensor | None] = {}
        starts: dict[str, torch.Tensor] = {}
        for idx, side in enumerate(_SIDES):
            wrist = hands_dev[idx, _WRIST_JOINT]
            controller_held = ctrl_dev is not None and float(ctrl_dev[idx, 7]) > 0.5
            if controller_held or float(wrist[:3].norm()) < 1e-6 or float(wrist[3:7].norm()) < 0.5:
                # Tracking lost (or replaced by a held controller, or the
                # sample has no valid quaternion): freeze. Re-pin at the last
                # pose with zero velocity — the base is a dynamic body, so a
                # stale finite-difference velocity (or a contact) would drift it.
                targets[side] = None
                prev = self._last_target[side]
                if prev is not None:
                    self._hands[side].write_root_pose_to_sim_index(root_pose=prev.unsqueeze(0), env_ids=self._env0)
                    self._hands[side].write_root_velocity_to_sim_index(
                        root_velocity=torch.zeros(1, 6, device=device), env_ids=self._env0
                    )
                continue
            quat = quat_mul(wrist[3:7].unsqueeze(0), self._wrist_quat_offset[side].unsqueeze(0))[0]
            off_pos, off_quat = self._wrist_to_root[side]
            root_pos, root_quat = combine_frame_transforms(
                wrist[:3].unsqueeze(0), quat.unsqueeze(0), off_pos.unsqueeze(0), off_quat.unsqueeze(0)
            )
            target = torch.cat([root_pos[0], root_quat[0]])

            prev = self._last_target[side]
            if prev is None:
                if not self._attach_clear(wrist[:3]):
                    targets[side] = None
                    continue  # stay parked until the wrist pulls clear of the objects
                # First sample after enter()/park/dropout: appear in place, stopped.
                self._hands[side].write_root_pose_to_sim_index(root_pose=target.unsqueeze(0), env_ids=self._env0)
                self._hands[side].write_root_velocity_to_sim_index(
                    root_velocity=torch.zeros(1, 6, device=device), env_ids=self._env0
                )
                self._hand_imageables[side].MakeVisible()
            else:
                # Rate-limit the base: it is kinematic (infinitely strong), so
                # an uncapped swing would ram objects at full hand speed —
                # and near the objects, contact speed IS launch speed, so the
                # caps tighten further there.
                if self._near_objects(target[:3]):
                    lin_cap, ang_cap = _NEAR_LIN_SPEED_MPS, _NEAR_ANG_SPEED_RADPS
                else:
                    lin_cap, ang_cap = _MAX_LIN_SPEED_MPS, _MAX_ANG_SPEED_RADPS
                delta = target[:3] - prev[:3]
                dist = float(delta.norm())
                max_step = lin_cap * step_dt
                if dist > max_step:
                    target[:3] = prev[:3] + delta * (max_step / dist)
                rot_prev = R.from_quat(prev[3:7].cpu().numpy())
                rotvec = (R.from_quat(target[3:7].cpu().numpy()) * rot_prev.inv()).as_rotvec()
                angle = float((rotvec**2).sum() ** 0.5)
                max_angle = ang_cap * step_dt
                if angle > max_angle:
                    limited = R.from_rotvec(rotvec * (max_angle / angle)) * rot_prev
                    target[3:7] = torch.tensor(limited.as_quat(), dtype=torch.float32, device=device)
                starts[side] = prev
            targets[side] = target
            self._last_target[side] = target
            # Fingers: reuse the DexPilot targets from the teleop action as-is.
            fingers = action_dev[14 + 22 * idx : 36 + 22 * idx].view(1, -1)
            self._hands[side].set_joint_position_target_index(target=fingers, joint_ids=self._finger_ids[side])

        # Constant root velocity over the control step (finite difference),
        # so contacts see a moving surface rather than a chain of teleports.
        velocities: dict[str, torch.Tensor] = {}
        for side, start in starts.items():
            target = targets[side]
            lin = (target[:3] - start[:3]) / step_dt
            rot_delta = R.from_quat(target[3:7].cpu().numpy()) * R.from_quat(start[3:7].cpu().numpy()).inv()
            ang = torch.tensor(rot_delta.as_rotvec() / step_dt, dtype=torch.float32, device=device)
            velocities[side] = torch.cat([lin, ang]).unsqueeze(0)

        render_interval = max(1, int(env.cfg.sim.render_interval))
        for k in range(decimation):
            frac = (k + 1) / decimation
            for side, start in starts.items():
                target = targets[side]
                pose = torch.empty(7, device=device)
                pose[:3] = torch.lerp(start[:3], target[:3], frac)
                # nlerp with sign alignment — the per-step rotation is small.
                q1 = target[3:7] if float(torch.dot(start[3:7], target[3:7])) >= 0.0 else -target[3:7]
                quat = torch.lerp(start[3:7], q1, frac)
                pose[3:7] = quat / quat.norm().clamp_min(1e-9)
                self._hands[side].write_root_pose_to_sim_index(root_pose=pose.unsqueeze(0), env_ids=self._env0)
                self._hands[side].write_root_velocity_to_sim_index(
                    root_velocity=velocities[side], env_ids=self._env0
                )
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            self._substep_count += 1
            if self._substep_count % render_interval == 0:
                env.sim.render()
            env.scene.update(env.physics_dt)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _near_objects(self, pos: torch.Tensor) -> bool:
        """Whether ``pos`` is within :data:`_NEAR_OBJECT_M` of any tracked object's footprint."""
        if not self._tracked:
            return False
        from usda_scene import FOOTPRINT_RADII  # noqa: PLC0415

        for name in self._tracked:
            key = f"object_{name}"
            if key not in self._env.scene.rigid_objects:
                continue
            obj_pos = self._env.scene[key].data.root_pos_w.torch[0]
            if float((pos - obj_pos).norm()) < FOOTPRINT_RADII.get(name, 0.05) + _NEAR_OBJECT_M:
                return True
        return False

    def _attach_clear(self, wrist_pos: torch.Tensor) -> bool:
        """Whether a hand may (re)appear at ``wrist_pos`` without hitting an object.

        Gates the first pose write after :meth:`enter` / :meth:`park_hands`:
        a hand materializing inside an object would eject it at the
        depenetration cap. The check is the wrist target against every tracked
        object's bounding sphere (footprint radius + :data:`_ATTACH_CLEARANCE_M`),
        so after a voice "reset" the hands stay stowed until the operator pulls
        clear of the restored layout.
        """
        if not self._tracked:
            return True
        from usda_scene import FOOTPRINT_RADII  # noqa: PLC0415

        for name in self._tracked:
            key = f"object_{name}"
            if key not in self._env.scene.rigid_objects:
                continue
            obj_pos = self._env.scene[key].data.root_pos_w.torch[0]
            if float((wrist_pos - obj_pos).norm()) < FOOTPRINT_RADII.get(name, 0.05) + _ATTACH_CLEARANCE_M:
                return False
        return True

    def _ensure_wrist_to_root(self) -> None:
        """Compute the constant wrist-body -> root-link transform per hand (FK, once).

        The wrist is mounted to the root link (the flange) through fixed joints,
        so the offset is configuration-independent; it maps a desired world
        wrist pose to the root pose the base write needs.
        """
        if self._wrist_to_root is not None:
            return
        self._wrist_to_root = {}
        for side in _SIDES:
            hand = self._hands[side]
            wid = self._wrist_body[side]
            off_pos, off_quat = subtract_frame_transforms(
                hand.data.body_pos_w.torch[0, wid].unsqueeze(0),
                hand.data.body_quat_w.torch[0, wid].unsqueeze(0),
                hand.data.root_pos_w.torch[0].unsqueeze(0),
                hand.data.root_quat_w.torch[0].unsqueeze(0),
            )
            self._wrist_to_root[side] = (off_pos[0].clone(), off_quat[0].clone())

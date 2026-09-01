# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""In-headset "adjust object" mode for the sharpa_duo teleop.

While the mode is open the operator repositions objects with two FREE-FLOATING
SharpaWave hands (see :mod:`floating_hands`): the arm rig is parked away and
each hand's base follows the tracked wrist directly, so nothing limits wrist
rotation — but the fingers keep the ordinary retargeting, contacts, and
actuator force limits, so an object still only ever moves because a compliant
robot hand pushed it. Whatever the recorder holds is discarded instead of
becoming a demo, and saying "done" turns the objects' resting poses into the
scene's new authored poses. Saying "reset" puts them back where they stood when
the mode opened.

Going through a robot hand is deliberate. An earlier version drove objects
straight from the tracked hand, which hung a rigid body off a noisy 240 Hz
position signal with no mass, damping or force limit in between; objects were
flung on contact. The floating hands keep the compliance that makes teleop
stable while dropping the arm kinematics that made rotation hard.

This class itself owns only the mode's bookkeeping and the panel pinch-taps;
hand tracking lives in :class:`floating_hands.FloatingHands`.

Persistence is a **sidecar** JSON next to the scene USDA (see
:func:`usda_scene.save_pose_overrides`); the USDA itself is never modified.
The next scene load automatically picks the sidecar up.

Import only after AppLauncher.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import torch

# OpenXR hand-joint indices in the 26-joint layout emitted by :mod:`xr_extras`.
# Only the two fingertips are read, and only to tap the in-headset panels.
_THUMB_TIP = 5
_INDEX_TIP = 10

_PINCH_ENGAGE_M = 0.03
"""Thumb-tip → index-tip distance below which a pinch is registered [m]."""

_PINCH_RELEASE_M = 0.06
"""Thumb-tip → index-tip distance above which a pinch releases [m]. Hysteresis
between engage and release avoids chatter — see :mod:`sharpa_retargeting`."""

_PRESS_DEPTH_M = 0.12
"""Tolerance [m] perpendicular to a panel for a pinch-tap to count as a press.

Buttons that report their panel normal are hit-tested anisotropically: tight
in the panel plane (the per-button radius) but slack in depth, because judging
distance to a floating billboard in a headset is hard. Buttons without a
normal fall back to a plain sphere test."""

_MISS_REPORT_RANGE_M = 0.35
"""A missed pinch is diagnosed on the console only when it lands within this
distance of some button — pinches are also the grasp gesture, and grasping an
object must not spam tap diagnostics."""

_TRIGGER_ENGAGE = 0.7
"""Controller trigger value above which a press is registered."""

_TRIGGER_RELEASE = 0.3
"""Controller trigger value below which the press releases (hysteresis)."""

_STICK_ENGAGE = 0.6
"""Thumbstick deflection beyond which a range step fires."""

_STICK_RELEASE = 0.4
"""Thumbstick deflection below which the axis re-arms (hysteresis)."""

_STICK_REPEAT_S = 0.30
"""Auto-repeat period [s] while a thumbstick axis stays deflected."""


class ObjectAdjuster:
    """Owns the "adjust object" mode lifecycle.

    One instance per teleop session. :meth:`enter` and :meth:`exit` bracket the
    mode and :meth:`step` runs every frame while it is open, dispatching panel
    pinch-taps and nothing else — object motion is the floating hands' job
    (:mod:`floating_hands`). All coordinate frames are the simulation world frame.

    Args:
        env: The live :class:`~isaaclab.envs.ManagerBasedRLEnv`.
        tracked_object_names: Names as returned by
            :func:`usda_scene.add_usda_scene` (without the ``object_`` prefix).
        scene_usda: Path to the scene USDA; the sidecar is written next to it.
    """

    def __init__(self, env, tracked_object_names: list[str], scene_usda: str):
        self._env = env
        self._names = list(tracked_object_names)
        self._scene_usda = scene_usda
        self._active = False
        # Per-hand "this pinch already dispatched", so one pinch fires one tap.
        self._button_consumed: list[bool] = [False, False]
        # Same, for the controllers' triggers (see :meth:`step_controllers`).
        self._trigger_consumed: list[bool] = [False, False]
        # Thumbstick auto-repeat state: next allowed fire time per controller
        # per axis (0 = x, 1 = y); None while the axis is centered (re-armed).
        self._stick_next_fire: list[list[float | None]] = [[None, None], [None, None]]
        # Optional in-VR pinch-tap plumbing: a callable returning the current
        # panel buttons, plus a callback fired on hit. See :meth:`set_button_dispatch`.
        self._panel_buttons_fn: Callable[[], list[tuple]] | None = None
        self._on_button_press: Callable[[str], None] | None = None
        self._button_hit_radius: float = 0.08
        # World poses at enter(), so :meth:`reset` can undo a session's edits.
        self._entry_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def set_button_dispatch(
        self,
        buttons_fn: Callable[[], list[tuple]] | None,
        on_press: Callable[[str], None] | None,
        hit_radius: float = 0.08,
    ) -> None:
        """Wire pinch-tap dispatch onto in-VR panel buttons.

        Args:
            buttons_fn: Callable returning ``[(kind, (x, y, z)), ...]``,
                ``[(kind, (x, y, z), radius), ...]`` or
                ``[(kind, (x, y, z), radius, normal), ...]`` in the world frame
                each frame. Entries with a ``normal`` are hit-tested in the
                panel plane with :data:`_PRESS_DEPTH_M` slack in depth; the
                others as plain spheres. ``None`` disables button dispatch.
            on_press: Called with ``kind`` whenever a pinch rising edge lands
                within a button's radius.
            hit_radius: Fallback radius [m] for entries that omit their own.
        """
        self._panel_buttons_fn = buttons_fn
        self._on_button_press = on_press
        self._button_hit_radius = float(hit_radius)

    # ------------------------------------------------------------------
    # Mode lifecycle
    # ------------------------------------------------------------------

    def enter(self) -> None:
        """Snapshot the current object poses and open the mode."""
        self._active = True
        self._button_consumed = [False, False]
        self._trigger_consumed = [False, False]
        self._stick_next_fire = [[None, None], [None, None]]
        self._entry_poses = self._snapshot_world()

    def reset(self) -> int:
        """Put every object back where it stood at :meth:`enter`, stopped dead.

        A pose write is right here: this is a teleport to a pose already known to
        be good, exactly what a scene reset does. Returns the number restored.
        """
        for name, (pos, quat) in self._entry_poses.items():
            self._teleport_object(name, pos, quat)
            self._zero_object_velocity(name)
        return len(self._entry_poses)

    def exit(self) -> int:
        """Save the sidecar and refresh the in-memory authored poses.

        The caller must let the scene settle first — poses are recorded exactly
        as they stand, so an object still moving would be written mid-flight and
        then reused as the centre of every future randomization. Returns the
        number of objects saved.
        """
        self._active = False
        overrides = self._current_poses_local()
        from usda_scene import save_pose_overrides  # noqa: PLC0415

        path = save_pose_overrides(self._scene_usda, overrides)
        print(f"[ADJUST] Saved authored poses for {len(overrides)} object(s) to {path}")

        # Refresh this session's cached authored pose. randomize_tracked_objects
        # reads cfg.init_state each call, so future resets perturb around the new
        # pose. default_root_pose is what reset_scene_to_default restores (the
        # --no_dr path); it was baked from cfg.init_state at init, so it has to
        # be refreshed too or plain resets would keep using the stale pose.
        for name, override in overrides.items():
            obj = self._env.scene[f"object_{name}"]
            obj.cfg.init_state.pos = tuple(override["pos"])
            obj.cfg.init_state.rot = tuple(override["rot"])
            try:
                obj.data.default_root_pose.torch[0] = torch.tensor(
                    [*override["pos"], *override["rot"]], dtype=torch.float32, device=self._env.device
                )
            except Exception as exc:
                print(f"[ADJUST] Could not refresh the default reset pose for '{name}': {exc}")
        return len(overrides)

    def is_active(self) -> bool:
        """Whether the mode is currently open."""
        return self._active

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def step(self, xr_hands: torch.Tensor | np.ndarray) -> None:
        """Dispatch panel pinch-taps.

        Deliberately the whole of it: objects are moved through the floating
        hands' contacts (:mod:`floating_hands`), so nothing here touches
        object state.

        Args:
            xr_hands: Shape ``(2, 26, 7)`` — ``[hand, joint, (px,py,pz,qx,qy,qz,qw)]``
                in the sim world frame. Untracked entries read as all zeros.
        """
        if not self._active or self._panel_buttons_fn is None:
            return
        hands = xr_hands.detach().cpu().numpy() if isinstance(xr_hands, torch.Tensor) else np.asarray(xr_hands)

        for hand in range(2):
            thumb, index = hands[hand][_THUMB_TIP, :3], hands[hand][_INDEX_TIP, :3]
            if np.linalg.norm(thumb) < 1e-6 or np.linalg.norm(index) < 1e-6:
                self._button_consumed[hand] = False  # tracking lost
                continue
            pinch = float(np.linalg.norm(thumb - index))
            if pinch > _PINCH_RELEASE_M:
                self._button_consumed[hand] = False
                continue
            if self._button_consumed[hand] or pinch >= _PINCH_ENGAGE_M:
                continue
            # Rising edge. Consume it either way, so one pinch is one tap.
            self._button_consumed[hand] = True
            self._handle_tap(0.5 * (thumb + index))

    def step_controllers(self, xr_controllers: torch.Tensor | np.ndarray) -> None:
        """Dispatch Quest-controller input: thumbstick range steps and trigger-taps.

        The primary use is the THUMBSTICK, which steps the randomization
        ranges directly — no aiming needed, from either controller:
        left/right = ``xy_range`` down/up, up/down = ``yaw_range`` up/down,
        auto-repeating while held. Each step goes through the same dispatch as
        the panel's ``-``/``+`` buttons, so the panel display updates
        identically. A trigger press with the controller tip at a panel key
        also works, as the controller twin of a pinch-tap.

        Args:
            xr_controllers: Shape ``(2, 11)`` — ``[controller, (aim px, py, pz,
                qx, qy, qz, qw, valid, trigger, thumb_x, thumb_y)]`` in the sim
                world frame. An absent controller reads as all zeros.
        """
        if not self._active or self._panel_buttons_fn is None:
            return
        ctrl = (
            xr_controllers.detach().cpu().numpy()
            if isinstance(xr_controllers, torch.Tensor)
            else np.asarray(xr_controllers)
        )
        now = time.monotonic()
        for hand in range(2):
            if ctrl[hand, 7] < 0.5:  # controller absent / aim pose invalid
                self._trigger_consumed[hand] = False
                self._stick_next_fire[hand] = [None, None]
                continue
            # Thumbstick → range steps (x: xy_range, y: yaw_range).
            for axis, (dec_kind, inc_kind) in enumerate((("xy_dec", "xy_inc"), ("yaw_dec", "yaw_inc"))):
                deflection = float(ctrl[hand, 9 + axis])
                if abs(deflection) < _STICK_RELEASE:
                    self._stick_next_fire[hand][axis] = None  # re-armed
                    continue
                if abs(deflection) < _STICK_ENGAGE:
                    continue
                next_fire = self._stick_next_fire[hand][axis]
                if next_fire is not None and now < next_fire:
                    continue
                self._stick_next_fire[hand][axis] = now + _STICK_REPEAT_S
                if self._on_button_press is not None:
                    self._on_button_press(inc_kind if deflection > 0.0 else dec_kind)
            # Trigger → tap at the controller tip.
            trigger = float(ctrl[hand, 8])
            if trigger < _TRIGGER_RELEASE:
                self._trigger_consumed[hand] = False
                continue
            if self._trigger_consumed[hand] or trigger < _TRIGGER_ENGAGE:
                continue
            # Rising edge. Consume it either way, so one press is one tap.
            self._trigger_consumed[hand] = True
            self._handle_tap(ctrl[hand, :3].astype(np.float64))

    def _handle_tap(self, world_xyz: np.ndarray) -> None:
        """Hit-test one tap point and dispatch it, or diagnose a near-miss."""
        kind, nearest = self._button_hit(world_xyz)
        if kind is not None:
            if self._on_button_press is not None:
                self._on_button_press(kind)
        elif nearest is not None and nearest[1] < _MISS_REPORT_RANGE_M:
            # Diagnose near-misses so "my taps do nothing" is self-explaining;
            # taps far from every button are grasps, not failed presses.
            name, _, in_plane, depth = nearest
            print(f"[ADJUST] tap missed: nearest '{name}' in-plane {in_plane:.2f} m, depth {depth:.2f} m")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _button_hit(self, world_xyz: np.ndarray) -> tuple[str | None, tuple[str, float, float, float] | None]:
        """Hit-test ``world_xyz`` against every panel button.

        Entries carrying a panel normal are tested anisotropically — the button
        radius applies in the panel plane, :data:`_PRESS_DEPTH_M` in depth —
        because depth is what operators misjudge on a floating billboard.

        Returns:
            ``(hit, nearest)``: the pressed button kind (or None), and the
            overall nearest button as ``(kind, distance, in_plane, depth)`` for
            miss diagnostics (None when there are no buttons).
        """
        best: tuple[float, str | None] = (float("inf"), None)
        nearest: tuple[str, float, float, float] | None = None
        for entry in self._panel_buttons_fn():
            kind, pos = entry[0], entry[1]
            # Panels may size their own keys; fall back to the dispatch default.
            radius = float(entry[2]) if len(entry) > 2 else self._button_hit_radius
            offset = np.asarray(pos, dtype=np.float64) - world_xyz
            distance = float(np.linalg.norm(offset))
            if len(entry) > 3:
                normal = np.asarray(entry[3], dtype=np.float64)
                depth = abs(float(offset @ normal))
                in_plane = float(np.linalg.norm(offset - (offset @ normal) * normal))
                hit = in_plane < radius and depth < _PRESS_DEPTH_M
            else:
                in_plane, depth = distance, 0.0
                hit = distance < radius
            if nearest is None or distance < nearest[1]:
                nearest = (kind, distance, in_plane, depth)
            if hit and in_plane < best[0]:
                best = (in_plane, kind)
        return best[1], nearest

    def _snapshot_world(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """World-frame ``(position, quaternion xyzw)`` for every tracked object."""
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in self._names:
            if f"object_{name}" not in self._env.scene.rigid_objects:
                continue
            data = self._env.scene[f"object_{name}"].data
            out[name] = (
                data.root_pos_w.torch[0].detach().cpu().numpy().copy(),
                data.root_quat_w.torch[0].detach().cpu().numpy().copy(),
            )
        return out

    def _current_poses_local(self) -> dict[str, dict[str, list[float]]]:
        """Every tracked object's current pose, in the env-local frame."""
        origin = self._env.scene.env_origins[0].detach().cpu().numpy()
        return {
            name: {"pos": [float(v) for v in (pos - origin)], "rot": [float(v) for v in quat]}
            for name, (pos, quat) in self._snapshot_world().items()
        }

    def _teleport_object(self, name: str, pos: np.ndarray, quat_xyzw: np.ndarray) -> None:
        """Place ``name`` at a known-good pose, bypassing contacts (reset only)."""
        pose = torch.tensor(
            [[*(float(v) for v in pos), *(float(q) for q in quat_xyzw)]],
            dtype=torch.float32,
            device=self._env.device,
        )
        self._env.scene[f"object_{name}"].write_root_pose_to_sim_index(
            root_pose=pose, env_ids=torch.tensor([0], device=self._env.device)
        )

    def _zero_object_velocity(self, name: str) -> None:
        """Kill linear + angular velocity on ``name`` (used by :meth:`reset`)."""
        if f"object_{name}" not in self._env.scene.rigid_objects:
            return
        self._env.scene[f"object_{name}"].write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(1, 6, device=self._env.device),
            env_ids=torch.tensor([0], device=self._env.device),
        )

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""In-headset "adjust object" mode for the sharpa_duo teleop: a kinematic pose editor.

While the mode is open, nothing is simulated in any way that matters: the arm
rig is parked out of sight, the desk turns translucent, a translucent ghost of
every object marks where it stood at entry, and the operator's tracked hand
becomes the cursor. Pinching directly on an object WELDS it to the hand
(``T_obj = H(t) ∘ O``, the standard VR grab): the grabbed spot stays under the
fingers, rotation pivots about the grip with its natural lever arm, and the
object stays exactly where the pinch releases — it moves like a real object
held at the pinch point. The hand frame comes from wrist/knuckle POSITIONS,
which track far better than the wrist orientation estimate. Pinching empty
air with BOTH hands grabs the WORLD instead: pan, rotate, and dolly-zoom the
view (restored exactly on exit). Physics keeps ticking (pose
writes reach the renderer through the physics→fabric sync on each step) but is
inert: every tracked object is re-pinned at its edited pose with zero velocity
every substep, so gravity and contacts can never move anything. What you place
is what is saved, verbatim — nothing settles, snaps, or falls. The one
constraint is the tabletop, which acts as a hard floor: a drag can never take
any part of an object's mesh below the support surface, so pushing down rests
the mesh exactly on the table.

This replaces two earlier manipulation designs. Driving objects straight from
the tracked hand *dynamically* hung a rigid body off a noisy 240 Hz signal and
flung objects; pushing them with simulated robot hands was stable but far too
imprecise to hit a target pose. The end goal was only ever the objects' initial
xy and yaw, so the editor sets them directly and skips simulation entirely.

Whatever the recorder holds is discarded instead of becoming a demo. Saying
"done" turns the edited poses into the scene's new authored poses; "reset"
puts every object back onto its ghost. Persistence is a **sidecar** JSON next
to the scene USDA (see :func:`usda_scene.save_pose_overrides`); the USDA
itself is never modified, and the next scene load picks the sidecar up.

Import only after AppLauncher.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# OpenXR hand-joint indices in the 26-joint layout emitted by :mod:`xr_extras`.
_THUMB_TIP = 5
_THUMB_DISTAL = 4
_INDEX_TIP = 10
_INDEX_DISTAL = 9
_WRIST = 1
_INDEX_PROXIMAL = 7
_LITTLE_PROXIMAL = 22

_PINCH_ENGAGE_M = 0.01
"""Thumb-tip → index-tip distance below which a pinch is registered [m].

Deliberately tight (operator feedback): fingers must visibly touch before a
grab or tap fires, so hovering near an object can't pick it up by accident."""

_PINCH_RELEASE_M = 0.03
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
distance of some button — a pinch is also the grab gesture, and grabbing an
object must not spam tap diagnostics."""

_GRAB_MARGIN_M = 0.08
"""Horizontal slack [m] beyond an object's footprint radius within which a
pinch grabs it."""

_GRAB_HEIGHT_M = 0.4
"""Vertical window [m] around an object's center within which a pinch grabs it."""

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

_GRIP_SMOOTHING = 0.4
"""Fraction of the newest grip-rotation sample blended in per frame while an
object is held. The grip axis runs across the pinch — a short baseline — so a
little low-pass keeps a held object from trembling with fingertip jitter."""

_NAV_DOLLY_GAIN = 2.0
"""World-grab zoom: metres the world dollies toward your head per metre of
extra hand separation (hands apart = zoom in, together = zoom out)."""

_ROBOT_PARK_LIFT_M = 6.0
"""How far straight up the rig is parked while adjust mode is open [m].

Up rather than down: scene floors are PhysX half-space planes, so parking
below them would put every link in deep penetration.
"""

_DESK_GHOST_MATERIAL = "/World/Looks/AdjustDeskGhostMaterial"
"""Stage path of the shared translucent material bound to the desk in adjust mode."""


def _hand_frame(joints: np.ndarray) -> np.ndarray | None:
    """Two unit vectors spanning the back of the hand, from joint POSITIONS.

    Wrist → index knuckle and wrist → little-finger knuckle. Their rotation
    since the grab IS the hand's rotation, measured from the joints Quest
    tracks best — the wrist's own orientation estimate under-reports twist
    while the fingers are pinched, which made rotating a held object feel
    dead. Returns None while any of the three joints is untracked or the
    vectors are degenerate.
    """
    wrist, index, little = joints[_WRIST, :3], joints[_INDEX_PROXIMAL, :3], joints[_LITTLE_PROXIMAL, :3]
    if min(np.linalg.norm(wrist), np.linalg.norm(index), np.linalg.norm(little)) < 1e-6:
        return None
    u, v = index - wrist, little - wrist
    if np.linalg.norm(u) < 1e-3 or np.linalg.norm(v) < 1e-3 or np.linalg.norm(np.cross(u, v)) < 1e-6:
        return None
    return np.stack([u / np.linalg.norm(u), v / np.linalg.norm(v)])


def _hand_rotation(joints: np.ndarray) -> R | None:
    """The hand's full 3-DoF orientation, orthonormalized from :func:`_hand_frame`."""
    frame = _hand_frame(joints)
    if frame is None:
        return None
    x = frame[0]
    z = np.cross(frame[0], frame[1])
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    return R.from_matrix(np.stack([x, y, z], axis=1))


def _grip_rotation(joints: np.ndarray) -> R | None:
    """Orientation of the pinch GRIP itself — what a held object should copy.

    The primary axis runs from the thumb to the index finger ACROSS the pinch
    (tip+distal midpoints, for a steadier baseline than the tips alone), so
    twirling the fingertips around each other — turning the pinch like a
    little dial — rotates this frame directly; the hand-back normal from
    :func:`_hand_frame` only supplies the remaining twist axis. Whole-hand
    turns rotate both references, so they still work as before. Returns None
    while any needed joint is untracked or the geometry is degenerate.
    """
    pts = joints[[_THUMB_DISTAL, _THUMB_TIP, _INDEX_DISTAL, _INDEX_TIP], :3]
    if float(np.min(np.linalg.norm(pts, axis=1))) < 1e-6:
        return None
    across = 0.5 * (pts[2] + pts[3]) - 0.5 * (pts[0] + pts[1])
    if np.linalg.norm(across) < 3e-3:
        return None
    x = across / np.linalg.norm(across)
    frame = _hand_frame(joints)
    if frame is None:
        return None
    up = np.cross(frame[0], frame[1])
    z = np.cross(x, up)
    if np.linalg.norm(z) < 1e-6:
        return None
    z /= np.linalg.norm(z)
    return R.from_matrix(np.stack([x, np.cross(z, x), z], axis=1))


class RobotParker:
    """Parks the teleop rig out of sight for adjust mode and restores it exactly.

    Args:
        env: The live :class:`~isaaclab.envs.ManagerBasedRLEnv`.
    """

    def __init__(self, env):
        from pxr import UsdGeom

        import isaaclab.sim as sim_utils

        self._env = env
        self._env0 = torch.tensor([0], device=env.device)
        self._imageable = UsdGeom.Imageable(sim_utils.get_current_stage().GetPrimAtPath("/World/envs/env_0/robot"))
        self._state: dict[str, torch.Tensor] | None = None

    def park(self) -> None:
        """Snapshot the rig, lift it straight up out of the workspace, hide its render."""
        if self._state is not None:
            return
        robot = self._env.scene["robot"]
        self._state = {
            "root_pose": torch.cat([robot.data.root_pos_w.torch[0], robot.data.root_quat_w.torch[0]]).clone(),
            "joint_pos": robot.data.joint_pos.torch.clone(),
            "joint_vel": robot.data.joint_vel.torch.clone(),
        }
        park_pose = self._state["root_pose"].clone()
        park_pose[2] += _ROBOT_PARK_LIFT_M
        robot.write_root_pose_to_sim_index(root_pose=park_pose.unsqueeze(0), env_ids=self._env0)
        if not robot.is_fixed_base:
            robot.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros(1, 6, device=self._env.device), env_ids=self._env0
            )
        robot.set_joint_position_target_index(target=self._state["joint_pos"].clone())
        self._imageable.MakeInvisible()

    def restore(self) -> None:
        """Put the rig back exactly where :meth:`park` found it."""
        if self._state is None:
            return
        robot = self._env.scene["robot"]
        robot.write_root_pose_to_sim_index(root_pose=self._state["root_pose"].unsqueeze(0), env_ids=self._env0)
        if not robot.is_fixed_base:
            robot.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros(1, 6, device=self._env.device), env_ids=self._env0
            )
        robot.write_joint_position_to_sim_index(position=self._state["joint_pos"])
        robot.write_joint_velocity_to_sim_index(velocity=self._state["joint_vel"])
        robot.set_joint_position_target_index(target=self._state["joint_pos"].clone())
        self._imageable.MakeVisible()
        self._state = None


class AdjustGhosts:
    """Translucent frozen copies of the tracked objects at their entry poses.

    A :class:`~isaaclab.markers.VisualizationMarkers` instancer with one
    prototype per object, referencing the object's own USD asset (resolved by
    :func:`usda_scene.add_usda_scene` into ``OBJECT_ASSET_PATHS``). The Kit
    marker backend strips all physics APIs off its prototypes, so the ghosts
    are pure visuals.
    """

    def __init__(self, env, tracked_object_names: list[str]):
        self._markers = None
        self._ghost_names: list[str] = []
        from usda_scene import OBJECT_ASSET_PATHS  # noqa: PLC0415

        names = [n for n in tracked_object_names if n in OBJECT_ASSET_PATHS]
        for name in tracked_object_names:
            if name not in OBJECT_ASSET_PATHS:
                print(f"[ADJUST] No asset path resolved for '{name}'; it gets no ghost.")
        if not names:
            return
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            self._markers = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/adjust_ghosts",
                    markers={
                        name: sim_utils.UsdFileCfg(
                            usd_path=OBJECT_ASSET_PATHS[name],
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.4, 0.9, 0.5), opacity=0.25, roughness=0.6
                            ),
                        )
                        for name in names
                    },
                )
            )
            self._markers.set_visibility(False)
            self._ghost_names = names
        except Exception as exc:
            print(f"[WARNING] Adjust ghosts unavailable ({exc}); the editor still works.")
            self._markers = None

    def show(self, entry_poses: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        """Place one ghost per object at its entry pose and show them."""
        if self._markers is None:
            return
        names = [n for n in self._ghost_names if n in entry_poses]
        if not names:
            return
        self._markers.set_visibility(True)  # hidden markers ignore visualize()
        self._markers.visualize(
            translations=torch.tensor(np.stack([entry_poses[n][0] for n in names]), dtype=torch.float32),
            orientations=torch.tensor(np.stack([entry_poses[n][1] for n in names]), dtype=torch.float32),
            marker_indices=[self._ghost_names.index(n) for n in names],
        )

    def hide(self) -> None:
        """Hide the ghosts (they are re-fed on the next :meth:`show`)."""
        if self._markers is not None:
            self._markers.set_visibility(False)


class SceneGhoster:
    """Makes the static scene geometry (the desk) translucent while adjusting.

    Binds a shared ghost material over each top-level scene child that is not a
    tracked object or a light, and removes exactly those bindings on restore —
    the originals live on descendant prims inside the payloads, so dropping the
    ancestor's direct binding brings them back.

    Real transparency needs the renderer's translucency support, which is on
    with the default ``--arm_visual transparent``; otherwise the desk renders
    solid in the ghost color (cosmetic only).
    """

    def __init__(self, env, tracked_object_names: list[str]):
        self._env = env
        # Top-level scene child that holds each tracked object (objects are
        # usually direct children, but the cfg path may nest deeper).
        self._tracked_roots = set()
        for name in tracked_object_names:
            key = f"object_{name}"
            if key in env.scene.rigid_objects:
                rel = env.scene[key].cfg.prim_path.split("/scene/", 1)[-1]
                self._tracked_roots.add(rel.split("/", 1)[0])
        self._bound: list[str] = []
        self._material_ready = False

    def ghost(self) -> None:
        """Bind the translucent material over the non-object scene geometry."""
        if self._bound:
            return
        from pxr import UsdGeom, UsdLux

        import isaaclab.sim as sim_utils

        if not self._material_ready:
            sim_utils.spawn_preview_surface(
                _DESK_GHOST_MATERIAL,
                sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.65, 0.7), opacity=0.15, roughness=0.4),
            )
            self._material_ready = True
        stage = sim_utils.get_current_stage()
        scene_prim = stage.GetPrimAtPath("/World/envs/env_0/scene")
        if not scene_prim:
            return
        for child in scene_prim.GetChildren():
            if child.GetName() in self._tracked_roots:
                continue
            if child.HasAPI(UsdLux.LightAPI) or not child.IsA(UsdGeom.Imageable):
                continue
            try:
                sim_utils.bind_visual_material(str(child.GetPath()), _DESK_GHOST_MATERIAL)
                self._bound.append(str(child.GetPath()))
            except Exception as exc:
                print(f"[ADJUST] Could not ghost '{child.GetPath()}': {exc}")

    def restore(self) -> None:
        """Remove exactly the bindings :meth:`ghost` authored."""
        from pxr import UsdShade

        import isaaclab.sim as sim_utils

        stage = sim_utils.get_current_stage()
        for path in self._bound:
            prim = stage.GetPrimAtPath(path)
            if prim:
                UsdShade.MaterialBindingAPI(prim).UnbindDirectBinding()
        self._bound = []


class ObjectAdjuster:
    """The "adjust object" kinematic pose editor.

    One instance per teleop session. :meth:`enter` and :meth:`exit` bracket the
    mode; while it is open, :meth:`step` handles pinches (panel taps win, then
    grabs), :meth:`step_controllers` handles the Quest thumbstick/trigger, and
    :meth:`step_sim` owns the physics substeps, pinning every object at its
    edited pose so nothing can move on its own. All coordinate frames are the
    simulation world frame.

    Args:
        env: The live :class:`~isaaclab.envs.ManagerBasedRLEnv`.
        tracked_object_names: Names as returned by
            :func:`usda_scene.add_usda_scene` (without the ``object_`` prefix).
        scene_usda: Path to the scene USDA; the sidecar is written next to it.
        shared_hand_markers: The session's ``--visualize_hands`` markers when
            present (updated by the main loop already); None makes the editor
            create its own hand cursor, shown only while the mode is open.
    """

    def __init__(self, env, tracked_object_names: list[str], scene_usda: str, shared_hand_markers=None):
        self._env = env
        self._env0 = torch.tensor([0], device=env.device)
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
        # The edited pose of every tracked object — the single source of truth
        # while the mode is open; :meth:`step_sim` re-pins the sim to it.
        self._edited: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        # Per hand: (name, offset_pos np(3), offset_quat np(4)) — the object's
        # pose expressed in the grip frame at grab time (the rigid weld).
        self._grabs: list[tuple | None] = [None, None]
        # Per hand: the low-passed grip rotation while an object is held.
        self._grip_smooth: list[R | None] = [None, None]
        # Per hand: a pinch that hit neither panel nor object ("empty air").
        # Both hands empty-pinching engages the world-grab view navigation.
        self._empty_pinch: list[bool] = [False, False]
        # (F0 mid, F0 rotation, hand separation, dolly direction) at nav engage.
        self._nav_ref: tuple[np.ndarray, R, float, np.ndarray | None] | None = None
        self._xr_cfg = None  # bound via set_view_control once the device exists
        self._anchor_entry: tuple[tuple, tuple] | None = None
        self._substep_count = 0

        self._parker = RobotParker(env)
        self._ghosts = AdjustGhosts(env, self._names)
        self._desk = SceneGhoster(env, self._names)
        # Hand cursor: the operator must see their hand to aim a grab.
        self._own_cursor = None
        if shared_hand_markers is None:
            from xr_extras import HandJointMarkers  # noqa: PLC0415

            self._own_cursor = HandJointMarkers()
            self._own_cursor.set_visibility(False)
        # Small emissive sphere on each held object, so a live grab is obvious.
        self._grab_markers = None
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            self._grab_markers = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/adjust_grab_markers",
                    markers={
                        "held": sim_utils.SphereCfg(
                            radius=0.03,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(1.0, 0.85, 0.2), emissive_color=(0.5, 0.4, 0.1), opacity=0.6
                            ),
                        )
                    },
                )
            )
            self._grab_markers.set_visibility(False)
        except Exception as exc:
            print(f"[WARNING] Grab markers unavailable ({exc}); grabbing still works.")

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

    def set_view_control(self, xr_cfg) -> None:
        """Enable world-grab view navigation by handing over the live XR config.

        Both hands pinching empty air grab the WORLD: move them to pan, turn
        them to rotate it (full 3-DoF — say "align" if you lose the horizon),
        and spread/close them to dolly-zoom. Mutates ``xr_cfg.anchor_pos`` /
        ``anchor_rot`` exactly like the voice "align" command; the anchor
        synchronizer re-pushes them every frame. The view snaps back to its
        entry pose when the mode exits, so teleop alignment is untouched.
        """
        self._xr_cfg = xr_cfg

    # ------------------------------------------------------------------
    # Mode lifecycle
    # ------------------------------------------------------------------

    def enter(self) -> None:
        """Open the editor: snapshot poses, park the rig, show the aids."""
        self._active = True
        self._button_consumed = [False, False]
        self._trigger_consumed = [False, False]
        self._stick_next_fire = [[None, None], [None, None]]
        self._grabs = [None, None]
        self._grip_smooth = [None, None]
        self._empty_pinch = [False, False]
        self._nav_ref = None
        if self._xr_cfg is not None:
            self._anchor_entry = (tuple(self._xr_cfg.anchor_pos), tuple(self._xr_cfg.anchor_rot))
        self._entry_poses = self._snapshot_world()
        self._edited = {
            name: (
                torch.tensor(pos, dtype=torch.float32, device=self._env.device),
                torch.tensor(quat, dtype=torch.float32, device=self._env.device),
            )
            for name, (pos, quat) in self._entry_poses.items()
        }
        self._parker.park()
        self._ghosts.show(self._entry_poses)
        self._desk.ghost()
        if self._own_cursor is not None:
            self._own_cursor.set_visibility(True)

    def reset(self) -> int:
        """Put every object back onto its ghost, stopped dead; drop all grabs.

        Returns the number of objects restored.
        """
        self._grabs = [None, None]
        self._grip_smooth = [None, None]
        for name, (pos, quat) in self._entry_poses.items():
            self._edited[name] = (
                torch.tensor(pos, dtype=torch.float32, device=self._env.device),
                torch.tensor(quat, dtype=torch.float32, device=self._env.device),
            )
            self._teleport_object(name, pos, quat)
            self._zero_object_velocity(name)
        return len(self._entry_poses)

    def exit(self, dr_params: dict | None = None) -> int:
        """Save the sidecar, refresh the authored poses, restore the rig and visuals.

        Poses are recorded from the editor's own edited state, verbatim —
        nothing settles or snaps, so a leaning arrangement survives untouched
        (and a pose left hovering will simply drop on the next real reset).
        Saving from :attr:`_edited` rather than a sim snapshot means even a
        solver depenetration nudge between the last pin and the save cannot
        leak into the sidecar. Returns the number of objects saved.

        Args:
            dr_params: The live ``dr_objects`` param dict (see
                ``_live_dr_params`` in the teleop script). Its ``xy_range`` [m]
                and ``yaw_range`` [rad] are written to the sidecar next to the
                poses, so the ranges tuned in the editor follow the scene across
                sessions exactly like the poses do. None keeps whatever block the
                sidecar already holds.
        """
        self._active = False
        self._grabs = [None, None]
        overrides = self._edited_poses_local() if self._edited else self._current_poses_local()
        from usda_scene import save_pose_overrides  # noqa: PLC0415

        randomization = None
        if dr_params is not None:
            randomization = {k: float(dr_params[k]) for k in ("xy_range", "yaw_range") if k in dr_params}
        path = save_pose_overrides(self._scene_usda, overrides, randomization)
        ranges = ""
        if randomization:
            ranges = (
                f" and randomization ranges (xy ±{randomization.get('xy_range', float('nan')):.3f} m,"
                f" yaw ±{math.degrees(randomization.get('yaw_range', float('nan'))):.1f} deg)"
            )
        print(f"[ADJUST] Saved authored poses for {len(overrides)} object(s){ranges} to {path}")

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

        self._parker.restore()
        self._ghosts.hide()
        self._desk.restore()
        self._nav_ref = None
        if self._xr_cfg is not None and self._anchor_entry is not None:
            # Snap the view back: teleop's arm-to-hand alignment depends on it.
            self._xr_cfg.anchor_pos, self._xr_cfg.anchor_rot = self._anchor_entry
            self._anchor_entry = None
        if self._own_cursor is not None:
            self._own_cursor.set_visibility(False)
        if self._grab_markers is not None:
            self._grab_markers.set_visibility(False)
        return len(overrides)

    def is_active(self) -> bool:
        """Whether the mode is currently open."""
        return self._active

    def held_objects(self) -> list[str]:
        """Names of the objects currently grabbed (at most one per hand)."""
        return [grab[0] for grab in self._grabs if grab is not None]

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def step(self, xr_hands: torch.Tensor | np.ndarray) -> None:
        """Handle pinches: panel taps, object grabs, or world-grab navigation.

        Each pinch classifies once, at its rising edge: on a panel key → tap;
        within grab range of an object → weld-grab it; otherwise "empty air".
        Both hands holding empty pinches grab the WORLD instead (pan / rotate /
        dolly-zoom the view).

        Args:
            xr_hands: Shape ``(2, 26, 7)`` — ``[hand, joint, (px,py,pz,qx,qy,qz,qw)]``
                in the sim world frame. Untracked entries read as all zeros
                (and release that hand's grab).
        """
        if not self._active:
            return
        hands = xr_hands.detach().cpu().numpy() if isinstance(xr_hands, torch.Tensor) else np.asarray(xr_hands)
        mids: list[np.ndarray | None] = [None, None]

        for hand in range(2):
            thumb, index = hands[hand][_THUMB_TIP, :3], hands[hand][_INDEX_TIP, :3]
            if np.linalg.norm(thumb) < 1e-6 or np.linalg.norm(index) < 1e-6:
                self._button_consumed[hand] = False  # tracking lost
                self._grabs[hand] = None
                self._grip_smooth[hand] = None
                self._empty_pinch[hand] = False
                continue
            pinch = float(np.linalg.norm(thumb - index))
            mid = 0.5 * (thumb + index)
            mids[hand] = mid
            if pinch > _PINCH_RELEASE_M:
                self._button_consumed[hand] = False
                self._grabs[hand] = None  # released: the object stays where it is
                self._grip_smooth[hand] = None
                self._empty_pinch[hand] = False
                continue
            if not self._button_consumed[hand] and pinch < _PINCH_ENGAGE_M:
                # Rising edge. Consume it either way, so one pinch is one action.
                self._button_consumed[hand] = True
                kind = None
                nearest = None
                if self._panel_buttons_fn is not None:
                    kind, nearest = self._button_hit(mid)
                if kind is not None:
                    if self._on_button_press is not None:
                        self._on_button_press(kind)  # panel taps win over grabs
                elif not self._try_grab(hand, mid, hands[hand]):
                    self._empty_pinch[hand] = True  # candidate for world-grab nav
                    if nearest is not None and nearest[1] < _MISS_REPORT_RANGE_M:
                        # Diagnose panel near-misses so "my taps do nothing" is
                        # self-explaining; pinches far from panel and objects
                        # alike are just empty air.
                        name, _, in_plane, depth = nearest
                        print(f"[ADJUST] tap missed: nearest '{name}' in-plane {in_plane:.2f} m, depth {depth:.2f} m")
            # While held: the object stays welded to the grip.
            grab = self._grabs[hand]
            if grab is not None:
                self._drag(hand, grab, mid, hands[hand])
        self._update_view_nav(hands, mids)
        self._update_grab_markers()

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
            kind, nearest = self._button_hit(ctrl[hand, :3].astype(np.float64))
            if kind is not None:
                if self._on_button_press is not None:
                    self._on_button_press(kind)
            elif nearest is not None and nearest[1] < _MISS_REPORT_RANGE_M:
                name, _, in_plane, depth = nearest
                print(f"[ADJUST] tap missed: nearest '{name}' in-plane {in_plane:.2f} m, depth {depth:.2f} m")

    def _update_view_nav(self, hands: np.ndarray, mids: list[np.ndarray | None]) -> None:
        """Grab-the-world view navigation while both hands pinch empty air.

        The pair of pinch points defines a frame; the anchor is servoed each
        frame so that frame maps back onto where it was at engage — i.e. the
        world sticks to your hands: move them to pan, turn them to rotate
        (full 3-DoF; "align" re-levels if you lose the horizon), spread or
        close them to dolly toward/away from your head. Pure view motion —
        no object pose is touched, and exit() restores the entry view.
        """
        engaged = (
            self._xr_cfg is not None
            and self._empty_pinch[0]
            and self._empty_pinch[1]
            and mids[0] is not None
            and mids[1] is not None
        )
        if not engaged:
            self._nav_ref = None
            return
        center = 0.5 * (mids[0] + mids[1])
        span = mids[1] - mids[0]
        separation = float(np.linalg.norm(span))
        if separation < 0.05:
            return  # hands nearly touching: the frame is degenerate
        # Second axis for the full 3-DoF frame: the hands' averaged palm normal
        # (falls back to world up while the knuckle frames are untracked).
        normals = [rot.apply([0.0, 0.0, 1.0]) for rot in (_hand_rotation(hands[0]), _hand_rotation(hands[1])) if rot]
        up = normals[0] + normals[1] if len(normals) == 2 else (normals[0] if normals else np.array([0.0, 0.0, 1.0]))
        x = span / separation
        z = np.cross(x, up)
        if np.linalg.norm(z) < 1e-6:
            return
        z /= np.linalg.norm(z)
        frame_rot = R.from_matrix(np.stack([x, np.cross(z, x), z], axis=1))
        if self._nav_ref is None:
            from xr_extras import current_head_pose  # noqa: PLC0415

            head = current_head_pose()
            dolly_dir = None
            if head is not None and np.linalg.norm(head[:3] - center) > 1e-3:
                dolly_dir = (head[:3] - center) / np.linalg.norm(head[:3] - center)
            self._nav_ref = (center, frame_rot, separation, dolly_dir)
            return
        ref_center, ref_rot, ref_separation, dolly_dir = self._nav_ref
        # Servo the anchor so the current hand frame maps back onto the engage
        # frame: world_T_anchor ← W ∘ world_T_anchor with W = F0 ∘ F1⁻¹. The
        # remap feeds back into next frame's hand coordinates, so holding still
        # converges and motion tracks 1:1 (one frame of lag).
        w_rot = ref_rot * frame_rot.inv()
        translation = ref_center - w_rot.apply(center)
        if dolly_dir is not None:
            translation = translation + _NAV_DOLLY_GAIN * (separation - ref_separation) * dolly_dir
        anchor_pos = np.asarray(self._xr_cfg.anchor_pos, dtype=np.float64)
        anchor_rot = R.from_quat(self._xr_cfg.anchor_rot)
        self._xr_cfg.anchor_pos = tuple(float(v) for v in (w_rot.apply(anchor_pos) + translation))
        self._xr_cfg.anchor_rot = tuple(float(v) for v in (w_rot * anchor_rot).as_quat())

    def update_cursor(self, xr_hands: torch.Tensor) -> None:
        """Feed the editor's own hand cursor (no-op when the session shares one)."""
        if self._active and self._own_cursor is not None:
            self._own_cursor.update(xr_hands)

    def step_sim(self) -> None:
        """Run one control step of inert physics: pin everything, step, render.

        Physics keeps stepping only because pose writes reach the renderer
        through the physics→fabric sync on each step — a render-only loop
        would freeze the edits on screen. Every tracked object is re-pinned at
        its edited pose with zero velocity every substep, so gravity and
        contacts can never move anything; the parked rig's joints are held by
        its PD targets.
        """
        env = self._env
        zero6 = torch.zeros(1, 6, device=env.device)
        decimation = max(1, int(env.cfg.decimation))
        render_interval = max(1, int(env.cfg.sim.render_interval))
        for _ in range(decimation):
            for name, (pos, quat) in self._edited.items():
                obj = env.scene[f"object_{name}"]
                obj.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat]).unsqueeze(0), env_ids=self._env0)
                obj.write_root_velocity_to_sim_index(root_velocity=zero6, env_ids=self._env0)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            self._substep_count += 1
            if self._substep_count % render_interval == 0:
                env.sim.render()
            env.scene.update(env.physics_dt)

    # ------------------------------------------------------------------
    # Grabbing
    # ------------------------------------------------------------------

    def _try_grab(self, hand: int, mid: np.ndarray, joints: np.ndarray) -> bool:
        """Weld the nearest object around ``mid`` to the hand, if any; returns success."""
        from usda_scene import FOOTPRINT_RADII  # noqa: PLC0415

        hand_rot = _grip_rotation(joints)
        if hand_rot is None:
            return False  # rotation reference untracked; try again next pinch
        held_elsewhere = {grab[0] for grab in self._grabs if grab is not None}
        best: tuple[float, str] | None = None
        for name, (pos, _) in self._edited.items():
            if name in held_elsewhere:
                continue
            dist_xy = math.hypot(mid[0] - float(pos[0]), mid[1] - float(pos[1]))
            if dist_xy > FOOTPRINT_RADII.get(name, 0.05) + _GRAB_MARGIN_M:
                continue
            if abs(mid[2] - float(pos[2])) > _GRAB_HEIGHT_M:
                continue
            if best is None or dist_xy < best[0]:
                best = (dist_xy, name)
        if best is None:
            return False
        name = best[1]
        pos, quat = self._edited[name]
        # The rigid weld: the object's pose expressed in the GRIP frame. While
        # held, re-applying it under the live grip frame reproduces exactly how
        # a real grabbed object moves — the grabbed spot stays under the
        # fingers, and twirling the fingertips (or turning the whole hand)
        # rotates the object about the grip with its natural lever arm.
        offset_pos = hand_rot.inv().apply(pos.detach().cpu().numpy() - mid)
        offset_rot = hand_rot.inv() * R.from_quat(quat.detach().cpu().numpy())
        self._grabs[hand] = (name, offset_pos, offset_rot.as_quat())
        self._grip_smooth[hand] = hand_rot
        print(f"[ADJUST] Grabbed '{name}' — welded to your fingertips 1:1; release to place.")
        return True

    def _drag(self, hand: int, grab: tuple, mid: np.ndarray, joints: np.ndarray) -> None:
        """Re-apply the rigid weld under the grip's current pose.

        ``T_obj = H(t) ∘ O`` with the offset ``O`` recorded at grab time: the
        object moves exactly like a real object held at the pinch point — the
        grabbed spot stays under the fingers, and twirling the fingertips or
        turning the hand pivots the object about the grip with its natural
        lever arm. The grip frame (:func:`_grip_rotation`) is low-passed by
        :data:`_GRIP_SMOOTHING` against fingertip jitter. While the frame is
        untracked, the held object freezes.

        The tabletop is a HARD FLOOR: the pose is clamped so the object's
        lowest mesh point (its rotated convex-hull vertices, see
        :data:`usda_scene.OBJECT_SUPPORT_POINTS`) never goes below the support
        surface. Pushing down therefore rests the mesh exactly on the table —
        the easiest way to place something flat — and nothing can be authored
        sunken into it.
        """
        from usda_scene import OBJECT_SUPPORT_POINTS, SUPPORT_SURFACE_Z  # noqa: PLC0415

        name, offset_pos, offset_quat = grab
        raw = _grip_rotation(joints)
        if raw is None:
            return  # rotation reference dropped out: freeze until it returns
        prev = self._grip_smooth[hand]
        if prev is None:
            hand_rot = raw
        else:
            hand_rot = R.from_rotvec((raw * prev.inv()).as_rotvec() * _GRIP_SMOOTHING) * prev
        self._grip_smooth[hand] = hand_rot
        pos = mid + hand_rot.apply(offset_pos)
        rot = hand_rot * R.from_quat(offset_quat)
        support = OBJECT_SUPPORT_POINTS.get(name)
        if support is not None and SUPPORT_SURFACE_Z is not None:
            lift = SUPPORT_SURFACE_Z - (pos[2] + float(rot.apply(support)[:, 2].min()))
            if lift > 0.0:
                pos = pos.copy()
                pos[2] += lift
        device = self._env.device
        self._edited[name] = (
            torch.tensor(pos, dtype=torch.float32, device=device),
            torch.tensor(rot.as_quat(), dtype=torch.float32, device=device),
        )

    def _update_grab_markers(self) -> None:
        """One highlight sphere per held object; hidden when nothing is held."""
        if self._grab_markers is None:
            return
        held = [self._edited[grab[0]][0] for grab in self._grabs if grab is not None]
        if held:
            self._grab_markers.set_visibility(True)
            self._grab_markers.visualize(translations=torch.stack(held).cpu())
        else:
            self._grab_markers.set_visibility(False)

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
        """Every tracked object's current SIM pose, in the env-local frame."""
        origin = self._env.scene.env_origins[0].detach().cpu().numpy()
        return {
            name: {"pos": [float(v) for v in (pos - origin)], "rot": [float(v) for v in quat]}
            for name, (pos, quat) in self._snapshot_world().items()
        }

    def _edited_poses_local(self) -> dict[str, dict[str, list[float]]]:
        """Every tracked object's EDITED pose, in the env-local frame."""
        origin = self._env.scene.env_origins[0].detach().cpu().numpy()
        return {
            name: {
                "pos": [float(v) for v in (pos.detach().cpu().numpy() - origin)],
                "rot": [float(v) for v in quat.detach().cpu().numpy()],
            }
            for name, (pos, quat) in self._edited.items()
        }

    def _teleport_object(self, name: str, pos: np.ndarray, quat_xyzw: np.ndarray) -> None:
        """Place ``name`` at a known-good pose (reset only)."""
        pose = torch.tensor(
            [[*(float(v) for v in pos), *(float(q) for q in quat_xyzw)]],
            dtype=torch.float32,
            device=self._env.device,
        )
        self._env.scene[f"object_{name}"].write_root_pose_to_sim_index(root_pose=pose, env_ids=self._env0)

    def _zero_object_velocity(self, name: str) -> None:
        """Kill linear + angular velocity on ``name`` (used by :meth:`reset`)."""
        if f"object_{name}" not in self._env.scene.rigid_objects:
            return
        self._env.scene[f"object_{name}"].write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(1, 6, device=self._env.device), env_ids=self._env0
        )

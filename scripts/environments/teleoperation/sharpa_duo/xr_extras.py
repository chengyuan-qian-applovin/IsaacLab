# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XR extras for the duo teleop: raw hand passthrough and visual helpers.

The IsaacTeleop device returns exactly one flat action tensor per frame, so the
raw hand-tracking data rides along INSIDE that tensor: :class:`HandsXrPassthrough`
is a pipeline node that emits the two hands' 26 joint poses as
``XR_EXTRAS_DIM = 364`` extra elements (2 hands x 26 joints x [x, y, z, qx, qy,
qz, qw], sim world frame, xyzw quats), which the pipeline appends after the 58-D
robot action. The teleop loop slices them off before ``env.step`` and feeds them
to the hand-joint markers and the ``obs/xr_hands`` recorder.
An untracked hand (or joint) reads as all zeros.

Import only after AppLauncher.
"""

from __future__ import annotations

import numpy as np
import torch

XR_EXTRAS_DIM = 2 * 26 * 7
"""Elements appended to the action tensor by :class:`HandsXrPassthrough`."""

XR_HAND_ELEMENTS = [
    f"xr_{hand}_j{j:02d}_{c}"
    for hand in ("left", "right")
    for j in range(26)
    for c in ("px", "py", "pz", "qx", "qy", "qz", "qw")
]
"""Element names of the appended block, for the pipeline's TensorReorderer."""


def make_hands_passthrough(name: str = "xr_hands_passthrough"):
    """Build the raw-hands passthrough pipeline node (see module docstring).

    Defined as a factory because isaacteleop must not be imported before the app
    launches; the class is created on first call.
    """
    from isaacteleop.retargeting_engine.interface import BaseRetargeter
    from isaacteleop.retargeting_engine.interface.tensor_group_type import OptionalType, TensorGroupType
    from isaacteleop.retargeting_engine.tensor_types import FloatType, HandInput, HandInputIndex

    class HandsXrPassthrough(BaseRetargeter):
        """Emits both hands' raw 26-joint poses as 364 named scalars."""

        def input_spec(self):
            return {"hand_left": OptionalType(HandInput()), "hand_right": OptionalType(HandInput())}

        def output_spec(self):
            return {"xr_hands": TensorGroupType("xr_hands", [FloatType(n) for n in XR_HAND_ELEMENTS])}

        def _compute_fn(self, inputs, outputs, context) -> None:
            out = outputs["xr_hands"]
            flat = np.zeros(XR_EXTRAS_DIM, dtype=np.float64)
            for h, key in enumerate(("hand_left", "hand_right")):
                group = inputs[key]
                if group.is_none:
                    continue
                pos = np.from_dlpack(group[HandInputIndex.JOINT_POSITIONS])  # (26, 3)
                quat = np.from_dlpack(group[HandInputIndex.JOINT_ORIENTATIONS])  # (26, 4) xyzw
                valid = np.from_dlpack(group[HandInputIndex.JOINT_VALID]).astype(bool)  # (26,)
                joints = np.concatenate([pos, quat], axis=1)  # (26, 7)
                joints[~valid] = 0.0
                flat[h * 182 : (h + 1) * 182] = joints.reshape(-1)
            for i, v in enumerate(flat):
                out[i] = float(v)

    return HandsXrPassthrough(name=name)


class HandJointMarkers:
    """Small red spheres on the 52 tracked hand joints (debug aid, render only)."""

    def __init__(self):
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        self._markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/xr_hand_joints",
                markers={
                    "joint": sim_utils.SphereCfg(
                        radius=0.005,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),
                    )
                },
            )
        )

    def update(self, xr_hands: torch.Tensor) -> None:
        pos = xr_hands[:, :, :3].reshape(-1, 3).clone()
        # Park untracked joints (all-zero poses) far below the scene.
        pos[pos.norm(dim=-1) < 1e-6, 2] = -1000.0
        self._markers.visualize(translations=pos)


def apply_arm_visual(mode: str) -> None:
    """Make the arms' render geometry 5% transparent or invisible (physics untouched).

    Targets each link's render prims individually — the instanceable ``visuals``
    roots and any loose Gprims (the panda links' render mesh doubles as their
    collider) — rather than the arm roots, so descendants keep working if a debug
    subtree is ever added.
    """
    if mode == "normal":
        return
    from pxr import Usd, UsdGeom

    import isaaclab.sim as sim_utils

    arm_paths = sim_utils.find_matching_prim_paths("/World/envs/env_.*/robot/(left|right)_arm")
    if not arm_paths:
        print("[WARNING] --arm_visual: no arm prims matched; skipping.")
        return
    stage = sim_utils.get_current_stage()
    targets = []
    for arm_path in arm_paths:
        it = iter(Usd.PrimRange(stage.GetPrimAtPath(arm_path)))
        for prim in it:
            name = prim.GetName()
            if name == "collisions":
                it.PruneChildren()
            elif name == "visuals":
                targets.append(prim)
                it.PruneChildren()
            elif prim.IsA(UsdGeom.Gprim):
                targets.append(prim)
    if mode == "hidden":
        for prim in targets:
            sim_utils.set_prim_visibility(prim, False)
        print(f"[INFO] Arms hidden (render only): {len(targets)} render prims under {arm_paths}")
    elif mode == "transparent":
        material_path = "/World/Looks/ArmGhostMaterial"
        sim_utils.spawn_preview_surface(
            material_path,
            sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.78, 0.85), opacity=0.05, roughness=0.0),
        )
        for prim in targets:
            sim_utils.bind_visual_material(str(prim.GetPath()), material_path, stronger_than_descendants=True)
        print(f"[INFO] Arms 5% transparent: {len(targets)} render prims under {arm_paths}")


def current_head_pose() -> np.ndarray | None:
    """Fresh head pose [x, y, z, qx, qy, qz, qw] in the Isaac world frame, from XRCore.

    Queried on demand (only when the voice "align" command fires), so no head
    tracker lives in the retargeting pipeline — a pipeline ``HeadSource`` made
    every DeviceIO session step fail on the Quest/Kit-bridge stack. Mirrors the
    source branch's ``current_head_pose``; returns None while the head is
    untracked (an exactly-origin position means "not tracked yet").
    """
    try:
        from omni.kit.xr.core import XRCore

        head_device = XRCore.get_singleton().get_input_device("/user/head")
        if not head_device:
            return None
        hmd = head_device.get_virtual_world_pose("")
        position = hmd.ExtractTranslation()
        quat = hmd.ExtractRotationQuat()
        imag = quat.GetImaginary()
        pose = np.array(
            [position[0], position[1], position[2], imag[0], imag[1], imag[2], quat.GetReal()],
            dtype=np.float64,
        )
    except Exception as e:
        print(f"[ALIGN] XRCore head query failed: {e}")
        return None
    if np.linalg.norm(pose[:3]) < 1e-6:
        return None
    return pose


class AnchorAligner:
    """Re-anchor the XR session so the workspace sits straight in front of the user.

    Port of the feature branch's aligner to the IsaacTeleop stack. The correction
    is the same world-frame rigid ΔT: rotate about the user's current head
    position until the head's forward axis (OpenXR: −Z) points along the robot's
    facing direction, then translate the head's xy onto ``target_head_xy`` and —
    when ``target_head_z`` is given — the head's height onto it (putting the
    world's z=0 floor exactly ``target_head_z`` below the head; with None the
    calibrated floor height holds). Because ΔT rigidly moves the whole XR→world
    mapping, the wrist offsets — which live in the wrist's own frame — are
    unaffected.

    Mechanism difference vs the 2.x branch: there is no direct XRCore write here.
    On this stack the :class:`~isaaclab_teleop.xr_anchor_utils.XrAnchorSynchronizer`
    re-pushes the anchor from ``XrCfg.anchor_pos`` / ``anchor_rot`` on every
    pre-sync update (a direct write would be clobbered a frame later), so the
    aligner mutates that shared config instead — the synchronizer then propagates
    the new anchor to both the renderer and the pipeline's ``world_T_anchor``.
    """

    def __init__(
        self, teleop, target_head_xy: tuple[float, float], robot_yaw: float, target_head_z: float | None = None
    ):
        self._xr_cfg = teleop._anchor_manager._xr_cfg
        self._target_xy = np.asarray(target_head_xy, dtype=np.float64)
        self._target_z = None if target_head_z is None else float(target_head_z)
        self._robot_yaw = float(robot_yaw)
        # Mutating the shared XrCfg only works if the synchronizer that pushes
        # the anchor every pre-sync frame holds the SAME object — the one
        # failure mode where align would "succeed" yet visibly do nothing.
        sync = getattr(teleop._anchor_manager, "_anchor_sync", None)
        if sync is not None and sync._xr_cfg is not self._xr_cfg:
            print("[WARNING] Align: the anchor synchronizer holds a different XrCfg; align will not take effect.")

    def align(self, head_pose_w: np.ndarray) -> bool:
        """Apply the correction for the given world-frame head pose [pos, quat xyzw]."""
        from scipy.spatial.transform import Rotation as R

        head_pos = head_pose_w[:3].astype(np.float64)
        # Head forward axis: OpenXR head frames look along -Z.
        fwd = R.from_quat(head_pose_w[3:].astype(np.float64)).apply([0.0, 0.0, -1.0])
        if np.linalg.norm(fwd[:2]) < 1e-6:
            print("[ALIGN] Looking straight up/down: yaw undefined, try again.")
            return False
        dyaw = self._robot_yaw - np.arctan2(fwd[1], fwd[0])
        q_dyaw = R.from_euler("z", dyaw)

        # ΔT: rotate the anchor about the head position (head stays put), then
        # translate the head's xy onto the target.
        anchor_pos = np.asarray(self._xr_cfg.anchor_pos, dtype=np.float64)
        new_pos = q_dyaw.apply(anchor_pos - head_pos) + head_pos
        new_pos[0] += self._target_xy[0] - head_pos[0]
        new_pos[1] += self._target_xy[1] - head_pos[1]
        height_note = ""
        if self._target_z is not None:
            new_pos[2] += self._target_z - head_pos[2]
            height_note = f", head height -> {self._target_z:.2f} m (was {head_pos[2]:.2f})"
        new_quat = q_dyaw * R.from_quat(self._xr_cfg.anchor_rot)

        self._xr_cfg.anchor_pos = tuple(float(v) for v in new_pos)
        self._xr_cfg.anchor_rot = tuple(float(v) for v in new_quat.as_quat())
        print(
            f"[ALIGN] Re-anchored: yaw {np.degrees(dyaw):+.1f} deg, head xy ->"
            f" ({self._target_xy[0]:.2f}, {self._target_xy[1]:.2f}){height_note}"
        )
        return True

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XR extras for the duo teleop: raw hand passthrough, stop gesture, visual helpers.

The IsaacTeleop device returns exactly one flat action tensor per frame, so the
raw hand-tracking data rides along INSIDE that tensor: :class:`HandsXrPassthrough`
is a pipeline node that emits the two hands' 26 joint poses as
``XR_EXTRAS_DIM = 364`` extra elements (2 hands x 26 joints x [x, y, z, qx, qy,
qz, qw], sim world frame, xyzw quats), which the pipeline appends after the 58-D
robot action. The teleop loop slices them off before ``env.step`` and feeds them
to the stop gesture, the hand-joint markers, and the ``obs/xr_hands`` recorder.
An untracked hand (or joint) reads as all zeros.

Import only after AppLauncher.
"""

from __future__ import annotations

import time

import numpy as np
import torch

# OpenXR fingertip joint indices (XR_EXT_hand_tracking order), thumb through little.
_TIP_INDICES = (5, 10, 15, 20, 25)
_WRIST_INDEX = 1

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


class CrossHandStopGesture:
    """All five same-finger tip pairs within ``touch_dist``, held ``hold_s`` seconds.

    Cross-hand by construction, so it cannot collide with the intra-hand pinches
    that drive DexPilot. After triggering, it re-arms only once any pair separates
    beyond ``release_dist`` (hysteresis against retriggering while the hands part).
    """

    def __init__(self, touch_dist: float = 0.02, release_dist: float = 0.10, hold_s: float = 0.5):
        self._touch_dist = touch_dist
        self._release_dist = release_dist
        self._hold_s = hold_s
        self._touch_since: float | None = None
        self._armed = True

    def reset(self) -> None:
        self._touch_since = None
        self._armed = True

    def update(self, xr_hands: torch.Tensor) -> bool:
        """Feed the (2, 26, 7) raw hand block; returns True exactly once per gesture."""
        pos = xr_hands[:, :, :3]
        # An untracked hand reads as all zeros; require both wrists to be live.
        if pos[0, _WRIST_INDEX].norm() < 1e-6 or pos[1, _WRIST_INDEX].norm() < 1e-6:
            self._touch_since = None
            return False
        dists = (pos[0, list(_TIP_INDICES)] - pos[1, list(_TIP_INDICES)]).norm(dim=-1)

        if not self._armed:
            if bool((dists > self._release_dist).any()):
                self._armed = True
            return False

        if bool((dists < self._touch_dist).all()):
            now = time.monotonic()
            if self._touch_since is None:
                self._touch_since = now
            elif now - self._touch_since >= self._hold_s:
                self._touch_since = None
                self._armed = False
                return True
        else:
            self._touch_since = None
        return False


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
    import isaacsim.core.utils.stage as stage_utils
    from pxr import Usd, UsdGeom

    import isaaclab.sim as sim_utils

    arm_paths = sim_utils.find_matching_prim_paths("/World/envs/env_.*/robot/(left|right)_arm")
    if not arm_paths:
        print("[WARNING] --arm_visual: no arm prims matched; skipping.")
        return
    stage = stage_utils.get_current_stage()
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

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dual-hand retargeting for the FR3 Duo + SharpaWave rig (RoboLab).

Maps Isaac Lab OpenXR hand tracking onto RoboLab's ``FrankaDuoSharpaIKActionCfg``
58-D action, following the GR1T2 teleop pattern (wrist passthrough + dex-retargeting
fingers):

    [ left wrist pose (7) | right wrist pose (7) | left fingers (22) | right fingers (22) ]

Wrist poses are emitted in the XR/world frame with a per-side constant orientation
offset (OpenXR wrist frame -> panda_link8 flange frame) and a positional offset
(human wrist -> flange, expressed in the flange frame so the robot's palm — not its
flange — lands where your palm is). The teleop script converts world -> robot-root
frame before ``env.step`` (the DiffIK action expects root-frame commands).

Finger joints are produced by DexPilot (``dex_retargeting``) against the vendored
SharpaWave URDFs, exactly as ``GR1TR2DexRetargeting`` does for the Fourier hands,
and scattered into RoboLab's ``HAND_JOINTS_ORDERED`` order by joint name.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_mul

with contextlib.suppress(Exception):
    # dex_retargeting needs pinocchio; unavailable on some platforms.
    import yaml
    from dex_retargeting.retargeting_config import RetargetingConfig
    from scipy.spatial.transform import Rotation as R

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sharpa_dex_retargeting")

# OpenXR 26-joint hand -> 21-joint MANO-style ordering used by dex_retargeting
# (drops palm and the four *_metacarpal joints). Same as the GR1T2 pipeline.
_HAND_JOINTS_INDEX = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25]

_OPERATOR2MANO = np.array([
    [0, -1, 0],
    [-1, 0, 0],
    [0, 0, -1],
])


class SharpaWaveDexRetargeting:
    """DexPilot retargeting for one pair of SharpaWave hands (GR1TR2DexRetargeting analogue)."""

    def __init__(self, left_config: str, right_config: str, pinch_separation: float = 1e-4):
        self._dex = {}
        for side, cfg_name in (("left", left_config), ("right", right_config)):
            cfg_path = os.path.join(_DATA_DIR, cfg_name)
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            # Resolve the URDF path relative to the data dir without mutating the yml
            # on disk (the GR1T2 flow rewrites its ymls in place; we avoid that).
            urdf = cfg["retargeting"]["urdf_path"]
            if not os.path.isabs(urdf):
                cfg["retargeting"]["urdf_path"] = os.path.join(os.path.dirname(_DATA_DIR), urdf)
            self._dex[side] = RetargetingConfig.from_dict(cfg["retargeting"]).build()
            # Commanded thumb-finger separation once a pinch is projected (library
            # default eta1 = +1e-4 m: touch). Negative = overshoot past contact so the
            # PD drives squeeze the object. Only the first 4 entries (thumb pairs) —
            # entries 4:10 are finger-finger spacing (eta2) and must stay +0.03.
            self._dex[side].optimizer.projected_dist[:4] = pinch_separation

        self.left_dof_names = self._dex["left"].optimizer.robot.dof_joint_names
        self.right_dof_names = self._dex["right"].optimizer.robot.dof_joint_names

    def _convert_hand_joints(self, hand_poses: dict[str, np.ndarray]) -> np.ndarray:
        """26 OpenXR joint poses -> wrist-relative 21x3 MANO-style positions."""
        joint_position = np.zeros((21, 3))
        hand_joints = list(hand_poses.values())
        for i, idx in enumerate(_HAND_JOINTS_INDEX):
            joint_position[i] = hand_joints[idx][:3]
        joint_position = joint_position - joint_position[0:1, :]
        wq = hand_poses["wrist"][3:]  # w,x,y,z
        wrist_rot = R.from_quat([wq[1], wq[2], wq[3], wq[0]]).as_matrix()
        return joint_position @ wrist_rot @ _OPERATOR2MANO

    def _compute_one(self, side: str, hand_poses: dict[str, np.ndarray]) -> np.ndarray:
        retargeting = self._dex[side]
        joint_pos = self._convert_hand_joints(hand_poses)
        indices = retargeting.optimizer.target_link_human_indices
        if retargeting.optimizer.retargeting_type == "POSITION":
            ref_value = joint_pos[indices, :]
        else:
            ref_value = joint_pos[indices[1, :], :] - joint_pos[indices[0, :], :]
        # dex_retargeting optimizes with gradients; escape inference mode.
        with torch.enable_grad(), torch.inference_mode(False):
            return retargeting.retarget(ref_value)

    def compute_left(self, hand_poses: dict[str, np.ndarray] | None) -> np.ndarray:
        if hand_poses is None:
            return np.zeros(len(self.left_dof_names))
        return self._compute_one("left", hand_poses)

    def compute_right(self, hand_poses: dict[str, np.ndarray] | None) -> np.ndarray:
        if hand_poses is None:
            return np.zeros(len(self.right_dof_names))
        return self._compute_one("right", hand_poses)


class FrankaDuoSharpaRetargeter(RetargeterBase):
    """AVP dual-hand tracking -> 58-D FrankaDuoSharpaIKActionCfg action (world-frame wrists)."""

    def __init__(self, cfg: FrankaDuoSharpaRetargeterCfg):
        super().__init__(cfg)
        self._cfg = cfg
        self._hands = SharpaWaveDexRetargeting(cfg.left_dex_config, cfg.right_dex_config, cfg.pinch_separation)

        # Map dex output joint names -> slot in the action's per-hand block.
        self._left_scatter = [cfg.left_hand_joint_names.index(n) for n in self._hands.left_dof_names]
        self._right_scatter = [cfg.right_hand_joint_names.index(n) for n in self._hands.right_dof_names]

        dev = cfg.sim_device
        self._left_rot_offset = torch.tensor(cfg.left_wrist_rot_offset, dtype=torch.float32, device=dev).unsqueeze(0)
        self._right_rot_offset = torch.tensor(cfg.right_wrist_rot_offset, dtype=torch.float32, device=dev).unsqueeze(0)
        self._pos_offset = torch.tensor(cfg.wrist_pos_offset, dtype=torch.float32, device=dev)

        # Red-sphere markers on the tracked OpenXR hand joints (GR1T2 teleop pattern).
        self._enable_visualization = cfg.enable_visualization
        if self._enable_visualization:
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/hand_joints",
                markers={
                    "joint": sim_utils.SphereCfg(
                        radius=0.005,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                },
            )
            self._markers = VisualizationMarkers(marker_cfg)

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.HAND_TRACKING]

    def _wrist_pose(self, hand_data: dict[str, np.ndarray], rot_offset: torch.Tensor) -> torch.Tensor:
        wrist = hand_data["wrist"]
        pos = torch.tensor(wrist[:3], dtype=torch.float32, device=self._cfg.sim_device)
        quat = torch.tensor(wrist[3:], dtype=torch.float32, device=self._cfg.sim_device).unsqueeze(0)
        quat = quat_mul(quat, rot_offset)  # q_flange = q_xr_wrist ⊗ q_offset
        # Pull the flange target back along the (rotated) flange frame so the robot's
        # palm, not its flange, tracks the human palm.
        if torch.any(self._pos_offset != 0.0):
            from isaaclab.utils.math import quat_apply

            pos = pos - quat_apply(quat, self._pos_offset.unsqueeze(0)).squeeze(0)
        return torch.cat([pos, quat.squeeze(0)])

    def retarget(self, data: dict) -> torch.Tensor:
        left = data.get(DeviceBase.TrackingTarget.HAND_LEFT)
        right = data.get(DeviceBase.TrackingTarget.HAND_RIGHT)
        dev = self._cfg.sim_device

        if self._enable_visualization:
            # Joint poses arrive already in the Isaac world frame (anchor applied),
            # so positions can be visualized directly.
            joints = np.array([pose[:3] for hand in (left, right) if hand for pose in hand.values()])
            if joints.size:
                self._markers.visualize(translations=torch.tensor(joints, dtype=torch.float32, device=dev))

        left_wrist = self._wrist_pose(left, self._left_rot_offset)
        right_wrist = self._wrist_pose(right, self._right_rot_offset)

        left_q = self._hands.compute_left(left)
        right_q = self._hands.compute_right(right)
        left_fingers = torch.zeros(len(self._cfg.left_hand_joint_names), dtype=torch.float32, device=dev)
        right_fingers = torch.zeros(len(self._cfg.right_hand_joint_names), dtype=torch.float32, device=dev)
        left_fingers[self._left_scatter] = torch.tensor(left_q, dtype=torch.float32, device=dev)
        right_fingers[self._right_scatter] = torch.tensor(right_q, dtype=torch.float32, device=dev)

        return torch.cat([left_wrist, right_wrist, left_fingers, right_fingers])


@dataclass
class FrankaDuoSharpaRetargeterCfg(RetargeterCfg):
    """Configuration for :class:`FrankaDuoSharpaRetargeter`.

    The wrist rotation offsets map the OpenXR wrist frame onto the panda_link8
    flange frame such that the rig's IK-solved ready pose (fingers forward, palms
    down) corresponds to the same human pose. Defaults are derived analytically
    from the OpenXR hand-joint convention and validated against the rig's ready
    pose by ``calibrate_sharpa_duo.py``; override via that script's output if the
    hands track with a constant twist.
    """

    left_hand_joint_names: list[str] = field(default_factory=list)
    right_hand_joint_names: list[str] = field(default_factory=list)
    left_dex_config: str = "sharpa_wave_left_dexpilot.yml"
    right_dex_config: str = "sharpa_wave_right_dexpilot.yml"
    # w,x,y,z — calibrated by calibrate_sharpa_duo.py at the rig's ready pose.
    # The clean cos/sin(22.5°) structure is the rig's ∓45° flange mounts.
    left_wrist_rot_offset: tuple[float, float, float, float] = (0.0, -0.9238795, 0.3826834, 0.0)
    right_wrist_rot_offset: tuple[float, float, float, float] = (0.0, 0.3826834, -0.9238795, 0.0)
    # Human-wrist -> flange pullback, in the flange local frame (meters).
    # Calibration measured |flange -> sharpa hand_wrist| = 0.0005 m: coincident.
    wrist_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Draw red spheres on the tracked OpenXR hand joints (GR1T2 teleop pattern).
    enable_visualization: bool = False
    retargeter_type: type[RetargeterBase] = FrankaDuoSharpaRetargeter
    pinch_separation: float = -0.02  # meters; Overshoot pinch closure to generate grip force.
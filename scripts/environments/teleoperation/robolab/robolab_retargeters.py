# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Retargeter adapters that map Isaac Lab XR hand tracking onto RoboLab's DROID/Franka
absolute-IK action space.

RoboLab's ``DroidIKActionCfg`` expects an 8-D action per env:

    [x, y, z, qw, qx, qy, qz, gripper]

where the pose targets the robot's gripper mount frame (``base_link``, i.e. the
Robotiq 2F-85 flange) and ``gripper`` is 0.0 (open) .. 1.0 (close), binarized at 0.5.

Isaac Lab's stock retargeters produce:

* :class:`~isaaclab.devices.openxr.retargeters.Se3AbsRetargeter` — 7-D absolute
  end-effector pose (w,x,y,z quaternion) in the *intuitive* end-effector frame.
* :class:`~isaaclab.devices.openxr.retargeters.GripperRetargeter` — 1-D command
  where -1.0 = close, +1.0 = open.

The two adapters below bridge the conventions:

* :class:`RobolabAbsIKRetargeter` post-multiplies the commanded orientation by the
  inverse of RoboLab's ``EEF_OFFSET_ROT`` so that a hand pose expressed in the
  natural end-effector frame becomes a valid ``base_link`` IK target. This mirrors
  the command-side conversion in RoboLab's ``examples/run_abs_ik_demo.py``
  (``target_base_quat = target_eef_quat ⊗ R_offset⁻¹``).
* :class:`RobolabGripperRetargeter` remaps -1/+1 to 1.0/0.0.

Concatenated in this order by the device layer, they emit exactly the 8-D action
``DroidIKActionCfg`` consumes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaaclab.devices.retargeter_base import RetargeterBase
from isaaclab.devices.openxr.retargeters import (
    GripperRetargeter,
    GripperRetargeterCfg,
    Se3AbsRetargeter,
    Se3AbsRetargeterCfg,
)
from isaaclab.utils.math import quat_inv, quat_mul

# Default matches robolab.robots.droid.EEF_OFFSET_ROT (w, x, y, z). The teleop
# script passes the value imported from robolab so the two never drift apart.
_DEFAULT_EEF_OFFSET_ROT = (0.5, -0.5, 0.5, -0.5)


class RobolabAbsIKRetargeter(Se3AbsRetargeter):
    """Se3AbsRetargeter whose output orientation is converted to RoboLab's base_link frame."""

    def __init__(self, cfg: RobolabAbsIKRetargeterCfg):
        super().__init__(cfg)
        offset = torch.tensor(cfg.eef_offset_rot, dtype=torch.float32, device=self._sim_device)
        self._eef_offset_rot_inv = quat_inv(offset.unsqueeze(0))

    def retarget(self, data: dict) -> torch.Tensor:
        pose = super().retarget(data)
        # target_base_quat = target_eef_quat ⊗ R_offset⁻¹  (all w,x,y,z)
        quat = quat_mul(pose[3:7].unsqueeze(0), self._eef_offset_rot_inv).squeeze(0)
        return torch.cat([pose[:3], quat])


@dataclass
class RobolabAbsIKRetargeterCfg(Se3AbsRetargeterCfg):
    """Configuration for :class:`RobolabAbsIKRetargeter`."""

    eef_offset_rot: tuple[float, float, float, float] = _DEFAULT_EEF_OFFSET_ROT
    retargeter_type: type[RetargeterBase] = RobolabAbsIKRetargeter


class RobolabGripperRetargeter(GripperRetargeter):
    """GripperRetargeter remapped to RoboLab's zero-to-one convention (0=open, 1=close)."""

    def retarget(self, data: dict) -> torch.Tensor:
        command = super().retarget(data)  # [-1.0] close / [+1.0] open
        return (1.0 - command) * 0.5  # -> [1.0] close / [0.0] open


@dataclass
class RobolabGripperRetargeterCfg(GripperRetargeterCfg):
    """Configuration for :class:`RobolabGripperRetargeter`."""

    retargeter_type: type[RetargeterBase] = RobolabGripperRetargeter

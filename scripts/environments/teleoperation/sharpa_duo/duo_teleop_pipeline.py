# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IsaacTeleop retargeting pipeline for the FR3 Duo + SharpaWave rig.

Builds the hand-tracking → 58-D action pipeline consumed by
:class:`isaaclab_teleop.IsaacTeleopDevice`, following the same structure as the
GR1T2 pick-place pipeline that ships with ``isaaclab_tasks``:

    XR hand tracking (26 joints per hand, via CloudXR)
      ├─ wrists → 2 × Se3AbsRetargeter   → absolute flange pose targets (world frame)
      └─ fingers → 2 × DexHandRetargeter → 22 SharpaWave joints per hand (DexPilot QP
                                            against the vendored URDFs)
      └─ TensorReorderer → [L wrist 7 | R wrist 7 | L fingers 22 | R fingers 22]

Wrist rotation offsets map the OpenXR wrist frame onto the ``panda_link8``
flange frame so that the rig's IK-solved ready pose (fingers forward, palms
down) corresponds to the same human pose. They were calibrated on the original
``feature/robolab-xr-teleop`` branch as quaternions — left ``(0, -0.924, 0.383, 0)``,
right ``(0, 0.383, -0.924, 0)`` (w,x,y,z); the clean cos/sin(22.5°) structure is
the rig's ∓45° flange mounts surfacing. ``Se3RetargeterConfig`` takes intrinsic
XYZ Euler angles in degrees and composes them wrist-side (``wrist ⊗ offset``,
same convention as the calibration), so the quaternions convert exactly to the
roll/pitch/yaw values below.

Quaternion order: ``Se3AbsRetargeter`` emits ``[pos, quat xyzw]``, which is
exactly what this Isaac Lab release's math utilities and IK actions consume
(the 3.0 data/math stack moved to the Warp ``xyzw`` layout), so wrist poses
pass through the reorderer unchanged.
"""

from __future__ import annotations

import os

from duo_robot import DEX_RETARGETING_DIR, FINGER_JOINTS, sided

# OpenXR-wrist-frame → MANO-frame change of basis used by dex_retargeting
# (row-major 3x3). Same constant as the GR1T2 pipeline and the original
# SharpaWave retargeter.
_OPERATOR2MANO = (0, -1, 0, -1, 0, 0, 0, 0, -1)

# Calibrated wrist offsets (see module docstring), as intrinsic XYZ Euler degrees.
_WRIST_OFFSET_RPY = {
    "left": (-180.0, 0.0, 45.0),
    "right": (180.0, 0.0, 135.0),
}


def build_duo_pipeline():
    """Build the duo teleop retargeting pipeline.

    Returns:
        A tuple ``(pipeline, retargeters)``: the ``OutputCombiner`` with the
        single ``"action"`` output that :class:`~isaaclab_teleop.IsaacTeleopDevice`
        expects, and the list of retargeter nodes for the tuning UI.
    """
    from isaacteleop.retargeters import (
        DexHandRetargeter,
        DexHandRetargeterConfig,
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    hands = HandsSource(name="hands")

    # World-to-anchor transform provided by IsaacTeleopDevice: wrist poses must be
    # expressed in the sim world frame. Finger retargeting is wrist-relative and
    # therefore frame-invariant, so the dex retargeters read the raw hands.
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_hands = hands.transformed(transform_input.output(ValueInput.VALUE))

    sides = {"left": HandsSource.LEFT, "right": HandsSource.RIGHT}
    se3_nodes = {}
    dex_nodes = {}
    retargeters = []
    for side, source in sides.items():
        roll, pitch, yaw = _WRIST_OFFSET_RPY[side]
        se3 = Se3AbsRetargeter(
            Se3RetargeterConfig(
                input_device=source,
                zero_out_xy_rotation=False,
                use_wrist_rotation=True,
                use_wrist_position=True,
                target_offset_roll=roll,
                target_offset_pitch=pitch,
                target_offset_yaw=yaw,
            ),
            name=f"{side}_ee_pose",
        )
        se3_nodes[side] = se3.connect({source: transformed_hands.output(source)})

        dex = DexHandRetargeter(
            DexHandRetargeterConfig(
                hand_retargeting_config=os.path.join(DEX_RETARGETING_DIR, f"sharpa_wave_{side}_dexpilot.yml"),
                hand_urdf=os.path.join(DEX_RETARGETING_DIR, f"{side}_sharpa_wave_with_flange.urdf"),
                hand_joint_names=sided(FINGER_JOINTS, side),
                hand_side=side,
                handtracking_to_baselink_frame_transform=_OPERATOR2MANO,
            ),
            name=f"{side}_hand",
        )
        dex_nodes[side] = dex.connect({source: hands.output(source)})
        retargeters += [se3, dex]

    # Se3AbsRetargeter output element order is [x, y, z, qx, qy, qz, qw] — already
    # the xyzw layout the IK actions expect, so the wrists pass through unchanged.
    ee_elements = {
        side: [f"{side[0]}_{el}" for el in ("pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w")]
        for side in sides
    }

    reorderer = TensorReorderer(
        input_config={
            "left_ee_pose": ee_elements["left"],
            "right_ee_pose": ee_elements["right"],
            "left_hand_joints": sided(FINGER_JOINTS, "left"),
            "right_hand_joints": sided(FINGER_JOINTS, "right"),
        },
        output_order=(
            ee_elements["left"] + ee_elements["right"] + sided(FINGER_JOINTS, "left") + sided(FINGER_JOINTS, "right")
        ),
        name="action_reorderer",
        input_types={
            "left_ee_pose": "array",
            "right_ee_pose": "array",
            "left_hand_joints": "scalar",
            "right_hand_joints": "scalar",
        },
    )
    connected_reorderer = reorderer.connect(
        {
            "left_ee_pose": se3_nodes["left"].output("ee_pose"),
            "right_ee_pose": se3_nodes["right"].output("ee_pose"),
            "left_hand_joints": dex_nodes["left"].output("hand_joints"),
            "right_hand_joints": dex_nodes["right"].output("hand_joints"),
        }
    )

    pipeline = OutputCombiner({"action": connected_reorderer.output("output")})
    return pipeline, retargeters

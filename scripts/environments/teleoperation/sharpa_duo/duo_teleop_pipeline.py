# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IsaacTeleop retargeting pipeline for the FR3 Duo + SharpaWave rig.

Builds the hand-tracking → 58-D action pipeline consumed by
:class:`isaaclab_teleop.IsaacTeleopDevice`, following the same structure as the
GR1T2 pick-place pipeline that ships with ``isaaclab_tasks``:

    XR hand tracking (26 joints per hand, via CloudXR)
      ├─ wrists → 2 × Se3AbsRetargeter        → absolute flange pose targets (world frame)
      └─ fingers → 2 × SharpaDexHandRetargeter → 22 SharpaWave joints per hand (DexPilot QP
                                                 + operator calibration, sharpa_retargeting.py)
      └─ TensorReorderer → [L wrist 7 | R wrist 7 | L fingers 22 | R fingers 22]

When the operator hand-shape calibration is loaded, its rotation additionally
composes into each wrist offset (``q_flange = q_wrist ⊗ q_corr ⊗ q_offset``) so
the arm tilts the robot hand until its fingers align with the operator's.

Wrist rotation offsets map the OpenXR wrist frame onto the frame of the body
each arm's IK commands, so the SharpaWave hand aligns with the operator's hand.
They are per-embodiment (:attr:`duo_robot.Embodiment.wrist_offsets_xyzw`):

- ``franka_duo`` commands the ``panda_link8`` flanges. Its offsets were
  calibrated on the original ``feature/robolab-xr-teleop`` branch for the rig's
  factory ∓45° flange mounts — left ``(0, -0.924, 0.383, 0)``, right
  ``(0, 0.383, -0.924, 0)`` (w,x,y,z). The hands are now re-clocked on the bolt
  circle to left ``Rz(-45°)`` / right ``Rz(-135°)`` (see the robot USD), and
  because ``hand = flange ⊗ Rz(mount)``, alignment is preserved by folding the
  re-clock into the offsets: ``offset_new = offset_calibrated ⊗ Rz(m_cal −
  m_new)`` = ``⊗ Rz(+90°)`` on both sides.
- ``yam_duo`` commands the hands' own wrist bodies, so its offsets are the pure
  OpenXR-wrist→SharpaWave-wrist rotation (the flange offsets above with the
  mount clocking folded back in), independent of the hand's mount on the arm.

``Se3RetargeterConfig`` takes intrinsic XYZ Euler angles in degrees and
composes them wrist-side (``wrist ⊗ offset``, same convention as the
calibration), so the quaternions convert exactly to roll/pitch/yaw.

Quaternion order: ``Se3AbsRetargeter`` emits ``[pos, quat xyzw]``, which is
exactly what this Isaac Lab release's math utilities and IK actions consume
(the 3.0 data/math stack moved to the Warp ``xyzw`` layout), so wrist poses
pass through the reorderer unchanged.
"""

from __future__ import annotations

from duo_robot import EMBODIMENTS, FINGER_JOINTS, sided
from scipy.spatial.transform import Rotation as R
from sharpa_retargeting import load_hand_calibration, make_sharpa_dex_node, wrist_correction


def build_duo_pipeline(
    include_xr_hands: bool = False,
    include_xr_controllers: bool = True,
    hand_calibration: str | None = "hand_calibration.yml",
    wrist_offsets_xyzw: dict[str, tuple[float, float, float, float]] | None = None,
):
    """Build the duo teleop retargeting pipeline.

    Args:
        include_xr_hands: Append the raw 26-joint hand poses (see
            :mod:`xr_extras`, ``XR_EXTRAS_DIM`` elements, sim world frame) after
            the 58 action elements. The teleop loop slices them off before
            ``env.step`` and uses them for the hand markers and the
            ``obs/xr_hands`` recording.
        include_xr_controllers: Also append the raw controller block
            (``XR_CONTROLLERS_DIM`` elements) after the hands — consumed only
            by the adjust mode. Ignored when ``include_xr_hands`` is False.
        hand_calibration: Operator hand-shape calibration yml (see
            :mod:`sharpa_retargeting`), resolved against the vendored
            ``assets/dex_retargeting`` directory. None or "" disables it; a
            missing file is announced and ignored.
        wrist_offsets_xyzw: Per-side ("left"/"right") rotation offsets, as xyzw
            quaternions, mapping the OpenXR wrist frame onto the frame of the
            body the arms' IK commands (``Embodiment.wrist_offsets_xyzw``).
            Defaults to the Franka duo flange offsets.

    Returns:
        A tuple ``(pipeline, retargeters)``: the ``OutputCombiner`` with the
        single ``"action"`` output that :class:`~isaaclab_teleop.IsaacTeleopDevice`
        expects, and the list of retargeter nodes for the tuning UI.
    """
    from isaacteleop.retargeters import Se3AbsRetargeter, Se3RetargeterConfig, TensorReorderer
    from isaacteleop.retargeting_engine.deviceio_source_nodes import HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    if wrist_offsets_xyzw is None:
        wrist_offsets_xyzw = EMBODIMENTS["franka_duo"].wrist_offsets_xyzw
    calibration = load_hand_calibration(hand_calibration) if hand_calibration else None
    if hand_calibration and calibration is None:
        print(f"[WARNING] Hand calibration '{hand_calibration}' not found; retargeting uncalibrated.")
    if calibration:
        for side in sorted(calibration):
            cal = calibration[side]
            print(
                f"[INFO] Hand calibration ({side}): scale {cal['scale']:.3f}, thumb ratio"
                f" {cal['thumb_ratio']:.3f}, pinky ratio {cal['pinky_ratio']:.3f}; rotation folded"
                " into the wrist offset."
            )

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
        # q_flange = q_wrist ⊗ q_corr ⊗ q_offset: the wrist-side calibration
        # rotation composes into the constant wrist→flange offset, so the arm
        # tilts the robot hand until its FINGERS align with the operator's.
        offset = R.from_quat(wrist_offsets_xyzw[side])
        cal = (calibration or {}).get(side)
        if cal is not None:
            offset = R.from_matrix(wrist_correction(cal["rotation"])) * offset
        roll, pitch, yaw = offset.as_euler("XYZ", degrees=True)
        se3 = Se3AbsRetargeter(
            Se3RetargeterConfig(
                input_device=source,
                zero_out_xy_rotation=False,
                use_wrist_rotation=True,
                use_wrist_position=True,
                target_offset_roll=float(roll),
                target_offset_pitch=float(pitch),
                target_offset_yaw=float(yaw),
            ),
            name=f"{side}_ee_pose",
        )
        se3_nodes[side] = se3.connect({source: transformed_hands.output(source)})

        dex = make_sharpa_dex_node(side, sided(FINGER_JOINTS, side), cal)
        dex_nodes[side] = dex.connect({f"hand_{side}": hands.output(source)})
        retargeters += [se3, dex]

    # Se3AbsRetargeter output element order is [x, y, z, qx, qy, qz, qw] — already
    # the xyzw layout the IK actions expect, so the wrists pass through unchanged.
    ee_elements = {
        side: [f"{side[0]}_{el}" for el in ("pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w")]
        for side in sides
    }

    input_config = {
        "left_ee_pose": ee_elements["left"],
        "right_ee_pose": ee_elements["right"],
        "left_hand_joints": sided(FINGER_JOINTS, "left"),
        "right_hand_joints": sided(FINGER_JOINTS, "right"),
    }
    output_order = (
        ee_elements["left"] + ee_elements["right"] + sided(FINGER_JOINTS, "left") + sided(FINGER_JOINTS, "right")
    )
    input_types = {
        "left_ee_pose": "array",
        "right_ee_pose": "array",
        "left_hand_joints": "scalar",
        "right_hand_joints": "scalar",
    }
    connections = {
        "left_ee_pose": se3_nodes["left"].output("ee_pose"),
        "right_ee_pose": se3_nodes["right"].output("ee_pose"),
        "left_hand_joints": dex_nodes["left"].output("hand_joints"),
        "right_hand_joints": dex_nodes["right"].output("hand_joints"),
    }

    if include_xr_hands:
        from xr_extras import XR_HAND_ELEMENTS, make_hands_passthrough

        # Raw hand poses ride along after the action elements, in the sim world
        # frame (same transform as the wrists). NOTE: deliberately no HeadSource
        # here — a head tracker in the pipeline made every DeviceIO session step
        # fail on the Quest/Kit-bridge stack (teardown/recreate loop, robot never
        # moved); the align command queries the head via XRCore on demand instead
        # (xr_extras.current_head_pose).
        passthrough = make_hands_passthrough()
        connected_passthrough = passthrough.connect(
            {
                "hand_left": transformed_hands.output(HandsSource.LEFT),
                "hand_right": transformed_hands.output(HandsSource.RIGHT),
            }
        )
        input_config["xr_hands"] = list(XR_HAND_ELEMENTS)
        output_order = output_order + list(XR_HAND_ELEMENTS)
        input_types["xr_hands"] = "scalar"
        connections["xr_hands"] = connected_passthrough.output("xr_hands")

    if include_xr_hands and include_xr_controllers:
        # Raw controller aim poses, trigger, and thumbstick ride along after
        # the hands block, also in the sim world frame — the adjust mode's
        # value input (thumbstick range steps, trigger-taps; see adjust_mode).
        from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
        from isaacteleop.retargeting_engine.utilities import ControllerTransform
        from xr_extras import XR_CONTROLLER_ELEMENTS, make_controllers_passthrough

        controllers = ControllersSource("xr_controllers_source")
        controller_xform = ControllerTransform("xr_controllers_world")
        transformed_controllers = controller_xform.connect(
            {
                ControllerTransform.LEFT: controllers.output(ControllersSource.LEFT),
                ControllerTransform.RIGHT: controllers.output(ControllersSource.RIGHT),
                "transform": transform_input.output(ValueInput.VALUE),
            }
        )
        controllers_passthrough = make_controllers_passthrough()
        connected_controllers = controllers_passthrough.connect(
            {
                "controller_left": transformed_controllers.output(ControllerTransform.LEFT),
                "controller_right": transformed_controllers.output(ControllerTransform.RIGHT),
            }
        )
        input_config["xr_controllers"] = list(XR_CONTROLLER_ELEMENTS)
        output_order = output_order + list(XR_CONTROLLER_ELEMENTS)
        input_types["xr_controllers"] = "scalar"
        connections["xr_controllers"] = connected_controllers.output("xr_controllers")

    reorderer = TensorReorderer(
        input_config=input_config,
        output_order=output_order,
        name="action_reorderer",
        input_types=input_types,
    )
    connected_reorderer = reorderer.connect(connections)

    pipeline = OutputCombiner({"action": connected_reorderer.output("output")})
    return pipeline, retargeters

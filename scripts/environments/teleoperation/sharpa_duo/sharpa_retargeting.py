# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Calibrated SharpaWave finger retargeting (DexPilot + operator hand-shape calibration).

Port of the ``feature/robolab-xr-teleop`` branch's ``SharpaWaveDexRetargeting``:
DexPilot (``dex_retargeting``) against the vendored SharpaWave URDFs, with the
operator hand-shape calibration from ``assets/dex_retargeting/hand_calibration.yml``
(written by that branch's ``calibrate_hand_shape.py``). Per side, the calibration
holds a global rotation + scale (wrist-pinned Procrustes on index/middle/ring
tips) and per-finger thumb/pinky corrections (length ratio + tip-direction
rotation), measured with the operator's hands flat.

How the pieces are applied (identical to the source branch):

- **Rotation, wrist-side**: the calibration rotation is inverted, moved into
  OpenXR wrist axes (``M @ R_cal.T @ M.T`` with ``M = OPERATOR2MANO``), and
  right-multiplied onto the tracked wrist rotation. Expressing the raw keypoints
  in that rotated wrist frame reproduces ``R_cal @ p`` exactly for the fingers,
  and the same constant can be composed into the wrist→flange offset so the arm
  tilts the robot hand until its FINGERS align with the operator's (see
  :func:`wrist_correction` and its use in ``duo_teleop_pipeline``).
- **Scale** multiplies the keypoints; the DexPilot yml ``scaling_factor`` is
  overridden to 1.0 so the optimizer does not scale again.
- **Thumb/pinky corrections** apply after the global transform, about the wrist:
  ``p' = ratio * R_f @ p`` on that finger's rows only (those fingers are excluded
  from the Procrustes fit, so they get their own).
- **Pinch detection on RAW distances**: DexPilot's internal project/escape
  hysteresis would see calibrated (inflated/distorted) distances, so its
  thresholds are disabled and the same hysteresis runs here on the operator's
  real fingertip gaps, writing ``optimizer.projected[:4]`` directly.

Quaternions are xyzw throughout (isaacteleop's HandInput layout; scipy's
native order). Import only after AppLauncher.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from duo_robot import DEX_RETARGETING_DIR
from scipy.spatial.transform import Rotation as R

# OpenXR 26-joint hand -> 21-joint MANO-style ordering used by dex_retargeting
# (drops palm and the four non-thumb *_metacarpal joints).
_HAND_JOINTS_INDEX = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25]
_WRIST_INDEX = 1  # OpenXR wrist joint

# OpenXR-wrist-frame -> MANO-frame change of basis.
OPERATOR2MANO = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], dtype=np.float64)

# MANO-21 rows: 0 wrist | 1-4 thumb | 5-8 index | 9-12 middle | 13-16 ring | 17-20 pinky
_MANO_TIP_ROWS = [4, 8, 12, 16, 20]
_FINGER_ROWS = {"thumb": slice(1, 5), "pinky": slice(17, 21)}


def load_hand_calibration(path: str) -> dict[str, dict] | None:
    """Load a hand-shape calibration yml written by ``calibrate_hand_shape.py``.

    Returns ``{side: {rotation 3x3, scale, {thumb,pinky}_ratio, {thumb,pinky}_rotation}}``
    for the sides present, or ``None`` if the file does not exist. Relative paths
    resolve against the vendored ``assets/dex_retargeting`` directory.
    """
    import yaml

    if not os.path.isabs(path):
        path = os.path.join(DEX_RETARGETING_DIR, path)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = yaml.safe_load(f)
    out = {}
    for side in ("left", "right"):
        if isinstance(data.get(side), dict):
            entry = {
                "rotation": np.asarray(data[side]["rotation"], dtype=np.float64),
                "scale": float(data[side]["scale"]),
            }
            for finger in ("thumb", "pinky"):
                entry[f"{finger}_ratio"] = float(data[side].get(f"{finger}_ratio", 1.0))
                entry[f"{finger}_rotation"] = np.asarray(
                    data[side].get(f"{finger}_rotation", np.eye(3).tolist()), dtype=np.float64
                )
            # File-level flag: whether the calibration was captured with the tips
            # extended to the skin surface (see extend_fingertips); runtime must match.
            entry["tip_extension"] = bool(data.get("tip_extension", False))
            out[side] = entry
    return out or None


def wrist_correction(cal_rotation: np.ndarray) -> np.ndarray:
    """The calibration rotation, inverted and moved into OpenXR wrist axes (3x3).

    Right-multiplies the tracked wrist rotation; see the module docstring.
    """
    return OPERATOR2MANO @ cal_rotation.T @ OPERATOR2MANO.T


def convert_hand_joints(
    positions: np.ndarray, wrist_quat_xyzw: np.ndarray, wrist_rot_corr: np.ndarray | None = None
) -> np.ndarray:
    """26 OpenXR joint positions -> wrist-relative 21x3 MANO-style positions.

    Frame-invariant: ``positions`` and the wrist quaternion just have to be in the
    SAME frame. ``wrist_rot_corr`` (3x3, optional) right-multiplies the tracked
    wrist rotation — i.e. it rotates the wrist FRAME the points are expressed in;
    passing :func:`wrist_correction`'s output reproduces ``R_cal @ p`` exactly.
    """
    pts = positions[_HAND_JOINTS_INDEX].astype(np.float64, copy=True)
    pts -= pts[0:1]
    wrist_rot = R.from_quat(np.asarray(wrist_quat_xyzw, dtype=np.float64)).as_matrix()
    if wrist_rot_corr is not None:
        wrist_rot = wrist_rot @ wrist_rot_corr
    return pts @ wrist_rot @ OPERATOR2MANO


class SharpaWaveDexPilot:
    """DexPilot retargeting for one SharpaWave hand, with optional calibration."""

    def __init__(
        self,
        side: str,
        calibration: dict | None = None,
        pinch_separation: float = 0.0,
        raw_pinch_detection: bool = True,
    ):
        import yaml
        from dex_retargeting.retargeting_config import RetargetingConfig

        cfg_path = os.path.join(DEX_RETARGETING_DIR, f"sharpa_wave_{side}_dexpilot.yml")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        # Resolve the URDF against the vendored dir without mutating the yml on disk.
        cfg["retargeting"]["urdf_path"] = os.path.join(
            DEX_RETARGETING_DIR, os.path.basename(cfg["retargeting"]["urdf_path"])
        )
        self._retargeting = RetargetingConfig.from_dict(cfg["retargeting"]).build()
        opt = self._retargeting.optimizer

        # Commanded thumb-finger separation once a pinch is projected (library
        # default eta1 = +1e-4 m: touch). Negative = overshoot past contact so the
        # PD drives squeeze the object. Only the first 4 entries (thumb pairs) —
        # entries 4:10 are finger-finger spacing (eta2) and must stay +0.03.
        opt.projected_dist[:4] = pinch_separation

        self._cal = calibration
        self._tip_extension = bool(calibration.get("tip_extension", False)) if calibration else False
        self._wrist_corr: np.ndarray | None = None
        self._finger_corrections: list[tuple[slice, float, np.ndarray]] = []
        self._pinch_state: np.ndarray | None = None
        if calibration is not None:
            self._wrist_corr = wrist_correction(calibration["rotation"])
            # The calibrated scale is baked into the keypoints; the optimizer must
            # not scale again.
            opt.scaling = 1.0
            for finger, rows in _FINGER_ROWS.items():
                ratio = calibration[f"{finger}_ratio"]
                rot = calibration[f"{finger}_rotation"]
                if ratio != 1.0 or not np.allclose(rot, np.eye(3)):
                    self._finger_corrections.append((rows, ratio, rot))
            # Pinch detection on RAW human distances: the optimizer's internal
            # projection update sees CALIBRATED vectors, so detection there no
            # longer reflects the physical fingers. Disable it (-inf/+inf make the
            # internal S1 update a no-op) and run the same project/escape
            # hysteresis on raw keypoints in :meth:`compute`.
            if raw_pinch_detection and hasattr(opt, "projected"):
                self._pinch_thresholds = (float(opt.project_dist), float(opt.escape_dist))
                self._pinch_state = np.zeros(4, dtype=bool)
                opt.project_dist = -np.inf
                opt.escape_dist = np.inf

        #: DexPilot output joint order (the URDF's actuated joints).
        self.dof_joint_names: list[str] = list(opt.robot.dof_joint_names)

    def reset(self) -> None:
        if self._pinch_state is not None:
            self._pinch_state[:] = False

    def compute(
        self, positions: np.ndarray, wrist_quat_xyzw: np.ndarray, radii: np.ndarray | None = None
    ) -> np.ndarray:
        """26 OpenXR joint positions + wrist quat -> joint angles in ``dof_joint_names`` order."""
        if self._tip_extension:
            positions = extend_fingertips(positions, radii)
        pts = convert_hand_joints(positions, wrist_quat_xyzw, self._wrist_corr)
        if self._cal is not None:
            pts = self._cal["scale"] * pts
            # Per-finger thumb/pinky corrections about the wrist, after the global
            # transform: p' = ratio * R_f @ p on that finger's rows only.
            for rows, ratio, rot in self._finger_corrections:
                pts[rows] = ratio * (pts[rows] @ rot.T)
        if self._pinch_state is not None:
            # Same project/escape hysteresis as DexPilot, on the ORIGINAL
            # (uncalibrated) fingertip distances. Pair order matches the
            # optimizer's S1 layout: thumb-index, thumb-middle, thumb-ring, thumb-pinky.
            raw_tips = convert_hand_joints(positions, wrist_quat_xyzw)[_MANO_TIP_ROWS]
            dists = np.linalg.norm(raw_tips[1:] - raw_tips[0], axis=1)
            project_dist, escape_dist = self._pinch_thresholds
            self._pinch_state[dists < project_dist] = True
            self._pinch_state[dists > escape_dist] = False
            self._retargeting.optimizer.projected[:4] = self._pinch_state
        indices = self._retargeting.optimizer.target_link_human_indices
        if self._retargeting.optimizer.retargeting_type == "POSITION":
            ref_value = pts[indices, :]
        else:
            ref_value = pts[indices[1, :], :] - pts[indices[0, :], :]
        # dex_retargeting optimizes with gradients; escape any ambient inference mode.
        with torch.enable_grad(), torch.inference_mode(False):
            return self._retargeting.retarget(ref_value)


def make_sharpa_dex_node(
    side: str,
    hand_joint_names: list[str],
    calibration: dict | None,
    pinch_separation: float = 0.0,
):
    """Build the isaacteleop pipeline node wrapping :class:`SharpaWaveDexPilot`.

    Outputs one named scalar per entry of ``hand_joint_names`` (the action-space
    order); DexPilot's outputs are scattered into it by joint name. An untracked
    hand emits zeros and resets the pinch hysteresis.
    """
    from isaacteleop.retargeting_engine.interface import BaseRetargeter
    from isaacteleop.retargeting_engine.interface.tensor_group_type import OptionalType, TensorGroupType
    from isaacteleop.retargeting_engine.tensor_types import FloatType, HandInput, HandInputIndex

    dex = SharpaWaveDexPilot(side, calibration=calibration, pinch_separation=pinch_separation)
    scatter = [hand_joint_names.index(n) for n in dex.dof_joint_names]
    needed = [_WRIST_INDEX, *_HAND_JOINTS_INDEX]

    class SharpaDexHandRetargeter(BaseRetargeter):
        """Calibrated DexPilot fingers for one SharpaWave hand."""

        def input_spec(self):
            return {f"hand_{side}": OptionalType(HandInput())}

        def output_spec(self):
            return {"hand_joints": TensorGroupType(f"hand_joints_{side}", [FloatType(n) for n in hand_joint_names])}

        def _compute_fn(self, inputs, outputs, context) -> None:
            out = outputs["hand_joints"]
            group = inputs[f"hand_{side}"]
            values = np.zeros(len(hand_joint_names))
            if group.is_none:
                dex.reset()
            else:
                positions = np.from_dlpack(group[HandInputIndex.JOINT_POSITIONS])  # (26, 3)
                orientations = np.from_dlpack(group[HandInputIndex.JOINT_ORIENTATIONS])  # (26, 4) xyzw
                radii = np.from_dlpack(group[HandInputIndex.JOINT_RADII])  # (26,)
                valid = np.from_dlpack(group[HandInputIndex.JOINT_VALID]).astype(bool)  # (26,)
                if valid[needed].all():
                    values[scatter] = dex.compute(positions, orientations[_WRIST_INDEX], radii)
                else:
                    dex.reset()
            for i, v in enumerate(values):
                out[i] = float(v)

    return SharpaDexHandRetargeter(name=f"{side}_hand")


# OpenXR tip joint -> its distal joint, for the fingertip extension direction.
_TIP_TO_DISTAL = {5: 4, 10: 9, 15: 14, 20: 19, 25: 24}


def extend_fingertips(positions: np.ndarray, radii: np.ndarray | None) -> np.ndarray:
    """Move the five tip joints from the capsule center to the skin surface.

    OpenXR (and Meta's runtime) place each ``*_TIP`` joint at the center of the
    fingertip capsule — about one tip-radius INSIDE the skin — while MANO
    fingertip keypoints (which the DexPilot configs were built against) sit on
    the skin surface, so Quest fingers read ~1 cm short and touching fingertips
    still read ~2 cm apart. Extend each tip along its distal-bone direction by
    the runtime-reported joint radius: ``tip' = tip + r * dir(tip - distal)``.

    Returns a copy; the input is untouched. No-op when ``radii`` is None.
    """
    if radii is None:
        return positions
    out = positions.astype(np.float64, copy=True)
    for tip, distal in _TIP_TO_DISTAL.items():
        r = float(radii[tip])
        if r <= 0.0:
            continue
        direction = out[tip] - out[distal]
        norm = float(np.linalg.norm(direction))
        if norm > 1e-6:
            out[tip] = out[tip] + (r / norm) * direction
    return out

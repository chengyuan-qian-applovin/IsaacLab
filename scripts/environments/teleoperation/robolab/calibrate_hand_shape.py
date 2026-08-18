# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hand-shape calibration scene: align YOUR hand proportions to the SharpaWave hand.

Empty world (no robot, no assets — just a dome light and the red joint markers).
Procedure:

1. Connect the AVP and click **Play**.
2. Hold BOTH hands with fingers straight (a flat, relaxed open hand) and keep them
   visible to the headset.
3. ``--delay`` seconds after Play (default 5), the first valid tracking frame is
   captured. Per hand, the wrist POSES are aligned first — your keypoints are
   expressed in your wrist's local frame (position subtracted, orientation removed:
   ``convert_hand_joints``), and the Sharpa fingertips at q = 0 (straight fingers)
   are expressed in ITS wrist link's local frame — then, treating the two wrist
   frames as the same (human wrist == robot wrist, always), a scale + residual
   rotation ABOUT that common wrist is solved for index/middle/ring fingertips:

       min_{s, R_res}  sum_i || s * R_res @ h_i  -  t_i^wrist ||^2   (closed-form SVD / Umeyama, no translation)

   The wrist itself is the fixed pivot: both point sets have the wrist at the
   origin and no translation is solved, so the wrist maps to the wrist exactly.
   What is SAVED as ``rotation`` is the composite ``R = R_wrist_in_base @ R_res``,
   because DexPilot compares vectors in the URDF base axes at apply time; the
   decomposition (``wrist_rot_in_base``, ``residual_rotation``) is stored alongside
   for inspection. The composite is mathematically identical to solving directly
   against base-frame targets — the wrist-frame formulation makes the residual
   interpretable.
4. The result is written to ``sharpa_dex_retargeting/hand_calibration.yml`` and
   printed. Teleop scenes (e.g. teleop_taco_scene.py) auto-load this file via
   ``FrankaDuoSharpaRetargeterCfg.hand_calibration`` and apply ``p' = s * R @ p`` to
   the MANO keypoints before DexPilot; the yml ``scaling_factor`` is then overridden
   to 1.0 so the calibrated scale is the single source of truth.

Run (same container/setup as the teleop scenes):

    ./isaaclab.sh -p scripts/environments/teleoperation/robolab/calibrate_hand_shape.py --headless
"""

# isort: skip_file
import argparse
import functools
import os
import time

import cv2  # noqa: F401  Must import before isaaclab/omni modules.
import pinocchio  # noqa: F401  Must import before AppLauncher (dex_retargeting builds in-kit).

print = functools.partial(print, flush=True)  # noqa: A001

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Calibrate hand shape (rotation + scale) against the SharpaWave hand.")
parser.add_argument(
    "--delay", type=float, default=5.0,
    help="Seconds after clicking Play before the calibration frame is captured (default 5).",
)
parser.add_argument(
    "--output", type=str, default=None,
    help="Output yml path (default: sharpa_dex_retargeting/hand_calibration.yml, auto-loaded by teleop scenes).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.xr = True

if not args_cli.headless and not os.environ.get("DISPLAY"):
    print("[WARNING] XR in GUI mode without a DISPLAY: the AR session will never start. Add --headless.")

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

"""Rest everything follows."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
import yaml

import isaaclab.sim as sim_utils
from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.openxr import OpenXRDevice, OpenXRDeviceCfg, XrCfg
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from sharpa_duo_retargeters import _DATA_DIR, SharpaWaveDexRetargeting, convert_hand_joints

# Wrist-relative MANO-21 rows of the five fingertips.
_MANO_TIP_ROWS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
# The Procrustes fit uses index/middle/ring (the most consistently tracked digits);
# thumb and pinky instead get PER-FINGER corrections measured after the fit — a
# length ratio (|sharpa tip| / |globally-calibrated tip|) plus the minimal rotation
# about the wrist that swings the calibrated tip direction onto the Sharpa's. Their
# direction differs too much from the Sharpa's (especially the thumb) for the global
# transform to capture without ruining the three fitted fingers.
_FIT_FINGERS = ("index", "middle", "ring")
_RATIO_FINGERS = ("thumb", "pinky")


def rotation_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Minimal rotation (Rodrigues) taking unit vector ``u`` onto unit vector ``v``."""
    c = float(np.clip(u @ v, -1.0, 1.0))
    axis = np.cross(u, v)
    s_ = float(np.linalg.norm(axis))
    if s_ < 1e-9:
        if c > 0:
            return np.eye(3)
        # 180 deg: rotate about any axis perpendicular to u
        perp = np.cross(u, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(u, [0.0, 1.0, 0.0])
        perp /= np.linalg.norm(perp)
        return 2.0 * np.outer(perp, perp) - np.eye(3)
    axis /= s_
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + s_ * K + (1.0 - c) * (K @ K)


@dataclass
class _CaptureCfg(RetargeterCfg):
    """Config for the raw-data capture retargeter."""


class _Capture(RetargeterBase):
    """Passthrough retargeter: declares HAND_TRACKING (so the device polls hands) and
    stores the latest raw data dict. Without any retargeter the OpenXR device requests
    no features and returns an empty dict."""

    def __init__(self, cfg: _CaptureCfg):
        super().__init__(cfg)
        self.latest: dict | None = None

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.HAND_TRACKING]

    def retarget(self, data: dict) -> torch.Tensor:
        self.latest = data
        return torch.zeros(1)


def hand_is_tracked(hand_poses: dict[str, np.ndarray] | None) -> bool:
    """True if the 26-joint dict looks like real tracking, not the untracked default
    (all joints at the origin) or a degenerate frame."""
    if hand_poses is None:
        return False
    pts = np.array([p[:3] for p in hand_poses.values()])
    if np.linalg.norm(hand_poses["wrist"][:3]) < 1e-6:
        return False
    spread = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    return spread > 0.08  # a real open hand spans well over 8 cm


def solve_similarity(h: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Solve min_{s,R} sum_i ||s R h_i - t_i||^2 (no translation: both point sets are
    wrist-relative, i.e. anchored at the origin). Kabsch/Umeyama closed form.

    Args:
        h: (N, 3) source points (human fingertips, wrist-local MANO frame).
        t: (N, 3) target points (Sharpa fingertips, URDF base frame).

    Returns:
        (R, s, rms): rotation (3x3, det +1), scale, and RMS residual in meters.
    """
    C = t.T @ h  # sum_i t_i h_i^T
    U, S, Vt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt
    s = float(np.trace(D @ np.diag(S)) / np.sum(h * h))
    resid = s * (h @ R.T) - t
    rms = float(np.sqrt((resid**2).sum(axis=1).mean()))
    return R, s, rms


def sharpa_reference_tips(
    dex: SharpaWaveDexRetargeting, side: str, cfg_name: str
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """All five SharpaWave fingertips at q = 0 (straight fingers), expressed in the
    WRIST LINK's local frame (full wrist pose removed: position and orientation),
    plus the wrist link's rotation in the URDF base frame.

    Returns:
        ({finger: tip (3,)}, wrist_rot_in_base (3, 3))
    """
    with open(os.path.join(_DATA_DIR, cfg_name)) as f:
        ycfg = yaml.safe_load(f)["retargeting"]
    wrist_link = ycfg["wrist_link_name"]

    robot = dex._dex[side].optimizer.robot
    robot.compute_forward_kinematics(np.zeros(robot.dof))
    wrist_pose = robot.get_link_pose(robot.get_link_index(wrist_link))
    wrist_rot, wrist_pos = wrist_pose[:3, :3], wrist_pose[:3, 3]
    tips = {}
    for finger in _MANO_TIP_ROWS:
        link = next(n for n in ycfg["finger_tip_link_names"] if finger in n)
        tips[finger] = wrist_rot.T @ (robot.get_link_pose(robot.get_link_index(link))[:3, 3] - wrist_pos)
    return tips, wrist_rot


def main():
    output_path = args_cli.output or os.path.join(_DATA_DIR, "hand_calibration.yml")

    # Empty world: a dome light so the compositor has something to render.
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60, device=args_cli.device))
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
    light_cfg.func("/World/DomeLight", light_cfg)

    # Red joint markers so you can see your tracked hands in AR (visual confirmation).
    markers = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/hand_joints",
            markers={
                "joint": sim_utils.SphereCfg(
                    radius=0.005,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        )
    )

    # Sharpa reference (built once; also validates the dex configs/URDFs early).
    dex = SharpaWaveDexRetargeting()
    sharpa_tips = {}
    sharpa_wrist_rot = {}
    for side, cfg_name in (("left", "sharpa_wave_left_dexpilot.yml"), ("right", "sharpa_wave_right_dexpilot.yml")):
        tips, wrist_rot = sharpa_reference_tips(dex, side, cfg_name)
        sharpa_tips[side] = tips
        sharpa_wrist_rot[side] = wrist_rot
        print(f"[INFO] Sharpa {side} reference tips (q=0, wrist frame, m): "
              + ", ".join(f"{f}={np.round(t, 4).tolist()}" for f, t in tips.items()))

    # OpenXR device with the capture retargeter (default anchor at the world origin —
    # only relative hand geometry matters here).
    capture = _Capture(_CaptureCfg())
    device = OpenXRDevice(OpenXRDeviceCfg(xr_cfg=XrCfg()), [capture])

    t_play: float | None = None

    def _start():
        nonlocal t_play
        if t_play is None:
            t_play = time.monotonic()
            print(f"[INFO] Play received: hold BOTH hands flat, fingers straight. "
                  f"Capturing in {args_cli.delay:.0f} s...")

    def _stop():
        nonlocal t_play
        t_play = None
        print("[INFO] Stop received: countdown cancelled. Click Play to re-arm.")

    device.add_callback("START", _start)
    device.add_callback("STOP", _stop)
    device.add_callback("RESET", _stop)

    sim.reset()
    print("[INFO] Hand-shape calibration scene ready.")
    print("[INFO] Connect the AVP, then click Play. Keep fingers of both hands STRAIGHT.")

    last_countdown = None
    warned_invalid = False
    result = None

    while simulation_app.is_running():
        device.advance()  # polls XR, fills capture.latest
        data = capture.latest
        left = data.get(DeviceBase.TrackingTarget.HAND_LEFT) if data else None
        right = data.get(DeviceBase.TrackingTarget.HAND_RIGHT) if data else None

        if left and right:
            pts = np.array([p[:3] for hand in (left, right) for p in hand.values()])
            markers.visualize(translations=torch.tensor(pts, dtype=torch.float32))

        if t_play is not None:
            remaining = args_cli.delay - (time.monotonic() - t_play)
            if remaining > 0:
                if last_countdown != int(remaining):
                    last_countdown = int(remaining)
                    print(f"[INFO] ... {last_countdown + 1}")
            elif hand_is_tracked(left) and hand_is_tracked(right):
                result = {}
                for side, hand in (("left", left), ("right", right)):
                    mano = convert_hand_joints(hand)  # (21, 3) human-wrist frame, uncalibrated
                    human = np.array([mano[_MANO_TIP_ROWS[f]] for f in _FIT_FINGERS])
                    targets = np.array([sharpa_tips[side][f] for f in _FIT_FINGERS])
                    # Wrist poses are aligned by construction (both point sets are in
                    # their own wrist frame, wrist at the origin); solve scale + the
                    # residual rotation ABOUT the common wrist.
                    R_res, s, rms = solve_similarity(human, targets)
                    # DexPilot compares vectors in URDF base axes -> save the composite.
                    R_total = sharpa_wrist_rot[side] @ R_res

                    def _angle(m: np.ndarray) -> float:
                        return float(np.degrees(np.arccos(np.clip((np.trace(m) - 1) / 2, -1.0, 1.0))))

                    # Thumb/pinky: excluded from the fit; measure per-finger corrections
                    # instead, both about the wrist and applied AFTER the global (s, R):
                    # a length ratio plus the minimal rotation that swings the
                    # globally-calibrated tip direction onto the Sharpa's (base axes).
                    finger_corr = {}
                    for f in _RATIO_FINGERS:
                        h_g = s * (R_total @ mano[_MANO_TIP_ROWS[f]])       # after global calib, base axes
                        t = sharpa_wrist_rot[side] @ sharpa_tips[side][f]   # base axes
                        ratio = float(np.linalg.norm(t) / np.linalg.norm(h_g))
                        R_f = rotation_between(h_g / np.linalg.norm(h_g), t / np.linalg.norm(t))
                        finger_corr[f] = (ratio, R_f)

                    result[side] = {
                        "rotation": [[float(v) for v in row] for row in R_total],
                        "scale": s,
                        **{f"{f}_ratio": ratio for f, (ratio, _r) in finger_corr.items()},
                        **{
                            f"{f}_rotation": [[float(v) for v in row] for row in R_f]
                            for f, (_ratio, R_f) in finger_corr.items()
                        },
                        **{f"{f}_angle_deg": _angle(R_f) for f, (_ratio, R_f) in finger_corr.items()},
                        "rms_m": rms,
                        "residual_rotation": [[float(v) for v in row] for row in R_res],
                        "residual_angle_deg": _angle(R_res),
                        "wrist_rot_in_base": [[float(v) for v in row] for row in sharpa_wrist_rot[side]],
                        "human_tips_wrist_frame": {
                            f: [float(v) for v in mano[_MANO_TIP_ROWS[f]]] for f in _MANO_TIP_ROWS
                        },
                        "sharpa_tips_wrist_frame": {
                            f: [float(v) for v in sharpa_tips[side][f]] for f in _MANO_TIP_ROWS
                        },
                    }
                    print(f"[RESULT] {side}: scale={s:.4f}  residual_wrist_rotation={_angle(R_res):.2f} deg  "
                          f"rms={rms * 1000:.2f} mm")
                    for f, (ratio, R_f) in finger_corr.items():
                        print(f"[RESULT] {side} {f}: ratio={ratio:.4f}  rotation={_angle(R_f):.2f} deg")
                    print(f"[RESULT] {side} residual R (about the common wrist) =\n"
                          f"{np.array2string(R_res, precision=5, suppress_small=True)}")
                    print(f"[RESULT] {side} applied composite R (base axes, saved) =\n"
                          f"{np.array2string(R_total, precision=5, suppress_small=True)}")
                break
            else:
                if not warned_invalid:
                    warned_invalid = True
                    print("[WARNING] Hands not (both) tracked at capture time — holding until the first "
                          "valid frame. Keep both hands flat and visible.")
        sim.render()

    if result is not None:
        payload = {
            "generated": datetime.now().isoformat(timespec="seconds"),
            "note": (
                "Hand-shape calibration (calibrate_hand_shape.py). Wrist poses aligned first (human and "
                "robot points each in their own wrist frame, wrists identified); scale + residual rotation "
                "solved about the common wrist on index/middle/ring. 'rotation' is applied WRIST-side "
                "(inverted, composed into the tracked wrist frame: arm target tilts, fingers derive from "
                "the rotated wrist); 'scale' multiplies the keypoints. Thumb and pinky get per-finger "
                "corrections applied after the global transform, about the wrist: {finger}_ratio (length "
                "multiplier) and {finger}_rotation (minimal rotation swinging the calibrated tip direction "
                "onto the Sharpa's; base axes). All hand-tunable. Loading sets the DexPilot scaling to 1.0."
            ),
            **result,
        }
        with open(output_path, "w") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
        print(f"[INFO] Calibration written to {output_path}")
        print("[INFO] teleop scenes auto-load it (FrankaDuoSharpaRetargeterCfg.hand_calibration).")
    else:
        print("[INFO] Exiting without a capture — no calibration written.")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

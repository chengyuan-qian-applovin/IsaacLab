# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hand-shape calibration scene: align YOUR hand proportions to the SharpaWave hand.

Per-user port of the source branch's ``calibrate_hand_shape.py`` onto the
IsaacTeleop stack. Empty world (a dome light and red joint markers only).
Procedure:

1. Run with ``--user <name>``, connect the headset, and press **Play**.
2. Hold BOTH hands with fingers straight (flat, relaxed open hands), visible
   to the headset.
3. ``--delay`` seconds after Play (default 5), the first valid tracking frame
   is captured. Per hand, the wrist POSES are aligned first — your keypoints
   are expressed in your wrist's local frame, and the Sharpa fingertips at
   q = 0 (straight fingers) in ITS wrist link's local frame — then, treating
   the two wrist frames as the same, a scale + residual rotation ABOUT that
   common wrist is solved for index/middle/ring tips (closed-form Umeyama, no
   translation). Thumb and pinky get per-finger corrections (length ratio +
   minimal tip-direction rotation) measured after the fit.
4. The result is written to ``assets/dex_retargeting/hand_calibration_<user>.yml``.
   Teleop then loads it with ``--user <name>``.

Fingertip convention: by default the OpenXR tip joints are extended from the
capsule center to the skin surface using the runtime-reported joint radii
(``sharpa_retargeting.extend_fingertips``) — Quest tips otherwise read ~1 cm
short of the MANO convention the DexPilot configs expect. The choice is
stamped into the yml (``tip_extension``) and the teleop retargeter applies the
same convention automatically, so calibration and runtime always agree.

Run:

    ./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/calibrate_hand_shape.py \\
        --user alice --headless
"""

# isort: skip_file
import argparse
import functools
import os
import time

print = functools.partial(print, flush=True)  # noqa: A001

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Calibrate hand shape (rotation + scale) against the SharpaWave hand.")
parser.add_argument("--user", type=str, required=True, help="User name; writes hand_calibration_<user>.yml.")
parser.add_argument(
    "--delay",
    type=float,
    default=5.0,
    help="Seconds after pressing Play before the calibration frame is captured (default 5).",
)
parser.add_argument(
    "--no_tip_extension",
    action="store_true",
    help=(
        "Calibrate against the raw OpenXR tip joints (capsule centers) instead of extending them to the"
        " skin surface by the runtime-reported radii. The choice is stamped into the yml and mirrored by"
        " the teleop retargeter."
    ),
)
parser.add_argument(
    "--cloudxr_env",
    type=str,
    default="cloudxrjs",
    help="CloudXR .env file path or shorthand ('cloudxrjs' default, 'avp', 'none').",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.xr = True

if not args_cli.headless and not os.environ.get("DISPLAY"):
    print("[WARNING] XR in GUI mode without a DISPLAY: the AR session will never start. Add --headless.")

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

"""Rest everything follows."""

from datetime import datetime

import numpy as np
import torch
import yaml

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from duo_robot import DEX_RETARGETING_DIR
from sharpa_retargeting import SharpaWaveDexPilot, convert_hand_joints, extend_fingertips

# Wrist-relative MANO-21 rows of the five fingertips.
_MANO_TIP_ROWS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
# The Procrustes fit uses index/middle/ring (the most consistently tracked digits);
# thumb and pinky instead get PER-FINGER corrections measured after the fit — a
# length ratio plus the minimal rotation about the wrist that swings the calibrated
# tip direction onto the Sharpa's. Their direction differs too much from the
# Sharpa's (especially the thumb) for the global transform to capture without
# ruining the three fitted fingers.
_FIT_FINGERS = ("index", "middle", "ring")
_RATIO_FINGERS = ("thumb", "pinky")

_WRIST_INDEX = 1


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


def solve_similarity(h: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Solve min_{s,R} sum_i ||s R h_i - t_i||^2 (no translation: both point sets are
    wrist-relative, i.e. anchored at the origin). Kabsch/Umeyama closed form.

    Args:
        h: (N, 3) source points (human fingertips, wrist-local MANO frame) [m].
        t: (N, 3) target points (Sharpa fingertips, URDF base frame) [m].

    Returns:
        (R, s, rms): rotation (3x3, det +1), scale, and RMS residual [m].
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


def sharpa_reference_tips(dex: SharpaWaveDexPilot, side: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """All five SharpaWave fingertips at q = 0 (straight fingers), expressed in the
    WRIST LINK's local frame (full wrist pose removed), plus the wrist link's
    rotation in the URDF base frame.

    Returns:
        ({finger: tip (3,) [m]}, wrist_rot_in_base (3, 3))
    """
    with open(os.path.join(DEX_RETARGETING_DIR, f"sharpa_wave_{side}_dexpilot.yml")) as f:
        ycfg = yaml.safe_load(f)["retargeting"]
    wrist_link = ycfg["wrist_link_name"]

    robot = dex._retargeting.optimizer.robot
    robot.compute_forward_kinematics(np.zeros(robot.dof))
    wrist_pose = robot.get_link_pose(robot.get_link_index(wrist_link))
    wrist_rot, wrist_pos = wrist_pose[:3, :3], wrist_pose[:3, 3]
    tips = {}
    for finger in _MANO_TIP_ROWS:
        link = next(n for n in ycfg["finger_tip_link_names"] if finger in n)
        tips[finger] = wrist_rot.T @ (robot.get_link_pose(robot.get_link_index(link))[:3, 3] - wrist_pos)
    return tips, wrist_rot


def solve_side(mano: np.ndarray, sharpa_tips: dict[str, np.ndarray], sharpa_wrist_rot: np.ndarray) -> dict:
    """Solve one hand's calibration from its wrist-frame MANO keypoints (source-branch math)."""
    human = np.array([mano[_MANO_TIP_ROWS[f]] for f in _FIT_FINGERS])
    targets = np.array([sharpa_tips[f] for f in _FIT_FINGERS])
    # Wrist poses are aligned by construction (both point sets are in their own
    # wrist frame, wrist at the origin); solve scale + the residual rotation
    # ABOUT the common wrist.
    R_res, s, rms = solve_similarity(human, targets)
    # DexPilot compares vectors in URDF base axes -> save the composite.
    R_total = sharpa_wrist_rot @ R_res

    def _angle(m: np.ndarray) -> float:
        return float(np.degrees(np.arccos(np.clip((np.trace(m) - 1) / 2, -1.0, 1.0))))

    # Thumb/pinky: excluded from the fit; per-finger corrections applied AFTER
    # the global (s, R): a length ratio plus the minimal rotation that swings
    # the globally-calibrated tip direction onto the Sharpa's (base axes).
    finger_corr = {}
    for f in _RATIO_FINGERS:
        h_g = s * (R_total @ mano[_MANO_TIP_ROWS[f]])
        t = sharpa_wrist_rot @ sharpa_tips[f]
        ratio = float(np.linalg.norm(t) / np.linalg.norm(h_g))
        R_f = rotation_between(h_g / np.linalg.norm(h_g), t / np.linalg.norm(t))
        finger_corr[f] = (ratio, R_f)

    return {
        "rotation": [[float(v) for v in row] for row in R_total],
        "scale": s,
        **{f"{f}_ratio": ratio for f, (ratio, _r) in finger_corr.items()},
        **{f"{f}_rotation": [[float(v) for v in row] for row in R_f] for f, (_ratio, R_f) in finger_corr.items()},
        **{f"{f}_angle_deg": _angle(R_f) for f, (_ratio, R_f) in finger_corr.items()},
        "rms_m": rms,
        "residual_rotation": [[float(v) for v in row] for row in R_res],
        "residual_angle_deg": _angle(R_res),
        "wrist_rot_in_base": [[float(v) for v in row] for row in sharpa_wrist_rot],
        "human_tips_wrist_frame": {f: [float(v) for v in mano[_MANO_TIP_ROWS[f]]] for f in _MANO_TIP_ROWS},
        "sharpa_tips_wrist_frame": {f: [float(v) for v in sharpa_tips[f]] for f in _MANO_TIP_ROWS},
    }


def build_capture_pipeline():
    """A minimal IsaacTeleop pipeline emitting both hands' raw joints + tip radii.

    Layout: per hand, 26 x [px, py, pz, qx, qy, qz, qw, r] = 208 elements; left
    then right (416 total). Untracked joints read as zeros.
    """
    from isaacteleop.retargeting_engine.deviceio_source_nodes import HandsSource
    from isaacteleop.retargeting_engine.interface import BaseRetargeter, OutputCombiner
    from isaacteleop.retargeting_engine.interface.tensor_group_type import OptionalType, TensorGroupType
    from isaacteleop.retargeting_engine.tensor_types import FloatType, HandInput, HandInputIndex
    from isaacteleop.retargeters import TensorReorderer

    elements = [
        f"c_{hand}_j{j:02d}_{c}"
        for hand in ("left", "right")
        for j in range(26)
        for c in ("px", "py", "pz", "qx", "qy", "qz", "qw", "r")
    ]

    class CaptureNode(BaseRetargeter):
        def input_spec(self):
            return {"hand_left": OptionalType(HandInput()), "hand_right": OptionalType(HandInput())}

        def output_spec(self):
            return {"joints": TensorGroupType("joints", [FloatType(n) for n in elements])}

        def _compute_fn(self, inputs, outputs, context) -> None:
            out = outputs["joints"]
            flat = np.zeros(len(elements))
            for h, key in enumerate(("hand_left", "hand_right")):
                group = inputs[key]
                if group.is_none:
                    continue
                pos = np.from_dlpack(group[HandInputIndex.JOINT_POSITIONS])
                quat = np.from_dlpack(group[HandInputIndex.JOINT_ORIENTATIONS])
                radii = np.from_dlpack(group[HandInputIndex.JOINT_RADII]).reshape(26, 1)
                valid = np.from_dlpack(group[HandInputIndex.JOINT_VALID]).astype(bool)
                joints = np.concatenate([pos, quat, radii], axis=1)
                joints[~valid] = 0.0
                flat[h * 208 : (h + 1) * 208] = joints.reshape(-1)
            for i, v in enumerate(flat):
                out[i] = float(v)

    hands = HandsSource(name="hands")
    node = CaptureNode(name="capture")
    connected = node.connect({"hand_left": hands.output(HandsSource.LEFT), "hand_right": hands.output(HandsSource.RIGHT)})
    reorderer = TensorReorderer(
        input_config={"joints": elements}, output_order=elements, name="reorder", input_types={"joints": "scalar"}
    )
    connected_reorderer = reorderer.connect({"joints": connected.output("joints")})
    return OutputCombiner({"action": connected_reorderer.output("output")})


def hand_is_tracked(joints: np.ndarray) -> bool:
    """True if a (26, 8) block looks like real tracking (wrist live, real spread)."""
    if float(np.linalg.norm(joints[_WRIST_INDEX, :3])) < 1e-6:
        return False
    spread = float(np.linalg.norm(joints[:, :3].max(axis=0) - joints[:, :3].min(axis=0)))
    return spread > 0.08  # a real open hand spans well over 8 cm


def main():
    tip_extension = not args_cli.no_tip_extension
    output_path = os.path.join(DEX_RETARGETING_DIR, f"hand_calibration_{args_cli.user}.yml")

    # Empty world: a dome light so the compositor has something to render.
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60, device=args_cli.device))
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
    light_cfg.func("/World/DomeLight", light_cfg)

    # Red joint markers so you can see your tracked hands in AR.
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
    sharpa_tips = {}
    sharpa_wrist_rot = {}
    for side in ("left", "right"):
        tips, wrist_rot = sharpa_reference_tips(SharpaWaveDexPilot(side), side)
        sharpa_tips[side] = tips
        sharpa_wrist_rot[side] = wrist_rot
        print(
            f"[INFO] Sharpa {side} reference tips (q=0, wrist frame, m): "
            + ", ".join(f"{f}={np.round(t, 4).tolist()}" for f, t in tips.items())
        )

    from isaaclab_teleop import CLOUDXR_AVP_ENV, CLOUDXR_JS_ENV, create_isaac_teleop_device, poll_control_events
    from isaaclab_teleop.isaac_teleop_cfg import IsaacTeleopCfg
    from isaaclab_teleop.xr_cfg import XrCfg

    cloudxr_env = {"cloudxrjs": CLOUDXR_JS_ENV, "avp": CLOUDXR_AVP_ENV, "none": None}.get(
        args_cli.cloudxr_env, args_cli.cloudxr_env
    )
    pipeline = build_capture_pipeline()
    teleop = create_isaac_teleop_device(
        IsaacTeleopCfg(xr_cfg=XrCfg(), pipeline_builder=lambda: pipeline, sim_device=args_cli.device),
        sim_device=args_cli.device,
        cloudxr_env_file=cloudxr_env,
        auto_launch_cloudxr=cloudxr_env is not None,
    )

    sim.reset()
    print(f"[INFO] Hand-shape calibration for user '{args_cli.user}' (tip extension: {tip_extension}).")
    print("[INFO] Connect the headset, then press Play. Keep fingers of BOTH hands STRAIGHT.")

    t_play = None
    last_countdown = None
    warned_invalid = False
    result = None

    with teleop, torch.inference_mode():
        while simulation_app.is_running():
            action = teleop.advance()
            ctrl = poll_control_events(teleop)
            if ctrl.is_active is True and t_play is None:
                t_play = time.monotonic()
                print(f"[INFO] Play received: hold BOTH hands flat. Capturing in {args_cli.delay:.0f} s...")
            elif ctrl.is_active is False and t_play is not None:
                t_play = None
                print("[INFO] Stop received: countdown cancelled. Press Play to re-arm.")

            if action is None:
                sim.render()
                continue
            joints = action.reshape(2, 26, 8).cpu().numpy().astype(np.float64)
            left, right = joints[0], joints[1]

            live = joints[:, :, :3].reshape(-1, 3)
            live = live[np.linalg.norm(live, axis=1) > 1e-6]
            if len(live):
                markers.visualize(translations=torch.tensor(live, dtype=torch.float32))

            if t_play is not None:
                remaining = args_cli.delay - (time.monotonic() - t_play)
                if remaining > 0:
                    if last_countdown != int(remaining):
                        last_countdown = int(remaining)
                        print(f"[INFO] ... {last_countdown + 1}")
                elif hand_is_tracked(left) and hand_is_tracked(right):
                    result = {}
                    for side, hand in (("left", left), ("right", right)):
                        positions = hand[:, :3]
                        if tip_extension:
                            positions = extend_fingertips(positions, hand[:, 7])
                        mano = convert_hand_joints(positions, hand[_WRIST_INDEX, 3:7])
                        result[side] = solve_side(mano, sharpa_tips[side], sharpa_wrist_rot[side])
                        print(
                            f"[RESULT] {side}: scale={result[side]['scale']:.4f}  residual_wrist_rotation="
                            f"{result[side]['residual_angle_deg']:.2f} deg  rms={result[side]['rms_m'] * 1000:.2f} mm"
                        )
                        for f in _RATIO_FINGERS:
                            print(
                                f"[RESULT] {side} {f}: ratio={result[side][f'{f}_ratio']:.4f}"
                                f"  rotation={result[side][f'{f}_angle_deg']:.2f} deg"
                            )
                    break
                elif not warned_invalid:
                    warned_invalid = True
                    print(
                        "[WARNING] Hands not (both) tracked at capture time — holding until the first"
                        " valid frame. Keep both hands flat and visible."
                    )
            sim.render()

    if result is not None:
        payload = {
            "generated": datetime.now().isoformat(timespec="seconds"),
            "user": args_cli.user,
            "tip_extension": tip_extension,
            "note": (
                "Hand-shape calibration (calibrate_hand_shape.py). Wrist poses aligned first (human and "
                "robot points each in their own wrist frame, wrists identified); scale + residual rotation "
                "solved about the common wrist on index/middle/ring. 'rotation' is applied WRIST-side "
                "(inverted, composed into the tracked wrist frame: arm target tilts, fingers derive from "
                "the rotated wrist); 'scale' multiplies the keypoints. Thumb and pinky get per-finger "
                "corrections applied after the global transform, about the wrist. 'tip_extension' records "
                "whether OpenXR tip joints were extended to the skin surface by their radii; the teleop "
                "retargeter mirrors it. Loading sets the DexPilot scaling to 1.0."
            ),
            **result,
        }
        with open(output_path, "w") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
        print(f"[INFO] Calibration written to {output_path}")
        print(f"[INFO] Load it in teleop with: --user {args_cli.user}")
    else:
        print("[INFO] Exiting without a capture — no calibration written.")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()

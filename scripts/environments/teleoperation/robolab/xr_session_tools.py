# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XR session tools for the TACO teleop pipeline. Import only after AppLauncher.

Contains the pieces that talk to the XR session beyond retargeting:

- :class:`RawXrCapture` — zero-dim passthrough retargeter exposing raw hand/head data.
- :class:`CrossHandStopGesture` — "all five fingertip pairs touching" stop gesture.
- :class:`TeleopCommandBridge` — extra ``teleop_command`` handling (record result,
  align) plus server→client messages via the CloudXR outgoing relay.
- :class:`AnchorAligner` — runtime re-anchoring so the table is in front of the user.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
import torch

from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg

# Fingertip joints, same-finger pair order: thumb, index, middle, ring, little.
_TIP_JOINTS = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip")


# ---------------------------------------------------------------------------
# Raw data capture
# ---------------------------------------------------------------------------


@dataclass
class RawXrCaptureCfg(RetargeterCfg):
    """Config for :class:`RawXrCapture`."""


class RawXrCapture(RetargeterBase):
    """Passthrough retargeter storing the latest raw XR data dict.

    Declares HAND_TRACKING (for the stop gesture) and HEAD_TRACKING (for align).
    Returns a 0-length tensor so concatenation leaves the main retargeter's
    action unchanged (the calibrate script's ``_Capture`` returns ``zeros(1)``,
    which would grow the 58-D action to 59).
    """

    def __init__(self, cfg: RawXrCaptureCfg):
        super().__init__(cfg)
        self.latest: dict | None = None

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.HAND_TRACKING, RetargeterBase.Requirement.HEAD_TRACKING]

    def retarget(self, data: dict) -> torch.Tensor:
        self.latest = data
        return torch.zeros(0)

    @property
    def head_pose(self) -> np.ndarray | None:
        """Latest head pose [x, y, z, qw, qx, qy, qz] in the Isaac world frame."""
        if self.latest is None:
            return None
        head = self.latest.get(DeviceBase.TrackingTarget.HEAD)
        if head is None or np.linalg.norm(head[:3]) < 1e-6:
            return None
        return head


def current_head_pose() -> np.ndarray | None:
    """Fresh head pose [x, y, z, qw, qx, qy, qz] straight from XRCore.

    Unlike :attr:`RawXrCapture.head_pose`, this does not depend on the teleop
    loop calling ``advance()`` — it works while teleop is paused (the Align
    button must work before Play is ever pressed). Mirrors
    ``OpenXRDevice._calculate_headpose``.
    """
    try:
        from omni.kit.xr.core import XRCore

        head_device = XRCore.get_singleton().get_input_device("/user/head")
        if not head_device:
            return None
        hmd = head_device.get_virtual_world_pose("")
        position = hmd.ExtractTranslation()
        quat = hmd.ExtractRotationQuat()
        quati = quat.GetImaginary()
        pose = np.array(
            [position[0], position[1], position[2], quat.GetReal(), quati[0], quati[1], quati[2]],
            dtype=np.float64,
        )
    except Exception as e:
        print(f"[current_head_pose] XRCore head query failed: {e}")
        return None
    # An exactly-origin position means "not tracked yet" (a real head is never
    # at the world origin in this scene).
    if np.linalg.norm(pose[:3]) < 1e-6:
        return None
    return pose


def hand_is_tracked(hand_poses: dict[str, np.ndarray] | None) -> bool:
    """True if the 26-joint dict looks like real tracking (untracked hands report
    all joints at the origin; stale hands can freeze but keep a plausible spread)."""
    if hand_poses is None:
        return False
    if np.linalg.norm(hand_poses["wrist"][:3]) < 1e-6:
        return False
    pts = np.array([p[:3] for p in hand_poses.values()])
    spread = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    return spread > 0.08  # a real open hand spans well over 8 cm


# ---------------------------------------------------------------------------
# Stop gesture
# ---------------------------------------------------------------------------


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

    def update(self, data: dict | None) -> bool:
        """Feed the latest raw XR data dict; returns True exactly once per gesture."""
        if data is None:
            self._touch_since = None
            return False
        left = data.get(DeviceBase.TrackingTarget.HAND_LEFT)
        right = data.get(DeviceBase.TrackingTarget.HAND_RIGHT)
        if not (hand_is_tracked(left) and hand_is_tracked(right)):
            self._touch_since = None
            return False

        dists = np.array([np.linalg.norm(left[j][:3] - right[j][:3]) for j in _TIP_JOINTS])

        if not self._armed:
            if np.any(dists > self._release_dist):
                self._armed = True
            return False

        if np.all(dists < self._touch_dist):
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


# ---------------------------------------------------------------------------
# Client messaging (both directions)
# ---------------------------------------------------------------------------


class TeleopCommandBridge:
    """Custom ``teleop_command`` handling + server→client messages.

    ``OpenXRDevice`` dispatches only the ``start``/``stop``/``reset`` substrings and
    silently drops everything else, so this class adds its own subscription to the
    same event type for the pipeline's extra commands:

    - ``record_success`` / ``record_failure`` → ``on_record_result(bool)``
    - ``align``                               → ``on_align()``

    Outbound, the Isaac Sim OpenXR plugin relays ``omni.kit.cloudxr.send_message``
    carb events to the client's ``MessageChannel.receivedMessageStream``
    ("CloudXR outgoing relay for message bus" in the plugin binary). The event must
    carry a ``message`` field.
    """

    SEND_EVENT_TYPE = "omni.kit.cloudxr.send_message"

    def __init__(self, on_record_result, on_align, use_dispatch: bool = False):
        import carb.events
        from omni.kit.xr.core import XRCore

        self._on_record_result = on_record_result
        self._on_align = on_align
        self._use_dispatch = use_dispatch
        self._bus = XRCore.get_singleton().get_message_bus()
        self._send_type = carb.events.type_from_string(self.SEND_EVENT_TYPE)
        self._subscription = self._bus.create_subscription_to_pop_by_type(
            carb.events.type_from_string("teleop_command"), self._on_command
        )

    def _on_command(self, event) -> None:
        # The payload's message field may surface as a string or a dict depending on
        # the plugin's JSON flattening; substring-match on its string form, exactly
        # like OpenXRDevice._on_teleop_command does.
        msg = str(event.payload["message"] if "message" in event.payload else event.payload)
        if "record_success" in msg:
            self._on_record_result(True)
        elif "record_failure" in msg:
            self._on_record_result(False)
        elif "align" in msg:
            self._on_align()

    def send_to_client(self, payload: dict) -> None:
        """Ship a JSON payload to the AVP client via the CloudXR outgoing relay."""
        message = json.dumps(payload)
        if self._use_dispatch:
            self._bus.dispatch(self._send_type, payload={"message": message})
        else:
            self._bus.push(self._send_type, payload={"message": message})

    def request_record_result(self, episode: int) -> None:
        self.send_to_client({"type": "recording_result_request", "episode": episode})


# ---------------------------------------------------------------------------
# Anchor align
# ---------------------------------------------------------------------------


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _quat_rotate_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    u = np.array([x, y, z])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


class AnchorAligner:
    """Re-anchor the XR session so the table sits straight in front of the user.

    The anchor prim (``/World/XRAnchor``) maps physical/XR space into the Isaac
    world. Align applies a world-frame rigid correction ΔT to it: rotate about the
    user's current head position until the head's forward axis (OpenXR: −Z) points
    along world +y (straight at the table), then translate the head's xy onto
    ``target_head_xy``. z is never touched, so the calibrated floor height holds.
    Because ΔT rigidly moves the whole XR→world mapping, the wrist offsets — which
    live in the wrist's own frame — are unaffected.
    """

    def __init__(self, anchor_pos, anchor_rot, target_head_xy=(0.0, -0.9), anchor_path="/World/XRAnchor"):
        self._pos = np.array(anchor_pos, dtype=np.float64)
        self._quat = np.array(anchor_rot, dtype=np.float64)  # wxyz
        self._target_xy = np.array(target_head_xy, dtype=np.float64)
        self._anchor_path = anchor_path
        self._layer_identifier = None

    def _resolve_layer(self) -> None:
        # Same resolution as XrAnchorSynchronizer: the layer the anchor prim's
        # strongest opinion lives in must be passed to set_world_transform_matrix.
        try:
            from isaacsim.core.utils.stage import get_current_stage

            prim = get_current_stage().GetPrimAtPath(self._anchor_path)
            prim_stack = prim.GetPrimStack() if prim is not None else None
            self._layer_identifier = prim_stack[0].layer.identifier if prim_stack else None
        except Exception:
            self._layer_identifier = None

    def align(self, head_pose_w: np.ndarray) -> bool:
        """Apply the correction for the given world-frame head pose. True on success."""
        from omni.kit.xr.core import XRCore
        from pxr import Gf

        head_pos = head_pose_w[:3].astype(np.float64)
        head_quat = head_pose_w[3:].astype(np.float64)

        # Head forward axis: OpenXR/Kit head frames look along -Z.
        fwd = _quat_rotate_wxyz(head_quat, np.array([0.0, 0.0, -1.0]))
        if np.linalg.norm(fwd[:2]) < 1e-6:
            return False  # looking straight up/down: yaw undefined
        yaw_head = np.arctan2(fwd[1], fwd[0])
        dyaw = np.pi / 2 - yaw_head  # face world +y (the table)
        q_dyaw = np.array([np.cos(dyaw / 2), 0.0, 0.0, np.sin(dyaw / 2)])

        # ΔT: rotate about the head position (head stays put), then translate its
        # xy onto the target. Compose into the tracked anchor transform.
        new_pos = _quat_rotate_wxyz(q_dyaw, self._pos - head_pos) + head_pos
        new_pos[0] += self._target_xy[0] - head_pos[0]
        new_pos[1] += self._target_xy[1] - head_pos[1]
        new_quat = _quat_mul_wxyz(q_dyaw, self._quat)

        if self._layer_identifier is None:
            self._resolve_layer()
        mat = Gf.Matrix4d()
        mat.SetRotateOnly(Gf.Quatd(new_quat[0], Gf.Vec3d(*new_quat[1:])))
        mat.SetTranslateOnly(Gf.Vec3d(*new_pos))
        try:
            XRCore.get_singleton().set_world_transform_matrix(self._anchor_path, mat, self._layer_identifier)
        except Exception as e:  # keep teleop alive; align is best-effort
            print(f"[AnchorAligner] set_world_transform_matrix failed: {e}")
            return False

        self._pos, self._quat = new_pos, new_quat
        yaw_deg = np.degrees(dyaw)
        print(f"[AnchorAligner] Re-anchored: yaw {yaw_deg:+.1f} deg, head xy -> ({self._target_xy[0]:.2f}, {self._target_xy[1]:.2f})")
        return True

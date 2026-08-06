# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for the RoboLab XR teleop scripts. Import only after AppLauncher."""

from __future__ import annotations

import time


class LoopProfiler:
    """Rolling per-stage wall-time profiler for the teleop loop.

    Usage per iteration:
        prof.begin()
        ...retargeting...      ; prof.lap("retarget")
        ...frame conversion... ; prof.lap("frame")
        ...env.step...         ; prof.lap("step")
        prof.end()             # prints a one-line report every `report_every_s`

    Overhead is a few perf_counter calls per loop — negligible next to a sim step.
    """

    def __init__(self, enabled: bool, report_every_s: float = 1.0):
        self.enabled = enabled
        self._report_every = report_every_s
        self._t_iter: float | None = None
        self._t_lap: float | None = None
        self._sums: dict[str, float] = {}
        self._iters = 0
        self._window_start = time.perf_counter()

    def begin(self) -> None:
        if not self.enabled:
            return
        self._t_iter = self._t_lap = time.perf_counter()

    def lap(self, stage: str) -> None:
        if not self.enabled or self._t_lap is None:
            return
        now = time.perf_counter()
        self._sums[stage] = self._sums.get(stage, 0.0) + (now - self._t_lap)
        self._t_lap = now

    def end(self) -> None:
        if not self.enabled or self._t_iter is None:
            return
        now = time.perf_counter()
        self._sums["total"] = self._sums.get("total", 0.0) + (now - self._t_iter)
        self._iters += 1
        window = now - self._window_start
        if window >= self._report_every and self._iters > 0:
            n = self._iters
            stages = "  ".join(
                f"{k}={v / n * 1000.0:6.1f}ms" for k, v in self._sums.items() if k != "total"
            )
            total_ms = self._sums["total"] / n * 1000.0
            print(
                f"[PROFILE] {n / window:5.1f} Hz loop | {stages}  total={total_ms:6.1f}ms",
                flush=True,
            )
            self._sums.clear()
            self._iters = 0
            self._window_start = now


def strip_cameras_for_xr(env_cfg) -> list[str]:
    """Remove camera sensors and their observation terms from a RoboLab env cfg.

    Under XR the renderer is paced by the headset session, and tiled-camera sensor
    updates deadlock env creation in headless XR mode. This mirrors what Isaac Lab's
    ``remove_camera_configs`` does for its own XR tasks, extended to RoboLab's custom
    observation groups. Camera imagery remains reproducible offline via RoboLab
    replay of the recorded states.

    Returns the names of the removed scene cameras.
    """
    from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
    from isaaclab.sensors import CameraCfg  # TiledCameraCfg subclasses CameraCfg

    removed: list[str] = []
    for attr_name in list(vars(env_cfg.scene).keys()):
        if isinstance(getattr(env_cfg.scene, attr_name, None), CameraCfg):
            delattr(env_cfg.scene, attr_name)
            removed.append(attr_name)
    if not removed:
        return removed

    for group_name in list(vars(env_cfg.observations).keys()):
        group = getattr(env_cfg.observations, group_name, None)
        if group is None:
            continue
        term_names = [
            n for n in list(vars(group).keys()) if isinstance(getattr(group, n, None), ObservationTermCfg)
        ]
        if not term_names:
            continue
        for term_name in term_names:
            params = getattr(getattr(group, term_name), "params", None) or {}
            if any(isinstance(v, SceneEntityCfg) and v.name in removed for v in params.values()):
                delattr(group, term_name)
        if not any(isinstance(getattr(group, n, None), ObservationTermCfg) for n in list(vars(group).keys())):
            setattr(env_cfg.observations, group_name, None)
    return removed

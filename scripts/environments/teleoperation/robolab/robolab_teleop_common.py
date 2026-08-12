# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for the RoboLab XR teleop scripts. Import only after AppLauncher."""

from __future__ import annotations

import time

from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction


class OncePerStepDiffIKAction(DifferentialInverseKinematicsAction):
    """Differential IK that solves once per *control* step instead of once per physics substep.

    The stock action recomputes the end-effector pose, re-reads the Jacobian from PhysX
    (a GPU readback) and re-solves DLS inside ``apply_actions`` — which the env calls once
    per decimation substep. At decimation 8 with two arms that is 16 IK solves per control
    step and dominates teleop loop time (~65 ms/step measured on the TACO duo scene).

    Teleop sends absolute pose targets at ~7-15 Hz, so re-linearizing the Jacobian at
    120 Hz buys nothing: this variant solves in ``process_actions`` (once per control
    step, against the freshest state) and has ``apply_actions`` only re-issue the cached
    joint-position target to the PD drives.
    """

    def process_actions(self, actions):
        super().process_actions(actions)
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            self._joint_pos_des = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
        else:
            self._joint_pos_des = joint_pos.clone()

    def apply_actions(self):
        self._asset.set_joint_position_target(self._joint_pos_des, self._joint_ids)


class LoopProfiler:
    """Rolling per-stage wall-time profiler for the teleop loop.

    Usage per iteration:
        prof.begin()
        ...retargeting...      ; prof.lap("retarget")
        ...frame conversion... ; prof.lap("frame")
        ...env.step...         ; prof.lap("step")
        prof.end()             # prints a one-line report every `report_every_s`

    Overhead is a few perf_counter calls per loop — negligible next to a sim step.

    Note on reading reports when ``wrap_render`` is used: the ``render_call``
    bucket is a *subset* of ``step`` (renders happen inside env.step). It is the
    CPU-blocked wall time of ``sim.render()`` — submission plus any XR pacing
    wait — NOT the async GPU render/encode, so it is not "render latency".
    ``total`` is true wall time per iteration and is less than the sum of buckets.
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

    def add_external(self, stage: str, seconds: float) -> None:
        """Accumulate time measured elsewhere (e.g. a wrapped env.sim.render)."""
        if not self.enabled:
            return
        self._sums[stage] = self._sums.get(stage, 0.0) + seconds

    def wrap_method(self, obj, method_name: str, stage: str) -> None:
        """Wrap ``obj.method_name`` so its wall time accumulates into ``stage``.

        Used to attribute sub-costs of env.step() to their own buckets:
        ``wrap_method(env.sim, "render", "render_call")`` for the renderer/XR
        compositor, ``wrap_method(env.sim, "step", "physx")`` for the raw
        physics call (which, under some kit configurations, also pumps app
        update work — separating it from the rest of ``step`` is the point).
        """
        if not self.enabled:
            return
        original = getattr(obj, method_name)

        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            result = original(*args, **kwargs)
            self.add_external(stage, time.perf_counter() - t0)
            return result

        setattr(obj, method_name, timed)

    def wrap_render(self, sim, stage: str = "render_call") -> None:
        """Back-compat alias: wrap ``sim.render`` into its own bucket."""
        self.wrap_method(sim, "render", stage)

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
        if now - self._window_start >= self._report_every:
            self.report()

    def report(self) -> None:
        """Print and reset the current window. Called automatically once per
        ``report_every_s``; call manually after a fixed-step benchmark loop so
        runs faster than one report window still produce output."""
        if not self.enabled or self._iters == 0:
            return
        now = time.perf_counter()
        window = now - self._window_start
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

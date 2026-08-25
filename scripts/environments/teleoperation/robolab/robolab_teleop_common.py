# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for the RoboLab XR teleop scripts. Import only after AppLauncher."""

from __future__ import annotations

import re
import time

from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

# Tip links of the SharpaWave hands: the only self-collision pairs kept live, and the
# ones drawn green by visualize_hand_collision_meshes. On the shipped asset the
# ``*_fingertip`` links carry no colliders (virtual sites); DP + elastomer do.
_TIP_LINK_RE = re.compile(r"^(?P<finger>.*)_(DP|elastomer|fingertip)$")

# Rigid links belonging to the hands (vs arms/torso), by link name.
_HAND_LINK_RE = re.compile(r"^(left|right)_(hand|thumb|index|middle|ring|pinky)_")


class OncePerStepDiffIKAction(DifferentialInverseKinematicsAction):
    """Differential IK that solves once per *control* step instead of once per physics substep.

    The stock action recomputes the end-effector pose, re-reads the Jacobian from PhysX
    (a GPU readback) and re-solves DLS inside ``apply_actions`` — which the env calls once
    per decimation substep. At decimation 8 with two arms that is 16 IK solves per control
    step and dominates teleop loop time (~65 ms/step measured on the TACO duo scene).

    Teleop delivers a new absolute pose target at most once per control step, so
    re-linearizing the Jacobian at the physics substep rate buys nothing: this variant
    solves in ``process_actions`` (once per control step, against the freshest state)
    and has ``apply_actions`` only re-issue the cached joint-position target to the
    PD drives.
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

    Two distortions to keep in mind when reading reports: wrapped methods also
    accrue time when called outside a begin()/end() iteration (e.g. inside
    env.reset()), inflating that window's stage averages; and a window spanning
    a teleop START/STOP transition mixes idle render-only iterations with full
    steps, diluting both the Hz and the averages.
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


def _links_with_colliders(stage, robot_root):
    """Map rigid-link prim -> collider prims under it (descending into instances).

    The SharpaWave links reference instanceable ``visuals``/``collisions`` subtrees,
    so the collider meshes are instance *proxies*: a plain ``Usd.PrimRange`` never
    reaches them, and no opinion (FilteredPairsAPI included) can be authored on them.
    """
    from pxr import Usd, UsdPhysics

    proxies = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    links: dict = {}
    for prim in Usd.PrimRange(robot_root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            colliders = [p for p in Usd.PrimRange(prim, proxies) if p.HasAPI(UsdPhysics.CollisionAPI)]
            if colliders:
                links[prim] = colliders
    return links


class JointSetpointsRecorder(RecorderTerm):
    """Recorder term: the PD joint-position targets actually sent to the drives.

    Writes ``obs/joint_setpoints`` with shape (T, num_joints), read post-step from
    ``asset.data.joint_pos_target`` — i.e. the control signal after ALL action terms
    applied: for the arms this is the DiffIK *output* (which ``actions`` does not
    contain — it only holds the commanded wrist poses), for the fingers the scaled
    joint-position targets. Joint order matches
    ``states/articulation/<asset>/joint_position``, so setpoint-vs-measured tracking
    error is a direct subtraction. Row t is the setpoint active during step t.
    """

    def record_post_step(self):
        asset = self._env.scene[self.cfg.asset_name]
        return "obs/joint_setpoints", asset.data.joint_pos_target.clone()


@configclass
class JointSetpointsRecorderCfg(RecorderTermCfg):
    """Configuration for :class:`JointSetpointsRecorder`."""

    class_type: type[RecorderTerm] = JointSetpointsRecorder
    asset_name: str = "robot"


def filter_self_collision_except_fingertips(env, env_ids, robot_name: str = "robot") -> None:
    """Startup event: with self-collision on, keep only cross-finger fingertip contacts.

    Why not plain ``enabled_self_collisions``: PhysX auto-excludes only
    directly-jointed link pairs, and the SharpaWave knuckles route through
    zero-length virtual links (``*_MCP_VL``) with their own convex hulls — so
    palm<->proximal and similar distance-2 pairs interpenetrate at rest, and the
    contact forces dwarf the finger drives' effort caps (0.19-1.86 Nm), jamming
    the fingers open.

    Policy here: filter EVERY link pair except tips of DIFFERENT fingers. Tip
    links are ``*_DP`` / ``*_elastomer`` / ``*_fingertip``; same-finger tip pairs
    (e.g. DP<->elastomer, one fixed joint apart) overlap rigidly and stay filtered.

    ``FilteredPairsAPI`` is authored on the rigid-link prims (which filters every
    shape under them, per the UsdPhysics spec). The collider prims themselves are
    instance proxies on this asset and cannot hold authored opinions — see
    ``_links_with_colliders``.
    """
    import omni.usd
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    kept, filtered = 0, 0
    for env_path in env.scene.env_prim_paths:
        robot_root = stage.GetPrimAtPath(f"{env_path}/{robot_name}")
        colliders_by_link = _links_with_colliders(stage, robot_root)
        if not colliders_by_link:
            print(f"[WARNING] Self-collision filter: no colliders found under {robot_root.GetPath()}; "
                  "self-collision is UNFILTERED (fingers will jam).")
            continue

        def finger_of(link_prim) -> str | None:
            m = _TIP_LINK_RE.match(link_prim.GetName())
            return m.group("finger") if m else None

        links = sorted(colliders_by_link, key=lambda p: str(p.GetPath()))
        for i, a in enumerate(links):
            targets = []
            finger_a = finger_of(a)
            for b in links[i + 1:]:
                finger_b = finger_of(b)
                if finger_a is not None and finger_b is not None and finger_a != finger_b:
                    kept += 1  # cross-finger tip pair: leave contact live
                    continue
                targets.append(b.GetPath())
                filtered += 1
            if targets:
                api = UsdPhysics.FilteredPairsAPI.Apply(a)
                rel = api.CreateFilteredPairsRel()
                for target_path in targets:
                    rel.AddTarget(target_path)
    print(f"[INFO] Self-collision: fingertips-only policy — {kept} cross-finger tip pairs "
          f"live, {filtered} link pairs filtered.")


def visualize_hand_collision_meshes(env, robot_name: str = "robot", opacity: float = 0.6) -> None:
    """Render the hands' collision shapes in-scene, color-coded by the self-collision policy.

    For every collider on a hand link, builds a render-only mesh of the shape PhysX
    actually collides with (the convex hull of the authored collision mesh — all
    SharpaWave hand colliders cook as ``convexHull``) and parents it under the link,
    so it follows the link through Fabric like the visual meshes do. Physics prims
    are never touched: the instanced ``collisions`` subtrees stay ``guide`` purpose.

    Colors: green = tip links (``*_DP``/``*_elastomer``) whose cross-finger contacts
    stay live under ``filter_self_collision_except_fingertips``; red = links whose
    self-contacts are filtered (they still collide with the table/objects).

    Semi-transparent materials need ``env_cfg.sim.render.enable_translucency`` set
    before sim-context creation (callers handle this).
    """
    import numpy as np

    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

    import isaaclab.sim as sim_utils

    stage = omni.usd.get_context().get_stage()
    materials = {}
    for key, color in (("tip", (0.1, 0.9, 0.2)), ("filtered", (0.9, 0.12, 0.08))):
        mat_path = f"/World/Looks/CollisionViz_{key}"
        if not stage.GetPrimAtPath(mat_path):
            sim_utils.spawn_preview_surface(
                mat_path,
                sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color,
                    emissive_color=tuple(0.4 * c for c in color),  # readable even inside shadowed palms
                    roughness=1.0,
                    opacity=opacity,
                ),
            )
        materials[key] = mat_path

    xf_cache = UsdGeom.XformCache()
    created, skipped = 0, 0
    for env_path in env.scene.env_prim_paths:
        robot_root = stage.GetPrimAtPath(f"{env_path}/{robot_name}")
        for link, colliders in _links_with_colliders(stage, robot_root).items():
            if not _HAND_LINK_RE.match(link.GetName()):
                continue
            scope_path = link.GetPath().AppendChild("collision_viz")
            if stage.GetPrimAtPath(scope_path):
                continue  # idempotent across resets
            UsdGeom.Scope.Define(stage, scope_path)
            link_world_inv = xf_cache.GetLocalToWorldTransform(link).GetInverse()
            color_key = "tip" if _TIP_LINK_RE.match(link.GetName()) else "filtered"
            for i, coll in enumerate(colliders):
                src = UsdGeom.Mesh(coll)
                if not src:
                    skipped += 1  # non-mesh collider (none on the SharpaWave hands)
                    continue
                points = np.asarray(src.GetPointsAttr().Get(), dtype=np.float32)
                approx = None
                if coll.HasAPI(UsdPhysics.MeshCollisionAPI):
                    approx = UsdPhysics.MeshCollisionAPI(coll).GetApproximationAttr().Get()
                if approx == "convexHull":
                    # Show the cooked shape, not the authored mesh: hulls fill concavities,
                    # which is exactly the kind of rest-pose overlap being debugged.
                    from scipy.spatial import ConvexHull

                    hull = ConvexHull(points.astype(np.float64))
                    remap = np.full(len(points), -1, dtype=np.int32)
                    remap[hull.vertices] = np.arange(len(hull.vertices), dtype=np.int32)
                    points = points[hull.vertices]
                    indices = remap[hull.simplices].reshape(-1)
                    counts = np.full(len(hull.simplices), 3, dtype=np.int32)
                else:
                    indices = np.asarray(src.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
                    counts = np.asarray(src.GetFaceVertexCountsAttr().Get(), dtype=np.int32)

                dst = UsdGeom.Mesh.Define(stage, scope_path.AppendChild(f"{coll.GetParent().GetName()}_{i}"))
                dst.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(points)))
                dst.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(np.ascontiguousarray(indices)))
                dst.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.ascontiguousarray(counts)))
                dst.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)  # else hulls render catmullClark-blobby
                dst.CreateDoubleSidedAttr().Set(True)  # hull triangle winding is arbitrary
                lo, hi = points.min(0).astype(float), points.max(0).astype(float)
                dst.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*lo), Gf.Vec3f(*hi)]))
                # Static collider-to-link offset (both from the same authored USD chain).
                rel = xf_cache.GetLocalToWorldTransform(coll) * link_world_inv
                UsdGeom.Xformable(dst).AddTransformOp().Set(Gf.Matrix4d(rel))
                sim_utils.bind_visual_material(str(dst.GetPath()), materials[color_key])
                created += 1
    print(f"[INFO] Collision viz: {created} hand collider hulls rendered "
          f"(green=live tips, red=filtered){f'; {skipped} non-mesh colliders skipped' if skipped else ''}.")


def apply_arm_visual(mode: str) -> None:
    """Make the arms' render geometry 5% transparent or invisible (physics untouched).

    Targets each link's render prims individually — the instanceable ``visuals``
    roots and any loose Gprims (the panda links' render mesh doubles as their
    collider) — rather than the arm roots, so a ``collision_viz`` debug subtree
    (see :func:`visualize_hand_collision_meshes`) keeps its own bright material
    instead of inheriting the stronger-than-descendants ghost binding.
    """
    if mode == "normal":
        return
    import isaacsim.core.utils.stage as stage_utils
    from pxr import Usd, UsdGeom

    import isaaclab.sim as sim_utils

    arm_paths = sim_utils.find_matching_prim_paths("/World/envs/env_.*/robot/(left|right)_arm")
    if not arm_paths:
        print("[WARNING] --arm_visual: no arm prims matched; skipping.")
        return
    stage = stage_utils.get_current_stage()
    targets = []
    for arm_path in arm_paths:
        it = iter(Usd.PrimRange(stage.GetPrimAtPath(arm_path)))
        for prim in it:
            name = prim.GetName()
            if name in ("collisions", "collision_viz"):
                it.PruneChildren()
            elif name == "visuals":
                targets.append(prim)
                it.PruneChildren()
            elif prim.IsA(UsdGeom.Gprim):
                targets.append(prim)
    if mode == "hidden":
        for prim in targets:
            sim_utils.set_prim_visibility(prim, False)
        print(f"[INFO] Arms hidden (render only): {len(targets)} render prims under {arm_paths}")
    elif mode == "transparent":
        material_path = "/World/Looks/ArmGhostMaterial"
        sim_utils.spawn_preview_surface(
            material_path,
            sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.78, 0.85), opacity=0.05, roughness=0.0),
        )
        for prim in targets:
            sim_utils.bind_visual_material(str(prim.GetPath()), material_path, stronger_than_descendants=True)
        print(f"[INFO] Arms 5% transparent: {len(targets)} render prims under {arm_paths}")


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

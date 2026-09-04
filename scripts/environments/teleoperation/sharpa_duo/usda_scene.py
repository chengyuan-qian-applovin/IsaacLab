# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load an arbitrary scene USDA into an Isaac Lab interactive scene.

The USDA is referenced into the stage untouched at ``{ENV_REGEX_NS}/scene`` —
whatever geometry, lights, materials, and physics it authors just work. On top
of that, :func:`add_usda_scene` optionally discovers the file's physics objects
and attaches one ``spawn=None`` stub per object, so Isaac Lab tracks them and
``reset_scene_to_default`` restores their authored state on every env reset:

- a prim with ``ArticulationRootAPI`` (drawer, box with a lid, ...) gets an
  :class:`~isaaclab.assets.ArticulationCfg` whose ``init_state.joint_pos`` is
  read from the joints' authored ``JointStateAPI`` positions, so the reset also
  puts every joint back where the file authored it;
- any other topmost ``RigidBodyAPI`` prim gets a
  :class:`~isaaclab.assets.RigidObjectCfg`.

Without the stubs the scene still loads and simulates; its objects just keep
their state across resets. Isaac Lab never reads default joint positions from
the stage itself — only from the cfg — which is why the articulation stubs
carry them explicitly.

Requirements on the USDA:

- It must have a default prim (that is what ``UsdFileCfg`` references).
- Object initial poses are read from the composed stage, so they may live on
  the object Xforms or inside payloads/references — both work.

Import only after AppLauncher.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import UsdFileCfg

#: Bounding-circle footprint radius [m] per tracked object, filled by
#: :func:`add_usda_scene` from the composed USD bounds; consumed by
#: :func:`randomize_tracked_objects` for its collision check.
FOOTPRINT_RADII: dict[str, float] = {}

#: Offset [m] from each tracked object's root origin down to its AABB bottom
#: (``aabb_min_z - authored_root_z``, usually negative), filled alongside
#: :data:`FOOTPRINT_RADII`; used to place visuals on the surface an object
#: rests on (see ``region_overlay``).
FOOTPRINT_BOTTOM_OFFSETS: dict[str, float] = {}

#: Height [m] of the support surface the tracked objects rest on (the min of
#: their AABB bottoms — objects sit ON the tabletop, so its top is where their
#: bottoms are; a stacked object's bottom sits higher). None until a scene with
#: tracked objects is loaded; used to place the in-headset panels above the table.
SUPPORT_SURFACE_Z: float | None = None

#: Absolute USD asset path per tracked object (the reference/payload the scene
#: prim brings in), filled by :func:`add_usda_scene`; used by the adjust mode to
#: spawn translucent ghost copies. Objects whose arc cannot be resolved are absent.
OBJECT_ASSET_PATHS: dict[str, str] = {}

#: Convex-hull vertices of each tracked object's meshes, in the object's own
#: frame [m], shape (N, 3) float32, filled by :func:`add_usda_scene`. The adjust
#: mode rotates them to find the object's lowest point under any orientation,
#: so an edited pose can be clamped to keep the whole mesh on or above the
#: tabletop. Falls back to the 8 local-AABB corners when hulling fails.
OBJECT_SUPPORT_POINTS: dict[str, np.ndarray] = {}


def sidecar_path(usda_path: str) -> str:
    """Path of the pose-override sidecar for ``usda_path`` — ``<scene>.usda.poses.json``."""
    return os.path.abspath(usda_path) + ".poses.json"


def _load_sidecar(usda_path: str) -> dict:
    """The parsed sidecar JSON, or an empty dict when absent."""
    path = sidecar_path(usda_path)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f) or {}


def load_pose_overrides(usda_path: str) -> dict[str, dict[str, list[float]]]:
    """Load ``{name: {"pos": [x,y,z], "rot": [x,y,z,w]}}`` from ``<scene>.usda.poses.json``.

    Returns an empty dict when the sidecar is absent. Rotations are stored in
    ``(x, y, z, w)`` order to match :attr:`RigidObjectCfg.InitialStateCfg.rot`.
    """
    return _load_sidecar(usda_path).get("objects", {}) or {}


def load_randomization_overrides(usda_path: str) -> dict[str, float] | None:
    """Load the randomization ranges saved with the scene, or None when absent.

    Returns ``{"xy_range": m, "yaw_range": rad}`` — the same keys and units as
    :func:`randomize_tracked_objects`'s parameters. The sidecar stores the yaw
    range in degrees (``yaw_range_deg``) because operators type and read it that
    way; it is converted here.
    """
    block = _load_sidecar(usda_path).get("randomization")
    if not isinstance(block, dict):
        return None
    out: dict[str, float] = {}
    if "xy_range" in block:
        out["xy_range"] = float(block["xy_range"])
    if "yaw_range_deg" in block:
        import math

        out["yaw_range"] = math.radians(float(block["yaw_range_deg"]))
    return out or None


def save_pose_overrides(
    usda_path: str,
    overrides: dict[str, dict[str, list[float]]],
    randomization: dict[str, float] | None = None,
) -> str:
    """Write the sidecar next to ``usda_path`` and return its path.

    ``overrides`` maps tracked-object name to ``{"pos": [x,y,z], "rot": [x,y,z,w]}``.
    ``randomization`` (optional) is ``{"xy_range": m, "yaw_range": rad}`` as
    used by :func:`randomize_tracked_objects`; it is stored as ``xy_range`` [m]
    and ``yaw_range_deg`` [deg]. When omitted, a previously saved block is kept.
    """
    path = sidecar_path(usda_path)
    data = {"objects": overrides}
    if randomization is not None:
        import math

        block: dict[str, float] = {}
        if "xy_range" in randomization:
            block["xy_range"] = float(randomization["xy_range"])
        if "yaw_range" in randomization:
            block["yaw_range_deg"] = math.degrees(float(randomization["yaw_range"]))
        data["randomization"] = block
    else:
        previous = _load_sidecar(usda_path).get("randomization")
        if isinstance(previous, dict):
            data["randomization"] = previous
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def _object_asset_path(prim, stage) -> str | None:
    """Absolute path of the USD asset a tracked object prim references/payloads.

    Walks the prim's composition arcs for the first reference or payload and
    resolves its asset path against the introducing layer; falls back to the
    root layer's authored payload/reference list. Returns None (with a warning
    upstream) when no arc is resolvable — such objects simply get no ghost.
    """
    from pxr import Pcp, Usd

    try:
        query = Usd.PrimCompositionQuery(prim)
        for arc in query.GetCompositionArcs():
            if arc.GetArcType() not in (Pcp.ArcTypePayload, Pcp.ArcTypeReference):
                continue
            _, entry = arc.GetIntroducingListEditor()
            layer = arc.GetIntroducingLayer()
            if entry is not None and layer is not None and entry.assetPath:
                return layer.ComputeAbsolutePath(entry.assetPath)
    except Exception:
        pass
    try:
        spec = stage.GetRootLayer().GetPrimAtPath(prim.GetPath())
        if spec is not None:
            for items in (spec.payloadList, spec.referenceList):
                for entry in [*items.prependedItems, *items.explicitItems, *items.appendedItems]:
                    if entry.assetPath:
                        return stage.GetRootLayer().ComputeAbsolutePath(entry.assetPath)
    except Exception:
        pass
    return None


def _object_support_points(prim, bbox_cache) -> np.ndarray:
    """Convex-hull vertices of ``prim``'s meshes, in the prim's own frame [m].

    Rotating these and taking the minimum z gives the object's lowest point
    under any orientation — how the adjust editor keeps a dragged mesh on or
    above the tabletop. Falls back to the 8 corners of the local AABB when the
    meshes yield no points or hulling fails (both conservative directions:
    the AABB box contains the mesh, so its corners can only over-clamp).
    """
    from pxr import Usd, UsdGeom

    points: list[np.ndarray] = []
    try:
        world_to_obj = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
        for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if not child.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(child).ComputePurpose() == UsdGeom.Tokens.guide:
                continue
            pts = UsdGeom.Mesh(child).GetPointsAttr().Get()
            if not pts:
                continue
            # USD matrices are row-vector convention: p' = p @ M[:3,:3] + M[3,:3].
            m = np.array(UsdGeom.Xformable(child).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) * world_to_obj)
            points.append(np.asarray(pts, dtype=np.float64) @ m[:3, :3] + m[3, :3])
        if points:
            pts = np.concatenate(points)
            if len(pts) > 50000:  # hulling cost cap; a subsample is plenty for support
                pts = pts[np.random.default_rng(0).choice(len(pts), 50000, replace=False)]
            from scipy.spatial import ConvexHull

            hull = ConvexHull(pts)
            return pts[hull.vertices].astype(np.float32)
    except Exception as exc:
        print(f"[WARNING] {prim.GetPath()}: support-point hull failed ({exc}); using the AABB corners.")
    box = bbox_cache.ComputeUntransformedBound(prim).ComputeAlignedBox()
    lo, hi = box.GetMin(), box.GetMax()
    return np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], dtype=np.float32
    )


def tracked_objects(env) -> dict[str, object]:
    """The tracked scene objects, ``{name: asset}`` without the ``object_`` prefix.

    Covers both kinds of stub :func:`add_usda_scene` registers — rigid objects
    and articulations — so callers do not have to know which one a given
    object is. Both asset types expose the root-pose/velocity API
    (``data.root_pos_w``, ``write_root_pose_to_sim_index``, ...) used here.
    """
    out: dict[str, object] = {}
    for coll in (env.scene.rigid_objects, env.scene.articulations):
        for key, asset in coll.items():
            if key.startswith("object_"):
                out[key.removeprefix("object_")] = asset
    return out


def _articulation_root_link(art_prim, stage):
    """The rigid link the physics engine treats as the articulation's root.

    That is the ``RigidBodyAPI`` prim under ``art_prim`` (or ``art_prim`` itself)
    that is never ``body1`` of a joint under it — the link every joint chain
    hangs from. Falls back to the first rigid descendant in traversal order.
    Returns None when the subtree has no rigid body at all.
    """
    from pxr import Usd, UsdPhysics

    links = []
    children = set()
    for prim in Usd.PrimRange(art_prim, Usd.PrimAllPrimsPredicate):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            links.append(prim)
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            body0 = joint.GetBody0Rel().GetTargets()
            body1 = joint.GetBody1Rel().GetTargets()
            if body0 and body1:  # a joint to the world (empty body0) does not parent anything
                children.add(body1[0])
    if not links:
        return None
    for link in links:
        if link.GetPath() not in children:
            return link
    return links[0]


def _articulation_joint_positions(art_prim) -> dict[str, float]:
    """Authored joint positions under ``art_prim``, ``{joint_prim_name: value}``.

    Reads the ``PhysxSchema.JointStateAPI`` attributes — ``state:angular:physics:position``
    [deg, converted to rad] for revolute joints and ``state:linear:physics:position``
    [m] for prismatic ones — which is where USD stores a joint's initial
    position. Joints without an authored state default to 0.0. Fixed and D6
    joints are skipped (Isaac Lab names their DOFs differently, and they
    usually carry no authored state).
    """
    import math

    from pxr import Usd, UsdPhysics

    joint_pos: dict[str, float] = {}
    for prim in Usd.PrimRange(art_prim, Usd.PrimAllPrimsPredicate):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            attr, scale = prim.GetAttribute("state:angular:physics:position"), math.pi / 180.0
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            attr, scale = prim.GetAttribute("state:linear:physics:position"), 1.0
        else:
            continue
        value = attr.Get() if attr and attr.HasAuthoredValue() else None
        joint_pos[prim.GetName()] = float(value) * scale if value is not None else 0.0
    return joint_pos


def add_usda_scene(scene_cfg: InteractiveSceneCfg, usda_path: str, track_objects: bool = True) -> list[str]:
    """Reference ``usda_path`` into the scene and optionally track its physics objects.

    Args:
        scene_cfg: The scene config instance to extend (attributes are added to it).
        usda_path: Path to the scene USD/USDA file (absolute, or relative to the cwd).
        track_objects: Register an articulation stub per ``ArticulationRootAPI``
            prim and a rigid-object stub per other topmost rigid body, so env
            resets restore the authored poses (and joint positions).

    Returns:
        The names of the tracked objects (empty when ``track_objects`` is off
        or the file authors none).
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    usda_path = os.path.abspath(usda_path)
    if not os.path.exists(usda_path):
        raise FileNotFoundError(f"scene USDA not found: {usda_path}")

    scene_cfg.scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/scene",
        spawn=UsdFileCfg(usd_path=usda_path),
    )
    if not track_objects:
        return []

    # Fully compose the file (payloads included): RigidBodyAPI and the effective
    # object poses may be authored inside payloads rather than on the top-level Xforms.
    stage = Usd.Stage.Open(usda_path, Usd.Stage.LoadAll)
    root = stage.GetDefaultPrim()
    if not root:
        raise ValueError(f"{usda_path} has no default prim; UsdFileCfg cannot reference it")

    global SUPPORT_SURFACE_Z
    objects: list[str] = []
    FOOTPRINT_RADII.clear()  # one scene is loaded at a time; drop the previous scene's entries
    FOOTPRINT_BOTTOM_OFFSETS.clear()
    OBJECT_ASSET_PATHS.clear()
    OBJECT_SUPPORT_POINTS.clear()
    SUPPORT_SURFACE_Z = None
    # <scene>.usda.poses.json overrides the authored pose per tracked object; the
    # in-headset "adjust object" mode writes it. Absent → the USDA is authoritative.
    overrides = load_pose_overrides(usda_path)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=True)
    articulated: list[str] = []
    it = iter(Usd.PrimRange(root, Usd.PrimAllPrimsPredicate))
    for prim in it:
        if prim == root:
            continue
        is_articulation = prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        if not is_articulation and not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        # Only the topmost physics prim of each subtree becomes a tracked object:
        # an articulation's links are part of it, never rigid objects of their own.
        it.PruneChildren()
        if prim.IsInstanceProxy():
            print(f"[WARNING] {prim.GetPath()}: physics object inside an instance; not tracking it.")
            continue
        # Isaac Lab reports/writes an articulation's root pose on its ROOT LINK,
        # so that link (not the ArticulationRootAPI Xform) is what init_state
        # must describe.
        pose_prim = prim
        if is_articulation:
            pose_prim = _articulation_root_link(prim, stage)
            if pose_prim is None:
                print(f"[WARNING] {prim.GetPath()}: ArticulationRootAPI without a rigid link; not tracking it.")
                continue
        xform = UsdGeom.Xformable(pose_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        transform = Gf.Transform(xform)
        t = transform.GetTranslation()
        q = transform.GetRotation().GetQuat()  # Gf quaternion: real part w + imaginary (x, y, z)
        imag = q.GetImaginary()
        name = prim.GetName()
        rel_path = str(prim.GetPath())[len(str(root.GetPath())) :]  # e.g. "/bowl"
        # Bounding-circle footprint radius for the DR collision check, from the
        # world AABB (the sim_benchmark scenegen convention: circle of the max
        # horizontal half-extent).
        aabb = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        size = aabb.GetMax() - aabb.GetMin()
        FOOTPRINT_RADII[name] = max(float(size[0]), float(size[1])) / 2.0 if not aabb.IsEmpty() else 0.05
        if not aabb.IsEmpty():
            bottom_z = float(aabb.GetMin()[2])
            FOOTPRINT_BOTTOM_OFFSETS[name] = bottom_z - float(t[2])
            # Objects rest on the support surface, so the LOWEST AABB bottom is
            # the tabletop (a stacked object's bottom sits above it, hence min).
            SUPPORT_SURFACE_Z = bottom_z if SUPPORT_SURFACE_Z is None else min(SUPPORT_SURFACE_Z, bottom_z)
        else:
            FOOTPRINT_BOTTOM_OFFSETS[name] = 0.0
        asset_path = _object_asset_path(prim, stage)
        if asset_path is not None:
            OBJECT_ASSET_PATHS[name] = asset_path
        OBJECT_SUPPORT_POINTS[name] = _object_support_points(prim, bbox_cache)
        # init_state repeats the authored pose because reset_scene_to_default
        # writes it back to the sim on every reset. NOTE: InitialStateCfg.rot is
        # (x, y, z, w) on this Isaac Lab release, while Gf stores w separately.
        pos = (float(t[0]), float(t[1]), float(t[2]))
        rot = (float(imag[0]), float(imag[1]), float(imag[2]), float(q.GetReal()))
        ovr = overrides.get(name, {})
        if "pos" in ovr and "rot" in ovr:
            pos = tuple(float(v) for v in ovr["pos"])
            rot = tuple(float(v) for v in ovr["rot"])
        if is_articulation:
            # Joint positions come from the authored JointStateAPI (Isaac Lab
            # does not read them from the stage); the sidecar may override
            # individual joints under "joint_pos".
            joint_pos = _articulation_joint_positions(prim)
            joint_pos.update({k: float(v) for k, v in (ovr.get("joint_pos") or {}).items()})
            cfg = ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/scene" + rel_path,
                spawn=None,
                init_state=ArticulationCfg.InitialStateCfg(pos=pos, rot=rot, joint_pos=joint_pos or {".*": 0.0}),
                # Passive object: keep whatever drive gains the USD authors
                # (None = "use the USD value"), just so the articulation has an
                # actuator group covering every joint, which the cfg requires.
                actuators={"passive": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)},
            )
            articulated.append(name)
        else:
            cfg = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/scene" + rel_path,
                spawn=None,
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
            )
        setattr(scene_cfg, f"object_{name}", cfg)
        objects.append(name)

    if objects:
        rigid = [n for n in objects if n not in articulated]
        print(
            f"[INFO] scene '{os.path.basename(usda_path)}': tracking rigid objects {rigid}, articulated objects"
            f" {articulated}"
        )
    else:
        print(f"[INFO] scene '{os.path.basename(usda_path)}': no physics objects found to track.")
    return objects


def randomize_tracked_objects(
    env,
    env_ids: torch.Tensor,
    xy_range: float = 0.05,
    yaw_range: float = 3.14159265,
    margin: float = 0.01,
    max_tries: int = 50,
    bias_toward: tuple[float, float] | None = None,
    bias_dist: float = 0.0,
) -> None:
    """Reset event: perturb every tracked object's authored pose, rejecting collisions.

    Each object gets a uniform xy offset in ``[-xy_range, xy_range]`` and a yaw
    offset in ``[-yaw_range, yaw_range]`` about its AUTHORED pose (so objects
    stay in their scene's workspace without needing table bounds). With
    ``bias_toward`` (an xy point [m], typically the robot base) and
    ``bias_dist``, every randomization center is first shifted ``bias_dist``
    [m] horizontally toward that point (never past it), bringing the objects
    within easier reach. A draw is rejected — and the whole set resampled, up
    to ``max_tries`` — while any two objects' bounding circles overlap (radius
    from the USD footprint plus ``margin``; the collision model used by
    sim_benchmark's scenegen solvers). Runs after ``reset_scene_to_default``,
    overwriting the poses it restored. Falls back to the (biased) authored
    poses if no collision-free draw is found.

    Stacked arrangements (objects authored xy-coincident, e.g. a box lid on its
    base) are moved as one: they share a single xy offset and receive no yaw
    randomization, so the stack never gets knocked apart at reset.
    """
    from isaaclab.utils.math import quat_mul

    assets = tracked_objects(env)  # rigid objects AND articulated ones
    names = list(assets.keys())
    if len(names) == 0:
        return
    device = env.device
    base_pos = torch.stack(
        [torch.tensor(assets[n].cfg.init_state.pos, dtype=torch.float32, device=device) for n in names]
    )
    base_rot = torch.stack(
        [torch.tensor(assets[n].cfg.init_state.rot, dtype=torch.float32, device=device) for n in names]
    )
    if bias_toward is not None and bias_dist > 0.0:
        # Shift every randomization center toward the target point. Coincident
        # (stacked) objects get identical shifts, so clusters stay intact.
        target_xy = torch.tensor(bias_toward[:2], dtype=torch.float32, device=device)
        to_target = target_xy - base_pos[:, :2]
        dist = to_target.norm(dim=-1, keepdim=True)
        direction = torch.where(dist > 1e-6, to_target / dist, torch.zeros_like(to_target))
        base_pos[:, :2] += direction * torch.clamp(dist, max=bias_dist)
    radii = torch.tensor([FOOTPRINT_RADII.get(n, 0.05) + margin for n in names], device=device)

    # Per-pair clearance requirement: the bounding-circle sum, but never MORE
    # than the (biased) layout already provides (some scenes author objects
    # closer than their conservative circles — e.g. a brush leaning on a bowl —
    # and demanding extra clearance would make every draw fail; the bias can
    # also converge objects slightly, so the cap uses the biased distances).
    authored_dists = (base_pos[:, None, :2] - base_pos[None, :, :2]).norm(dim=-1)
    min_dists = torch.minimum(radii[:, None] + radii[None, :], authored_dists)

    # Cluster stacked objects (authored xy-coincident): one offset per cluster,
    # no yaw for stack members, so the stack moves as a rigid group.
    cluster = list(range(len(names)))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if float(authored_dists[i, j]) < 0.03:
                cluster[j] = cluster[i]
    stacked = [cluster.count(cluster[i]) > 1 for i in range(len(names))]
    cluster_t = torch.tensor(cluster, device=device)

    positions = base_pos
    yaws = torch.zeros(len(names), device=device)
    for _ in range(max_tries):
        cluster_offsets = (torch.rand(len(names), 2, device=device) * 2.0 - 1.0) * xy_range
        offsets = cluster_offsets[cluster_t]  # members of a cluster share their root's draw
        candidate = base_pos.clone()
        candidate[:, :2] += offsets
        dists = (candidate[:, None, :2] - candidate[None, :, :2]).norm(dim=-1)
        collided = (dists < min_dists - 1e-6) & ~torch.eye(len(names), dtype=torch.bool, device=device)
        if not bool(collided.any()):
            positions = candidate
            yaws = (torch.rand(len(names), device=device) * 2.0 - 1.0) * yaw_range
            yaws[torch.tensor(stacked, device=device)] = 0.0
            break
    else:
        print("[DR] No collision-free object placement found; keeping the authored poses.")

    half = yaws / 2.0
    yaw_quat = torch.zeros(len(names), 4, device=device)  # xyzw, rotation about world z
    yaw_quat[:, 2] = torch.sin(half)
    yaw_quat[:, 3] = torch.cos(half)
    rots = quat_mul(yaw_quat, base_rot)

    origin = env.scene.env_origins[env_ids[0]]
    for i, name in enumerate(names):
        pose = torch.cat([positions[i] + origin, rots[i]]).unsqueeze(0)
        assets[name].write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        assets[name].write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=device), env_ids=env_ids)

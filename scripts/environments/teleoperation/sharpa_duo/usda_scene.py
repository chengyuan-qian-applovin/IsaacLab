# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load an arbitrary scene USDA into an Isaac Lab interactive scene.

The USDA is referenced into the stage untouched at ``{ENV_REGEX_NS}/scene`` —
whatever geometry, lights, materials, and physics it authors just work. On top
of that, :func:`add_usda_scene` optionally discovers the file's rigid bodies
and attaches one ``spawn=None`` :class:`~isaaclab.assets.RigidObjectCfg` stub
per body, so Isaac Lab tracks them and ``reset_scene_to_default`` restores
their authored poses on every env reset. Without the stubs the scene still
loads and simulates; its objects just keep their state across resets.

Requirements on the USDA:

- It must have a default prim (that is what ``UsdFileCfg`` references).
- Object initial poses are read from the composed stage, so they may live on
  the object Xforms or inside payloads/references — both work.

Import only after AppLauncher.
"""

from __future__ import annotations

import os

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import UsdFileCfg


def add_usda_scene(scene_cfg: InteractiveSceneCfg, usda_path: str, track_objects: bool = True) -> list[str]:
    """Reference ``usda_path`` into the scene and optionally track its rigid bodies.

    Args:
        scene_cfg: The scene config instance to extend (attributes are added to it).
        usda_path: Path to the scene USD/USDA file (absolute, or relative to the cwd).
        track_objects: Register a rigid-object stub per discovered rigid body so
            env resets restore the authored poses.

    Returns:
        The names of the tracked rigid objects (empty when ``track_objects`` is off
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

    objects: list[str] = []
    it = iter(Usd.PrimRange(root, Usd.PrimAllPrimsPredicate))
    for prim in it:
        if prim == root or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        # Only the topmost rigid prim of each subtree becomes a tracked object.
        it.PruneChildren()
        if prim.IsInstanceProxy():
            print(f"[WARNING] {prim.GetPath()}: rigid body inside an instance; not tracking it.")
            continue
        xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        transform = Gf.Transform(xform)
        t = transform.GetTranslation()
        q = transform.GetRotation().GetQuat()  # Gf quaternion: real part w + imaginary (x, y, z)
        imag = q.GetImaginary()
        name = prim.GetName()
        rel_path = str(prim.GetPath())[len(str(root.GetPath())) :]  # e.g. "/bowl"
        # init_state repeats the authored pose because reset_scene_to_default
        # writes it back to the sim on every reset. NOTE: InitialStateCfg.rot is
        # (x, y, z, w) on this Isaac Lab release, while Gf stores w separately.
        setattr(
            scene_cfg,
            f"object_{name}",
            RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/scene" + rel_path,
                spawn=None,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(float(t[0]), float(t[1]), float(t[2])),
                    rot=(float(imag[0]), float(imag[1]), float(imag[2]), float(q.GetReal())),
                ),
            ),
        )
        objects.append(name)

    if objects:
        print(f"[INFO] scene '{os.path.basename(usda_path)}': tracking rigid objects {objects}")
    else:
        print(f"[INFO] scene '{os.path.basename(usda_path)}': no rigid bodies found to track.")
    return objects

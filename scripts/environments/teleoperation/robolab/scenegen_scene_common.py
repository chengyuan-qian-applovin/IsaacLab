# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load any ``sim_benchmark.scenegen`` scene USDA into the duo teleop env.

Import only after AppLauncher.

scenegen scenes share one layout contract (see the generator's header comment in
each file): a ``world`` default prim holding lights, an ``ObjectMaterial``, the
standard table Cube (top at z = 0.5421 — identical to the TACO scene, so the
robot placement, XR anchor, and Align defaults all transfer unchanged), an
invisible ground plane, and one Xform per manipulable object that payloads a
rigid-body mesh USD (RigidBodyAPI + convex-decomposition collider + mass,
produced by ``02_mesh/06_usd_conversion``) and authors its initial
``xformOp:translate`` / ``xformOp:orient``.

:func:`load_scenegen_env_cfg` therefore reuses :class:`TacoTeleopEnvCfg`
wholesale (robot, IK action space, physics settings) and only swaps the scene:
the USDA replaces the TACO one, the brush/bowl stubs are dropped, and one
``spawn=None`` :class:`RigidObjectCfg` stub is attached per payload Xform so
Isaac Lab tracks it (and the recorder captures its pose) — the same pattern
``TacoSceneCfg`` uses for its objects. Payloads must resolve, so keep the
scene's relative directory layout when copying from GCS (e.g.
``<root>/04_episode_scenegen/runs/scenes/*.usda`` next to
``<root>/02_mesh/06_usd_conversion/runs/usd/...``).
"""

from __future__ import annotations

import os

from isaaclab.assets import RigidObjectCfg

from taco_scene_common import TacoTeleopEnvCfg


def load_scenegen_env_cfg(usda_path: str) -> TacoTeleopEnvCfg:
    """Build a duo teleop env cfg for a scenegen scene USDA.

    Args:
        usda_path: Path to the scene USDA (absolute, or relative to the cwd).

    Returns:
        A :class:`TacoTeleopEnvCfg` whose scene spawns ``usda_path`` with one
        tracked rigid-object stub per payload Xform found under the default prim.
    """
    from pxr import Usd

    usda_path = os.path.abspath(usda_path)
    if not os.path.exists(usda_path):
        raise FileNotFoundError(f"scenegen scene not found: {usda_path}")

    # LoadNone: object poses live on the Xforms in this layer; the payloads
    # (mesh geometry) are only needed later, when the sim composes the stage.
    stage = Usd.Stage.Open(usda_path, Usd.Stage.LoadNone)
    world = stage.GetDefaultPrim()
    if not world:
        raise ValueError(f"{usda_path} has no default prim")

    env_cfg = TacoTeleopEnvCfg()
    scene = env_cfg.scene
    scene.scene.spawn.usd_path = usda_path
    scene.brush = None  # InteractiveScene skips None entries
    scene.bowl = None

    objects = []
    # PrimAllPrimsPredicate: with LoadNone the payload prims are UNLOADED, and the
    # default child predicate silently filters unloaded prims out.
    for child in world.GetFilteredChildren(Usd.PrimAllPrimsPredicate):
        if not child.HasAuthoredPayloads():
            continue  # lights, table, materials, ground plane
        name = child.GetName()
        t_attr = child.GetAttribute("xformOp:translate")
        q_attr = child.GetAttribute("xformOp:orient")
        t = t_attr.Get() if t_attr and t_attr.HasAuthoredValue() else (0.0, 0.0, 0.0)
        if q_attr and q_attr.HasAuthoredValue():
            q = q_attr.Get()
            imag = q.GetImaginary()
            rot = (float(q.GetReal()), float(imag[0]), float(imag[1]), float(imag[2]))
        else:
            rot = (1.0, 0.0, 0.0, 0.0)
        # Mirrors TacoSceneCfg's brush/bowl: track the prim the scene USDA already
        # carries; init_state repeats the authored pose because reset_scene_to_default
        # writes it back to the sim on every reset.
        setattr(
            scene,
            name,
            RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/scene/" + name,
                spawn=None,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(float(t[0]), float(t[1]), float(t[2])), rot=rot
                ),
            ),
        )
        objects.append(name)

    if not objects:
        raise ValueError(f"{usda_path}: no payload-bearing object Xforms under {world.GetPath()}")
    print(f"[INFO] scenegen scene '{os.path.basename(usda_path)}': tracking objects {objects}")
    return env_cfg

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Domain randomization for the TACO replay scene. Import only after AppLauncher.

Patterned on RoboLab's ``robolab/variations`` + ``run_table_variation.py``, with its
sharp edges fixed: materials are resolved by explicit prim path (not stage-traversal
by name), the dome/key lights are mutated in place between grid combos (no env
rebuild per variation), and everything is applied through Isaac Lab's tested
spawner/binding utilities.

Assets are referenced read-only from a RoboLab checkout (default ``~/RoboLab``):

- backgrounds: ``assets/backgrounds/**/*.{hdr,exr}`` (equirect HDRIs)
- table materials: ``assets/materials/**/*.mdl`` (vMaterials)
"""

from __future__ import annotations

import os
from itertools import product

import isaaclab.sim as sim_utils

def default_robolab_dir() -> str:
    """Container mount first (docker-compose.robolab.patch.yaml puts the checkout at
    /workspace/robolab), then the host-side location."""
    for candidate in ("/workspace/robolab", os.path.expanduser("~/RoboLab")):
        if os.path.isdir(os.path.join(candidate, "assets")):
            return candidate
    return os.path.expanduser("~/RoboLab")


DEFAULT_ROBOLAB_DIR = default_robolab_dir()

# Curated defaults (all overridable from the replay CLI).
DEFAULT_BACKGROUNDS = ["empty_warehouse", "brown_photostudio", "billiard_hall"]
DEFAULT_LIGHTINGS = ["default", "dim", "warm"]
DEFAULT_TABLE_MATERIALS = ["Oak", "Walnut_Planks", "Bamboo", "Carpaint_Solid"]

# name -> MDL path relative to <robolab>/assets/materials. The material prim name
# spawned by spawn_from_mdl_file is the file stem, which for vMaterials matches the
# default subidentifier.
TABLE_MATERIAL_MDLS = {
    "Oak": "Base/Wood/Oak.mdl",
    "Walnut_Planks": "Base/Wood/Walnut_Planks.mdl",
    "Bamboo": "Base/Wood/Bamboo.mdl",
    "Carpaint_Solid": "2023_1/vMaterials_2/Paint/Carpaint/Carpaint_Solid.mdl",  # RoboLab's "Black_Matte"
}

# name -> (dome intensity, key intensity, key color). The TACO scene authors
# DomeLight at 1000 and KeyLight (DistantLight) at 3000, white.
LIGHTING_PRESETS = {
    "default": (1000.0, 3000.0, (1.0, 1.0, 1.0)),
    "dim": (300.0, 800.0, (1.0, 1.0, 1.0)),
    "bright": (2000.0, 6000.0, (1.0, 1.0, 1.0)),
    "warm": (800.0, 3000.0, (1.0, 0.8, 0.6)),
    "cool": (800.0, 3000.0, (0.7, 0.85, 1.0)),
}


def discover_backgrounds(robolab_dir: str = DEFAULT_ROBOLAB_DIR) -> dict[str, str]:
    """name (file stem) -> absolute HDRI path, from <robolab>/assets/backgrounds.

    Mirrors RoboLab's ``find_background_files``: recursive, skips ``_``-prefixed
    files and directories, accepts .hdr/.exr.
    """
    root = os.path.join(robolab_dir, "assets", "backgrounds")
    found: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith("_")]
        for name in filenames:
            if name.startswith("_"):
                continue
            stem, ext = os.path.splitext(name)
            if ext.lower() in (".hdr", ".exr"):
                found[stem] = os.path.join(dirpath, name)
    return found


def _first_prim(path_regex: str):
    import isaacsim.core.utils.stage as stage_utils

    paths = sim_utils.find_matching_prim_paths(path_regex)
    if not paths:
        raise RuntimeError(f"No prim matches {path_regex}")
    return stage_utils.get_current_stage().GetPrimAtPath(paths[0])


def apply_background(hdr_path: str | None, dome_intensity: float | None = None) -> None:
    """Point the scene's existing dome light at an HDRI (mutated in place, no respawn)."""
    from pxr import Sdf, UsdLux

    dome = UsdLux.DomeLight(_first_prim("/World/envs/env_.*/scene/DomeLight"))
    if hdr_path is not None:
        dome.CreateTextureFileAttr().Set(Sdf.AssetPath(hdr_path))
        dome.CreateTextureFormatAttr().Set("latlong")
    if dome_intensity is not None:
        dome.CreateIntensityAttr().Set(float(dome_intensity))


def apply_lighting(preset: str) -> None:
    """Apply a named lighting preset to the scene's dome + key lights."""
    from pxr import Gf, UsdLux

    dome_intensity, key_intensity, key_color = LIGHTING_PRESETS[preset]
    apply_background(None, dome_intensity=dome_intensity)
    key = UsdLux.DistantLight(_first_prim("/World/envs/env_.*/scene/KeyLight"))
    key.CreateIntensityAttr().Set(float(key_intensity))
    key.CreateColorAttr().Set(Gf.Vec3f(*key_color))


def spawn_table_materials(names: list[str], robolab_dir: str = DEFAULT_ROBOLAB_DIR) -> dict[str, str]:
    """Spawn the requested MDL materials once, under /World/Looks/TableVariants.

    Returns name -> material prim path. Missing MDL files are skipped with a warning.
    """
    spawned: dict[str, str] = {}
    for name in names:
        rel = TABLE_MATERIAL_MDLS.get(name)
        if rel is None:
            print(f"[WARNING] Unknown table material '{name}' (known: {sorted(TABLE_MATERIAL_MDLS)}); skipping.")
            continue
        mdl_path = os.path.join(robolab_dir, "assets", "materials", rel)
        if not os.path.exists(mdl_path):
            print(f"[WARNING] MDL not found: {mdl_path}; skipping '{name}'.")
            continue
        prim_path = f"/World/Looks/TableVariants/{name}"
        sim_utils.spawn_from_mdl_file(prim_path, sim_utils.MdlFileCfg(mdl_path=mdl_path))
        spawned[name] = prim_path
    return spawned


def apply_table_material(material_prim_path: str) -> None:
    """Bind a spawned material to the table cube (explicit path — every env)."""
    for table_path in sim_utils.find_matching_prim_paths("/World/envs/env_.*/scene/table"):
        sim_utils.bind_visual_material(table_path, material_prim_path, stronger_than_descendants=True)


def build_grid(
    backgrounds: list[str],
    lightings: list[str],
    materials: list[str],
    robolab_dir: str = DEFAULT_ROBOLAB_DIR,
) -> list[dict]:
    """Deterministic Cartesian grid of variation combos (RoboLab's product pattern).

    Each combo is ``{"label", "background", "background_path", "lighting", "table_material"}``.
    """
    available = discover_backgrounds(robolab_dir)
    for bg in backgrounds:
        if bg not in available:
            raise FileNotFoundError(f"Background '{bg}' not under {robolab_dir}/assets/backgrounds "
                                    f"(available: {sorted(available)})")
    for lt in lightings:
        if lt not in LIGHTING_PRESETS:
            raise KeyError(f"Unknown lighting preset '{lt}' (available: {sorted(LIGHTING_PRESETS)})")
    combos = []
    for bg, lt, mat in product(backgrounds, lightings, materials):
        combos.append({
            "label": f"bg-{bg}__light-{lt}__table-{mat}",
            "background": bg,
            "background_path": available[bg],
            "lighting": lt,
            "table_material": mat,
        })
    return combos

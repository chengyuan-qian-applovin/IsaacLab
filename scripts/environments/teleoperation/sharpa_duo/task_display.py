# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Show the scene's task description to the operator, in the headset.

Scene-generation runs ship a JSON next to their USDAs (e.g.
``scenes_first50.json``) listing, per scene, a ``task_description``.
:func:`find_task_description` looks that text up for a scene file, and
:func:`spawn_task_display` puts it in the world as a floating billboard — a
textured quad whose texture is the text rendered offline with PIL — so the
operator reads the task in XR while teleoperating. The material is emissive,
making the panel readable regardless of the scene's lighting.

Import only after AppLauncher.
"""

from __future__ import annotations

import glob
import json
import os
import tempfile

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)

#: Per-directory cache of {scene basename: task description}.
_DIR_CACHE: dict[str, dict[str, str]] = {}


def find_task_description(scene_path: str) -> str | None:
    """Task description for a scene file, from any instructions JSON in its directory.

    Every ``*.json`` next to the scene is scanned once (per directory) for the
    scene-generation format: a ``scenes`` list of entries with ``scene`` (a path,
    matched by basename — the authored paths come from another machine) and
    ``task_description``. Returns None when no JSON describes this scene.
    """
    directory = os.path.dirname(os.path.abspath(scene_path))
    if directory not in _DIR_CACHE:
        mapping: dict[str, str] = {}
        for json_path in sorted(glob.glob(os.path.join(directory, "*.json"))):
            try:
                with open(json_path) as f:
                    data = json.load(f)
            except (OSError, ValueError):
                continue
            entries = data.get("scenes") if isinstance(data, dict) else data
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if isinstance(entry, dict) and entry.get("scene") and entry.get("task_description"):
                    # Some descriptions arrive wrapped in stray quotes.
                    text = str(entry["task_description"]).strip().strip("'\"")
                    mapping[os.path.basename(entry["scene"])] = text
        _DIR_CACHE[directory] = mapping
    return _DIR_CACHE[directory].get(os.path.basename(scene_path))


def _render_text_png(text: str, width_px: int = 1024) -> tuple[str, float]:
    """Render ``text`` (white on dark, pixel-wrapped) to a temp PNG.

    Returns:
        (png_path, aspect) where aspect is image height / width.
    """
    from PIL import Image, ImageDraw, ImageFont

    font = None
    for candidate in _FONT_CANDIDATES:
        if os.path.exists(candidate):
            font = ImageFont.truetype(candidate, 58)
            break
    if font is None:
        font = ImageFont.load_default(size=58)

    margin = 48
    max_line_px = width_px - 2 * margin
    measurer = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        line = ""
        for word in paragraph.split():
            trial = f"{line} {word}".strip()
            if measurer.textlength(trial, font=font) <= max_line_px or not line:
                line = trial
            else:
                lines.append(line)
                line = word
        lines.append(line)

    line_height = 74
    height_px = 2 * margin + line_height * len(lines)
    img = Image.new("RGB", (width_px, height_px), (18, 20, 26))
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        line_px = measurer.textlength(line, font=font)
        draw.text(((width_px - line_px) / 2, margin + i * line_height), line, font=font, fill=(240, 240, 245))

    png_path = os.path.join(tempfile.mkdtemp(prefix="duo_task_display_"), "task.png")
    img.save(png_path)
    return png_path, height_px / width_px


def spawn_task_display(text: str, position: tuple[float, float, float], yaw_deg: float, width: float = 1.2) -> None:
    """Spawn the task text as an emissive billboard at ``position``.

    Args:
        text: The task description to display.
        position: Billboard center in the world frame [m].
        yaw_deg: Rotation about world z [deg]; at 0 the panel faces -y (an
            operator north-facing along +y reads it head-on).
        width: Panel width [m]; the height follows the wrapped text.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    import isaaclab.sim as sim_utils

    stage = sim_utils.get_current_stage()
    prim_path = "/World/TaskDisplay"
    png_path, aspect = _render_text_png(text)
    half_w, half_h = width / 2.0, width * aspect / 2.0

    xform = UsdGeom.Xform.Define(stage, prim_path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    xform.AddRotateZOp().Set(float(yaw_deg))

    # A single quad in the local xz-plane, normal -y; +x is the text's reading
    # direction. st(0,0) samples the texture's LOWER-left, so v runs bottom-up.
    mesh = UsdGeom.Mesh.Define(stage, prim_path + "/panel")
    mesh.CreatePointsAttr([(-half_w, 0, -half_h), (half_w, 0, -half_h), (half_w, 0, half_h), (-half_w, 0, half_h)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateExtentAttr([(-half_w, 0.0, -half_h), (half_w, 0.0, half_h)])
    st = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    st.Set([(0, 0), (1, 0), (1, 1), (0, 1)])

    material = UsdShade.Material.Define(stage, prim_path + "/material")
    shader = UsdShade.Shader.Define(stage, prim_path + "/material/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    reader = UsdShade.Shader.Define(stage, prim_path + "/material/st_reader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    texture = UsdShade.Shader.Define(stage, prim_path + "/material/texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(png_path)
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    rgb = texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb)
    # Emissive so the panel reads the same under any scene lighting.
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

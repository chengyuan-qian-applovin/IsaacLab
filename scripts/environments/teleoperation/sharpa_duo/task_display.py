# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Show text to the operator in the headset, as floating billboards.

Scene-generation runs ship a JSON next to their USDAs (e.g.
``scenes_first50.json``) listing, per scene, a ``task_description``.
:func:`find_task_description` looks that text up for a scene file, and
:func:`spawn_task_display` puts it in the world as a floating billboard — a
textured quad whose texture is the text rendered offline with PIL — so the
operator reads the task in XR while teleoperating. The material is emissive,
making the panel readable regardless of the scene's lighting.

:class:`MessageDisplay` is the transient counterpart: a fixed-size panel whose
text is replaced at runtime and which hides itself after a hold time, used to
echo voice commands back to the operator.

Import only after AppLauncher.
"""

from __future__ import annotations

import contextlib
import glob
import itertools
import json
import os
import tempfile
import time

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)

#: Per-directory cache of {scene basename: task description}.
_DIR_CACHE: dict[str, dict[str, str]] = {}

#: Scratch space for :class:`MessageDisplay` textures. Each message needs its
#: own file: Hydra caches textures by asset path and would not reload a reused
#: one, so the names are serialized by a counter rather than overwritten.
_MESSAGE_DIR = tempfile.mkdtemp(prefix="duo_message_display_")
_MESSAGE_COUNTER = itertools.count()


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


def _build_billboard(
    prim_path: str,
    png_path: str,
    position: tuple[float, float, float],
    yaw_deg: float,
    half_w: float,
    half_h: float,
):
    """Author an emissive textured quad and return its ``UsdUVTexture`` shader.

    Args:
        prim_path: Stage path to author the billboard under.
        png_path: Texture image to bind.
        position: Billboard center in the world frame [m].
        yaw_deg: Rotation about world z [deg]; at 0 the panel faces -y (an
            operator north-facing along +y reads it head-on).
        half_w: Half the panel width [m].
        half_h: Half the panel height [m].

    Returns:
        The texture shader, so callers can retarget its ``file`` input later.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    import isaaclab.sim as sim_utils

    stage = sim_utils.get_current_stage()
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
    return texture


def spawn_task_display(text: str, position: tuple[float, float, float], yaw_deg: float, width: float = 1.2) -> None:
    """Spawn the task text as an emissive billboard at ``position``.

    Args:
        text: The task description to display.
        position: Billboard center in the world frame [m].
        yaw_deg: Rotation about world z [deg]; at 0 the panel faces -y (an
            operator north-facing along +y reads it head-on).
        width: Panel width [m]; the height follows the wrapped text.
    """
    png_path, aspect = _render_text_png(text)
    _build_billboard("/World/TaskDisplay", png_path, position, yaw_deg, width / 2.0, width * aspect / 2.0)


def _render_message_png(text: str, width_px: int = 1024, aspect: float = 0.28, max_lines: int = 3) -> str:
    """Render ``text`` centered on a fixed-size canvas and return the PNG path.

    Unlike :func:`_render_text_png` the canvas size never varies with the text,
    so a panel can swap this texture in without re-authoring its quad. Text
    beyond ``max_lines`` is truncated with an ellipsis.

    Args:
        text: The message to draw; empty renders a blank panel.
        width_px: Canvas width [px].
        aspect: Canvas height / width.
        max_lines: Lines kept before truncating.

    Returns:
        Path to a freshly written PNG. Each call writes a new file because
        Hydra caches textures by asset path and would not reload a reused one.
    """
    from PIL import Image, ImageDraw, ImageFont

    font = None
    for candidate in _FONT_CANDIDATES:
        if os.path.exists(candidate):
            font = ImageFont.truetype(candidate, 52)
            break
    if font is None:
        font = ImageFont.load_default(size=52)

    height_px = int(round(width_px * aspect))
    margin, line_height = 32, 66
    max_line_px = width_px - 2 * margin
    measurer = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    lines: list[str] = []
    line = ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if measurer.textlength(trial, font=font) <= max_line_px or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    if len(lines) > max_lines:
        lines = [*lines[: max_lines - 1], lines[max_lines - 1] + " ..."]

    img = Image.new("RGB", (width_px, height_px), (18, 20, 26))
    draw = ImageDraw.Draw(img)
    block_h = line_height * len(lines)
    top = max(margin, (height_px - block_h) // 2)
    for i, entry in enumerate(lines):
        line_px = measurer.textlength(entry, font=font)
        draw.text(((width_px - line_px) / 2, top + i * line_height), entry, font=font, fill=(240, 240, 245))

    png_path = os.path.join(_MESSAGE_DIR, f"msg_{next(_MESSAGE_COUNTER)}.png")
    img.save(png_path)
    return png_path


class MessageDisplay:
    """A billboard whose text is replaced at runtime and which hides itself.

    Transient operator feedback in the headset — what the voice labeler heard
    and what it did about it. The quad is authored once at a fixed aspect, so
    :meth:`show` only retargets the texture rather than rebuilding geometry.

    Call :meth:`update` once per frame to let the panel time out.
    """

    def __init__(
        self,
        position: tuple[float, float, float],
        yaw_deg: float,
        width: float = 1.2,
        aspect: float = 0.28,
        seconds: float = 4.0,
        prim_path: str = "/World/VoiceDisplay",
    ):
        """Author the (initially hidden) panel.

        Args:
            position: Panel center in the world frame [m].
            yaw_deg: Rotation about world z [deg]; matches
                :func:`spawn_task_display`'s convention.
            width: Panel width [m].
            aspect: Panel height / width.
            seconds: How long a message stays up before auto-hiding.
            prim_path: Stage path to author the panel under.
        """
        from pxr import UsdGeom

        import isaaclab.sim as sim_utils

        self._aspect = aspect
        self._seconds = seconds
        self._hide_at: float | None = None
        #: The live texture, plus the last few it replaced (see :meth:`show`).
        self._pngs = [_render_message_png("", aspect=aspect)]
        self._texture = _build_billboard(prim_path, self._pngs[0], position, yaw_deg, width / 2.0, width * aspect / 2.0)
        self._imageable = UsdGeom.Imageable(sim_utils.get_current_stage().GetPrimAtPath(prim_path))
        self._imageable.MakeInvisible()

    def show(self, text: str) -> None:
        """Display ``text`` and restart the auto-hide timer."""
        from pxr import Sdf

        png = _render_message_png(text, aspect=self._aspect)
        self._texture.GetInput("file").Set(Sdf.AssetPath(png))
        self._imageable.MakeVisible()
        self._hide_at = time.monotonic() + self._seconds
        # Retire old textures a couple of messages late: Hydra can still be
        # sampling the one just swapped out, and would show a hole if it went.
        self._pngs.append(png)
        while len(self._pngs) > 3:
            with contextlib.suppress(OSError):
                os.remove(self._pngs.pop(0))

    def update(self) -> None:
        """Hide the panel once the current message has had its hold time."""
        if self._hide_at is not None and time.monotonic() >= self._hide_at:
            self._imageable.MakeInvisible()
            self._hide_at = None

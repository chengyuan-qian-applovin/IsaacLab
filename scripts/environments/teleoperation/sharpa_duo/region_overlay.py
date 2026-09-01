# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""In-headset visualization of the object-randomization region (adjust mode).

For every tracked object, a translucent square on the tabletop shows the XY
region ``randomize_tracked_objects`` samples (axis-aligned, side ``2*xy_range``,
centered on the object's pose after the reach bias). The yaw range is shown
only as a number on the RangePanel — earlier yaw-indicator bars around each
object were dropped as visual clutter (operator feedback). The square is a
:class:`~isaaclab.markers.VisualizationMarkers` PointInstancer — real USD
geometry, so it renders in the CloudXR headset (unlike the viewport-only
``debug_draw`` overlay).

The overlay is redrawn every adjust-mode frame from the LIVE DR params dict and
the objects' CURRENT poses: it previews the region that will apply once "done"
makes the resting poses the new authored centers, and it follows every input
path (panel, keypad, stdin) with no extra wiring. Stacked objects (within 3 cm
in xy) mirror the randomizer's cluster rule: one shared square.

Import only after AppLauncher.
"""

from __future__ import annotations

import math

import torch

#: Objects closer than this in xy are one stack sharing a single square
#: (mirror of the cluster rule in :func:`usda_scene.randomize_tracked_objects`).
_STACK_XY_M = 0.03

#: Height of the square above the surface the object rests on [m] (z-fighting guard).
_SQUARE_LIFT_M = 0.003


class RegionOverlay:
    """Live randomization-region markers for the adjust mode.

    Args:
        env: The live :class:`~isaaclab.envs.ManagerBasedRLEnv`.
        tracked_object_names: Names as returned by :func:`usda_scene.add_usda_scene`
            (without the ``object_`` prefix).
    """

    def __init__(self, env, tracked_object_names: list[str]):
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        self._env = env
        self._names = [n for n in tracked_object_names if f"object_{n}" in env.scene.rigid_objects]
        self._visible = False
        self._markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/dr_region_overlay",
                markers={
                    # The XY region, scaled to (2*xy_range, 2*xy_range) per instance.
                    "square": sim_utils.CuboidCfg(
                        size=(1.0, 1.0, 0.002),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.15, 0.5, 1.0), emissive_color=(0.05, 0.2, 0.45), opacity=0.25
                        ),
                    ),
                },
            )
        )
        self._markers.set_visibility(False)

    def show(self) -> None:
        """Show the overlay (the next :meth:`update` draws it)."""
        self._visible = True
        self._markers.set_visibility(True)

    def hide(self) -> None:
        """Hide the overlay."""
        self._visible = False
        self._markers.set_visibility(False)

    def update(self, dr_params: dict) -> None:
        """Redraw the overlay from the live DR params and current object poses.

        Args:
            dr_params: The live param dict of the ``dr_objects`` event term —
                ``xy_range`` [m], ``yaw_range`` [rad], and optionally
                ``bias_toward`` (xy [m]) with ``bias_dist`` [m].
        """
        if not self._visible or not self._names:
            return
        from usda_scene import FOOTPRINT_BOTTOM_OFFSETS

        xy_range = float(dr_params.get("xy_range", 0.0))
        bias_toward = dr_params.get("bias_toward")
        bias_dist = float(dr_params.get("bias_dist", 0.0))

        # Current world poses (env 0).
        centers: list[tuple[float, float, float]] = []
        for name in self._names:
            data = self._env.scene[f"object_{name}"].data
            px, py, pz = (float(v) for v in data.root_pos_w.torch[0, :3])
            # The randomization center is the (future) authored pose after the
            # reach bias — mirror of the shift in randomize_tracked_objects.
            if bias_toward is not None and bias_dist > 0.0:
                dx, dy = float(bias_toward[0]) - px, float(bias_toward[1]) - py
                dist = math.hypot(dx, dy)
                if dist > 1e-6:
                    shift = min(dist, bias_dist) / dist
                    px, py = px + dx * shift, py + dy * shift
            z = pz + FOOTPRINT_BOTTOM_OFFSETS.get(name, 0.0) + _SQUARE_LIFT_M
            centers.append((px, py, z))

        # Cluster stacked objects exactly like the randomizer (transitive chain).
        cluster = list(range(len(self._names)))
        for i in range(len(self._names)):
            for j in range(i + 1, len(self._names)):
                if math.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1]) < _STACK_XY_M:
                    cluster[j] = cluster[i]

        side = max(2.0 * xy_range, 0.01)  # a hairline square still marks the center at range 0
        translations = [centers[i] for i in range(len(self._names)) if cluster[i] == i]  # one per cluster
        count = len(translations)
        self._markers.visualize(
            translations=torch.tensor(translations, dtype=torch.float32),
            orientations=torch.tensor([(0.0, 0.0, 0.0, 1.0)] * count, dtype=torch.float32),
            scales=torch.tensor([(side, side, 1.0)] * count, dtype=torch.float32),
            marker_indices=[0] * count,
        )

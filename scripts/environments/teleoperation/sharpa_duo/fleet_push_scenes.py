# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Push scene packages (with targets and task descriptions) to the fleet server.

Uploads over HTTP, so it works from any collector machine — no access to the
server's disk needed. Re-pushing a scene replaces its file and keeps its
collected-episode history. No simulator involved; runs with plain Python.

The server's convention is one self-contained ``.usdz`` per scene (flattened
composition, geometry and textures inside, no external references except the
runtime-resolved ``OmniPBR.mdl``): collectors download exactly that one file.
Generator output (``.usda`` referencing a ``02_mesh/...`` tree) is NOT
self-contained — convert it first (see the teleop-data-server README, "File
organization convention"). Pushing a non-``.usdz`` file is allowed but warned
about, since the server does not check what it references.

Examples:

    # everything in the vendored scene list, 20 successes each
    ./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/fleet_push_scenes.py \\
        --fleet_server http://fleet-host:8080 --scene_list scenes/scene_list.json --target 20

    # every .usdz (or .usda/.usd) under a directory
    ./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/fleet_push_scenes.py \\
        --fleet_server http://fleet-host:8080 --scene_dir ~/scenes_usdz --target 10

    # individual files, higher priority (suggested to collectors first)
    ./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/fleet_push_scenes.py \\
        --fleet_server http://fleet-host:8080 --priority 5 scenes/taco/scene/taco_hoi_178_023.usda
"""

from __future__ import annotations

import argparse
import json
import os

from fleet_client import FleetClient, FleetError
from task_display import find_task_description

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
SCENE_SUFFIXES = (".usdz", ".usda", ".usd")

parser.add_argument("scenes", nargs="*", help="Individual scene files (.usdz preferred) to push.")
parser.add_argument("--fleet_server", type=str, required=True, help="Fleet server URL (e.g. http://fleet-host:8080).")
parser.add_argument(
    "--fleet_token", type=str, default=None, help="Auth token (default: the FLEET_TOKEN environment variable)."
)
parser.add_argument(
    "--scene_dir", type=str, default=None, help="Push every .usdz/.usda/.usd under this directory (recursive)."
)
parser.add_argument(
    "--scene_list",
    type=str,
    default=None,
    help="Push the scenes named in this scene_list JSON (same format the teleop script accepts).",
)
parser.add_argument("--target", type=int, default=None, help="Target successes per scene (server default: keep/20).")
parser.add_argument("--priority", type=int, default=None, help="Priority: higher is suggested to collectors first.")


def collect_entries(args: argparse.Namespace) -> list[tuple[str, str | None]]:
    """The (path, task_description) pairs selected by the CLI arguments."""
    entries: list[tuple[str, str | None]] = []
    if args.scene_dir:
        for root, _dirs, files in os.walk(args.scene_dir):
            entries += [(os.path.join(root, f), None) for f in sorted(files) if f.endswith(SCENE_SUFFIXES)]
    if args.scene_list:
        base = os.path.dirname(os.path.abspath(args.scene_list))
        with open(args.scene_list) as f:
            data = json.load(f)
        for entry in data["scenes"] if isinstance(data, dict) else data:
            if isinstance(entry, dict):
                path, description = entry.get("scene", ""), entry.get("task_description")
                description = str(description).strip().strip("'\"") if description else None
            else:
                path, description = entry, None
            if not os.path.isabs(path):
                path = os.path.join(base, path)
            elif not os.path.exists(path):
                path = os.path.join(base, os.path.basename(path))
            entries.append((path, description))
    entries += [(p, None) for p in args.scenes]
    return entries


def main() -> None:
    args = parser.parse_args()
    entries = collect_entries(args)
    if not entries:
        parser.error("nothing to push: pass scene files, --scene_dir, or --scene_list")
    client = FleetClient(args.fleet_server, token=args.fleet_token)
    pushed = 0
    for path, description in entries:
        if not os.path.exists(path):
            print(f"[PUSH] MISSING {path} — skipped")
            continue
        scene_id = os.path.basename(path)
        if not scene_id.endswith(".usdz"):
            print(
                f"[PUSH] WARNING {scene_id}: not a .usdz package — collectors download only this one file,"
                " so any external asset reference in it will fail to resolve on their machines."
            )
        description = description or find_task_description(path)
        try:
            row = client.push_scene(
                path, target_successes=args.target, priority=args.priority, task_description=description
            )
        except FleetError as exc:
            raise SystemExit(f"[PUSH] Failed on {scene_id}: {exc}")
        print(
            f"[PUSH] {scene_id}: target {row['target_successes']}, priority {row['priority']},"
            f" {row['size_bytes']} bytes"
        )
        pushed += 1
    print(f"[PUSH] Done: {pushed}/{len(entries)} scenes on {args.fleet_server}")


if __name__ == "__main__":
    main()

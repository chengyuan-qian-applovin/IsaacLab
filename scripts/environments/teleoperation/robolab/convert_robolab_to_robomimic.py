# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert RoboLab teleop recordings into a single robomimic/Isaac Lab-style HDF5 dataset.

RoboLab's ``StreamingHDF5DatasetFileHandler`` already writes the Isaac Lab
``EpisodeData`` layout (``data/demo_N/{initial_state, actions, states, obs, ...}``
with robomimic-style ``env_args`` attrs), one ``run_<N>.hdf5`` file per teleop demo.
This script merges those runs into one dataset file, optionally keeping only
successful demos, renumbering them ``demo_0..demo_K`` — the format that Isaac Lab's
imitation-learning tooling (``replay_demos.py``, Isaac Lab Mimic, robomimic training
configs) consumes.

Pure h5py — no Isaac Sim required. Example:

    python convert_robolab_to_robomimic.py \\
        --input_dir /workspace/robolab/output --output ./robomimic_dataset.hdf5 \\
        --env_name Isaac-RoboLab-Teleop-v0
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import h5py


def find_input_files(input_dir: str) -> list[str]:
    """Return run_*.hdf5 (sorted by run index) plus any data.hdf5 in the directory."""
    runs = glob.glob(os.path.join(input_dir, "run_*.hdf5"))
    runs.sort(key=lambda p: int(os.path.basename(p)[len("run_"):-len(".hdf5")]))
    default = os.path.join(input_dir, "data.hdf5")
    if os.path.isfile(default):
        runs.append(default)
    return runs


def demo_num_samples(demo_group: h5py.Group) -> int:
    """Number of steps in a demo, from the attr if present, else from the actions dataset."""
    if "num_samples" in demo_group.attrs:
        return int(demo_group.attrs["num_samples"])
    if "actions" in demo_group:
        return int(demo_group["actions"].shape[0])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", type=str, help="Directory containing RoboLab run_*.hdf5 files.")
    parser.add_argument("--inputs", type=str, nargs="*", default=None, help="Explicit input HDF5 files.")
    parser.add_argument("--output", type=str, required=True, help="Output dataset path (.hdf5).")
    parser.add_argument("--env_name", type=str, default=None, help="Override env_name in env_args attrs.")
    parser.add_argument(
        "--include_failed", action="store_true", help="Also include demos not marked successful."
    )
    args = parser.parse_args()

    inputs = list(args.inputs) if args.inputs else []
    if args.input_dir:
        inputs.extend(find_input_files(args.input_dir))
    if not inputs:
        print("error: no input files (use --input_dir and/or --inputs)", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_demo_idx = 0
    total_samples = 0
    skipped = 0
    env_args: dict = {}

    with h5py.File(args.output, "w") as out:
        out_data = out.create_group("data")

        for path in inputs:
            try:
                src = h5py.File(path, "r")
            except OSError as exc:
                print(f"warning: skipping unreadable file {path}: {exc}", file=sys.stderr)
                continue
            with src:
                if "data" not in src:
                    print(f"warning: {path} has no 'data' group, skipping", file=sys.stderr)
                    continue
                src_data = src["data"]
                # Carry over env_args from the first file that has them.
                if not env_args and "env_args" in src_data.attrs:
                    env_args = json.loads(src_data.attrs["env_args"])

                demo_names = sorted(
                    (n for n in src_data.keys() if n.startswith("demo_")),
                    key=lambda n: int(n.split("_")[1]),
                )
                for name in demo_names:
                    demo = src_data[name]
                    success = bool(demo.attrs.get("success", False))
                    if not success and not args.include_failed:
                        skipped += 1
                        continue
                    dst_name = f"demo_{out_demo_idx}"
                    src.copy(demo, out_data, name=dst_name)
                    n = demo_num_samples(demo)
                    out_data[dst_name].attrs["num_samples"] = n
                    total_samples += n
                    out_demo_idx += 1
                    print(f"  {os.path.basename(path)}:{name} -> {dst_name} (success={success}, samples={n})")

        if args.env_name:
            env_args["env_name"] = args.env_name
        env_args.setdefault("type", 2)
        out_data.attrs["env_args"] = json.dumps(env_args)
        out_data.attrs["total"] = total_samples

    print(
        f"\nWrote {out_demo_idx} demos ({total_samples} samples) to {args.output}"
        + (f"; skipped {skipped} unsuccessful demos" if skipped else "")
    )
    if out_demo_idx == 0:
        print("warning: output contains no demos", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

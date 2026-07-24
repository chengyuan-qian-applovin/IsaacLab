#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Install RoboLab into the Isaac Lab container's Isaac Sim Python environment.
#
# Run INSIDE the isaac-lab-base container, with the RoboLab checkout mounted at
# /workspace/robolab (see docker/docker-compose.robolab.patch.yaml):
#
#   bash scripts/environments/teleoperation/robolab/install_robolab.sh
#
# Why --no-deps: RoboLab's pyproject pins pip-installed isaacsim/isaaclab wheels
# (its standalone uv install path). Inside this container Isaac Sim is the /isaac-sim
# binary install and Isaac Lab comes from /workspace/isaaclab source, so letting pip
# resolve RoboLab's dependency pins would try to install a second, conflicting stack.
# We install RoboLab itself without dependencies and add only the small utility
# libraries it actually imports on top of the container stack.

set -euo pipefail

ROBOLAB_PATH="${ROBOLAB_PATH:-/workspace/robolab}"
PYTHON="${ISAACLAB_PATH:-/workspace/isaaclab}/_isaac_sim/python.sh"

if [[ ! -d "${ROBOLAB_PATH}" ]]; then
    echo "error: RoboLab not found at ${ROBOLAB_PATH}." >&2
    echo "Mount it via: ./docker/container.py start --files docker-compose.robolab.patch.yaml ..." >&2
    exit 1
fi

# 1) Fix the torch CUDA library resolution in the Isaac Sim 5.1 container image
#    (torch 2.7.0+cu128 vs mismatched CUDA-13 nvidia packages in site-packages).
#    Registers Isaac Sim's bundled, version-matched CUDA 12.8 libs with the loader.
if [[ ! -f /etc/ld.so.conf.d/isaac-torch-cuda.conf ]]; then
    ls -d /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/*/lib \
          /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/cusparselt/lib \
          > /etc/ld.so.conf.d/isaac-torch-cuda.conf
    ldconfig
    echo "[INFO] Registered Isaac Sim prebundled CUDA libs with ldconfig."
fi

# 2) RoboLab's small runtime dependencies that the container python lacks.
#    (h5py, psutil, tqdm, numpy, warp etc. already ship with Isaac Sim.)
"${PYTHON}" -m pip install --quiet opencv-python-headless

# 3) RoboLab itself, editable, without its pinned simulator stack.
"${PYTHON}" -m pip install --quiet --no-deps -e "${ROBOLAB_PATH}"

# 4) Verify.
"${PYTHON}" - <<'EOF'
import cv2  # noqa: F401  (must precede isaaclab imports in entry scripts)
import robolab
import robolab.constants
print(f"[OK] robolab {getattr(robolab, '__version__', '(no __version__)')} at {robolab.__file__}")
print(f"[OK] TASK_DIR: {robolab.constants.TASK_DIR}")
import os
if not os.path.isdir(robolab.constants.TASK_DIR):
    raise SystemExit(f"error: TASK_DIR does not exist: {robolab.constants.TASK_DIR}")
EOF

echo "[INFO] RoboLab installed. Missing imports at runtime? Install them the same way:"
echo "       ${PYTHON} -m pip install <package>"

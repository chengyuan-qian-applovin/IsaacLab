# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode recording for the duo teleop: robomimic-style HDF5, one demo per episode.

Built on Isaac Lab's recorder manager. Each demo carries:

- ``initial_state`` and per-step ``states`` (robot joint state + tracked object poses),
- ``actions`` — the 58-D commands sent to ``env.step`` (root-frame wrist poses +
  finger joint targets),
- ``obs/xr_hands`` — the raw 26-joint XR hand poses, shape (T, 2, 26, 7) as
  [x, y, z, qx, qy, qz, qw] in the sim world frame: the retargeter INPUT, before
  any retargeting, so retargeters can be re-tuned offline,
- ``obs/joint_setpoints`` — the PD drive targets, shape (T, 58): the control
  signal after all action terms, i.e. the differential-IK *output* for the arms,
- a boolean ``success`` attribute (the voice label).

All demos of a session land in one timestamped file (the HDF5 handler truncates
its file at env creation, so a fixed name would wipe earlier sessions). Exports
happen only through the teleop script's explicit calls — an env reset means
"discard", never "export".

Import only after AppLauncher.
"""

from __future__ import annotations

import os

import torch

from isaaclab.envs.mdp.recorders.recorders_cfg import (
    InitialStateRecorderCfg,
    PostStepStatesRecorderCfg,
    PreStepActionsRecorderCfg,
)
from isaaclab.managers.recorder_manager import DatasetExportMode, RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.datasets import HDF5DatasetFileHandler


class AppendableHDF5DatasetFileHandler(HDF5DatasetFileHandler):
    """HDF5 handler that APPENDS to an existing dataset file instead of truncating.

    The stock handler opens its file in "w" mode at env creation, wiping earlier
    demos — fine for one file per session, wrong for a shared dataset accumulated
    across sessions and scene switches. When the file already exists, open it in
    append mode; demo numbering continues from the episodes already present.
    """

    def create(self, file_path: str, env_name: str | None = None):
        if not file_path.endswith(".hdf5"):
            file_path += ".hdf5"
        if os.path.exists(file_path):
            self.open(file_path, mode="a")
        else:
            super().create(file_path, env_name=env_name)


class XrHandsRecorder(RecorderTerm):
    """Records the raw XR hand block each control step (``obs/xr_hands``).

    The recorder manager is constructed with the env, before the teleop device
    exists, so the teleop loop late-binds the data source by assigning
    ``XrHandsRecorder.latest`` (a (2, 26, 7) tensor) after every device poll.
    Steps taken before the first poll record zeros. Recorded pre-step, so row t
    is the tracking frame that produced ``actions[t]``.
    """

    latest: torch.Tensor | None = None

    def record_pre_step(self):
        data = XrHandsRecorder.latest
        if data is None:
            data = torch.zeros(2, 26, 7)
        return "obs/xr_hands", data.unsqueeze(0).to(self._env.device)


@configclass
class XrHandsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = XrHandsRecorder


class JointSetpointsRecorder(RecorderTerm):
    """Records the PD joint-position targets actually sent to the drives.

    Read post-step from ``asset.data.joint_pos_target`` — the control signal
    after ALL action terms applied: for the arms this is the differential-IK
    output (which ``actions`` does not contain — it only holds the commanded
    wrist poses), for the fingers the joint-position targets. Joint order
    matches the recorded articulation joint state, so setpoint-vs-measured
    tracking error is a direct subtraction. Row t is the setpoint active
    during step t.
    """

    def record_post_step(self):
        asset = self._env.scene[self.cfg.asset_name]
        return "obs/joint_setpoints", asset.data.joint_pos_target.torch.clone()


@configclass
class JointSetpointsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = JointSetpointsRecorder
    asset_name: str = "robot"


@configclass
class DuoRecorderManagerCfg(RecorderManagerBaseCfg):
    """See the module docstring for what each term captures."""

    record_initial_state = InitialStateRecorderCfg()
    record_post_step_states = PostStepStatesRecorderCfg()
    record_pre_step_actions = PreStepActionsRecorderCfg()
    record_pre_step_xr_hands = XrHandsRecorderCfg()
    record_post_step_joint_setpoints = JointSetpointsRecorderCfg()

    dataset_export_mode = DatasetExportMode.EXPORT_ALL
    export_in_record_pre_reset = False

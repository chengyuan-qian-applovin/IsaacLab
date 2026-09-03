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
- ``obs/joint_setpoints`` — the PD drive targets, shape (T, num actuated
  joints — 58 on the Franka duo, 56 on the YAM duo): the control signal after
  all action terms, i.e. the differential-IK *output* for the arms,
- a boolean ``success`` attribute (the voice label).

Every episode is exported to its OWN HDF5 file (as ``demo_0``), named
``<prefix>_<timestamp>_<uuid8>.hdf5`` — one trajectory per file, so a fleet
uploader can ship each labeled trajectory the moment it is closed, and a
partially written file can never corrupt earlier demos. Exports happen only
through the teleop script's explicit calls — an env reset means "discard",
never "export".

Import only after AppLauncher.
"""

from __future__ import annotations

import json
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


class PerEpisodeHDF5DatasetFileHandler(HDF5DatasetFileHandler):
    """HDF5 handler that writes each exported episode to its own file.

    The recorder manager treats its handler as one dataset file created at env
    creation; this handler instead remembers the directory + filename prefix it
    was created with and opens a fresh file per :meth:`write_episode` (each
    holding a single ``demo_0``). The teleop flow names the next file by
    assigning :attr:`next_episode_stem` before exporting, tags extra attributes
    on the freshly written (still open) file, then calls :meth:`close`; the
    completed file's path is left in :attr:`last_file_path`.
    """

    def __init__(self):
        super().__init__()
        self._dir = "."
        self._prefix = "episode"
        self._env_name = None
        self._session_count = 0
        self.next_episode_stem: str | None = None
        self.last_file_path: str | None = None

    def create(self, file_path: str, env_name: str | None = None):
        # Called once by the recorder manager with <export_dir>/<dataset_filename>;
        # only remember where episode files should go — no file is opened yet.
        if file_path.endswith(".hdf5"):
            file_path = file_path[:-5]
        self._dir = os.path.dirname(file_path) or "."
        self._prefix = os.path.basename(file_path)
        self._env_name = env_name
        os.makedirs(self._dir, exist_ok=True)

    def add_env_args(self, env_args: dict):
        # The recorder manager sets env args before write_episode, when no file
        # is open yet: buffer them, and (re-)stamp them whenever a file IS open.
        self._env_args.update(env_args)
        if self._hdf5_file_stream is not None:
            self._hdf5_data_group.attrs["env_args"] = json.dumps(self._env_args)

    def write_episode(self, episode, demo_id: int | None = None, dataset_compression: bool = True):
        if episode.is_empty():
            return
        self.close()  # the previous episode's file, if the flow left it open
        stem = self.next_episode_stem or f"{self._prefix}_{self._session_count:04d}"
        self.next_episode_stem = None
        path = os.path.join(self._dir, f"{stem}.hdf5")
        super().create(path, env_name=self._env_name)
        super().write_episode(episode, demo_id=demo_id, dataset_compression=dataset_compression)
        self._session_count += 1
        self.last_file_path = path

    def flush(self):
        if self._hdf5_file_stream is not None:
            super().flush()

    def get_num_episodes(self) -> int:
        return self._session_count


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

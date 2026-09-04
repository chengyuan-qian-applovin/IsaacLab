# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment config shared by the duo teleop and replay scripts.

Holds the scene skeleton (dome light + robot slot; the USDA scene and its
objects are added at runtime via :func:`usda_scene.add_usda_scene`) and the
manager configs. Each script subclasses :class:`DuoEnvCfg` and sets its own
physics/timing in ``__post_init__`` — teleop runs real physics at 240 Hz,
replay is purely kinematic and floors every solver knob.

Import only after AppLauncher.
"""

from __future__ import annotations

from duo_robot import DuoIKActionsCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import joint_pos_rel, reset_scene_to_default, time_out
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass


@configclass
class DuoSceneCfg(InteractiveSceneCfg):
    """The rig plus a dome light; the USDA scene and its objects are added at runtime."""

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.95, 0.95, 0.92)),
    )
    robot = None  # placed from CLI args by the scripts


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=joint_pos_rel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    # reset_joint_targets: an articulated scene object whose USD authors a stiff
    # drive would otherwise spring back to its pre-reset target right after the
    # joints were written to their authored positions. For the robot it just
    # re-arms the drives at the default pose the IK action overwrites next step.
    reset = EventTermCfg(func=reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=time_out, time_out=True)


@configclass
class RewardsCfg:
    pass


@configclass
class DuoEnvCfg(ManagerBasedRLEnvCfg):
    """Base env config; subclasses own the physics/timing in ``__post_init__``."""

    scene: DuoSceneCfg = DuoSceneCfg(num_envs=1, env_spacing=3.0)
    actions: DuoIKActionsCfg = DuoIKActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()

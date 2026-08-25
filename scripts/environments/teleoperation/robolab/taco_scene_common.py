# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TACO teleop scene/env configs shared by the teleop and replay scripts.

Import only after AppLauncher. The configs carry no CLI state: scripts mutate the
instance after construction (``cfg.sim.device``, ``cfg.sim.render_interval``,
``cfg.scene.robot.spawn.articulation_props.enabled_self_collisions``, ...).

Composes three sources (see teleop_taco_scene.py's module docstring for the full
story): sim_benchmark's TACO scene + franka_duo placement, and the RoboLab fork's
58-D ``FrankaDuoSharpaIKActionCfg`` action space.
"""

from __future__ import annotations

import os
import sys

# sim_benchmark is not pip-installed; it lives inside the IsaacLab checkout.
_SIM_BENCHMARK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "sim_benchmark"))
if _SIM_BENCHMARK_ROOT not in sys.path:
    sys.path.insert(0, _SIM_BENCHMARK_ROOT)

from sim_benchmark.franka_duo import FRANKA_DUO_USD  # noqa: E402
from sim_benchmark.taco_hoi import TacoSceneCfg, robot_spawn_props  # noqa: E402

from robolab.robots.franka_duo_sharpa_wave import FrankaDuoSharpaIKActionCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.envs.mdp import joint_pos_rel, reset_scene_to_default, time_out  # noqa: E402
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, TerminationTermCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from robolab_teleop_common import OncePerStepDiffIKAction  # noqa: E402

# Arm ready pose shared by both repos (IK-solved: fingers forward, palms down).
_ARM_INIT = {
    "left_panda_joint1": 1.145, "left_panda_joint2": 1.048, "left_panda_joint3": -0.464,
    "left_panda_joint4": -1.516, "left_panda_joint5": -2.540, "left_panda_joint6": 2.045,
    "left_panda_joint7": 0.108,
    "right_panda_joint1": -1.144, "right_panda_joint2": 1.047, "right_panda_joint3": 0.462,
    "right_panda_joint4": -1.517, "right_panda_joint5": 2.541, "right_panda_joint6": 2.044,
    "right_panda_joint7": -0.107,
}


@configclass
class TacoTeleopSceneCfg(TacoSceneCfg):
    """sim_benchmark's TACO scene (table + brush + bowl + lights) plus the duo rig.

    Robot placement mirrors sim_benchmark's franka_duo: torso at (0, -0.7, 1.0),
    +90° yaw — standing south of the table (top at z = 0.5421), facing +y.
    """

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(FRANKA_DUO_USD), **robot_spawn_props()),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, -0.70, 1.0),
            rot=(0.7071068, 0.0, 0.0, 0.7071068),
            joint_pos={**_ARM_INIT, "(left|right)_(thumb|index|middle|ring|pinky)_.*": 0.0},
        ),
        soft_joint_pos_limit_factor=1.05,
        actuators={
            "shoulders": ImplicitActuatorCfg(
                joint_names_expr=["(left|right)_panda_joint[1-4]"],
                effort_limit=87.0, velocity_limit=2.175, stiffness=400.0, damping=80.0,
            ),
            "forearms": ImplicitActuatorCfg(
                joint_names_expr=["(left|right)_panda_joint[5-7]"],
                effort_limit=12.0, velocity_limit=2.61, stiffness=400.0, damping=80.0,
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=["(left|right)_(thumb|index|middle|ring|pinky)_.*"],
                stiffness=400.0, damping=4.0,  # keep Sharpa's USD-calibrated gains
            ),
        },
    )


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
    reset = EventTermCfg(func=reset_scene_to_default, mode="reset")


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=time_out, time_out=True)


@configclass
class RewardsCfg:
    pass


@configclass
class TacoTeleopEnvCfg(ManagerBasedRLEnvCfg):
    scene: TacoTeleopSceneCfg = TacoTeleopSceneCfg(num_envs=1, env_spacing=3.0)
    actions: FrankaDuoSharpaIKActionCfg = FrankaDuoSharpaIKActionCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        self.episode_length_s = 300
        self.decimation = 4
        self.sim.dt = 1 / 240           # teleop timing, not sim_benchmark's 20 Hz clip timing
        self.sim.render_interval = 8   # 8 = 30 Hz at dt=1/240; scripts override from CLI
        self.sim.physx.solver_type = 1  # TGS (sim_benchmark's solver_type=2 is invalid on this stack)
        self.sim.physx.max_position_iteration_count = 32
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.enable_ccd = True                   # CCD
        # solve arm IK once per control step instead of once per physics substep
        # (decimation× cheaper; the stock per-substep resolve cost ~65 ms/step on this
        # scene at decimation 8 — see profiler).
        self.actions.left_arm.class_type = OncePerStepDiffIKAction
        self.actions.right_arm.class_type = OncePerStepDiffIKAction

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The FR3 Duo + SharpaWave rig: articulation config and 58-D IK action space.

One PhysX articulation: a fixed torso, two 7-DoF Panda arms, and two 22-DoF
SharpaWave dexterous hands mounted on the arm flanges (``*_panda_link8``).
The robot USD is vendored under ``assets/robots/`` (originally authored in the
``sim_benchmark`` project); joint names, actuator gains, and the arm ready pose
are carried over unchanged from the validated teleop setup on the
``feature/robolab-xr-teleop`` branch.

The action layout matches the teleop pipeline's output exactly (term
declaration order defines the layout):

    [ left wrist pose 7 | right wrist pose 7 | left fingers 22 | right fingers 22 ]

Wrist poses are absolute ``[pos(3), quat xyzw(4)]`` targets for the flanges,
expressed in the ROBOT ROOT frame — the differential-IK action terms do no
frame conversion themselves, so the teleop script converts world-frame wrist
targets to the root frame before ``env.step`` (see ``make_teleop_scene.py``).

Import only after AppLauncher.
"""

from __future__ import annotations

import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg, JointPositionActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils.configclass import configclass

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

DUO_ROBOT_USD = os.path.join(_ASSETS_DIR, "robots", "franka_duo_sharpa_wave.usda")
"""Vendored robot USD: FR3 Duo torso + 2 Panda arms + 2 SharpaWave hands (58 actuated joints)."""

DEX_RETARGETING_DIR = os.path.join(_ASSETS_DIR, "dex_retargeting")
"""Vendored SharpaWave URDFs + DexPilot configs consumed by the finger retargeters."""

# -- Joint naming (must match the robot USD) -----------------------------------

#: 22 SharpaWave finger joints per hand, thumb through pinky. The list ORDER is
#: load-bearing: the hand action terms use ``preserve_order=True`` and the teleop
#: pipeline emits finger joints in exactly this order.
FINGER_JOINTS = [
    "thumb_CMC_FE",
    "thumb_CMC_AA",
    "thumb_MCP_FE",
    "thumb_MCP_AA",
    "thumb_IP",
    "index_MCP_FE",
    "index_MCP_AA",
    "index_PIP",
    "index_DIP",
    "middle_MCP_FE",
    "middle_MCP_AA",
    "middle_PIP",
    "middle_DIP",
    "ring_MCP_FE",
    "ring_MCP_AA",
    "ring_PIP",
    "ring_DIP",
    "pinky_CMC",
    "pinky_MCP_FE",
    "pinky_MCP_AA",
    "pinky_PIP",
    "pinky_DIP",
]

#: 7 Panda arm joints per side, shoulder through wrist.
ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]


def sided(names: list[str], side: str) -> list[str]:
    """Prefix each joint name with ``left_`` or ``right_``."""
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    return [f"{side}_{name}" for name in names]


# Ready pose: the standard Franka home configuration, applied identically to
# both arms (the mirrored arm mounts make the resulting postures mirror each
# other in the workspace).
ARM_READY_POSE = {
    "left_panda_joint1": 0.0,
    "left_panda_joint2": -0.785,
    "left_panda_joint3": 0.0,
    "left_panda_joint4": -2.356,
    "left_panda_joint5": 0.0,
    "left_panda_joint6": 1.571,
    "left_panda_joint7": 0.785,
    "right_panda_joint1": 0.0,
    "right_panda_joint2": -0.785,
    "right_panda_joint3": 0.0,
    "right_panda_joint4": -2.356,
    "right_panda_joint5": 0.0,
    "right_panda_joint6": 1.571,
    "right_panda_joint7": 0.785,
}


def duo_robot_cfg(pos: tuple[float, float, float], rot: tuple[float, float, float, float]) -> ArticulationCfg:
    """Build the duo rig's articulation config at the given root pose.

    Args:
        pos: Root position of the torso in the environment frame [m].
        rot: Root orientation quaternion (x, y, z, w).

    Returns:
        The articulation config, ready to drop into an :class:`~isaaclab.scene.InteractiveSceneCfg`.
    """
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=DUO_ROBOT_USD,
            activate_contact_sensors=True,
            # The rig is position-driven rather than gravity-compensated.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=pos,
            rot=rot,
            joint_pos={**ARM_READY_POSE, "(left|right)_(thumb|index|middle|ring|pinky)_.*": 0.0},
        ),
        soft_joint_pos_limit_factor=1.05,
        actuators={
            "shoulders": ImplicitActuatorCfg(
                joint_names_expr=["(left|right)_panda_joint[1-4]"],
                effort_limit_sim=87.0,
                stiffness=400.0,
                damping=80.0,
            ),
            "forearms": ImplicitActuatorCfg(
                joint_names_expr=["(left|right)_panda_joint[5-7]"],
                effort_limit_sim=12.0,
                stiffness=400.0,
                damping=80.0,
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=["(left|right)_(thumb|index|middle|ring|pinky)_.*"],
                stiffness=400.0,
                damping=4.0,  # keep Sharpa's USD-calibrated gains
            ),
        },
    )


# -- Actions --------------------------------------------------------------------


class OncePerStepDiffIKAction(DifferentialInverseKinematicsAction):
    """Differential IK that solves once per *control* step instead of once per physics substep.

    The stock action recomputes the end-effector pose, re-reads the Jacobian from
    PhysX (a GPU readback) and re-solves DLS inside ``apply_actions`` — which the
    env calls once per decimation substep. Teleop delivers a new target at most
    once per control step, so re-linearizing at the substep rate buys nothing:
    this variant solves in ``process_actions`` (once per control step, against
    the freshest state) and has ``apply_actions`` only re-issue the cached
    joint-position target to the PD drives. Measured ~65 ms/step savings on the
    two-arm rig at decimation 8 on the original branch.
    """

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos.torch[:, self._joint_ids]
        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            self._joint_pos_des = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
        else:
            self._joint_pos_des = joint_pos.clone()

    def apply_actions(self):
        self._asset.set_joint_position_target_index(target=self._joint_pos_des, joint_ids=self._joint_ids)


def _arm_ik_action(side: str) -> DifferentialInverseKinematicsActionCfg:
    """Absolute wrist-pose differential IK (damped least squares) for one arm."""
    return DifferentialInverseKinematicsActionCfg(
        class_type=OncePerStepDiffIKAction,
        asset_name="robot",
        joint_names=sided(ARM_JOINTS, side),
        body_name=f"{side}_panda_link8",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
    )


def _hand_action(side: str) -> JointPositionActionCfg:
    """Direct joint-position targets for one hand's 22 finger joints."""
    return JointPositionActionCfg(
        asset_name="robot",
        joint_names=sided(FINGER_JOINTS, side),
        preserve_order=True,
        use_default_offset=False,
    )


@configclass
class DuoIKActionsCfg:
    """58-D action space; attribute order defines the action layout."""

    left_arm: DifferentialInverseKinematicsActionCfg = _arm_ik_action("left")
    right_arm: DifferentialInverseKinematicsActionCfg = _arm_ik_action("right")
    left_hand: JointPositionActionCfg = _hand_action("left")
    right_hand: JointPositionActionCfg = _hand_action("right")

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The teleop robot embodiments: articulation configs and the 58-D IK action space.

Two selectable embodiments (``--embodiment`` on the teleop/replay scripts),
each one PhysX articulation carrying two 22-DoF SharpaWave dexterous hands:

- ``franka_duo`` (default): a fixed torso with two 7-DoF Panda arms, hands
  mounted on the arm flanges (``*_panda_link8``). The robot USD is vendored
  under ``assets/robots/`` (originally authored in the ``sim_benchmark``
  project); joint names, actuator gains, and the arm ready pose are carried
  over unchanged from the validated teleop setup on the
  ``feature/robolab-xr-teleop`` branch.
- ``yam_duo``: two 6-DoF I2RT YAM Ultra (v2) arms on a table-edge mounting
  rail, bases 0.565 m apart. Built by
  ``assets/robots/yam_ultra/make_yam_duo_assets.py`` from the vendored I2RT
  URDF; place it with the rail ON the tabletop at the near edge.

The action layout matches the teleop pipeline's output exactly (term
declaration order defines the layout), identical for both embodiments:

    [ left wrist pose 7 | right wrist pose 7 | left fingers 22 | right fingers 22 ]

Wrist poses are absolute ``[pos(3), quat xyzw(4)]`` targets for the arms' IK
bodies (the Panda flanges, or the SharpaWave wrist bodies on the YAM rig),
expressed in the ROBOT ROOT frame — the differential-IK action terms do no
frame conversion themselves, so the teleop script converts world-frame wrist
targets to the root frame before ``env.step`` (see ``make_teleop_scene.py``).

Import only after AppLauncher.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

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


# Ready pose: square 6-DoF IK solution (joint 2 fixed at 0.6 on both arms --
# the requested 0.3 is infeasible for these targets, feasibility starts at
# ~0.40) for both hands palm-down, yawed 30 deg inward, at
# (+-0.30, -0.60, 1.00) in the scene frame, on the re-clocked mounts
# (left Rz(-45) "flipped", right Rz(-135)). Best of two IK branches per arm;
# free-joint min-margin 0.55 rad.
ARM_READY_POSE = {
    "left_panda_joint1": 0.6024,
    "left_panda_joint2": 0.6,
    "left_panda_joint3": 0.8563,
    "left_panda_joint4": -2.5242,
    "left_panda_joint5": -1.9285,
    "left_panda_joint6": 1.3715,
    "left_panda_joint7": 1.3574,
    "right_panda_joint1": -0.6024,
    "right_panda_joint2": 0.6,
    "right_panda_joint3": -0.8563,
    "right_panda_joint4": -2.5242,
    "right_panda_joint5": 1.9285,
    "right_panda_joint6": 1.3715,
    "right_panda_joint7": 1.7842,
}


def duo_robot_cfg(
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
    arm_stiffness: float = 400.0,
    arm_damping: float = 80.0,
    hand_stiffness: float = 400.0,
    hand_damping: float = 4.0,
) -> ArticulationCfg:
    """Build the duo rig's articulation config at the given root pose.

    Args:
        pos: Root position of the torso in the environment frame [m].
        rot: Root orientation quaternion (x, y, z, w).
        arm_stiffness: Joint drive stiffness kp [N·m/rad] for all 14 Panda arm joints.
        arm_damping: Joint drive damping kd [N·m·s/rad] for all 14 Panda arm joints.
        hand_stiffness: Joint drive stiffness kp [N·m/rad] for all 44 SharpaWave finger joints.
        hand_damping: Joint drive damping kd [N·m·s/rad] for all 44 SharpaWave finger joints
            (the 4.0 default keeps Sharpa's USD-calibrated gain).

    Returns:
        The articulation config, ready to drop into an :class:`~isaaclab.scene.InteractiveSceneCfg`.
    """
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=DUO_ROBOT_USD,
            activate_contact_sensors=True,
            # Ideal gravity compensation: gravity is disabled on the ROBOT's
            # links only (scene objects keep gravity), so the position drives
            # never fight the rig's own weight. Applies to every embodiment.
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
                stiffness=arm_stiffness,
                damping=arm_damping,
            ),
            "forearms": ImplicitActuatorCfg(
                joint_names_expr=["(left|right)_panda_joint[5-7]"],
                effort_limit_sim=12.0,
                stiffness=arm_stiffness,
                damping=arm_damping,
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=["(left|right)_(thumb|index|middle|ring|pinky)_.*"],
                stiffness=hand_stiffness,
                damping=hand_damping,
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


def _arm_ik_action(
    side: str, arm_joints: list[str] | None = None, body_fmt: str = "{side}_panda_link8"
) -> DifferentialInverseKinematicsActionCfg:
    """Absolute pose differential IK (damped least squares) for one arm."""
    return DifferentialInverseKinematicsActionCfg(
        class_type=OncePerStepDiffIKAction,
        asset_name="robot",
        joint_names=sided(arm_joints if arm_joints is not None else ARM_JOINTS, side),
        body_name=body_fmt.format(side=side),
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


# ==============================================================================
# YAM Ultra duo (two I2RT YAM Ultra v2 arms on a table-edge rail)
# ==============================================================================

YAM_DUO_ROBOT_USD = os.path.join(_ASSETS_DIR, "robots", "yam_duo_sharpa_wave.usda")
"""Generated robot USD: mounting rail + 2 YAM Ultra arms + 2 SharpaWave hands (56 actuated joints)."""

#: 6 YAM Ultra arm joints per side, base through wrist roll.
YAM_ARM_JOINTS = [f"yam_joint{i}" for i in range(1, 7)]

# Ready pose: maximin-margin 6-DoF IK solution for both hands palm-down,
# yawed 30 deg inward, at (-+0.26, -0.30, 1.15) in the scene frame with the
# rail at its default table-edge placement (0, -0.55, 1.0) facing +y
# (see assets/robots/yam_ultra/make_yam_duo_assets.py for the rig geometry).
# Tightest limit margin is joint 4 at 0.168 rad; all other joints >= 0.5 rad.
# Joint 6 differs by 180 deg between the sides because the LEFT hand is
# re-clocked 180 deg on the flange (see make_yam_duo_assets.py): same wrist
# pose, but the left wrist-roll window favors the operator's inward direction.
YAM_READY_POSE: dict[str, float] = {
    "left_yam_joint1": -0.0118,
    "left_yam_joint2": 1.6584,
    "left_yam_joint3": 0.2558,
    "left_yam_joint4": 1.4026,
    "left_yam_joint5": 0.5117,
    "left_yam_joint6": 1.5708,
    "right_yam_joint1": 0.0024,
    "right_yam_joint2": 1.6584,
    "right_yam_joint3": 0.2558,
    "right_yam_joint4": 1.4026,
    "right_yam_joint5": -0.5211,
    "right_yam_joint6": -1.5708,
}


def yam_duo_robot_cfg(
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
    arm_stiffness: float = 400.0,
    arm_damping: float = 80.0,
    hand_stiffness: float = 400.0,
    hand_damping: float = 4.0,
) -> ArticulationCfg:
    """Build the YAM duo rig's articulation config at the given root pose.

    The root sits between the two arm base plates at mounting-surface height:
    place it ON the tabletop at the table's near edge (the default
    ``robot_pos`` puts it on the raised TACO table, ``(0, -0.55, 1.0)``).

    Args:
        pos: Root position (between the base plates) in the environment frame [m].
        rot: Root orientation quaternion (x, y, z, w).
        arm_stiffness: Joint drive stiffness kp [N·m/rad] for all 12 YAM arm joints.
        arm_damping: Joint drive damping kd [N·m·s/rad] for all 12 YAM arm joints.
        hand_stiffness: Joint drive stiffness kp [N·m/rad] for all 44 SharpaWave finger joints.
        hand_damping: Joint drive damping kd [N·m·s/rad] for all 44 SharpaWave finger joints.

    Returns:
        The articulation config, ready to drop into an :class:`~isaaclab.scene.InteractiveSceneCfg`.
    """
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=YAM_DUO_ROBOT_USD,
            activate_contact_sensors=True,
            # Ideal gravity compensation, robot links only (see duo_robot_cfg).
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
            joint_pos={**YAM_READY_POSE, "(left|right)_(thumb|index|middle|ring|pinky)_.*": 0.0},
        ),
        soft_joint_pos_limit_factor=1.05,
        actuators={
            # DM4340 joints (1-4): 27 N·m peak; DM4310 wrist joints (5-6): 10 N·m.
            # Hardware position gains are far softer (kp 80/40/10, kd 5/3/1.5 —
            # see assets/robots/yam_ultra/yam_ultra_v2_gains.yml); teleop runs the
            # same stiff PD defaults as the Franka rig for responsive tracking,
            # with torque clamped at the real motor limits.
            "shoulders": ImplicitActuatorCfg(
                joint_names_expr=["(left|right)_yam_joint[1-4]"],
                effort_limit_sim=27.0,
                stiffness=arm_stiffness,
                damping=arm_damping,
            ),
            "wrists": ImplicitActuatorCfg(
                joint_names_expr=["(left|right)_yam_joint[5-6]"],
                effort_limit_sim=10.0,
                stiffness=arm_stiffness,
                damping=arm_damping,
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=["(left|right)_(thumb|index|middle|ring|pinky)_.*"],
                stiffness=hand_stiffness,
                damping=hand_damping,
            ),
        },
    )


@configclass
class YamDuoIKActionsCfg:
    """58-D action space for the YAM duo; same layout as :class:`DuoIKActionsCfg`.

    The arms' IK targets the SharpaWave WRIST bodies directly (the YAM flange
    plus two fixed joints), so the wrist retargeting offsets are a property of
    the hand alone, independent of how it is clocked onto the arm.
    """

    left_arm: DifferentialInverseKinematicsActionCfg = _arm_ik_action("left", YAM_ARM_JOINTS, "{side}_hand_wrist")
    right_arm: DifferentialInverseKinematicsActionCfg = _arm_ik_action("right", YAM_ARM_JOINTS, "{side}_hand_wrist")
    left_hand: JointPositionActionCfg = _hand_action("left")
    right_hand: JointPositionActionCfg = _hand_action("right")


# ==============================================================================
# Embodiment registry
# ==============================================================================


@dataclass(frozen=True)
class Embodiment:
    """Everything the teleop/replay scripts need to know about one robot rig."""

    name: str
    """Registry key (the ``--embodiment`` CLI value)."""

    robot_cfg: Callable[..., ArticulationCfg]
    """``(pos, rot, arm_stiffness, arm_damping, hand_stiffness, hand_damping) -> ArticulationCfg``."""

    actions_cfg: Callable[[], object]
    """Factory for the 58-D action config (``DuoIKActionsCfg``-shaped)."""

    arm_joint_regex: str
    """Regex matching all arm joints (domain randomization, gain groups)."""

    ik_body_fmt: str
    """Per-side body the wrist-pose IK terms command (``{side}`` placeholder)."""

    wrist_offsets_xyzw: dict[str, tuple[float, float, float, float]]
    """Per-side rotation offsets mapping the OpenXR wrist frame onto the IK body frame."""

    default_robot_pos: tuple[float, float, float]
    """Default rig root position in the scene frame [m] (raised TACO table placement)."""

    default_robot_rot: tuple[float, float, float, float]
    """Default rig root orientation (x, y, z, w); +90 deg yaw faces the rig +y."""

    def ik_body(self, side: str) -> str:
        """Name of the IK-commanded body for ``side`` ('left' or 'right')."""
        return self.ik_body_fmt.format(side=side)


# Wrist retargeting offsets, as xyzw quaternions composed wrist-side
# (q_target = q_wrist ⊗ q_corr ⊗ q_offset, see duo_teleop_pipeline):
#
# - franka_duo commands the Panda flanges: the offsets fold the calibrated
#   OpenXR-wrist→flange rotation together with the Rz(-45°)/Rz(-135°) hand
#   mount clocking (see duo_teleop_pipeline's module docstring).
# - yam_duo commands the SharpaWave wrist bodies directly, so its offsets are
#   the pure OpenXR-wrist→SharpaWave-wrist rotation: offset_flange ⊗ Rz(mount)
#   evaluated on the franka rig, which collapses to the SAME clean rotation on
#   both sides — 180° about the wrist-frame axis (-1, 1, 0)/√2 (the two xyzw
#   quats below differ only by sign, i.e. they are one rotation).
_FRANKA_WRIST_OFFSETS = {
    "left": (-0.3826834, 0.9238795, 0.0, 0.0),
    "right": (-0.3826834, -0.9238795, 0.0, 0.0),
}
_SHARPA_WRIST_OFFSETS = {
    "left": (-0.7071068, 0.7071068, 0.0, 0.0),
    "right": (0.7071068, -0.7071068, 0.0, 0.0),
}

EMBODIMENTS: dict[str, Embodiment] = {
    "franka_duo": Embodiment(
        name="franka_duo",
        robot_cfg=duo_robot_cfg,
        actions_cfg=DuoIKActionsCfg,
        arm_joint_regex="(left|right)_panda_joint[1-7]",
        ik_body_fmt="{side}_panda_link8",
        wrist_offsets_xyzw=_FRANKA_WRIST_OFFSETS,
        default_robot_pos=(0.0, -0.8, 1.3),
        default_robot_rot=(0.0, 0.0, 0.7071068, 0.7071068),
    ),
    "yam_duo": Embodiment(
        name="yam_duo",
        robot_cfg=yam_duo_robot_cfg,
        actions_cfg=YamDuoIKActionsCfg,
        arm_joint_regex="(left|right)_yam_joint[1-6]",
        ik_body_fmt="{side}_hand_wrist",
        wrist_offsets_xyzw=_SHARPA_WRIST_OFFSETS,
        # Rail between the base plates, ON the raised TACO tabletop (top z=1.0)
        # at its near edge, facing the table (+y).
        default_robot_pos=(0.0, -0.55, 1.0),
        default_robot_rot=(0.0, 0.0, 0.7071068, 0.7071068),
    ),
}

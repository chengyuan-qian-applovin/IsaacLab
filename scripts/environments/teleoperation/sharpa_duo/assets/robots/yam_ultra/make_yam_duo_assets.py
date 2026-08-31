# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the YAM Ultra duo + SharpaWave robot USD from the vendored sources.

Inputs (this directory):

- ``yam_ultra.urdf`` + ``assets/*.stl`` — the I2RT YAM Ultra v2 arm, vendored
  verbatim from https://github.com/i2rt-robotics/i2rt
  (``i2rt/robot_models/arm/yam_ultra/v2``, MIT license). See ``README.md`` for
  its full kinematic documentation and ``yam_ultra_v2_gains.yml`` for the
  hardware motor/gain config the sim actuator values are derived from.
- ``../sharpa_wave/{left,right}_sharpa_wave_with_flange/`` — the SharpaWave
  hand USDs already used by the Franka duo rig.

Outputs:

1. ``{left,right}_yam.urdf`` — per-side arm URDFs: links/joints prefixed with
   the side, the stock parallel gripper (``gripper`` visual + ``tip_*`` fingers)
   removed so the ``gripper`` mount frame becomes a bare ``*_yam_flange`` stub
   the hand bolts onto, and real actuator limits (URDF placeholders are 1.0).
2. ``{left,right}_yam/{left,right}_yam.usd`` — the converted arm USDs
   (collisions from visuals, convex decomposition).
3. ``../yam_duo_sharpa_wave.usda`` — the duo rig: a table-edge mounting rail,
   both arms 0.565 m apart, and the SharpaWave hands fixed to the arm flanges.

Run from the repo root (needs Isaac Sim for the URDF conversion):

    ./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/assets/robots/yam_ultra/make_yam_duo_assets.py
"""

# isort: skip_file
import argparse
import os
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Build the YAM duo + SharpaWave robot USD.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
from scipy.spatial.transform import Rotation as R

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOTS_DIR = os.path.dirname(THIS_DIR)

# -- Rig geometry ---------------------------------------------------------------
#
# At identity the rig faces +x with the arm bases 0.565 m apart along y (the
# same convention as the Franka duo torso, whose left arm is on +y). The rig
# root sits between the two base plates, ON the mounting surface (table top):
# place it with e.g. robot_pos=(0, -0.55, 1.0) on the raised TACO table.
BASE_SEPARATION = 0.565
BASE_Y = {"left": +BASE_SEPARATION / 2, "right": -BASE_SEPARATION / 2}

# Hand mount: the SharpaWave flange bolts onto the YAM ``gripper`` mount frame
# (the child frame of joint6). The stock YAM gripper extends along the mount
# frame's -z, so the hand flange (whose +z runs toward the fingers) mounts as
# Rx(180°)·Rz(clock). The clock angle re-centers joint6's ±120° range for
# palm-down tabletop teleop; 90° won the reachability study (see the mount
# study in the PR/changelog notes).
HAND_MOUNT_CLOCK_DEG = {"left": 90.0, "right": 90.0}

# Per-arm base yaw about z, relative to the rig's +x facing.
BASE_YAW_DEG = {"left": 0.0, "right": 0.0}

# DM4340 (joints 1-4) / DM4310 (joints 5-6) peak torque [N·m] and a
# conservative velocity limit [rad/s] (URDF placeholders are effort=1, vel=1).
JOINT_LIMITS = {  # joint index -> (effort [N·m], velocity [rad/s])
    1: (27.0, 10.0),
    2: (27.0, 10.0),
    3: (27.0, 10.0),
    4: (27.0, 10.0),
    5: (10.0, 20.0),
    6: (10.0, 20.0),
}

# Mount stub inertial for the flange frame (the stock gripper's mass is
# removed with its visual; the hand's own links carry their real inertia).
FLANGE_MASS = 0.05
FLANGE_INERTIA = 2.0e-5


def build_side_urdf(side: str) -> str:
    """Write ``{side}_yam.urdf`` next to the vendored sources; returns its path."""
    tree = ET.parse(os.path.join(THIS_DIR, "yam_ultra.urdf"))
    root = tree.getroot()
    root.set("name", f"{side}_yam")

    def new_name(old: str) -> str | None:
        if old in ("tip_left", "tip_right"):
            return None
        if old == "gripper":
            return f"{side}_yam_flange"
        if old == "base":
            return f"{side}_yam_base"
        return f"{side}_yam_{old}"  # link1..link5, joint1..joint6

    for link in list(root.findall("link")):
        name = new_name(link.get("name"))
        if name is None:
            root.remove(link)
            continue
        link.set("name", name)
        if name.endswith("_flange"):
            # Bare mount frame: drop the stock gripper visual, keep a stub inertial.
            for visual in link.findall("visual"):
                link.remove(visual)
            inertial = link.find("inertial")
            inertial.find("mass").set("value", str(FLANGE_MASS))
            inertial.find("origin").set("xyz", "0 0 0")
            inertia = inertial.find("inertia")
            for key in ("ixx", "iyy", "izz"):
                inertia.set(key, str(FLANGE_INERTIA))
            for key in ("ixy", "ixz", "iyz"):
                inertia.set(key, "0")

    for joint in list(root.findall("joint")):
        old = joint.get("name")
        if old in ("joint7", "joint8"):  # stock gripper fingers
            root.remove(joint)
            continue
        index = int(old.removeprefix("joint"))
        joint.set("name", f"{side}_yam_{old}")
        joint.find("parent").set("link", new_name(joint.find("parent").get("link")))
        joint.find("child").set("link", new_name(joint.find("child").get("link")))
        effort, velocity = JOINT_LIMITS[index]
        limit = joint.find("limit")
        limit.set("effort", str(effort))
        limit.set("velocity", str(velocity))

    out_path = os.path.join(THIS_DIR, f"{side}_yam.urdf")
    ET.indent(tree, space="    ")
    tree.write(out_path, xml_declaration=True, encoding="utf-8")
    print(f"[URDF] wrote {out_path}")
    return out_path


def convert_side(side: str, urdf_path: str) -> str:
    """URDF -> USD; returns the USD path (relative references stay local)."""
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=os.path.join(THIS_DIR, f"{side}_yam"),
        usd_file_name=f"{side}_yam.usd",
        force_usd_conversion=True,
        fix_base=True,
        merge_fixed_joints=False,
        collision_from_visuals=True,
        collision_type="Convex Decomposition",
        self_collision=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            # USD-default gains only; the teleop ArticulationCfg overrides them.
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=400.0, damping=80.0),
        ),
    )
    usd_path = UrdfConverter(cfg).usd_path
    print(f"[USD] {side}: {usd_path}")
    return usd_path


# -- FK at zero joints (for authoring the hand spawn poses in the USDA) ----------


def _transform(xyz, rpy=None, rot: R | None = None) -> np.ndarray:
    out = np.eye(4)
    out[:3, 3] = xyz
    if rpy is not None:
        rot = R.from_euler("xyz", rpy)
    if rot is not None:
        out[:3, :3] = rot.as_matrix()
    return out


def yam_flange_at_zero() -> np.ndarray:
    """Mount-frame (``gripper``) pose at zero joints, in the arm base frame."""
    tree = ET.parse(os.path.join(THIS_DIR, "yam_ultra.urdf"))
    frames = {}
    for joint in tree.getroot().findall("joint"):
        origin = joint.find("origin")
        frames[joint.get("name")] = _transform(
            [float(v) for v in origin.get("xyz").split()],
            [float(v) for v in origin.get("rpy").split()],
        )
    X = np.eye(4)
    for i in range(1, 7):
        X = X @ frames[f"joint{i}"]
    return X


def quat_usd(rot: R) -> str:
    """Format a rotation as a USD quat literal (w, x, y, z)."""
    x, y, z, w = rot.as_quat()
    return f"({w:.10g}, {x:.10g}, {y:.10g}, {z:.10g})"


def vec_usd(vec) -> str:
    return f"({vec[0]:.10g}, {vec[1]:.10g}, {vec[2]:.10g})"


def write_duo_usda() -> str:
    flange_zero = yam_flange_at_zero()
    out_path = os.path.join(ROBOTS_DIR, "yam_duo_sharpa_wave.usda")

    blocks = [
        """#usda 1.0
(
    defaultPrim = "robot"
)

# YAM Ultra duo + SharpaWave rig, generated by yam_ultra/make_yam_duo_assets.py.
# Two I2RT YAM Ultra v2 arms on a table-edge mounting rail, 0.565 m apart,
# each carrying a SharpaWave hand on its flange. At identity the rig faces +x
# with the left arm on +y; the root sits between the base plates at mounting-
# surface height.

def Xform "robot" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"]
)
{
    def Xform "rail" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 4

        # Slim mounting rail between the two base plates; floats 2 mm above
        # the mounting surface so it never fights the tabletop contact.
        def Cube "geom" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double size = 1
            float3 xformOp:scale = (0.06, 0.45, 0.016)
            double3 xformOp:translate = (0, 0, 0.01)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }

        def PhysicsFixedJoint "root_joint"
        {
            rel physics:body1 = </robot/rail>
        }
"""
    ]

    def base_body(side: str) -> str:
        return f"/robot/{side}_arm/Geometry/{side}_yam_base"

    def flange_body(side: str) -> str:
        links = "/".join(f"{side}_yam_link{i}" for i in range(1, 6))
        return f"{base_body(side)}/{links}/{side}_yam_flange"

    for side in ("left", "right"):
        base_pos = np.array([0.0, BASE_Y[side], 0.0])
        base_rot = R.from_euler("z", BASE_YAW_DEG[side], degrees=True)

        blocks.append(f"""
        def PhysicsFixedJoint "{side}_arm_mount"
        {{
            rel physics:body0 = </robot/rail>
            rel physics:body1 = <{base_body(side)}>
            point3f physics:localPos0 = {vec_usd(base_pos)}
            point3f physics:localPos1 = (0, 0, 0)
            quatf physics:localRot0 = {quat_usd(base_rot)}
            quatf physics:localRot1 = (1, 0, 0, 0)
        }}
""")

    blocks.append("    }\n")

    for side in ("left", "right"):
        base_pos = np.array([0.0, BASE_Y[side], 0.0])
        base_rot = R.from_euler("z", BASE_YAW_DEG[side], degrees=True)
        mount_rot = R.from_euler("XZ", [180.0, HAND_MOUNT_CLOCK_DEG[side]], degrees=True)
        hand_x = _transform(base_pos, rot=base_rot) @ flange_zero @ _transform((0, 0, 0), rot=mount_rot)
        hand_pos = hand_x[:3, 3]
        hand_rot = R.from_matrix(hand_x[:3, :3])
        hand_usd = f"./sharpa_wave/{side}_sharpa_wave_with_flange/{side}_sharpa_wave_with_flange.usd"

        blocks.append(f"""
    def Xform "{side}_arm" (
        prepend references = @./yam_ultra/{side}_yam/{side}_yam/{side}_yam.usda@</{side}_yam>
    )
    {{
        quatd xformOp:orient = {quat_usd(base_rot)}
        double3 xformOp:translate = {vec_usd(base_pos)}
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]

        # The converted arm USD carries its own articulation root (on the
        # Geometry scope) and a fixed-base root joint; both give way to the
        # duo articulation root and the rail mount joints.
        over "Geometry" (
            delete apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI", "NewtonArticulationRootAPI"]
        )
        {{
        }}

        over "Physics"
        {{
            over "root_joint" (
                active = false
            )
            {{
            }}
        }}
    }}

    def Xform "{side}_hand" (
        prepend references = @{hand_usd}@</{side}_sharpa_wave>
    )
    {{
        quatd xformOp:orient = {quat_usd(hand_rot)}
        double3 xformOp:translate = {vec_usd(hand_pos)}
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]

        # Same skin the Franka duo rig uses, so the hands render identically.
        over "Looks" (
            prepend references = @./sharpa_wave_skin.usda@</Looks>
        )
        {{
        }}

        # The hand USD's own articulation root becomes the fixed mount joint:
        # flange bolted onto the YAM mount frame (stock-gripper direction is
        # the mount's -z, hence the Rx(180); the Rz clock recenters joint6).
        over "root_joint" (
            delete apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"]
        )
        {{
            rel physics:body0 = <{flange_body(side)}>
            rel physics:body1 = </robot/{side}_hand/{side}_hand_flange>
            point3f physics:localPos0 = (0, 0, 0)
            point3f physics:localPos1 = (0, 0, 0)
            quatf physics:localRot0 = {quat_usd(mount_rot)}
            quatf physics:localRot1 = (1, 0, 0, 0)
        }}
    }}
""")

    blocks.append("}\n")
    with open(out_path, "w") as f:
        f.write("".join(blocks))
    print(f"[USDA] wrote {out_path}")
    return out_path


def main():
    for side in ("left", "right"):
        urdf_path = build_side_urdf(side)
        convert_side(side, urdf_path)
    write_duo_usda()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()

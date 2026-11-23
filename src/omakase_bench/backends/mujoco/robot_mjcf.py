"""A robot preset as an MJCF document, ready for scene elements to be added.

The URDF goes through MuJoCo's own importer with the settings of
manipulation-kit's ``description/mjcf.py`` (Apache-2.0; see NOTICE): visuals
kept, static links not fused, URDF ``<mimic>`` joints as joint equalities,
visual geoms in group 2 and collision geoms in group 3.

On top of that this module adds what a simulation needs: collision bits (the
robot collides with fixtures and objects, never with itself or the floor),
gravity compensation, high-friction finger pads, position actuators for every
driven joint, TCP sites and cameras at the optical frames.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from ...robots.spec import Actuation, RobotSpec

ROBOT_CONTYPE, ROBOT_CONAFFINITY = 2, 0
VISUAL_GROUP, COLLISION_GROUP = 2, 3
#: optical frame (z forward, y down) -> MuJoCo camera (looks along -z, y up)
_OPTICAL_TO_MJ_CAMERA = "0 1 0 0"


def _mimics(root: ET.Element) -> dict[str, tuple[str, float, float]]:
    out = {}
    for joint in root.iter("joint"):
        m = joint.find("mimic")
        if m is not None:
            out[joint.get("name")] = (m.get("joint"), float(m.get("multiplier", "1")),
                                      float(m.get("offset", "0")))
    return out


def urdf_to_mjcf_string(urdf_path: Path) -> tuple[str, dict[str, tuple[str, float, float]]]:
    """MuJoCo's URDF import with the kit's settings; returns the MJCF text and the mimic map."""
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()
    mimics = _mimics(root)
    for mesh in root.iter("mesh"):
        f = mesh.get("filename", "")
        if f and not os.path.isabs(f):
            mesh.set("filename", str((urdf_path.parent / f).resolve()))
    ext = ET.SubElement(root, "mujoco")
    ET.SubElement(ext, "compiler", strippath="false", discardvisual="false", fusestatic="false")
    tmpdir = tempfile.mkdtemp(prefix="omakase-bench-urdf-")
    try:
        tmp = os.path.join(tmpdir, "robot.urdf")
        tree.write(tmp, encoding="unicode", xml_declaration=True)
        spec = mujoco.MjSpec.from_file(tmp)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    spec.modelname = root.get("name") or urdf_path.stem
    existing = {}
    for eq in list(spec.equalities):
        if eq.type == mujoco.mjtEq.mjEQ_JOINT and eq.name1 in mimics:
            if eq.name1 in existing:
                spec.delete(eq)
            else:
                existing[eq.name1] = eq
    for follower, (leader, multiplier, offset) in mimics.items():
        eq = existing.get(follower) or spec.add_equality()
        eq.type = mujoco.mjtEq.mjEQ_JOINT
        eq.name = f"{follower}_mimic"
        eq.name1, eq.name2 = follower, leader
        eq.data[:5] = [offset, multiplier, 0.0, 0.0, 0.0]
    for mesh in spec.meshes:
        # inertia comes from <inertial>; a shell volume avoids rejecting thin parts
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    for geom in spec.geoms:
        if geom.contype == 0 and geom.conaffinity == 0:
            geom.group = VISUAL_GROUP
            geom.density = 0
        else:
            geom.group = COLLISION_GROUP
    with warnings.catch_warnings():
        # the vendored OBJ files mix textured and untextured faces; MuJoCo drops the
        # texture coordinates, which the colour-only materials never use
        warnings.filterwarnings("ignore", message=".*obj_decoder.*")
        spec.compile()
    return spec.to_xml(), mimics


#: default of ``actuation.base_lag_s``
DIFFERENTIAL_BASE_LAG = Actuation.base_lag_s


def _gains(spec: RobotSpec, joint: str) -> tuple[float, float]:
    """(kp, force limit) of a joint's position servo, from the robot's ``actuation`` block."""
    act = spec.actuation
    if spec.mobile_base and joint in (spec.mobile_base.x, spec.mobile_base.y):
        return act.base_linear
    if spec.mobile_base and joint == spec.mobile_base.yaw:
        return act.base_yaw
    if joint == spec.lift_joint:
        return act.lift
    for ee in spec.end_effectors.values():
        if any(f.joint == joint for f in ee.fingers):
            return act.finger_kp, ee.force
    for arm in spec.arms.values():
        if joint in arm.joints:
            return act.arm_proximal if arm.joints.index(joint) < act.arm_proximal_joints else act.arm_distal
    return act.other


def build_robot_mjcf(spec: RobotSpec, visual: str | None = None) -> ET.Element:
    """The robot as a complete ``<mujoco>`` element (compiler, worldbody, equality, actuator).
    ``visual`` picks the preset's visual variant ("mesh", "primitive")."""
    xml, mimics = urdf_to_mjcf_string(spec.visual_urdf(visual) if visual != "kinematics" else spec.urdf_path)
    root = ET.fromstring(xml)
    for tag in ("visual",):
        for el in root.findall(tag):
            root.remove(el)
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("angle", "radian")
    compiler.set("autolimits", "true")
    world = root.find("worldbody")

    pad_bodies = {b for ee in spec.end_effectors.values() for b in ee.pad_links}
    tcp_bodies = {ee.tcp_link: ee for ee in spec.end_effectors.values()}
    camera_bodies = {c.link: c for c in spec.cameras.values()}
    for body in world.iter("body"):
        body.set("gravcomp", "1")
        name = body.get("name")
        for geom in body.findall("geom"):
            if geom.get("group") == str(COLLISION_GROUP):
                geom.set("contype", str(ROBOT_CONTYPE))
                geom.set("conaffinity", str(ROBOT_CONAFFINITY))
                if name in pad_bodies:
                    geom.set("friction", "2.0 0.02 0.002")
                    geom.set("condim", "4")
                    geom.set("priority", "1")
                    geom.set("solref", "0.004 1")
        if name in tcp_bodies:
            ET.SubElement(body, "site", name=f"{tcp_bodies[name].name}_tcp", pos="0 0 0",
                          size="0.006", rgba="1 0.3 0.1 0.6", group="4")
        if name in camera_bodies:
            c = camera_bodies[name]
            ET.SubElement(body, "camera", name=c.name, pos="0 0 0", quat=_OPTICAL_TO_MJ_CAMERA,
                          fovy=f"{c.fovy:.4g}")
    for joint in world.iter("joint"):
        if joint.get("name") not in mimics:
            joint.set("armature", "0.05")
            joint.set("damping", "1.0")
        else:
            joint.set("armature", "0.001")

    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    mb = spec.mobile_base
    for joint in world.iter("joint"):
        name = joint.get("name")
        if name in mimics:
            continue
        kp, force = _gains(spec, name)
        attrs = {"name": f"act_{name}", "joint": name, "kp": f"{kp:g}", "dampratio": "1",
                 "forcerange": f"{-force:g} {force:g}"}
        if mb is not None and mb.differential and name in (mb.x, mb.y, mb.yaw):
            # A servo following a ramp lags it by kv / kp seconds. With the same lag on x, y and
            # yaw the measured pose trails the commanded unicycle path in time instead of
            # slipping sideways; critical damping per joint would give yaw a 10x shorter lag.
            del attrs["dampratio"]
            attrs["kv"] = f"{kp * spec.actuation.base_lag_s:g}"
        rng = joint.get("range")
        if rng and joint.get("limited") != "false":
            attrs["ctrlrange"] = rng
        ET.SubElement(actuator, "position", attrs)
    return root

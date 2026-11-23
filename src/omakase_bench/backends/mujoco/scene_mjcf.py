"""A backend-neutral :class:`SceneSpec` as MJCF around a robot preset.

Collision bits: fixtures and objects have contype 1 and conaffinity 3, so they
touch each other and the robot (contype 2); the floor (conaffinity 1) touches
objects but not the robot, whose base is driven kinematically. In a scene with
several robots each robot also has its own bit (8 << k) and collides with the
others.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from typing import Sequence

import numpy as np

from ...core.task import Articulated, SceneSpec, Shape, rgba
from ...robots.spec import RobotSpec
from .robot_mjcf import ROBOT_CONTYPE, build_robot_mjcf

STATIC = {"contype": "1", "conaffinity": "3"}
OBJECT = {"contype": "1", "conaffinity": "3"}
NO_COLLIDE = {"contype": "0", "conaffinity": "0"}
#: moving parts of articulated fixtures touch only the robot (not walls, floor or objects)
MOVING = {"contype": "4", "conaffinity": "2"}


def f(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in v)


def yaw_quat(yaw: float) -> str:
    return f((math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)))


def lookat_xyaxes(pos, target) -> str:
    fwd = np.array(target, float) - np.array(pos, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return f((*right, *up))


def _shape_geoms(parent: ET.Element, s: Shape, extra: dict[str, str]) -> None:
    color = f(rgba(s.color))
    fr = f"{s.friction:g} 0.01 0.001"
    if s.type == "box":
        ET.SubElement(parent, "geom", name=f"{s.name}_g", type="box", size=f(np.array(s.size) / 2),
                      rgba=color, friction=fr, **extra)
    elif s.type == "cylinder":
        ET.SubElement(parent, "geom", name=f"{s.name}_g", type="cylinder",
                      size=f((s.size[0], s.size[1] / 2)), rgba=color, friction=fr, **extra)
    elif s.type == "sphere":
        ET.SubElement(parent, "geom", name=f"{s.name}_g", type="sphere", size=f(s.size[:1]),
                      rgba=color, friction=fr, **extra)
    elif s.type == "tray":
        x, y, z = s.size
        w = s.wall
        ET.SubElement(parent, "geom", name=f"{s.name}_floor", type="box", size=f((x / 2, y / 2, w / 2)),
                      pos=f((0, 0, -z / 2 + w / 2)), rgba=color, friction=fr, **extra)
        for k, (px, py, sx, sy) in enumerate(((x / 2 - w / 2, 0, w / 2, y / 2), (-x / 2 + w / 2, 0, w / 2, y / 2),
                                              (0, y / 2 - w / 2, x / 2, w / 2), (0, -y / 2 + w / 2, x / 2, w / 2))):
            ET.SubElement(parent, "geom", name=f"{s.name}_wall{k}", type="box", size=f((sx, sy, z / 2)),
                          pos=f((px, py, 0)), rgba=color, friction=fr, **extra)
    else:
        raise ValueError(f"unknown shape type {s.type!r} for {s.name}")


def _staunton(body: ET.Element, asset: ET.Element, s: Shape) -> None:
    """A chess piece: its lathe mesh as the visible geom (no contacts) and its collision primitives
    (hidden group 3, contacts as any object). Mass sits low, as in a weighted piece."""
    from ...objects import staunton
    height, base_d = float(s.size[0]), float(s.size[1])
    mesh = f"staunton_{s.variant}_{round(height * 1e4)}_{round(base_d * 1e4)}"
    if asset.find(f"mesh[@name='{mesh}']") is None:
        verts, faces = staunton.mesh(s.variant, height, base_d)
        ET.SubElement(asset, "mesh", name=mesh, vertex=" ".join(f"{v:.6g}" for v in verts.ravel()),
                      face=" ".join(str(int(i)) for i in faces.ravel()), inertia="shell")
    r = base_d / 2
    ET.SubElement(body, "inertial", pos=f((0, 0, staunton.com_height(s.variant, height))), mass=f"{s.mass:g}",
                  diaginertia=f((s.mass * (3 * r * r + height * height) / 12,) * 2 + (s.mass * r * r / 2,)))
    color = f(rgba(s.color))
    ET.SubElement(body, "geom", name=f"{s.name}_vis", type="mesh", mesh=mesh, rgba=color, group="1", mass="0",
                  **NO_COLLIDE)
    fr = f"{s.friction:g} 0.01 0.001"
    for k, p in enumerate(staunton.collision(s.variant, height, base_d)):
        ET.SubElement(body, "geom", name=f"{s.name}_c{k}", type=p.type, size=f(p.size), pos=f(p.pos),
                      rgba=color, friction=fr, group="3", mass="0", condim="4", solref="0.004 1", **OBJECT)


def _button(world: ET.Element, a: Articulated) -> None:
    """A plate standing ``plate_thickness`` proud of its mount point, a lamp ring and a
    spring-loaded cap that slides into the plate along the mount's -x."""
    p = a.params
    radius = float(p.get("radius", 0.022))
    travel = float(p.get("travel", 0.008))
    protrude = float(p.get("protrusion", 0.012))
    stiffness = float(p.get("stiffness", 300.0))     # N/m: about 1.5 N at the threshold
    plate_t = float(p.get("plate_thickness", 0.006))
    plate = a.size or (plate_t, 0.11, 0.16)
    mount = ET.SubElement(world, "body", name=f"{a.name}_mount", pos=f(a.pos), quat=yaw_quat(a.yaw))
    ET.SubElement(mount, "geom", name=f"{a.name}_plate", type="box", size=f(np.array(plate) / 2),
                  pos=f((plate[0] / 2, 0, 0)), rgba=f(rgba(p.get("plate_color", "steel"))), **STATIC)
    ET.SubElement(mount, "geom", name=f"{a.name}_lamp", type="cylinder", size=f((radius + 0.008, 0.001)),
                  pos=f((plate[0] + 0.0005, 0, 0)), quat="0.7071068 0 0.7071068 0", rgba="0.35 0.35 0.38 1",
                  **NO_COLLIDE)
    cap = ET.SubElement(mount, "body", name=a.name, pos=f((plate[0], 0, 0)))
    ET.SubElement(cap, "joint", name=f"{a.name}_joint", type="slide", axis="-1 0 0",
                  range=f((0, travel)), stiffness=f"{stiffness:g}", damping="2", springref="0",
                  armature="0.01")
    ET.SubElement(cap, "geom", name=f"{a.name}_cap", type="cylinder", size=f((radius, protrude / 2)),
                  pos=f((protrude / 2, 0, 0)), quat="0.7071068 0 0.7071068 0", mass="0.02",
                  rgba=f(rgba(a.color)), friction="1 0.01 0.001", **MOVING)


def _sliding_door(world: ET.Element, actuators: ET.Element, a: Articulated) -> None:
    """Two panels in the local y-z plane that slide apart along local y."""
    thick, width, height = a.size or (0.03, 0.9, 2.0)
    mount = ET.SubElement(world, "body", name=f"{a.name}_mount", pos=f(a.pos), quat=yaw_quat(a.yaw))
    half = width / 2
    for side, sign in (("l", 1.0), ("r", -1.0)):
        panel = ET.SubElement(mount, "body", name=f"{a.name}_{side}", pos=f((0, sign * half / 2, height / 2)))
        ET.SubElement(panel, "joint", name=f"{a.name}_{side}_joint", type="slide", axis=f((0, sign, 0)),
                      range=f((0, half)), damping="50", armature="1")
        ET.SubElement(panel, "geom", name=f"{a.name}_{side}_panel", type="box",
                      size=f((thick / 2, half / 2, height / 2)), rgba=f(rgba(a.color)), mass="20",
                      **MOVING)
        ET.SubElement(actuators, "position", name=f"act_{a.name}_{side}", joint=f"{a.name}_{side}_joint",
                      kp="3000", dampratio="1", ctrlrange=f((0, half)), forcerange="-400 400")


def _hinged(world: ET.Element, a: Articulated) -> None:
    thick, width, height = a.size or (0.03, 0.8, 2.0)
    mount = ET.SubElement(world, "body", name=f"{a.name}_mount", pos=f(a.pos), quat=yaw_quat(a.yaw))
    leaf = ET.SubElement(mount, "body", name=a.name)
    ET.SubElement(leaf, "joint", name=f"{a.name}_joint", type="hinge", axis="0 0 1",
                  range=f((0, math.radians(float(a.params.get("max_deg", 100))))), damping="5")
    ET.SubElement(leaf, "geom", name=f"{a.name}_leaf", type="box", size=f((thick / 2, width / 2, height / 2)),
                  pos=f((0, -width / 2, height / 2)), rgba=f(rgba(a.color)), mass="8", **MOVING)
    ET.SubElement(leaf, "geom", name=f"{a.name}_handle", type="box", size="0.03 0.012 0.012",
                  pos=f((thick / 2 + 0.03, -width + 0.08, height * 0.48)), rgba="0.2 0.2 0.2 1", mass="0.1",
                  **MOVING)


def _drawer(world: ET.Element, a: Articulated) -> None:
    depth, width, height = a.size or (0.35, 0.4, 0.12)
    mount = ET.SubElement(world, "body", name=f"{a.name}_mount", pos=f(a.pos), quat=yaw_quat(a.yaw))
    box = ET.SubElement(mount, "body", name=a.name)
    ET.SubElement(box, "joint", name=f"{a.name}_joint", type="slide", axis="1 0 0",
                  range=f((0, depth * 0.8)), damping="5")
    ET.SubElement(box, "geom", name=f"{a.name}_front", type="box", size=f((0.01, width / 2, height / 2)),
                  rgba=f(rgba(a.color)), mass="1", **MOVING)
    ET.SubElement(box, "geom", name=f"{a.name}_handle", type="box", size="0.012 0.05 0.012",
                  pos="0.03 0 0", rgba="0.2 0.2 0.2 1", mass="0.05", **MOVING)


def build_scene(robot: RobotSpec, scene: SceneSpec, visual: str | None = None) -> str:
    root = build_robot_mjcf(robot, visual)
    root.set("model", f"{robot.name}_scene")
    _add_scene(root, scene)
    return ET.tostring(root, encoding="unicode")


def build_team_scene(robots: Sequence[tuple[str, RobotSpec]], scene: SceneSpec, visual: str | None = None) -> str:
    """The scene with several robots: each robot's MJCF attached at the world origin with every name
    prefixed ``<id>/`` (its base joints carry its pose). Robot ``k`` gets collision bit ``8 << k``
    on top of the robot bit and collides with every other robot, never with itself."""
    import mujoco
    if len(robots) > 24:
        raise ValueError("at most 24 robots per scene (one collision bit each)")
    root = ET.Element("mujoco", model="team_scene")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "worldbody")
    ET.SubElement(root, "actuator")
    _add_scene(root, scene)
    parent = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    cache: dict[str, str] = {}
    for k, (rid, spec) in enumerate(robots):
        if spec.name not in cache:
            cache[spec.name] = ET.tostring(build_robot_mjcf(spec, visual), encoding="unicode")
        robot_root = ET.fromstring(cache[spec.name])
        others = sum(8 << j for j in range(len(robots)) if j != k)
        for geom in robot_root.iter("geom"):
            if geom.get("contype") == str(ROBOT_CONTYPE):
                geom.set("contype", str(ROBOT_CONTYPE | (8 << k)))
                geom.set("conaffinity", str(others))
        child = mujoco.MjSpec.from_string(ET.tostring(robot_root, encoding="unicode"))
        parent.attach(child, frame=parent.worldbody.add_frame(), prefix=f"{rid}/")
    return parent.to_xml()


def _add_scene(root: ET.Element, scene: SceneSpec) -> None:
    """Options, lights, floor, fixtures, objects, articulated fixtures and scene cameras."""
    pos = 1
    for el in list(root):
        if el.tag == "compiler":
            pos = list(root).index(el) + 1
    option = ET.Element("option", timestep="0.002", integrator="implicitfast", cone="elliptic",
                        impratio="5")
    root.insert(pos, option)
    visual = ET.Element("visual")
    ET.SubElement(visual, "global", offwidth="1920", offheight="1080")
    ET.SubElement(visual, "headlight", ambient="0.35 0.35 0.35", diffuse="0.45 0.45 0.45", specular="0.1 0.1 0.1")
    ET.SubElement(visual, "quality", shadowsize="4096")
    root.insert(pos + 1, visual)
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(pos + 2, asset)
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker", width="512", height="512",
                  rgb1=f(rgba(scene.floor_color)[:3]), rgb2=f(np.array(rgba(scene.floor_color)[:3]) * 0.9))
    ET.SubElement(asset, "material", name="floor", texture="grid", texrepeat="8 8", reflectance="0.05")
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient", rgb1="0.75 0.8 0.86",
                  rgb2="0.95 0.95 0.97", width="256", height="256")

    world = root.find("worldbody")
    ET.SubElement(world, "light", name="key", pos="1.5 -1.5 3.5", dir="-0.3 0.3 -1", directional="true",
                  diffuse="0.55 0.55 0.55", castshadow="true")
    ET.SubElement(world, "light", name="fill", pos="-2 2 3", dir="0.4 -0.4 -1", directional="true",
                  diffuse="0.25 0.25 0.25", castshadow="false")
    ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 0.05", material="floor",
                  contype="1", conaffinity="1", friction="1 0.01 0.001")
    for s in scene.fixtures:
        body = ET.SubElement(world, "body", name=s.name, pos=f(s.pos), quat=yaw_quat(s.yaw))
        _shape_geoms(body, s, STATIC)
    actuators = root.find("actuator")
    for s in scene.objects:
        body = ET.SubElement(world, "body", name=s.name, pos=f(s.pos), quat=yaw_quat(s.yaw))
        ET.SubElement(body, "freejoint", name=f"{s.name}_free")
        if s.type == "staunton":
            _staunton(body, asset, s)
            continue
        ET.SubElement(body, "inertial", pos="0 0 0", mass=f"{s.mass:g}",
                      diaginertia=f([s.mass * 0.4 * (max(s.size) / 2) ** 2] * 3))
        _shape_geoms(body, s, {**OBJECT, "condim": "4", "solref": "0.004 1"})
    for a in scene.articulated:
        if a.type == "button":
            _button(world, a)
        elif a.type == "sliding_door":
            _sliding_door(world, actuators, a)
        elif a.type == "hinged_door":
            _hinged(world, a)
        elif a.type == "drawer":
            _drawer(world, a)
        else:
            raise ValueError(f"unknown articulated type {a.type!r}")
    for c in scene.cameras:
        ET.SubElement(world, "camera", name=c.name, pos=f(c.pos), xyaxes=lookat_xyaxes(c.pos, c.lookat),
                      fovy=f"{c.fovy:g}")

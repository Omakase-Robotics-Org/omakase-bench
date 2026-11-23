"""The kinematic model of a robot preset, independent of the backend: forward kinematics and rotations.

It is a robot-only MuJoCo model built from the same URDF every backend loads, with the mobile base at
the origin, so every pose here is in the robot base frame (x forward, y left, z up). IK solvers
(:mod:`.ik`) run on it, so a backend needs no IK of its own: it only has to follow joint targets.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Mapping, Sequence

import mujoco
import numpy as np

from ...robots.spec import RobotSpec
from .ik import DlsSolver, IKResult

#: named approach directions in the base frame, with the jaw-closing axis at wrist angle 0
APPROACHES: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "down": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "forward": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "left": ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
    "right": ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0)),
}


def _rotate(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return (v * np.cos(angle) + np.cross(axis, v) * np.sin(angle)
            + axis * np.dot(axis, v) * (1 - np.cos(angle)))


class Kinematics:
    def __init__(self, spec: RobotSpec):
        from ...backends.mujoco.robot_mjcf import build_robot_mjcf   # the MJCF builder is shared
        self.spec = spec
        root = build_robot_mjcf(spec, visual="kinematics")
        for tag in ("actuator", "equality"):
            for el in root.findall(tag):
                root.remove(el)
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        self._qadr = {self.model.joint(i).name: int(self.model.jnt_qposadr[i])
                      for i in range(self.model.njnt)}
        self._dofadr = {self.model.joint(i).name: int(self.model.jnt_dofadr[i])
                        for i in range(self.model.njnt)}
        self._range = {self.model.joint(i).name: tuple(self.model.jnt_range[i])
                       if self.model.jnt_limited[i] else (-np.pi * 2, np.pi * 2)
                       for i in range(self.model.njnt)}
        base = spec.mobile_base
        self._base_joints = {base.x, base.y, base.yaw} if base else set()
        self._closing_local: dict[str, np.ndarray] = {}
        mujoco.mj_kinematics(self.model, self.data)
        for name, ee in spec.end_effectors.items():
            body = self.model.body(ee.tcp_link).id
            if ee.fingers:
                j = self.model.joint(ee.fingers[0].joint)
                axis_world = self.data.xmat[j.bodyid[0]].reshape(3, 3) @ j.axis
                local = self.data.xmat[body].reshape(3, 3).T @ axis_world
                a = np.array(ee.approach_axis, float)
                local = local - a * np.dot(local, a)
                self._closing_local[name] = local / np.linalg.norm(local)
            else:
                self._closing_local[name] = np.array([0.0, 1.0, 0.0])

    # --- state -------------------------------------------------------------
    def set_q(self, q: Mapping[str, float]) -> None:
        for name, v in q.items():
            if name in self._qadr and name not in self._base_joints:
                self.data.qpos[self._qadr[name]] = v
        mujoco.mj_kinematics(self.model, self.data)

    def tcp_pose(self, arm: str) -> tuple[np.ndarray, np.ndarray]:
        """(position, rotation matrix) of the arm's TCP site in the base frame."""
        site = self.data.site(f"{self.spec.arms[arm].end_effector}_tcp")
        return site.xpos.copy(), site.xmat.reshape(3, 3).copy()

    def body_position(self, link: str) -> np.ndarray:
        return self.data.body(link).xpos.copy()

    def fk(self, q: Mapping[str, float]) -> dict[str, np.ndarray]:
        self.set_q(q)
        return {arm: self.tcp_pose(arm)[0] for arm in self.spec.arms}

    # --- orientation helpers ----------------------------------------------
    def target_rotation(self, arm: str, approach: Sequence[float], closing: Sequence[float]) -> np.ndarray:
        """Rotation of the EE link whose approach axis points along ``approach`` and whose
        jaw-closing axis lies along ``closing`` (made orthogonal to the approach)."""
        ee = self.spec.hand_of(arm)
        a_l = np.array(ee.approach_axis, float)
        c_l = self._closing_local[ee.name]
        a_w = np.array(approach, float)
        a_w /= np.linalg.norm(a_w)
        c_w = np.array(closing, float)
        c_w = c_w - a_w * np.dot(c_w, a_w)
        if np.linalg.norm(c_w) < 1e-6:
            raise ValueError("closing axis is parallel to the approach axis")
        c_w /= np.linalg.norm(c_w)
        local = np.column_stack([a_l, c_l, np.cross(a_l, c_l)])
        world = np.column_stack([a_w, c_w, np.cross(a_w, c_w)])
        return world @ local.T

    def named_rotation(self, arm: str, approach: str, wrist_angle: float) -> np.ndarray:
        a, c = APPROACHES[approach]
        # "down": turn about +z, so a positive angle is counter-clockwise seen from above and an
        # object's yaw closes the jaws across its faces; otherwise turn about the approach axis
        axis = np.array((0.0, 0.0, 1.0)) if approach == "down" else np.array(a, float)
        c_rot = _rotate(np.array(c, float), axis, wrist_angle)
        return self.target_rotation(arm, a, c_rot)

    def approach_axis(self, arm: str) -> np.ndarray:
        ee = self.spec.hand_of(arm)
        _, rot = self.tcp_pose(arm)
        return rot @ np.array(ee.approach_axis, float)

    # --- IK with the default solver -------------------------------------------
    def solve_ik(self, arm: str, position: Sequence[float], rotation: np.ndarray | None,
                 q_current: Mapping[str, float], **options) -> IKResult:
        return DlsSolver(self).solve_ik(arm, position, rotation, q_current, **options)

    def solve_ik_near(self, arm: str, position: Sequence[float], rotation: np.ndarray | None,
                      q_current: Mapping[str, float], **options) -> IKResult:
        return DlsSolver(self).solve_ik_near(arm, position, rotation, q_current, **options)

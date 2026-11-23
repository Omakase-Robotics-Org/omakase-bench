"""The scripted subject: it runs the reference solution a task package registers.

Reference solutions are hand-written programs that read ground truth. They are the upper bound of each
task on a backend: if one fails, the task, the scene or the robot API is broken, not a model. They call
the same robot API as every other subject and are scored the same way. This module also holds what such
programs share.
"""
from __future__ import annotations

import math

import numpy as np

from ..harness.api import RobotAPI
from ..harness.session import Session
from .base import SubjectCard


def object_pose(session: Session, name: str) -> tuple[np.ndarray, float, tuple]:
    """Ground truth of a scene object in the robot's base frame: position, yaw (rad), size."""
    snap = session.oracle()
    o = snap.objects[name]
    api = session.api
    w, x, y, z = o.quat
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)) - api._base_pose()[2]
    return api.world_to_base(o.pos), yaw, o.size


def tip_reach(api: RobotAPI, arm: str) -> float:
    """Fingertip reach past the TCP along the approach direction (``end_effectors.<name>.tip_reach``)."""
    return api.spec.hand_of(arm).tip_reach


def grasp_wrist(yaw: float) -> float:
    """Jaws close along a cube face normal: wrist angle in (-45, 45] degrees."""
    d = math.degrees(yaw) % 90.0
    return d - 90.0 if d > 45.0 else d


def drive_to(api: RobotAPI, x: float, y: float, turn_after_deg: float | None = None) -> None:
    """Reach (x, y) in the current base frame by turn, drive, turn, which a differential base
    can follow; the final turn restores the starting heading unless ``turn_after_deg`` is given
    as the heading change relative to it."""
    heading = math.degrees(math.atan2(y, x)) if math.hypot(x, y) > 1e-3 else 0.0
    if abs(heading) > 0.05:
        api.move_base(turn_deg=heading)
    api.move_base(forward_m=math.hypot(x, y))
    final = (0.0 if turn_after_deg is None else turn_after_deg) - heading
    if abs(final) > 0.05:
        api.move_base(turn_deg=final)


class ScriptedSubject:
    kind = "scripted"
    interface = "scripted"

    def __init__(self, name: str = "oracle"):
        self.name = f"scripted/{name}"

    def card(self) -> SubjectCard:
        from ..tasks import packages
        return SubjectCard(self.name, self.kind, self.interface,
                           detail={"reads_ground_truth": True,
                                   "programs": sorted(t for p in packages() for t in p.scripted)})

    def run(self, session: Session) -> str:
        from ..tasks import scripted_program
        program = scripted_program(session.task.id)
        if program is None:
            raise NotImplementedError(f"no scripted program for {session.task.id}")
        program(session.api, session)
        session.advance(0.5)
        return "finished"

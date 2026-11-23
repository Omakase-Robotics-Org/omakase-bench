"""Reference solutions that act through end-effector rows: a task package registers its pick-and-place
moves (``TaskPackage.moves``) and ``solve`` or ``solve_turn`` as its program.

It reads ground truth (an upper bound and a check of the task, never a model
condition) and turns the task's pick-and-place moves into fingertip waypoints.
One row is emitted per step and the next waypoint is taken up when the
measured fingertip pose has arrived, so the step limit and the IK of the
end-effector interface are part of what it exercises.
"""
from __future__ import annotations

import numpy as np

from ..harness.ee_control import ArmTarget, EEController, EELimits, format_row, parse_row
from ..harness.session import Session

GRASP_INSERTION = 0.014  # m, fingertip centre below the object top when the jaws start closing
HOVER = 0.03             # m, fingertips above the object top before descending
RELEASE_GAP = 0.003      # m, object bottom above its support when the jaws open
CLEARANCE = 0.02         # m, carried object's bottom above the tallest obstacle on the way
RETREAT = 0.05           # m, straight up after the release, before moving sideways
ARRIVE_M = 0.004         # grasp and release poses
PASS_M = 0.015           # poses the arm only passes through
ARRIVE_DEG = 3.0


def move_waypoints(move: dict) -> list[list]:
    """[arm, fingertip target, yaw, gripper, steps to stay once there, arrival tolerance] for one
    pick and place (or lift)."""
    arm, top, yaw = move["arm"], np.asarray(move["top"], float), float(move.get("yaw", 0.0))
    grasp = top - [0, 0, GRASP_INSERTION]
    hover = top + [0, 0, HOVER]
    hover[2] = max(hover[2], move.get("approach_z", 0.0))
    out = [[arm, hover, yaw, 0, 0, PASS_M], [arm, grasp, yaw, 0, 0, ARRIVE_M], [arm, grasp, yaw, 1, 2, ARRIVE_M]]
    if "lift" in move:
        return out + [[arm, grasp + [0, 0, move["lift"]], yaw, 1, 2, ARRIVE_M]]
    xy = np.asarray(move["place"], float)
    bottom_to_tip = move["height"] - GRASP_INSERTION
    carry_z = move["clear_z"] + CLEARANCE + bottom_to_tip
    release = np.array([xy[0], xy[1], move["support_z"] + RELEASE_GAP + bottom_to_tip])
    return out + [[arm, np.array([top[0], top[1], carry_z]), yaw, 1, 0, PASS_M],
                  [arm, np.array([xy[0], xy[1], carry_z]), yaw, 1, 0, PASS_M],
                  [arm, release, yaw, 1, 0, ARRIVE_M], [arm, release, yaw, 0, 1, ARRIVE_M],
                  [arm, release + [0, 0, RETREAT], yaw, 0, 0, PASS_M]]


class Follower:
    """Emits the next row of a waypoint list, given the measured fingertip pose."""

    def __init__(self, controller: EEController, moves: list[dict]):
        self.controller = controller
        self.wps = [w for m in moves for w in move_waypoints(m)]

    def _arrived(self) -> bool:
        arm, tip, yaw, _, _, tol = self.wps[0]
        pos, cur = self.controller.tip_pose(arm)
        return bool(np.linalg.norm(tip - pos) < tol and abs((yaw - cur + 180) % 360 - 180) < ARRIVE_DEG)

    def finished(self) -> bool:
        """The last waypoint is reached and held for its steps."""
        return len(self.wps) == 1 and self._arrived() and self.wps[0][4] <= 0

    def row(self) -> str:
        """The next action row (canonical text, without ``a:``)."""
        while len(self.wps) > 1 and self._arrived() and self.wps[0][4] <= 0:
            self.wps.pop(0)
        arm, tip, yaw, g, _, _ = self.wps[0]
        if self._arrived():
            self.wps[0][4] -= 1
        row = {a: None for a in self.controller.arms}
        row[arm] = ArmTarget(tuple(int(v) for v in np.round(tip * 1000)), int(round(yaw)), g)
        return format_row(row, self.controller.arms)


def make_controller(session: Session) -> EEController:
    """The task's end-effector controller, with the arms at the task's start pose."""
    cfg = session.task.raw.get("ee", {})
    controller = EEController(session, EELimits.from_mapping(cfg.get("limits")))
    controller.move_to_start(cfg.get("start", {}), cfg.get("start_path", "joint") == "staged")
    return controller


def make_follower(session: Session, controller: EEController) -> Follower:
    from ..tasks import package_for
    plan = package_for(session.task.id).moves[session.task.id]
    moves = plan(session.oracle(), session.episode.values, session.task.raw.get("params", {}))
    return Follower(controller, moves)


def solve(api, session: Session) -> None:
    controller = make_controller(session)
    follower = make_follower(session, controller)
    for _ in range(int(session.task.raw.get("ee", {}).get("horizon", 40))):
        controller.step(parse_row(follower.row(), controller.arms))


def solve_turn(api, session: Session) -> None:
    """One turn of a multi-robot task: the robot's move for this turn, ending when it is done."""
    controller = make_controller(session)
    follower = make_follower(session, controller)
    for _ in range(int(session.task.raw.get("ee", {}).get("horizon", 40))):
        if follower.finished():
            return
        controller.step(parse_row(follower.row(), controller.arms))

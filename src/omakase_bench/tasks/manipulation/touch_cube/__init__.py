"""Touch the cube on the table with a closed gripper.

Definition: ``suites/core/v1/touch_cube.yaml`` (published in ``core@1``).
"""
from __future__ import annotations

from pathlib import Path

from ....harness.api import RobotAPI
from ....harness.session import Session
from ....subjects.scripted import grasp_wrist, object_pose, tip_reach
from ...base import TaskPackage


def solve(api: RobotAPI, session: Session) -> None:
    p, yaw, size = object_pose(session, "cube")
    arm = api.spec.arm_for(p[1])
    top = p[2] + size[2] / 2
    api.close_gripper(arm)
    api.move_ee(arm, p[0], p[1], top + 0.10, "down", grasp_wrist(yaw))
    api.move_ee(arm, p[0], p[1], top + tip_reach(api, arm) - 0.004, "down", grasp_wrist(yaw))
    api.move_ee_by(arm, dz=0.15)


PACKAGE = TaskPackage(
    name="manipulation/touch_cube",
    summary="touch a cube with a closed gripper",
    scripted={"core/touch_cube": solve},
    directory=Path(__file__).parent,
)

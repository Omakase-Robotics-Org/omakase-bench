"""Pick the cube from the table and put it into the tray: the template for a new task.

One robot, one subject. The pieces of the task and where they are:

    definition          suites/core/v1/cube_to_tray.yaml (published in ``core@1``): the scene (table, tray,
                        cube), what varies per seed (cube pose and colour, tray pose), the instruction,
                        ``success`` (the cube is inside the tray and released) and three weighted sub-goals
    reference solution  ``solve`` below, run by ``--model scripted/oracle``; it reads ground truth and
                        calls the same robot API a model does
    registration        ``PACKAGE`` at the end of this file

``docs/new_task.md`` walks through writing a task like this one.
"""
from __future__ import annotations

from pathlib import Path

from ....harness.api import Refused, RobotAPI
from ....harness.session import Session
from ....subjects.scripted import grasp_wrist, object_pose
from ...base import TaskPackage


def solve(api: RobotAPI, session: Session) -> None:
    p, yaw, size = object_pose(session, "cube")
    t, tyaw, tsize = object_pose(session, "tray")
    arm = api.spec.arm_for(p[1])
    wrist = grasp_wrist(yaw)
    api.open_gripper(arm)
    try:
        api.move_ee(arm, p[0], p[1], p[2] + 0.12, "down", wrist)
    except Refused:
        wrist += 90.0    # a cube can be gripped across either pair of faces
        api.move_ee(arm, p[0], p[1], p[2] + 0.12, "down", wrist)
    api.move_ee(arm, p[0], p[1], p[2] + 0.006, "down", wrist)
    api.close_gripper(arm)
    api.move_ee_by(arm, dz=0.12)
    rim = t[2] + tsize[2] / 2
    api.move_ee(arm, t[0], t[1], rim + 0.10, "down", grasp_wrist(tyaw))
    api.move_ee(arm, t[0], t[1], rim + 0.035, "down", grasp_wrist(tyaw))
    api.open_gripper(arm)
    api.move_ee_by(arm, dz=0.10)


PACKAGE = TaskPackage(
    name="manipulation/cube_to_tray",
    summary="pick a cube and place it in a tray",
    scripted={"core/cube_to_tray": solve},
    directory=Path(__file__).parent,
)

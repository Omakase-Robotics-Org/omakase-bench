"""Press a wall-mounted push button.

Definition: ``suites/core/v1/press_button.yaml`` (published in ``core@1``). ``press`` is also the
button press of tasks that contain one (``mobile_manipulation/elevator_call``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ....harness.api import RobotAPI
from ....harness.session import Session
from ....subjects.scripted import tip_reach
from ...base import TaskPackage


def press(api: RobotAPI, session: Session, name: str = "button") -> str:
    """Press the named button; returns the arm that pressed it."""
    item = next(o for o in api.list_objects() if o["name"] == name)
    face = np.array(item["position"])
    n = np.array(item["facing"])
    arm = api.spec.arm_for(face[1])
    api.close_gripper(arm)
    pre = face + n * 0.10
    api.move_ee(arm, *pre, "forward", 90.0)
    press = face + n * (tip_reach(api, arm) - 0.007)
    api.move_ee(arm, *press, "forward", 90.0)
    api.move_ee(arm, *pre, "forward", 90.0)
    return arm


PACKAGE = TaskPackage(
    name="manipulation/press_button",
    summary="press a wall-mounted push button",
    scripted={"core/press_button": press},
    directory=Path(__file__).parent,
)

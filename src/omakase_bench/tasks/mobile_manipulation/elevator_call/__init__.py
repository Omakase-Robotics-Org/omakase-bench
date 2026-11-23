"""Call the elevator, wait for the door and drive into the car.

Definition: ``suites/drafts/v0/elevator_call.yaml`` (draft). The base is differential drive, so every
leg is turn, drive, turn.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ....harness.api import RobotAPI
from ....harness.session import Session
from ....subjects.scripted import drive_to
from ...base import TaskPackage
from ...manipulation.press_button import press


def solve(api: RobotAPI, session: Session) -> None:
    objs = {o["name"]: o for o in api.list_objects()}
    # face the wall squarely, then stop 0.62 m in front of the button with the button at y = -0.22
    yaw = math.atan2(-objs["call_button"]["facing"][1], -objs["call_button"]["facing"][0])
    api.move_base(turn_deg=math.degrees(yaw))
    objs = {o["name"]: o for o in api.list_objects()}
    button = np.array(objs["call_button"]["position"])
    drive_to(api, button[0] - 0.62, button[1] + 0.22)
    api.go_home(press(api, session, "call_button"))
    for _ in range(20):
        door = next(o for o in api.list_objects() if o["name"] == "car_door")
        if door["open_fraction"] > 0.95:
            break
        api.wait(1.0)
    # line up with the doorway before driving through it: sideways leg first, then straight in
    car = np.array(next(o for o in api.list_objects() if o["name"] == "car")["position"])
    drive_to(api, 0.0, car[1])
    car = np.array(next(o for o in api.list_objects() if o["name"] == "car")["position"])
    api.move_base(forward_m=car[0])


PACKAGE = TaskPackage(
    name="mobile_manipulation/elevator_call",
    summary="press the call button, wait for the door, drive into the car",
    scripted={"drafts/elevator_call": solve},
    directory=Path(__file__).parent,
)

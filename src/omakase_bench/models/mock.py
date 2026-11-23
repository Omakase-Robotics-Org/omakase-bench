"""Deterministic stand-in models for tests and plumbing checks; no network, no cost.

``mock/finish``      declares the task finished at once (a lower bound).
``mock/press``       solves press_button-style tasks: reads ``list_objects`` and presses
                     the first button it lists (tools interface), or writes the
                     equivalent program (code interface).
``mock/place``       solves cube_to_tray-style tasks: reads ``list_objects``, grasps the object
                     labelled ``cube`` and releases it above the one labelled ``tray`` (tools
                     interface), or writes the equivalent program (code interface). It takes the
                     arm names from the robot it is given, so it runs on any robot.
``mock/loop``        calls ``get_state`` forever; used to exercise budget caps.
``mock/script:<f>``  replays a JSON list of replies from a file.
"""
from __future__ import annotations

import json
from pathlib import Path

from .base import Capabilities, Reply, Request, Text, ToolCall, Usage

PRESS_PROGRAM = '''
objs = robot.list_objects()
button = next(o for o in objs if o["kind"] == "push button")
fx, fy, fz = button["position"]
nx, ny, nz = button["facing"]
arm = "left" if fy >= 0 else "right"
robot.close_gripper(arm)
robot.move_ee(arm, fx + 0.10 * nx, fy + 0.10 * ny, fz + 0.10 * nz, approach="forward", wrist_deg=90)
robot.move_ee(arm, fx + 0.004 * nx, fy + 0.004 * ny, fz + 0.004 * nz, approach="forward", wrist_deg=90)
robot.move_ee(arm, fx + 0.10 * nx, fy + 0.10 * ny, fz + 0.10 * nz, approach="forward", wrist_deg=90)
print("pressed", button["name"])
'''


PLACE_PROGRAM = '''
objs = {o["kind"]: o for o in robot.list_objects() if "kind" in o}
cube, tray = objs["cube"], objs["tray"]
arms = list(robot.get_state()["arms"])
side = "left" if cube["position"][1] >= 0 else "right"
arm = side if side in arms else arms[0]

def wrist(yaw_deg):
    d = yaw_deg % 90.0
    return d - 90.0 if d > 45.0 else d

x, y, z = cube["position"]
robot.open_gripper(arm)
w = wrist(cube["yaw_deg"])
try:
    robot.move_ee(arm, x, y, z + 0.12, approach="down", wrist_deg=w)
except RobotError:
    w += 90.0
    robot.move_ee(arm, x, y, z + 0.12, approach="down", wrist_deg=w)
robot.move_ee(arm, x, y, z + 0.006, approach="down", wrist_deg=w)
robot.close_gripper(arm)
robot.move_ee_by(arm, dz=0.12)
tx, ty, tz = tray["position"]
rim = tz + tray["size"][2] / 2
robot.move_ee(arm, tx, ty, rim + 0.10, approach="down", wrist_deg=wrist(tray["yaw_deg"]))
robot.move_ee(arm, tx, ty, rim + 0.035, approach="down", wrist_deg=wrist(tray["yaw_deg"]))
robot.open_gripper(arm)
robot.move_ee_by(arm, dz=0.10)
print("placed", cube["name"])
'''


def _wrist(yaw_deg: float) -> float:
    """Jaws across a face of a square object: a wrist angle in (-45, 45] degrees."""
    d = yaw_deg % 90.0
    return d - 90.0 if d > 45.0 else d


def _last_result_text(request: Request) -> str:
    for m in reversed(request.messages):
        for r in m.results:
            return "".join(p.text for p in r.parts if isinstance(p, Text))
    return ""


#: a nominal price per mock request, so cost caps can be exercised without a provider
MOCK_USD = 0.001


def _usage() -> Usage:
    return Usage(input_tokens=100, output_tokens=20, seconds=0.0, usd=MOCK_USD)


BEHAVIOURS = ("finish", "loop", "press", "place")


class MockAdapter:
    capabilities = Capabilities(images=False)

    def __init__(self, behaviour: str):
        self.behaviour = behaviour
        self.name = f"mock/{behaviour}"
        self.requests: list[Request] = []
        self._script: list[dict] | None = None
        if not behaviour.startswith("script:") and behaviour not in BEHAVIOURS:
            raise ValueError(f"unknown mock behaviour {behaviour!r}; known: {', '.join(BEHAVIOURS)}, script:<file>")
        if behaviour.startswith("script:"):
            self._script = json.loads(Path(behaviour.split(":", 1)[1]).read_text())

    def respond(self, request: Request) -> Reply:
        self.requests.append(request)
        n = len(self.requests)
        call = lambda name, **a: Reply(calls=(ToolCall(f"m{n}", name, a),), stop="tool_use", usage=_usage(),
                                        model=self.name)
        is_code = not request.tools
        if self._script is not None:
            if not self._script:
                return Reply(stop="end", usage=_usage(), text="done")
            step = self._script.pop(0)
            if "code" in step:
                return Reply(text=f"```python\n{step['code']}\n```", usage=_usage())
            return call(step["name"], **step.get("arguments", {}))
        if self.behaviour == "finish":
            if is_code:
                return Reply(text="```python\nprint('nothing to do')\n```", usage=_usage())
            return call("finish", note="done")
        if self.behaviour == "loop":
            if is_code:
                return Reply(text="```python\nwhile True:\n    robot.get_state()\n```", usage=_usage())
            return call("get_state")
        if self.behaviour == "press":
            if is_code:
                return Reply(text=f"```python\n{PRESS_PROGRAM}\n```", usage=_usage())
            return self._press(request, call)
        if self.behaviour == "place":
            if is_code:
                return Reply(text=f"```python\n{PLACE_PROGRAM}\n```", usage=_usage())
            return self._place(request, call)
        raise ValueError(f"unknown mock behaviour {self.behaviour!r}")

    def _press(self, request: Request, call) -> Reply:
        names = [c.name for m in request.messages for c in m.calls]
        if not names:
            return call("list_objects")
        if names[-1] == "list_objects":
            objs = json.loads(_last_result_text(request))
            b = next(o for o in objs if o["kind"] == "push button")
            self._button = b
            self._arm = "left" if b["position"][1] >= 0 else "right"
            return call("close_gripper", arm=self._arm)
        f, nrm = self._button["position"], self._button["facing"]
        pre = [f[i] + 0.10 * nrm[i] for i in range(3)]
        press = [f[i] + 0.004 * nrm[i] for i in range(3)]
        step = sum(1 for x in names if x == "move_ee")
        if step < 3:
            target = (pre, press, pre)[step]
            return call("move_ee", arm=self._arm, x=target[0], y=target[1], z=target[2],
                        approach="forward", wrist_deg=90)
        return call("finish", note="pressed")

    def _place(self, request: Request, call) -> Reply:
        names = [c.name for m in request.messages for c in m.calls]
        if not names:
            return call("list_objects")
        if names == ["list_objects"]:
            objs = {o["kind"]: o for o in json.loads(_last_result_text(request)) if "kind" in o}
            cube, tray = objs["cube"], objs["tray"]
            arms = next(t for t in request.tools if t.name == "open_gripper").parameters["properties"]["arm"]["enum"]
            side = "left" if cube["position"][1] >= 0 else "right"
            arm = side if side in arms else arms[0]
            (x, y, z), (tx, ty, tz) = cube["position"], tray["position"]
            rim, w, tw = tz + tray["size"][2] / 2, _wrist(cube["yaw_deg"]), _wrist(tray["yaw_deg"])
            down = lambda px, py, pz, wr: ("move_ee", dict(arm=arm, x=px, y=py, z=pz, approach="down", wrist_deg=wr))
            self._plan = [("open_gripper", dict(arm=arm)), down(x, y, z + 0.12, w), down(x, y, z + 0.006, w),
                          ("close_gripper", dict(arm=arm)), ("move_ee_by", dict(arm=arm, dz=0.12)),
                          down(tx, ty, rim + 0.10, tw), down(tx, ty, rim + 0.035, tw),
                          ("open_gripper", dict(arm=arm)), ("move_ee_by", dict(arm=arm, dz=0.10))]
        elif _last_result_text(request).startswith("refused") and self._plan[0][0] == "move_ee":
            # a cube can be gripped across either pair of faces: the other one for the approach and the grasp
            for _, arguments in self._plan[:2]:
                arguments["wrist_deg"] += 90.0
        else:
            self._plan.pop(0)
        if not self._plan:
            return call("finish", note="placed")
        name, arguments = self._plan[0]
        return call(name, **arguments)

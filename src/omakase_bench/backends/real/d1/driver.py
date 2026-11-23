"""The D1 for the real backend: joint targets as requests of the daemon API, and its state operations as
joint positions.

Motion. Over the WebSocket transport both arms go in one ``move_joints_both`` request with ``wait: false``:
the daemon treats such requests arriving within 250 ms of each other as a stream, which is what a
teleoperation client sends at up to 200 Hz, and slows a streamed target on approach to a clearance margin
instead of refusing it. Over HTTP the fallback is one ``move_joints`` request per arm that changed, each
checked by the daemon's motion guard as a single move. The lift takes a height, the neck pitch and yaw, a
gripper an openness between 0 and 1.

Base. The harness moves the base by targets for its planar joints; the daemon takes a body velocity
(``cmd_vel``) and holds the last one. A change of the base target within one control period becomes the
velocity that covers it in that period, and the backend sends a zero velocity in the first period without
a new target. A sideways component is refused: the base is differential drive. Only the daemon's routes
are used, never the base's own API.

State. ``GET /v1/arm/feedback`` gives both arms' measured joint positions in degrees with the controller's
timestamp, ``GET /v1/neck/state`` pitch and yaw, ``GET /v1/end_effectors`` each gripper's openness, and
``GET /v1/slider/state`` the lift height; the lift is a serial read of 100-200 ms, so it is read as a slow
device. The base pose is the odometry pose of ``GET /v1/chassis/amr/localization`` (x, y in metres, heading
in radians), which the daemon serves only with the in-house base firmware; where it answers ``unavailable``
the base joints hold their targets and ``base_pose`` says so.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ....core.backend import BackendUnsupported
from ....harness.command_api import Request
from ....robots.spec import RobotSpec
from .api import RobotAPIConfig, robot_api

#: index of each daemon side in the ``sides`` array of the arm feedback record
SIDE_INDEX = {"a": 0, "b": 1}


#: m of sideways base motion in one control period taken as none
LATERAL_TOLERANCE_M = 0.002


@dataclass
class D1Driver:
    config: RobotAPIConfig
    #: "measured", or "commanded: <why>" once the daemon has said it cannot report a base pose
    base_pose: str = "measured"
    #: the daemon's arm lease: read, take (also renews) and hand back
    lease = {"read": ("GET", "/v1/arm/lease"), "take": ("POST", "/v1/arm/lease"), "release": ("DELETE", "/v1/arm/lease")}

    def requests(self, robot: RobotSpec, q: Mapping[str, float], changed: Sequence[str], streaming: bool,
                 previous: Mapping[str, float], dt: float) -> list[Request]:
        todo, out = set(changed), []
        base = robot.mobile_base
        if base and todo & {base.x, base.y, base.yaw}:
            dx, dy = q[base.x] - previous[base.x], q[base.y] - previous[base.y]
            yaw = previous[base.yaw]
            forward, left = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
            if abs(left) > LATERAL_TOLERANCE_M:
                raise BackendUnsupported(f"a base target {left * 1000:.0f} mm to the side: the base is differential "
                                         "drive and takes a forward velocity and a turn rate only")
            # within the ceilings the daemon clamps to: a target further away than one period covers is
            # approached at that speed, not asked for at a higher one
            top_v, top_w = self.config.max_linear_m_s, self.config.max_angular_rad_s
            out.append(Request("POST", "/v1/chassis/cmd_vel",
                               {"vx": round(min(top_v, max(-top_v, forward / dt)), 4),
                                "omega": round(min(top_w, max(-top_w, (q[base.yaw] - yaw) / dt)), 4)}))
        degrees = {self.config.arm_sides[name]: [round(math.degrees(q[j]), 3) for j in arm.joints]
                   for name, arm in robot.arms.items()}
        moved = [self.config.arm_sides[name] for name, arm in robot.arms.items() if todo & set(arm.joints)]
        if moved and streaming:
            out.append(Request("POST", "/v1/arm/move_joints_both", {**degrees, "wait": False}))
        else:
            out += [Request("POST", f"/v1/arm/{side}/move_joints", {"joints_deg": degrees[side], "wait": False})
                    for side in moved]
        for name in robot.arms:
            hand = robot.hand_of(name)
            fingers = [f for f in hand.fingers if f.joint in todo]
            if fingers:
                f = fingers[0]
                closed = (q[f.joint] - f.open) / (f.closed - f.open)
                out.append(Request("POST", f"/v1/end_effectors/{self.config.arm_sides[name]}/set",
                                   {"openness": round(min(1.0, max(0.0, 1.0 - closed)), 4), "wait": False}))
        if robot.lift_joint in todo:
            out.append(Request("POST", "/v1/slider/set_height", {"height_m": round(q[robot.lift_joint], 4), "wait": False}))
        tilt, pan = robot.neck.get("tilt"), robot.neck.get("pan")
        if todo & {tilt, pan}:
            # the API's pitch is positive looking up, the description's tilt joint positive looking down
            out.append(Request("POST", "/v1/neck/cmd", {"pitch": round(-q[tilt], 4), "yaw": round(q[pan], 4), "relative": False}))
        return out

    def stop(self, robot: RobotSpec) -> list[Request]:
        return [Request("POST", "/v1/chassis/cmd_vel", {"vx": 0.0, "omega": 0.0})]

    def read(self, robot: RobotSpec, get: Callable[[str], Any], slow: bool) -> dict[str, float]:
        q: dict[str, float] = {}
        sides = get("/v1/arm/feedback")["sides"]
        effectors = {e["side"]: e for e in get("/v1/end_effectors")}
        for name, arm in robot.arms.items():
            side = self.config.arm_sides[name]
            q.update(zip(arm.joints, (math.radians(v) for v in sides[SIDE_INDEX[side]]["pos"])))
            openness = ((effectors.get(side) or {}).get("state") or {}).get("openness")
            if openness is not None:
                closed = 1.0 - float(openness)
                q.update({f.joint: f.open + closed * (f.closed - f.open) for f in robot.hand_of(name).fingers})
        neck = get("/v1/neck/state")
        tilt, pan = robot.neck.get("tilt"), robot.neck.get("pan")
        if tilt and pan:
            q[tilt], q[pan] = -float(neck["pitch"]), float(neck["yaw"])
        if slow and robot.lift_joint:
            q[robot.lift_joint] = float(get("/v1/slider/state")["height_m"])
        base = robot.mobile_base
        if base and self.base_pose == "measured":
            try:
                data = get("/v1/chassis/amr/localization")
            except BackendUnsupported as exc:
                # another base firmware: the daemon has no pose with a heading to give
                self.base_pose = f"commanded: {exc}"
            else:
                pose = (data.get("mission_frame") or {}).get("odometry") or data["pose"]
                q[base.x], q[base.y], q[base.yaw] = float(pose["x"]), float(pose["y"]), float(pose["theta"])
        return q


def driver(robot: RobotSpec) -> D1Driver:
    return D1Driver(robot_api(robot))

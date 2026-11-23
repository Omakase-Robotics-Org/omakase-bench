"""The D1 daemon API served by a simulated robot.

:class:`SimDaemon` answers the requests of ``api.py`` on a simulated robot with the documented semantics:
units, which out-of-range values are refused (422 ``unprocessable`` for an openness or force outside its
range, 400 ``invalid`` for a lift height outside the stroke or a speed ratio outside 0..1, 409 ``refused``
for a joint target outside its travel or a joint-space path that comes closer than the clearance margin)
and which are clamped (neck angles to their envelope, base velocities to their ceilings), and the response
envelope. The arm lease is granted and handed back but does not gate the simulated arms. What it does not
model: the lease's motion fence, soft-kill latches, modes other than
``position``, a multi-finger hand's grip presets, and the base's own safety and navigation stack. Its
base drives with the bench's base noise model, whose values are assumptions, not measurements.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ....harness.command_api import Envelope, _why, error, load_spec, member, ok, response_data, skeleton

#: the state operations the simulated daemon serves
STATE = ("/v1/arm/feedback", "/v1/end_effectors", "/v1/slider/state", "/v1/neck/state")
LEASE_RANK = {"ambient": 10, "policy": 50, "operator": 90}

@dataclass
class SimDaemon:
    """The daemon API on a simulated robot, through an :class:`~omakase_bench.harness.ee_control.EEController`
    (its joint limits, clearance checks, gripper limits and base noise)."""
    controller: Any
    sides: Mapping[str, str]                                  # daemon side -> arm name
    max_linear_m_s: float = 0.5
    max_angular_rad_s: float = 0.6
    vel_ratio: dict = field(default_factory=dict)            # side -> ratio set by the mode call
    velocity: tuple[float, float] = (0.0, 0.0)
    lease: dict | None = None                                # the arm lease's current grant
    lease_epoch: int = 0

    def request(self, method: str, path: str, body: Any = None) -> Envelope:
        body = body or {}
        m = re.fullmatch(r"/v1/arm/(\w+)/(mode|move_joints|move_joint|estop)", path)
        if m:
            arm = self.sides.get(m.group(1))
            if arm is None:
                return error("not_found", f"no arm {m.group(1)!r}", 404)
            return getattr(self, "_" + m.group(2))(arm, m.group(1), body)
        if path == "/v1/arm/move_joints_both":
            targets = {}
            for s, arm in self.sides.items():
                targets.update(self._joint_targets(arm, body[s]))
            return self._move(targets, min(self.vel_ratio.get(s, 0.3) for s in self.sides))
        m = re.fullmatch(r"/v1/end_effectors/(\w+)/(set|stop)", path)
        if m:
            arm = self.sides.get(m.group(1))
            if arm is None:
                return error("not_found", f"no end effector {m.group(1)!r}", 404)
            if m.group(2) == "stop":
                return ok()
            return self._gripper(arm, body)
        if path == "/v1/slider/set_height":
            return self._slider(body)
        if path == "/v1/neck/cmd":
            return self._neck(body)
        if path == "/v1/chassis/cmd_vel":
            vx = max(-self.max_linear_m_s, min(self.max_linear_m_s, float(body["vx"])))
            om = max(-self.max_angular_rad_s, min(self.max_angular_rad_s, float(body["omega"])))
            self.velocity = (vx, om)
            env = ok()
            if (vx, om) != (float(body["vx"]), float(body["omega"])):
                env["note"] = f"clamped to vx {vx:g} m/s, omega {om:g} rad/s"
            return env
        if path == "/v1/limits":
            return ok(self.limits())
        if method == "GET" and path in STATE:
            return ok(self.state(path))
        if path == "/v1/chassis/amr/localization":
            return ok(self.localization())
        if path == "/v1/arm/lease":
            return self._lease(method, body)
        return error("not_found", f"{method} {path} is not served by the simulated daemon", 404)

    @property
    def document(self) -> Mapping:
        spec = self.controller.spec
        return load_spec(str(Path(spec.preset_dir) / spec.api["spec"]))

    def localization(self) -> dict:
        """The base's odometry pose (x, y in metres, heading in radians), as the in-house base reports it."""
        base = self.controller.spec.mobile_base
        q = self.controller.session.backend.joint_positions()
        pose = {"x": q[base.x], "y": q[base.y], "theta": q[base.yaw]}
        return {"pose": dict(pose), "mission_frame": {"frame": "odom", "odometry": pose}}

    def _lease(self, method: str, body: Mapping) -> Envelope:
        """The lease's grants: first come within a class, a higher class takes it, only the holder hands it
        back. The simulated arms are not gated by it."""
        held = self.lease
        if method == "GET":
            return ok(None if held is None else {k: v for k, v in held.items() if k != "token"})
        if method == "DELETE":
            if held is not None and held["holder"] != body.get("holder"):
                return error("lease_held", f"the arms are held by {held['holder']} ({held['class']})", 409)
            self.lease = None
            return ok()
        rank = LEASE_RANK.get(body.get("class") or "ambient")
        if rank is None or not body.get("holder"):
            return error("invalid", "a lease names its holder and one of the classes ambient, policy, operator", 400)
        if held is not None and held["holder"] != body["holder"]:
            if rank <= LEASE_RANK[held["class"]]:
                return error("lease_held", f"the arms are held by {held['holder']} ({held['class']})", 409)
        fresh = held is None or held["holder"] != body["holder"]
        ttl = int(body.get("ttl_s") or 30)
        self.lease = {"holder": body["holder"], "class": body.get("class") or "ambient",
                      "epoch": (held["epoch"] if held else self.lease_epoch) + (1 if fresh else 0),
                      "expires_in_s": ttl, "ttl_s": ttl, "preempted_from": held["holder"] if held and fresh else None,
                      "token": "0" * 16}
        self.lease_epoch = self.lease["epoch"]
        return ok(dict(self.lease))

    def state(self, path: str) -> Any:
        """The ``data`` of a state operation, from the simulated joints: the fields a client reads for joint
        state, the rest of the record at the document's zero values."""
        c = self.controller
        spec, q = c.spec, c.session.backend.joint_positions()
        data = skeleton(self.document, response_data(self.document, "GET", path))
        if path == "/v1/arm/feedback":
            side = skeleton(self.document, member(self.document, response_data(self.document, "GET", path), "properties", "sides", "items"))
            data["sides"] = [{**side, "pos": [math.degrees(q[j]) for j in spec.arms[self.sides[s]].joints]}
                             for s in sorted(self.sides)]
            data["header"]["mono_ns"] = data["rx_mono_ns"] = int(c.session.backend.sim_time() * 1e9)
        elif path == "/v1/end_effectors":
            schema = member(self.document, response_data(self.document, "GET", path), "items")
            item, state = skeleton(self.document, schema), skeleton(self.document, member(self.document, schema, "properties", "state"))
            data = [{**item, "side": s, "fitted": True, "available": True,
                     "state": {**state, "openness": 1.0 - c._closed_fraction(self.sides[s], q)}} for s in sorted(self.sides)]
        elif path == "/v1/slider/state":
            data["height_m"] = q[spec.lift_joint]
        elif path == "/v1/neck/state":
            data["pitch"], data["yaw"] = -q[spec.neck["tilt"]], q[spec.neck["pan"]]
        return data

    def wait(self, seconds: float) -> None:
        """Let ``seconds`` pass with the held base velocity. The motion is checked like any base move before
        it starts and refused as a whole when it is over the command limits or would come too close to
        something (``last_base`` holds the feedback line)."""
        vx, om = self.velocity
        if vx or om:
            fb = self.controller.base_move(vx * seconds, math.degrees(om * seconds), arc=True)
            self.last_base = fb
        else:
            self.controller.session.advance(seconds)

    # --- arms ---
    def _mode(self, arm: str, side: str, body: Mapping) -> Envelope:
        if body.get("mode") != "position":
            return error("precondition_unmet", "the simulated daemon runs the arms in position mode only", 409)
        for k in ("vel_ratio", "acc_ratio"):
            v = body.get(k)
            if v is None:
                return error("invalid", f"position mode needs {k}", 400)
            if not 0.0 <= float(v) <= 1.0:
                return error("invalid", f"{k} {v} is outside 0.0..=1.0", 400)
        self.vel_ratio[side] = float(body["vel_ratio"])
        return ok()

    def _joint_targets(self, arm: str, degrees: Sequence[float]) -> dict[str, float]:
        joints = self.controller.spec.arms[arm].joints
        if len(degrees) != len(joints):
            raise ValueError(f"{len(joints)} joints expected")
        return {j: math.radians(float(v)) for j, v in zip(joints, degrees)}

    def _move_joints(self, arm: str, side: str, body: Mapping) -> Envelope:
        try:
            targets = self._joint_targets(arm, body["joints_deg"])
        except ValueError as exc:
            return error("invalid", str(exc), 400)
        return self._move(targets, self.vel_ratio.get(side, 0.3))

    def _move_joint(self, arm: str, side: str, body: Mapping) -> Envelope:
        joints = self.controller.spec.arms[arm].joints
        k = int(body["joint"])
        if not 0 <= k < len(joints):
            return error("invalid", f"joint index {k} is outside 0..={len(joints) - 1}", 400)
        return self._move({joints[k]: math.radians(float(body["position"]))}, self.vel_ratio.get(side, 0.3))

    def _estop(self, arm: str, side: str, body: Mapping) -> Envelope:
        return ok()

    def _move(self, targets: Mapping[str, float], vel_ratio: float) -> Envelope:
        fb = self.controller.joints_move(dict(targets), max(vel_ratio, 1e-3))
        if fb.startswith("joints: moved"):
            return ok()
        return error("refused", _why(fb), 409)

    # --- end effectors, lift, neck ---
    def _gripper(self, arm: str, body: Mapping) -> Envelope:
        o = float(body["openness"])
        if not 0.0 <= o <= 1.0:
            return error("unprocessable", f"openness {o} is outside [0, 1]", 422)
        f = body.get("force_percent")
        if f is not None and not 0.0 <= float(f) <= 100.0:
            return error("unprocessable", f"force_percent {f} is outside [0, 100]", 422)
        if body.get("grip") is not None or body.get("torque_nm") is not None:
            return error("unprocessable", "this end effector lists no grip presets or torque caps", 422)
        fb = self.controller.gripper_set(arm, o, None if f is None else float(f))
        if "refused" in fb:
            return error("refused", _why(fb), 409)
        env = ok()
        if "clamped" in fb:
            env["note"] = fb[fb.index("(") + 1:fb.rindex(")")] + ", a limit of this task"
        return env

    def _slider(self, body: Mapping) -> Envelope:
        if (body.get("height_m") is None) == (body.get("delta_m") is None):
            return error("invalid", "send exactly one of height_m and delta_m", 400)
        r = body.get("speed_ratio")
        if r is not None and not 0.0 <= float(r) <= 1.0:
            return error("invalid", f"speed_ratio {r} is outside 0.0..=1.0", 400)
        spec = self.controller.spec
        now = self.controller.session.backend.joint_positions()[spec.lift_joint]
        target = float(body["height_m"]) if body.get("height_m") is not None else now + float(body["delta_m"])
        lo, hi = spec.lift_range
        if not lo - 1e-9 <= target <= hi + 1e-9:
            return error("invalid", f"height {target:.3f} m is outside the travel {lo:g}..{hi:g} m", 400)
        fb = self.controller.lift_move(target, 1.0 if r is None else float(r))
        return ok() if fb.startswith("lift: at") else error("refused", _why(fb), 409)

    def _neck(self, body: Mapping) -> Envelope:
        c = self.controller
        spec = c.spec
        q = c.session.backend.joint_positions()
        tilt_j, pan_j = spec.neck.get("tilt"), spec.neck.get("pan")
        pitch = body.get("pitch")
        yaw = body.get("yaw")
        if body.get("relative"):
            pitch = None if pitch is None else -q[tilt_j] + float(pitch)
            yaw = None if yaw is None else q[pan_j] + float(yaw)
        pitch = -q[tilt_j] if pitch is None else float(pitch)
        yaw = q[pan_j] if yaw is None else float(yaw)
        lo_t, hi_t = c.kin._range[tilt_j]
        lo_p, hi_p = c.kin._range[pan_j]
        cp, cy = min(max(pitch, -hi_t), -lo_t), min(max(yaw, lo_p), hi_p)
        c.head_move(cp, cy)
        env = ok()
        if (cp, cy) != (pitch, yaw):
            env["note"] = f"clamped to pitch {cp:.2f}, yaw {cy:.2f} rad"
        return env

    def limits(self) -> dict:
        """``GET /v1/limits`` from the robot preset: joint travel, HOME and STOW, neck envelope, lift stroke and
        the base ceilings. Values the simulation has no counterpart for are 0 (watchdog intervals, pulses)."""
        c = self.controller
        spec, kin = c.spec, c.kin
        first = next(iter(spec.arms.values()))
        joints = [{"min_deg": round(math.degrees(kin._range[j][0]), 2), "max_deg": round(math.degrees(kin._range[j][1]), 2)}
                  for j in first.joints]

        def pose_deg(name: str) -> dict:
            out = {}
            for side, arm in self.sides.items():
                js = spec.arms[arm].joints
                q = spec.pose_arm_targets(name) if name in spec.poses else dict(zip(js, spec.arms[arm].home))
                out[side] = [round(math.degrees(q[j]), 2) for j in js]
            return out

        tilt_j, pan_j = spec.neck.get("tilt"), spec.neck.get("pan")
        lo, hi = spec.lift_range
        return {"arm": {"joints": joints, "home_deg": pose_deg("home"), "stow_deg": pose_deg("stow")},
                "chassis": {"max_linear_m_s": self.max_linear_m_s, "max_angular_rad_s": self.max_angular_rad_s,
                            "deadman_ms": 0, "cmd_vel_repeat_ms": 0},
                "neck": {"pitch_min_rad": round(-kin._range[tilt_j][1], 4), "pitch_max_rad": round(-kin._range[tilt_j][0], 4),
                         "yaw_min_rad": round(kin._range[pan_j][0], 4), "yaw_max_rad": round(kin._range[pan_j][1], 4),
                         "home_pitch_rad": 0.0, "home_yaw_rad": 0.0, "default_velocity_rad_s": 1.0,
                         "max_velocity_rad_s": 1.0},
                "slider": {"min_height_m": lo, "max_height_m": hi, "travel_pulses": 0}}

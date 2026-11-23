"""The D1 daemon's HTTP API as the bench speaks it: the command lines a model may write and the adapter the
harness calls (:class:`~omakase_bench.harness.command_api.CommandAdapter`).

The subset of the daemon's OpenAPI document the bench uses is data of the robot description
(``robots/d1/api/openapi.json``, with its provenance beside it). Joint-level commands of the ``ee``
interface are requests of that API, the same method, path and body a real robot accepts:

    joints: L 10 -20 0 -90 0 0 0 vel=0.3   ->  POST /v1/arm/a/mode        {"mode": "position", "vel_ratio": 0.3, "acc_ratio": 0.3}
                                               POST /v1/arm/a/move_joints {"joints_deg": [10, -20, ...], "wait": true}
    gripper: R 0.4 force=30                ->  POST /v1/end_effectors/b/set {"openness": 0.4, "force_percent": 30, "wait": true}
    lift: 0.15 speed=0.5                   ->  POST /v1/slider/set_height {"height_m": 0.15, "speed_ratio": 0.5, "wait": true}
    head: -0.3 0.2                         ->  POST /v1/neck/cmd {"pitch": -0.3, "yaw": 0.2, "relative": false}
    base: vel 0.1 0 0.6                    ->  POST /v1/chassis/cmd_vel {"vx": 0.1, "omega": 0}, wait 0.6 s,
                                               POST /v1/chassis/cmd_vel {"vx": 0, "omega": 0}

``base: drive <m>`` and ``base: turn <rad>`` are conveniences that expand to such ``cmd_vel`` sequences.
Every request is checked against the document before it is sent
(:func:`~omakase_bench.harness.command_api.validate`).

In simulation the requests are served by :class:`~.mirror.SimDaemon`; on hardware by the real backend's
HTTP transport (:mod:`omakase_bench.backends.real.http`).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ....harness.command_api import Request, RobotCommandAPI, load_spec, summary, validate


# --- command lines ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LineSyntax:
    """How command lines name things: arm letter -> daemon side, joints per arm, base conveniences."""
    sides: Mapping[str, str]                 # "L" -> "a"
    joints_per_arm: int = 7
    drive_speed_m_s: float = 0.1             # base: drive <m> runs at this speed
    turn_speed_rad_s: float = 0.3            # base: turn <rad> at this rate
    default_vel_ratio: float = 0.3


_NUM = r"-?\d+(?:\.\d+)?"
LINE = re.compile(r"^[ \t]*(joints|joint|gripper|lift|head|base)[ \t]*:[ \t]*([^\n]*)$", re.IGNORECASE | re.MULTILINE)


def translate(kind: str, rest: str, syn: LineSyntax) -> list[Request]:
    """The API requests of one command line (``kind: rest``); ValueError when the line is malformed."""
    kind = kind.lower()
    opts = {k.lower(): float(v) for k, v in re.findall(r"(\w+)\s*=\s*(" + _NUM + ")", rest)}
    words = re.sub(r"\w+\s*=\s*" + _NUM, " ", rest).replace("|", " | ").split()
    n = syn.joints_per_arm

    def side(word: str) -> str:
        key = word.upper()
        if key not in syn.sides:
            raise ValueError(f"unknown arm {word!r}; arms: {', '.join(syn.sides)}")
        return syn.sides[key]

    def nums(ws: Sequence[str], count: int) -> list[float]:
        vals = [float(w) for w in ws if w != "|"]
        if len(vals) != count:
            raise ValueError(f"expected {count} numbers, got {len(vals)}")
        return vals

    def mode(s: str) -> Request:
        v = opts.get("vel", syn.default_vel_ratio)
        return Request("POST", f"/v1/arm/{s}/mode", {"mode": "position", "vel_ratio": v, "acc_ratio": opts.get("acc", v)})

    if kind == "joints":
        if not words:
            raise ValueError("expected an arm and joint angles")
        if words[0].lower() == "both":
            vals = nums(words[1:], 2 * n)
            sides = list(dict.fromkeys(syn.sides.values()))
            out = [mode(s) for s in sides] if "vel" in opts else []
            return out + [Request("POST", "/v1/arm/move_joints_both",
                                  {sides[0]: vals[:n], sides[1]: vals[n:], "wait": True})]
        s = side(words[0])
        out = [mode(s)] if "vel" in opts else []
        return out + [Request("POST", f"/v1/arm/{s}/move_joints", {"joints_deg": nums(words[1:], n), "wait": True})]
    if kind == "joint":
        if len(words) != 3:
            raise ValueError("expected 'joint: <arm> <index> <deg>'")
        s = side(words[0])
        out = [mode(s)] if "vel" in opts else []
        return out + [Request("POST", f"/v1/arm/{s}/move_joint",
                              {"joint": int(words[1]), "position": float(words[2]), "wait": True})]
    if kind == "gripper":
        if len(words) != 2:
            raise ValueError("expected 'gripper: <arm> <openness>'")
        body: dict[str, Any] = {"openness": float(words[1]), "wait": True}
        if "force" in opts:
            body["force_percent"] = opts["force"]
        return [Request("POST", f"/v1/end_effectors/{side(words[0])}/set", body)]
    if kind == "lift":
        if len(words) != 1:
            raise ValueError("expected 'lift: <height_m>'")
        body = {"height_m": float(words[0]), "wait": True}
        if "speed" in opts:
            body["speed_ratio"] = opts["speed"]
        return [Request("POST", "/v1/slider/set_height", body)]
    if kind == "head":
        pitch, yaw = nums(words, 2)
        return [Request("POST", "/v1/neck/cmd", {"pitch": pitch, "yaw": yaw, "relative": False})]
    if kind == "base":
        if not words:
            raise ValueError("expected 'drive <m>', 'turn <rad>' or 'vel <vx> <omega> <s>'")
        what = words[0].lower()
        stop = Request("POST", "/v1/chassis/cmd_vel", {"vx": 0.0, "omega": 0.0})
        if what == "vel":
            vx, omega, seconds = nums(words[1:], 3)
            return [Request("POST", "/v1/chassis/cmd_vel", {"vx": vx, "omega": omega}, seconds), stop]
        if what == "drive":
            (d,) = nums(words[1:], 1)
            v = math.copysign(syn.drive_speed_m_s, d)
            return [Request("POST", "/v1/chassis/cmd_vel", {"vx": v, "omega": 0.0}, abs(d) / syn.drive_speed_m_s), stop]
        if what == "turn":
            (a,) = nums(words[1:], 1)
            w = math.copysign(syn.turn_speed_rad_s, a)
            return [Request("POST", "/v1/chassis/cmd_vel", {"vx": 0.0, "omega": w}, abs(a) / syn.turn_speed_rad_s), stop]
        raise ValueError(f"unknown base command {what!r}")
    raise ValueError(f"unknown command {kind!r}")


def run_lines(api: RobotCommandAPI, spec: Mapping, syn: LineSyntax, text: str) -> list[str]:
    """Carry out the command lines of ``text`` in order through ``api``; one feedback line per request.
    A request that does not fit the document is not sent. A line stops at its first failed request."""
    out = []
    for m in LINE.finditer(text):
        try:
            reqs = translate(m.group(1), m.group(2), syn)
        except ValueError as exc:
            out.append(f"{m.group(1).lower()}: unusable line ({exc})")
            continue
        for req in reqs:
            bad = validate(spec, req.method, req.path, req.body)
            if bad:
                out.append(f"{req.method} {req.path} -> not sent: {'; '.join(bad)}")
                break
            env = api.request(req.method, req.path, req.body)
            out.append(summary(req, env))
            if env["status"] != "ok":
                break
            if req.then_wait_s:
                try:
                    api.wait(req.then_wait_s)
                except BaseException:
                    # a velocity is held: whatever interrupts the wait, the stop that follows is still sent
                    for rest in reqs[reqs.index(req) + 1:]:
                        api.request(rest.method, rest.path, rest.body)
                    raise
    return out


# --- the preset's configuration and adapter ----------------------------------------------------------------

@dataclass(frozen=True)
class RobotAPIConfig:
    """A preset's daemon API (``robot.yaml`` ``api``): the document, the side of each arm, the base ceilings."""
    spec: Mapping
    arm_sides: Mapping[str, str]             # arm name -> daemon side
    max_linear_m_s: float
    max_angular_rad_s: float

    def syntax(self, keys: Mapping[str, str], joints_per_arm: int) -> LineSyntax:
        """``keys``: arm name -> row letter (left -> L)."""
        return LineSyntax({keys[a]: s for a, s in self.arm_sides.items()}, joints_per_arm)


def robot_api(robot) -> RobotAPIConfig | None:
    """The daemon API of a robot preset, or None when its robot.yaml names none."""
    api = robot.api
    if not api:
        return None
    ceil = api.get("base_ceilings") or {}
    return RobotAPIConfig(load_spec(str(Path(robot.preset_dir) / api["spec"])), dict(api["arm_sides"]),
                          float(ceil.get("max_linear_m_s", 0.5)), float(ceil.get("max_angular_rad_s", 0.6)))


@dataclass
class D1Adapter:
    """The D1 daemon behind the harness's command-adapter interface."""
    config: RobotAPIConfig

    def describe(self, robot, body, keys: Mapping[str, str]) -> tuple[str, list[str]]:
        cfg = self.config
        n = len(next(iter(robot.arms.values())).joints)
        sides = ", ".join(f"{keys[a]} = side {s}" for a, s in cfg.arm_sides.items())
        k = next(iter(keys.values()))
        lines = []
        if body.joints:
            lines += [
                f"'joints: {k} <{n} angles, deg> [vel=<0..1>]' = POST /v1/arm/<side>/move_joints {{joints_deg, wait: true}} "
                f"(J1 first), 'joints: both <{n} angles> | <{n} angles>' = POST /v1/arm/move_joints_both {{a, b}} (side a "
                f"first), 'joint: {k} <index 0..{n - 1}> <deg>' = POST /v1/arm/<side>/move_joint {{joint, position}}; "
                "vel=<v> first sends POST /v1/arm/<side>/mode {mode: position, vel_ratio: v, acc_ratio: v} (a fraction "
                "of full joint speed, kept for later moves; 0.3 until set). The arm moves in a straight line in joint "
                "space; targets outside a joint's travel and paths that come too close are refused (409 refused) "
                "before anything moves",
                f"'gripper: {k} <openness 0..1> [force=<0..100>]' = POST /v1/end_effectors/<side>/set {{openness, "
                "force_percent} (1 fully open, 0 closed; outside the range: 422 unprocessable); the task's opening "
                "limits still apply and the reply says when they did"]
        if body.lift:
            lo, hi = robot.lift_range
            lines.append(f"'lift: <height_m> [speed=<0..1>]' = POST /v1/slider/set_height {{height_m, speed_ratio}} "
                         f"(travel {lo:g}..{hi:g} m, outside: 400 invalid; at most {body.max_lift_step_m:g} m per line)")
        if body.head:
            lines.append("'head: <pitch_rad> <yaw_rad>' = POST /v1/neck/cmd {pitch, yaw} (pitch positive looks up, yaw "
                         "positive turns right; clamped to the neck's envelope)")
        if body.base:
            lines.append(f"'base: vel <vx m/s> <omega rad/s> <seconds>' = POST /v1/chassis/cmd_vel {{vx, omega}}, the "
                         "seconds pass, then cmd_vel {vx: 0, omega: 0} (omega positive counter-clockwise; velocities are "
                         f"clamped to {cfg.max_linear_m_s:g} m/s and {cfg.max_angular_rad_s:g} "
                         "rad/s); 'base: drive <m>' and 'base: turn <rad>' are the same at 0.1 m/s or 0.3 rad/s. One "
                         f"line drives at most {body.max_forward_m:g} m and turns at most "
                         f"{math.radians(body.max_turn_deg):.2f} rad; a longer one is refused before the base moves. "
                         "The base is differential drive and never moves sideways")
        return sides, lines

    def run(self, controller, text: str) -> list[str]:
        from ....harness.ee_control import arm_keys
        from .mirror import SimDaemon
        cfg = self.config
        sim = getattr(controller, "daemon", None)
        if sim is None:
            sim = controller.daemon = SimDaemon(controller, {s: a for a, s in cfg.arm_sides.items()},
                                                cfg.max_linear_m_s, cfg.max_angular_rad_s)
        syn = cfg.syntax(arm_keys(controller.arms), len(next(iter(controller.spec.arms.values())).joints))
        out = run_lines(sim, cfg.spec, syn, text)
        if getattr(sim, "last_base", ""):
            out.append(sim.last_base)
            sim.last_base = ""
        return out


def adapter(robot) -> D1Adapter:
    return D1Adapter(robot_api(robot))

"""End-effector rows: Cartesian fingertip targets with per-step limits, joint targets from an IK solver.

An action row gives, per arm, a target for the fingertip centre (the midpoint
of the jaw tips) in millimetres in the robot base frame, a yaw about the
vertical in degrees and a gripper command (0 open, 1 closed), or ``hold``:

    a: L 400 170 900 0 0 | R hold

The approach axis points straight down. Yaw 0 is the ``down`` orientation of
:mod:`kinematics` at the wrist angle the preset names (``yaw_zero_deg``, 0 by
default: jaws closing along the base y axis); a positive yaw is
counter-clockwise seen from above.

Each step moves the fingertip centre at most ``max_step_m`` and the yaw at most
``max_yaw_step_deg`` from the measured pose towards the target, so a far
target takes several steps and a blocked arm does not wind up. A step goal
without an IK solution leaves that arm's joint targets unchanged and is
reported, while its gripper command still applies.

The solver of the row steps is anything with :class:`IKSolver`'s ``solve_ik``:
a task names one in ``ee.ik`` (registered with
:func:`~omakase_bench.harness.kinematics.ik.register_ik_solver`); the default is the robot's own
(``robot.yaml`` ``ik.solver``). The move to the start pose is solved by the robot's solver whatever
the task names.
Limits and tolerances are an :class:`EELimits` value (task YAML ``ee.limits``);
fingertip geometry comes from the robot preset.

When the task limits a gripper's opening (task YAML ``gripper:``), 0 opens the
jaws to the task's pre-shape opening, 1 closes them down to its minimum
closing width, and the command may also be an opening in millimetres written
``w<mm>`` (``a: L 410 25 795 0 w30``). An opening outside the limits is
clamped and the next observation row says so (``gL: opening clamped to 40 mm``).

When the task allows tilted grasps (task YAML ``ee.tilt: {max_deg: N}``), an
arm's values may include a tilt ``t<deg>`` after the yaw
(``a: L 560 25 790 -45 t25 w30``): the gripper leans by that angle about its
jaw-closing axis, so the jaws stay level and the fingertips point past the
wrist, away from the robot (negative: towards it). Without the block a ``t``
value is an invalid reply, as before.
"""
from __future__ import annotations

import math
import re
import zlib
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from .kinematics.ik import IKSolver, ik_solver, solve_near
from .kinematics.model import Kinematics

if TYPE_CHECKING:
    from .session import Session

_NUM = r"-?\d+(?:\.\d+)?"
_LINE = re.compile(r"^[ \t]*(?:t\d+[^\n]*?)?\ba[ \t]*:[ \t]*([^\n]*)", re.IGNORECASE | re.MULTILINE)
_ARM = re.compile(rf"\b([A-Z])\b[ \t:=]*(hold\b|{_NUM}(?:[ \t,]+{_NUM})*)", re.IGNORECASE)
#: the same, with an opening command ``w<mm>`` allowed in place of the gripper value
_ARM_W = re.compile(rf"\b([A-Z])\b[ \t:=]*(hold\b|{_NUM}(?:[ \t,]+{_NUM})*(?:[ \t,]+w[ \t]*{_NUM})?)", re.IGNORECASE)
_WIDTH = re.compile(rf"w[ \t]*({_NUM})\s*$", re.IGNORECASE)
#: with tilted grasps allowed: an optional ``t<deg>`` after the yaw
_ARM_TW = re.compile(rf"\b([A-Z])\b[ \t:=]*(hold\b|{_NUM}(?:[ \t,]+{_NUM}){{3}}(?:[ \t,]+t[ \t]*{_NUM})?"
                     rf"(?:[ \t,]+(?:w[ \t]*{_NUM}|{_NUM}))?)", re.IGNORECASE)
_TILT = re.compile(rf"[ \t,]+t[ \t]*({_NUM})", re.IGNORECASE)


class ParseError(ValueError):
    pass


@dataclass(frozen=True)
class EELimits:
    max_step_m: float = 0.045
    max_yaw_step_deg: float = 30.0
    ik_pos_tol_m: float = 0.003
    ik_ang_tol_deg: float = 3.0
    #: a solution further than this from the measured joints (rad, any joint) is another arm
    #: configuration, not a step, and counts as unreachable
    max_joint_step_rad: float = 1.0
    step_seconds: float = 0.25
    #: share of a step over which the joint targets ramp from the previous ones
    ramp_fraction: float = 0.8

    @classmethod
    def from_mapping(cls, d: Mapping[str, Any] | None) -> "EELimits":
        known = {f.name for f in fields(cls)}
        unknown = set(d or {}) - known
        if unknown:
            raise ValueError(f"unknown ee limit(s) {sorted(unknown)}; known: {sorted(known)}")
        return cls(**{k: float(v) for k, v in (d or {}).items()})


@dataclass(frozen=True)
class ArmTarget:
    pos_mm: tuple[int, int, int]
    yaw_deg: int
    g: int
    #: jaw opening (mm) commanded with ``w<mm>`` instead of g; only where the task limits the gripper
    width_mm: int | None = None
    #: lean of the gripper about its jaw-closing axis (deg); only where the task allows tilted grasps
    tilt_deg: int = 0


#: one action row: arm -> target, or None when the arm holds
EERow = dict[str, ArmTarget | None]


def arm_keys(arms: Sequence[str]) -> dict[str, str]:
    """Row letter of each arm: the first letter of its name, upper case (left -> L)."""
    keys = {a: a[0].upper() for a in arms}
    if len(set(keys.values())) != len(keys):
        raise ValueError(f"arm names {list(arms)} do not have distinct first letters")
    return keys


def format_row(row: EERow, arms: Sequence[str]) -> str:
    """Canonical text of a row, without the leading ``a:``."""
    keys = arm_keys(arms)
    parts = []
    for arm in arms:
        t = row.get(arm)
        if t is None:
            parts.append(f"{keys[arm]} hold")
            continue
        g = t.g if t.width_mm is None else f"w{t.width_mm}"
        tilt = f" t{t.tilt_deg}" if t.tilt_deg else ""
        parts.append(f"{keys[arm]} {t.pos_mm[0]} {t.pos_mm[1]} {t.pos_mm[2]} {t.yaw_deg}{tilt} {g}")
    return " | ".join(parts)


def parse_row(body: str, arms: Sequence[str], widths: bool = False, tilt: bool = False) -> EERow:
    """Parse the part of an action line after ``a:``. An arm that is not mentioned holds.
    ``widths`` accepts an opening ``w<mm>`` as the gripper command, ``tilt`` a tilt ``t<deg>``
    after the yaw."""
    if tilt:
        return _parse_row_tilt(body, arms, widths)
    by_key = {k: a for a, k in arm_keys(arms).items()}
    row: EERow = {arm: None for arm in arms}
    seen = set()
    for m in (_ARM_W if widths else _ARM).finditer(body):
        key = m.group(1).upper()
        if key not in by_key:
            continue
        arm = by_key[key]
        if arm in seen:
            raise ParseError(f"arm {key} given twice: {body.strip()!r}")
        seen.add(arm)
        if m.group(2).lower() == "hold":
            continue
        width = _WIDTH.search(m.group(2)) if widths else None
        if width is not None:
            vals = [float(v) for v in re.findall(_NUM, m.group(2)[:width.start()])]
            if len(vals) != 4:
                raise ParseError(f"expected 'x y z yaw w<mm>' for arm {key}, got {len(vals)} values before the opening")
            if float(width.group(1)) < 0:
                raise ParseError(f"opening must not be negative, got w{width.group(1)}")
            row[arm] = ArmTarget(tuple(int(round(v)) for v in vals[:3]), int(round(vals[3])), 0,
                                 int(round(float(width.group(1)))))
            continue
        vals = [float(v) for v in re.findall(_NUM, m.group(2))]
        if len(vals) != 5:
            raise ParseError(f"expected 'x y z yaw g' for arm {key}, got {len(vals)} values")
        if vals[4] not in (0.0, 1.0):
            raise ParseError(f"gripper command must be 0 or 1, got {vals[4]:g}")
        row[arm] = ArmTarget(tuple(int(round(v)) for v in vals[:3]), int(round(vals[3])), int(vals[4]))
    if not seen:
        raise ParseError(f"no arm target in action line: {body.strip()!r}")
    return row


def _parse_row_tilt(body: str, arms: Sequence[str], widths: bool) -> EERow:
    by_key = {k: a for a, k in arm_keys(arms).items()}
    row: EERow = {arm: None for arm in arms}
    seen = set()
    for m in _ARM_TW.finditer(body):
        key = m.group(1).upper()
        if key not in by_key:
            continue
        arm = by_key[key]
        if arm in seen:
            raise ParseError(f"arm {key} given twice: {body.strip()!r}")
        seen.add(arm)
        text = m.group(2)
        if text.lower() == "hold":
            continue
        tilt = 0
        t = _TILT.search(text)
        if t is not None:
            tilt = int(round(float(t.group(1))))
            text = text[:t.start()] + text[t.end():]
        width = _WIDTH.search(text) if widths else None
        if width is not None:
            if float(width.group(1)) < 0:
                raise ParseError(f"opening must not be negative, got w{width.group(1)}")
            vals = [float(v) for v in re.findall(_NUM, text[:width.start()])]
            if len(vals) != 4:
                raise ParseError(f"expected 'x y z yaw [t<tilt>] w<mm>' for arm {key}, got {len(vals)} values")
            row[arm] = ArmTarget(tuple(int(round(v)) for v in vals[:3]), int(round(vals[3])), 0,
                                 int(round(float(width.group(1)))), tilt)
            continue
        vals = [float(v) for v in re.findall(_NUM, text)]
        if len(vals) != 5:
            raise ParseError(f"expected 'x y z yaw [t<tilt>] g' for arm {key}, got {len(vals)} values")
        if vals[4] not in (0.0, 1.0):
            raise ParseError(f"gripper command must be 0 or 1, got {vals[4]:g}")
        row[arm] = ArmTarget(tuple(int(round(v)) for v in vals[:3]), int(round(vals[3])), int(vals[4]), None, tilt)
    if not seen:
        raise ParseError(f"no arm target in action line: {body.strip()!r}")
    return row


def parse_rows(text: str, arms: Sequence[str], max_rows: int = 1, widths: bool = False,
               tilt: bool = False) -> list[str]:
    """1..max_rows action rows of a reply, each in canonical text form."""
    bodies = [m.group(1) for m in _LINE.finditer(text)]
    if not bodies:
        raise ParseError("no action rows found")
    return [format_row(parse_row(b, arms, widths, tilt), arms) for b in bodies[:max_rows]]


def _mm1(v: float) -> int:
    return int(round(v * 1000))


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def step_goal(pos: np.ndarray, yaw: float, target: ArmTarget, limits: EELimits) -> tuple[np.ndarray, float]:
    """The pose aimed at this step: at most the step limits from the measured (pos, yaw)."""
    delta = np.asarray(target.pos_mm, float) / 1000.0 - pos
    dist = float(np.linalg.norm(delta))
    goal = pos + delta * min(1.0, limits.max_step_m / dist) if dist > 0 else pos.copy()
    turn = float(np.clip(_wrap(target.yaw_deg - yaw), -limits.max_yaw_step_deg, limits.max_yaw_step_deg))
    return goal, yaw + turn


def tilt_step(tilt: float, target: ArmTarget, limits: EELimits) -> float:
    """The tilt aimed at this step: at most the yaw step limit from the measured tilt."""
    return tilt + float(np.clip(target.tilt_deg - tilt, -limits.max_yaw_step_deg, limits.max_yaw_step_deg))


def lean_axis(closing: np.ndarray) -> np.ndarray:
    """The horizontal direction a positive tilt moves the fingertips to: perpendicular to the jaw-closing
    axis, on the side away from the robot (+x; +y when the jaws close along x)."""
    n = np.array([-closing[1], closing[0], 0.0])
    norm = float(np.linalg.norm(n))
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0])
    n /= norm
    if n[0] < -1e-9 or (abs(n[0]) <= 1e-9 and n[1] < 0):
        n = -n
    return n


@dataclass(frozen=True)
class WholeBody:
    """What an ee task lets the subject move besides the arms (task YAML ``ee.whole_body``), with the
    limits of one command, and the base execution noise (``base_noise``): a drive of ``d`` metres ends
    with a lateral offset drawn from N(0, (lateral_per_m * d)^2) and a heading error from
    N(0, (heading_deg_per_m * d)^2); a turn of ``a`` degrees ends with a heading error from
    N(0, (heading_deg_per_rad * a_rad)^2). The commanded motion is what odometry reports."""
    base: bool = False
    max_forward_m: float = 0.10
    max_turn_deg: float = 15.0
    #: base speed ceilings of velocity commands, and the longest one command may run
    max_linear_m_s: float = 0.5
    max_angular_rad_s: float = 0.6
    max_duration_s: float = 2.0
    lift: bool = False
    max_lift_step_m: float = 0.05
    head: bool = False
    #: joint-space commands for the arms (7 joints each, degrees) and the grippers (openness 0..1)
    joints: bool = False
    #: commands that would bring the robot closer than this to anything else are refused
    clearance_m: float = 0.01
    lateral_per_m: float = 0.0
    heading_deg_per_m: float = 0.0
    heading_deg_per_rad: float = 0.0

    @property
    def any(self) -> bool:
        return self.base or self.lift or self.head or self.joints

    @classmethod
    def from_mapping(cls, d: Mapping[str, Any] | None, noise: Mapping[str, Any] | None = None) -> "WholeBody":
        d, noise = dict(d or {}), dict(noise or {})
        known = {f.name for f in fields(cls)}
        unknown = (set(d) | set(noise)) - known
        if unknown:
            raise ValueError(f"unknown whole_body / base_noise field(s) {sorted(unknown)}")
        return cls(**{k: (bool(v) if k in ("base", "lift", "head", "joints") else float(v))
                      for k, v in {**d, **noise}.items()})


#: spacing of the poses along a base motion that are checked for clearance before it starts
BASE_SAMPLE_M = 0.02
BASE_SAMPLE_DEG = 2.0


class EEController:
    """Executes end-effector rows on a session's backend, one row per step."""

    def __init__(self, session: "Session", limits: EELimits | None = None, ik: IKSolver | None = None):
        self.session = session
        self.spec = session.robot
        self.kin: Kinematics = session.api.kin
        named = (session.task.raw.get("ee") or {}).get("ik")
        self.ik: IKSolver = ik or (ik_solver(named, self.kin) if named else session.api.ik)
        self.limits = limits or EELimits()
        self.arms = tuple(self.spec.arms)
        q = session.backend.joint_positions()
        self._targets = {j: q[j] for a in self.spec.arms.values() for j in a.joints}
        #: per arm the last gripper command as the rows show it: 0, 1 or "w<mm>"
        self.gripper_cmd: dict[str, int | str] = {arm: 0 for arm in self.arms}
        #: per arm the commanded closed fraction of the jaws (0 open, 1 closed)
        self._closed = {arm: 0 for arm in self.arms}
        #: parallel-gripper limits of the task, per arm (``gripper:``); rows then accept w<mm>
        self.grip_limits = dict(session.gripper)
        for arm, lim in self.grip_limits.items():
            self._closed[arm] = lim.closed_fraction(lim.preshape)
        #: clamp notes of the last step, shown in the next observation row
        self.notes: list[str] = []
        #: arm -> why its last step goal was refused although IK solved it (a joint would jump)
        self.jumps: dict[str, str] = {}
        self.unreachable_total = 0
        #: largest tilt (deg) rows may command; 0: tilted grasps are not part of the task
        self.max_tilt_deg = 0.0
        #: base, torso and head commands the task allows, and the base noise model
        self.body = WholeBody()
        #: commanded base motion since the episode began: forward (m), left (m), heading (rad) in the
        #: start frame; the true pose differs by the base noise
        self.odometry = np.zeros(3)
        # one stream per episode, robot and turn: a controller is built for every turn of a multi-robot task,
        # and the same stream every turn would repeat the same drift instead of drawing it afresh
        turn = getattr(session, "turn", None)
        key = f"{session.episode.seed}/{getattr(session, 'robot_id', '')}" + (f"/{turn}" if turn else "")
        self._noise = np.random.default_rng(zlib.crc32(key.encode()))

    @property
    def widths(self) -> bool:
        """Rows may command a jaw opening ``w<mm>`` (the task limits the grippers)."""
        return bool(self.grip_limits)

    def _gripper_target(self, arm: str, t: ArmTarget) -> tuple[float, str]:
        """Closed fraction for a row's gripper command, and a clamp note ('' if none)."""
        lim = self.grip_limits.get(arm)
        if lim is None:
            return int(t.g), ""
        if t.width_mm is None:
            return lim.closed_fraction(lim.min_closing if t.g else lim.preshape), ""
        opening, note = lim.clamp_opening(t.width_mm / 1000.0)
        return lim.closed_fraction(opening), note

    # --- measured state ---------------------------------------------------------
    def _q(self) -> dict[str, float]:
        q = self.session.backend.joint_positions()
        self.kin.set_q(q)
        return q

    def _closed_fraction(self, arm: str, q: Mapping[str, float]) -> float:
        fingers = self.spec.hand_of(arm).fingers
        if not fingers:
            return 0.0
        return float(np.clip(np.mean([(q[f.joint] - f.open) / (f.closed - f.open) for f in fingers]), 0.0, 1.0))

    def _tip_offset(self, arm: str, q: Mapping[str, float]) -> float:
        lo, hi = self.spec.hand_of(arm).fingertip_offset
        return lo + (hi - lo) * self._closed_fraction(arm, q)

    def tip_pose(self, arm: str, q: Mapping[str, float] | None = None) -> tuple[np.ndarray, float]:
        """Fingertip-centre position (m, base frame) and yaw (deg) from the measured joints."""
        q = q or self._q()
        pos, rot = self.kin.tcp_pose(arm)
        closing = rot @ self.kin._closing_local[self.spec.hand_of(arm).name]
        tip = pos + self.kin.approach_axis(arm) * self._tip_offset(arm, q)
        wrist = math.degrees(math.atan2(-closing[0], closing[1]))
        return tip, _wrap(wrist - self.spec.hand_of(arm).yaw_zero_deg)

    def joints_deg(self, arm: str, q: Mapping[str, float] | None = None) -> list[int]:
        q = q or self._q()
        return [int(round(math.degrees(q[j]))) for j in self.spec.arms[arm].joints]

    def _rotation(self, arm: str, yaw_deg: float, tilt_deg: float = 0.0) -> np.ndarray:
        rot = self.kin.named_rotation(arm, "down", math.radians(yaw_deg + self.spec.hand_of(arm).yaw_zero_deg))
        if not tilt_deg:
            return rot
        hand = self.spec.hand_of(arm)
        closing = rot @ self.kin._closing_local[hand.name]
        approach = rot @ np.array(hand.approach_axis, float)
        axis = closing / np.linalg.norm(closing)
        if np.dot(np.cross(axis, approach), lean_axis(closing)) < 0:
            axis = -axis
        t = math.radians(tilt_deg)
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        turn = np.eye(3) + math.sin(t) * k + (1 - math.cos(t)) * (k @ k)
        return turn @ rot

    def tilt(self, arm: str, q: Mapping[str, float] | None = None) -> float:
        """Measured tilt (deg): the angle of the approach axis from straight down, positive when the
        fingertips lean to the side :func:`lean_axis` names."""
        if q is not None:
            self.kin.set_q(q)
        _, rot = self.kin.tcp_pose(arm)
        closing = rot @ self.kin._closing_local[self.spec.hand_of(arm).name]
        a = self.kin.approach_axis(arm)
        angle = math.degrees(math.acos(float(np.clip(-a[2], -1.0, 1.0))))
        return angle if float(np.dot(a, lean_axis(closing))) >= 0 else -angle

    # --- stepping ---------------------------------------------------------------
    def solve(self, arm: str, target: ArmTarget, q: Mapping[str, float]) -> np.ndarray | None:
        """Joint targets for one step towards ``target``, or None without an IK solution."""
        lim = self.limits
        pos, yaw = self.tip_pose(arm, q)
        goal, yaw_goal = step_goal(pos, yaw, target, lim)
        if self.max_tilt_deg:
            tilt_goal = tilt_step(self.tilt(arm), target, lim)
            rotation = self._rotation(arm, yaw_goal, tilt_goal)
            approach = rotation @ np.array(self.spec.hand_of(arm).approach_axis, float)
            tcp_goal = goal - approach * self._tip_offset(arm, q)
        else:
            tcp_goal = goal + np.array([0.0, 0.0, self._tip_offset(arm, q)])   # the TCP sits above the tips
            rotation = self._rotation(arm, yaw_goal)
        res = self.ik.solve_ik(arm, tcp_goal, rotation, q, pos_tol=lim.ik_pos_tol_m,
                               ang_tol=math.radians(lim.ik_ang_tol_deg), restarts=0)
        if not res.ok:
            return None
        sol = np.array(res.q)
        now = np.array([q[j] for j in self.spec.arms[arm].joints])
        jump = np.abs(sol - now)
        if float(np.max(jump)) > lim.max_joint_step_rad:
            k = int(np.argmax(jump))
            self.jumps[arm] = (f"would turn joint {k + 1} by {int(round(np.degrees(jump[k])))} deg in one step, "
                               "try another yaw or a nearer target")
            return None
        return sol

    def step(self, row: EERow) -> list[str]:
        """Run one row for one step. Returns the arms whose step goal had no IK solution."""
        args = {"row": format_row(row, self.arms)}
        self.session.before_api_call("ee_step", args)
        q = self._q()
        start = dict(self._targets)
        unreachable = []
        g_from = dict(self._closed)
        self.notes = []
        self.jumps = {}
        for arm in self.arms:
            t = row.get(arm)
            if t is None:
                continue
            if self.max_tilt_deg and abs(t.tilt_deg) > self.max_tilt_deg:
                lim_t = int(self.max_tilt_deg) * (1 if t.tilt_deg > 0 else -1)
                self.notes.append(f"t{arm_keys(self.arms)[arm]}: tilt clamped to {lim_t} deg")
                t = ArmTarget(t.pos_mm, t.yaw_deg, t.g, t.width_mm, lim_t)
            self.gripper_cmd[arm] = int(t.g) if t.width_mm is None else f"w{t.width_mm}"
            self._closed[arm], note = self._gripper_target(arm, t)
            if note:
                self.notes.append(f"g{arm_keys(self.arms)[arm]}: {note}")
            sol = self.solve(arm, t, q)
            if sol is None:
                unreachable.append(arm)
            else:
                self._targets.update(zip(self.spec.arms[arm].joints, sol))
        self.kin.set_q(q)
        self._ramp(start, g_from)
        self.unreachable_total += len(unreachable)
        result: dict[str, Any] = {"unreachable": unreachable}
        if self.jumps:
            result["ik"] = dict(self.jumps)
        if self.notes:
            result["gripper"] = list(self.notes)
        self.session.after_api_call("ee_step", args, result)
        return unreachable

    def arrived(self, row: EERow, pos_m: float = 0.003, deg: float = 3.0) -> bool:
        """Every moving arm of ``row`` has its fingertips at the row's target (and yaw, tilt)."""
        q = self._q()
        for arm, t in row.items():
            if t is None:
                continue
            pos, yaw = self.tip_pose(arm, q)
            if float(np.linalg.norm(np.asarray(t.pos_mm, float) / 1000.0 - pos)) > pos_m:
                return False
            if abs(_wrap(t.yaw_deg - yaw)) > deg:
                return False
            if self.max_tilt_deg and abs(t.tilt_deg - self.tilt(arm, q)) > deg:
                return False
        return True

    def step_until(self, row: EERow, max_steps: int, stall_m: float = 0.001) -> tuple[list[str], int]:
        """Repeat ``row`` for up to ``max_steps`` steps, until the arms arrive, a target has no IK
        solution, or the fingertips stop moving (less than ``stall_m`` in a step, e.g. against an
        object). Returns the unreachable arms of the last step and the steps taken."""
        moving = [a for a, t in row.items() if t is not None]

        def pose(arm: str) -> tuple[np.ndarray, float, float]:
            pos, yaw = self.tip_pose(arm)
            return pos, yaw, self.tilt(arm) if self.max_tilt_deg else 0.0

        for k in range(max(1, max_steps)):
            before = {a: pose(a) for a in moving}
            unreachable = self.step(row)
            if unreachable or k + 1 >= max_steps or self.arrived(row):
                return unreachable, k + 1
            after = {a: pose(a) for a in moving}
            if k and all(float(np.linalg.norm(after[a][0] - before[a][0])) < stall_m
                         and abs(_wrap(after[a][1] - before[a][1])) < 1.0 and abs(after[a][2] - before[a][2]) < 1.0
                         for a in moving):
                return unreachable, k + 1
        return [], max_steps

    # --- whole-body commands ---------------------------------------------------------------------------
    def _clearance(self, joints: Mapping[str, float]) -> tuple[float, str] | None:
        probe = getattr(self.session.backend, "pose_clearance", None)
        if probe is None:
            return None
        cap = max(0.05, 2 * self.body.clearance_m)
        if hasattr(self.session.backend, "prefix"):          # a robot's view of a multi-robot backend
            return probe(dict(joints), cap)
        return probe("", dict(joints), cap)

    def _refusal(self, joints: Mapping[str, float], path: Sequence[Mapping[str, float]] = ()) -> str:
        """Why the pose ``joints`` (or any pose along ``path`` ending there) is refused: a clearance under the
        margin that is also smaller than the robot has where it stands now; '' when it is accepted."""
        start = self._clearance({})
        if start is None:
            return ""
        floor = min(self.body.clearance_m, start[0] - 0.001)
        for k, pose in enumerate([*path, joints]):
            res = self._clearance(pose)
            if res[0] < floor:
                where = f" at sample {k + 1} of {len(path) + 1}" if path else ""
                return f"refused{where}: {res[1] or 'two parts'} would come within {_mm1(res[0])} mm"
        return ""

    def base_move(self, forward_m: float = 0.0, turn_deg: float = 0.0, arc: bool = False) -> str:
        """Drive the base straight by ``forward_m`` (negative: backwards) or turn it in place by ``turn_deg``
        (positive: counter-clockwise), or with ``arc`` both at once along an arc; never sideways. Checked
        before it moves: within the command limits, and clear of everything by ``body.clearance_m`` along
        the path, sampled every ``BASE_SAMPLE_M`` and ``BASE_SAMPLE_DEG``. Returns the feedback line."""
        from .api import _unicycle
        b, spec = self.body, self.spec
        if not b.base or spec.mobile_base is None:
            return "base: not available in this task"
        if forward_m and turn_deg and not arc:
            return "base: refused, drive straight or turn in place, not both in one command"
        if abs(forward_m) > b.max_forward_m + 1e-9 or abs(turn_deg) > b.max_turn_deg + 1e-9:
            return (f"base: refused, one command drives at most {b.max_forward_m:g} m or turns at most "
                    f"{math.radians(b.max_turn_deg):.2f} rad")
        mb = spec.mobile_base
        q = self.session.backend.joint_positions()
        pose = _unicycle(np.array([q[mb.x], q[mb.y], q[mb.yaw]]), forward_m, math.radians(turn_deg))
        n = max(2, int(math.ceil(abs(forward_m) / BASE_SAMPLE_M)), int(math.ceil(abs(turn_deg) / BASE_SAMPLE_DEG)))
        samples = [dict(zip((mb.x, mb.y, mb.yaw), (float(v) for v in pose(k / n)))) for k in range(1, n + 1)]
        why = self._refusal(samples[-1], samples[:-1])
        if why:
            return "base: " + why
        self.session.before_api_call("base_move", {"forward_m": forward_m, "turn_deg": turn_deg})
        api = self.session.api
        # the undecorated function: this command is logged once, as base_move, after the noise is applied
        out = type(api).move_base.__wrapped__(api, forward_m=forward_m, turn_deg=turn_deg)
        moved = abs(forward_m)
        sd_lat = b.lateral_per_m * moved
        sd_yaw = math.radians(b.heading_deg_per_m * moved + b.heading_deg_per_rad * abs(math.radians(turn_deg)))
        if not out.get("bumped_into") and (sd_lat or sd_yaw):
            q = self.session.backend.joint_positions()
            x1, y1, yaw1 = q[mb.x], q[mb.y], q[mb.yaw]
            lat, dyaw = float(self._noise.normal(0, sd_lat)) if sd_lat else 0.0, \
                float(self._noise.normal(0, sd_yaw)) if sd_yaw else 0.0
            target = {mb.x: x1 - math.sin(yaw1) * lat, mb.y: y1 + math.cos(yaw1) * lat, mb.yaw: yaw1 + dyaw}
            self.session.advance(0.3, lambda t: self.session.command_joints(target))
        if not out.get("bumped_into"):
            o = self.odometry
            o[0] += math.cos(o[2]) * forward_m
            o[1] += math.sin(o[2]) * forward_m
            o[2] += math.radians(turn_deg)
        self.session.after_api_call("base_move", {"forward_m": forward_m, "turn_deg": turn_deg}, out)
        if out.get("bumped_into"):
            return f"base: bumped into {', '.join(out['bumped_into'])} and backed off"
        parts = ([f"drove {forward_m:.3f} m"] if forward_m else []) + (
            [f"turned {math.radians(turn_deg):.2f} rad"] if turn_deg else [])
        return "base: " + (", ".join(parts) or "stayed")

    def lift_move(self, target_m: float, speed_ratio: float = 1.0) -> str:
        """Move the torso lift to ``target_m``, at most ``body.max_lift_step_m`` from where it is, checked
        for clearance at the target first."""
        spec = self.spec
        if not self.body.lift or not spec.lift_joint:
            return "lift: not available in this task"
        lo, hi = spec.lift_range
        now = self.session.backend.joint_positions()[spec.lift_joint]
        if not lo - 1e-9 <= target_m <= hi + 1e-9:
            return f"lift: refused, the range is {lo:g}..{hi:g} m"
        if abs(target_m - now) > self.body.max_lift_step_m + 1e-6:
            return f"lift: refused, one command moves the lift at most {self.body.max_lift_step_m:g} m"
        why = self._refusal({spec.lift_joint: target_m})
        if why:
            return "lift: " + why
        self.session.before_api_call("lift_move", {"lift_m": target_m})
        self.session.api._joint_move([spec.lift_joint], [target_m], spec.lift_speed * max(0.05, min(1.0, speed_ratio)))
        self.session.after_api_call("lift_move", {"lift_m": target_m}, {})
        return f"lift: at {target_m:.3f} m"

    def head_move(self, pitch_rad: float, yaw_rad: float) -> str:
        """Point the head: ``pitch_rad`` positive looks up, ``yaw_rad`` positive turns right."""
        spec = self.spec
        if not self.body.head or not spec.neck:
            return "head: not available in this task"
        targets = {}
        for key, v in (("tilt", -pitch_rad), ("pan", yaw_rad)):
            j = spec.neck.get(key)
            if j is None:
                continue
            lo, hi = self.kin._range[j]
            if not lo - 1e-9 <= v <= hi + 1e-9:
                name, (a, b) = ("pitch", (-hi, -lo)) if key == "tilt" else ("yaw", (lo, hi))
                return f"head: refused, {name} must be in {a:.2f}..{b:.2f} rad"
            targets[j] = v
        args = {"pitch_rad": pitch_rad, "yaw_rad": yaw_rad}
        self.session.before_api_call("head_move", args)
        self.session.api._joint_move(list(targets), list(targets.values()), 1.0)
        self.session.after_api_call("head_move", args, {})
        return f"head: pitch {pitch_rad:.2f} yaw {yaw_rad:.2f} rad"

    def base_velocity(self, vx: float, omega: float, seconds: float) -> str:
        """Drive at ``vx`` m/s forward and ``omega`` rad/s counter-clockwise for ``seconds``, then stop: an arc,
        a straight drive or a turn in place, never sideways."""
        b = self.body
        if not b.base or self.spec.mobile_base is None:
            return "base: not available in this task"
        if abs(vx) > b.max_linear_m_s + 1e-9 or abs(omega) > b.max_angular_rad_s + 1e-9 or not 0 < seconds <= b.max_duration_s:
            return (f"base: refused, |vx| at most {b.max_linear_m_s:g} m/s, |omega| at most {b.max_angular_rad_s:g} rad/s, "
                    f"duration 0-{b.max_duration_s:g} s")
        return self.base_move(vx * seconds, math.degrees(omega * seconds), arc=True)

    def joints_move(self, targets: Mapping[str, float], vel_ratio: float = 0.5) -> str:
        """Move arm joints (rad) in a straight joint-space path at ``vel_ratio`` of the arm's speed, after
        checking the targets against the joint limits and the path, sampled every 2 degrees per joint,
        against the clearance margin. Gripper limits apply as for every command."""
        if not self.body.joints:
            return "joints: not available in this task"
        if not 0 < vel_ratio <= 1:
            return "joints: refused, vel must be in 0..1"
        for j, v in targets.items():
            lo, hi = self.kin._range[j]
            if not lo - 1e-9 <= v <= hi + 1e-9:
                return (f"joints: refused, {j} target {math.degrees(v):.1f} deg is outside its travel "
                        f"{math.degrees(lo):.0f}..{math.degrees(hi):.0f} deg")
        q = self.session.backend.joint_positions()
        names = list(targets)
        now = np.array([q[j] for j in names])
        end = np.array([targets[j] for j in names])
        n = max(1, int(math.ceil(float(np.max(np.abs(end - now))) / math.radians(2.0))))
        path = [dict(zip(names, now + (end - now) * (k / n))) for k in range(1, n)]
        why = self._refusal(dict(zip(names, end)), path)
        if why:
            return "joints: " + why
        args = {"joints_deg": [round(math.degrees(v), 1) for v in end], "vel": vel_ratio}
        self.session.before_api_call("joints_move", args)
        speed = min(a.max_joint_speed for a in self.spec.arms.values()) * vel_ratio
        self.session.api._joint_move(names, end, speed)
        self._targets.update(zip(names, end))
        self.session.after_api_call("joints_move", args, {})
        return f"joints: moved {len(names)} joint(s)"

    def gripper_set(self, arm: str, openness: float, force_percent: float | None = None) -> str:
        """Open the arm's gripper to ``openness`` (1 fully open, 0 closed) of its hardware stroke; task limits
        clamp it, with a note. ``force_percent`` caps the force (0..100 % of the hardware force)."""
        if not self.body.joints:
            return "gripper: not available in this task"
        if not 0 <= openness <= 1:
            return "gripper: refused, openness must be in 0..1"
        if force_percent is not None and not 0 <= force_percent <= 100:
            return "gripper: refused, force must be in 0..100 %"
        hand = self.spec.hand_of(arm)
        note = ""
        lim = self.grip_limits.get(arm)
        opening = openness * hand.max_opening
        if lim is not None:
            opening, note = lim.clamp_opening(opening)
            self._closed[arm] = lim.closed_fraction(opening)
        else:
            self._closed[arm] = 1.0 - openness
        self.gripper_cmd[arm] = f"w{_mm1(opening)}"
        if force_percent is not None:
            setter = getattr(self.session.backend, "set_gripper_force", None)
            if setter is None:
                return "gripper: refused, this robot cannot cap the grip force"
            cap = hand.force * force_percent / 100.0
            if lim is not None and lim.max_force is not None:
                cap = min(cap, lim.max_force)
            setter(arm, cap)
        self.session.before_api_call("gripper_set", {"arm": arm, "openness": openness})
        self._ramp(dict(self._targets), {**self._closed})
        self.session.after_api_call("gripper_set", {"arm": arm, "openness": openness}, {})
        return f"gripper {arm}: opening {_mm1(opening)} mm" + (f" ({note})" if note else "")

    def hold(self) -> None:
        """One step with the current targets (an unusable reply)."""
        self.session.before_api_call("ee_step", {"row": "hold"})
        self.notes = []
        self._ramp(dict(self._targets), dict(self._closed))
        self.session.after_api_call("ee_step", {"row": "hold"}, {"unreachable": []})

    def _ramp(self, start: Mapping[str, float], g_from: Mapping[str, float]) -> None:
        lim = self.limits
        ramp = max(lim.step_seconds * lim.ramp_fraction, 1e-6)
        dt = self.session.control_dt

        def on_tick(t: float) -> None:
            k = min(1.0, (t + dt) / ramp)
            cmd = {j: start[j] + k * (v - start[j]) for j, v in self._targets.items()}
            for arm in self.arms:
                g = g_from[arm] + k * (self._closed[arm] - g_from[arm])
                for f in self.spec.hand_of(arm).fingers:
                    cmd[f.joint] = f.open + g * (f.closed - f.open)
            self.session.command_joints(cmd)

        self.session.advance(lim.step_seconds, on_tick)

    def move_to_start(self, start: Mapping[str, Sequence[float]], staged: bool = False) -> None:
        """Bring each arm to its start pose (TCP x, y, z in m and yaw in deg) in joint space. With
        ``staged`` the joints the preset names in ``leave_home_first`` reach their targets first and
        the others follow, so that an arm leaving the robot's home lifts off what stands below it
        before it turns (robot.yaml ``arms.<arm>.leave_home_first``)."""
        api = self.session.api
        for arm, (x, y, z, *yaw) in start.items():
            q = self._q()
            rotation = self._rotation(arm, yaw[0] if yaw else 0.0)
            # the staged start (new tasks) takes the solution nearest the current joints, so that no
            # joint turns by half a revolution; the single move keeps its published behaviour
            robot_ik = self.session.api.ik
            res = (solve_near(robot_ik, self.kin, arm, (x, y, z), rotation, q) if staged
                   else robot_ik.solve_ik(arm, (x, y, z), rotation, q))
            if not res.ok:
                raise ValueError(f"start pose of the {arm} arm is unreachable: {res.reason}")
            a = self.spec.arms[arm]
            at_home = all(abs(q[j] - v) < 0.02 for j, v in zip(a.joints, a.home))
            first = [i for i, j in enumerate(a.joints) if j in a.leave_home_first] if staged and at_home else []
            if first:
                now, end = np.array([q[j] for j in a.joints]), np.array(res.q)
                mid = now.copy()
                mid[first] = end[first]
                durations = [max(0.2, float(np.max(np.abs(b - c))) / a.max_joint_speed) for c, b in ((now, mid), (mid, end))]
                api._run_joint_path(list(a.joints), [now, mid, end], durations)
            else:
                api._joint_move(list(a.joints), res.q, a.max_joint_speed)
            self._targets.update(zip(a.joints, res.q))

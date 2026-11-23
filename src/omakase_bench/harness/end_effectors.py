"""Per-task end-effector limits: a schema per end-effector type, validated against the robot preset.

A task may narrow what its end effectors do (task YAML ``gripper:``); the
robot preset (``robot.yaml``) keeps the hardware range. The block holds the
fields of the limit schema of each arm's end-effector type, with optional
per-arm overrides under ``arms``::

    gripper:
      max_opening_mm: 40      # parallel_gripper fields
      arms:
        right: {max_opening_mm: 30}

Each end-effector type registers a parser (:func:`register_limit_type`) that
turns the merged mapping into an :class:`EndEffectorLimits` value and rejects
values outside the hardware range. The session applies
:meth:`EndEffectorLimits.clamp_joint_targets` to every joint command it sends,
so no interface can command past a limit; interfaces use the type's own
methods (for a parallel gripper, openings in metres) for their feedback.

A task without the block gets no limits and behaves exactly as before.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from ..robots.spec import EndEffector, RobotSpec

#: a clamp smaller than this (m of jaw opening) is rounding, not a limit, and is not reported
REPORT_TOLERANCE = 0.0005


class EndEffectorLimits(Protocol):
    arm: str
    #: cap on each finger actuator's force (N), applied through the backend's
    #: ``gripper_force_limit`` capability; None leaves the hardware value
    max_force: float | None

    def clamp_joint_targets(self, ee: EndEffector, targets: Mapping[str, float]) -> tuple[dict[str, float], list[str]]:
        """``targets`` with this end effector's joints inside the limits, and a note per clamp."""
        ...

    def start_targets(self, ee: EndEffector) -> dict[str, float]:
        """Joint targets the end effector takes before the subject starts (may be empty)."""
        ...

    def grip_targets(self, ee: EndEffector, close: bool) -> dict[str, float]:
        """Joint targets of the robot API's open (``close`` False) and close commands."""
        ...

    def grip_note(self, ee: EndEffector, close: bool) -> str:
        """What the limits change about that command, for the call's result ('' = nothing)."""
        ...

    def describe(self) -> str:
        """One sentence stating the limits to a model."""
        ...

    def to_json(self) -> dict[str, Any]: ...


#: end-effector type (``robot.yaml`` ``type``) -> parser(fields, hardware, arm, where)
LIMIT_TYPES: dict[str, Callable[[Mapping[str, Any], EndEffector, str, str], EndEffectorLimits]] = {}


def register_limit_type(ee_type: str, parser: Callable[[Mapping[str, Any], EndEffector, str, str], EndEffectorLimits]) -> None:
    LIMIT_TYPES[ee_type] = parser


def _mm(v: float) -> int:
    return int(round(v * 1000))


@dataclass(frozen=True)
class ParallelJawLimits:
    """Opening range of a parallel gripper for one task, in metres of pad-face distance.

    ``max_opening``: the widest the jaws may open. ``min_closing``: the narrowest they close to
    (a close command stops there). ``preshape``: the opening that open commands go to and that
    the episode starts with. ``max_force``: cap on each jaw's actuator force (N), or None.
    """
    arm: str
    hardware_opening: float
    max_opening: float
    min_closing: float
    preshape: float
    max_force: float | None = None

    # --- opening <-> finger joints (the jaw opening is linear in the finger coordinate) ---------
    def closed_fraction(self, opening: float) -> float:
        return 1.0 - opening / self.hardware_opening

    def opening_of(self, ee: EndEffector, joint: str, value: float) -> float:
        f = next(f for f in ee.fingers if f.joint == joint)
        return self.hardware_opening * (1.0 - (value - f.open) / (f.closed - f.open))

    def clamp_opening(self, opening: float) -> tuple[float, str]:
        """``opening`` inside [min_closing, max_opening] and a note if it was outside."""
        if opening > self.max_opening + REPORT_TOLERANCE:
            return self.max_opening, f"opening clamped to {_mm(self.max_opening)} mm"
        if opening < self.min_closing - REPORT_TOLERANCE:
            return self.min_closing, f"closing clamped to {_mm(self.min_closing)} mm"
        return min(max(opening, self.min_closing), self.max_opening), ""

    def finger_targets(self, ee: EndEffector, opening: float) -> dict[str, float]:
        frac = self.closed_fraction(opening)
        return {f.joint: f.open + frac * (f.closed - f.open) for f in ee.fingers}

    def clamp_joint_targets(self, ee: EndEffector, targets: Mapping[str, float]) -> tuple[dict[str, float], list[str]]:
        out, notes = dict(targets), []
        for f in ee.fingers:
            if f.joint not in out:
                continue
            opening, note = self.clamp_opening(self.opening_of(ee, f.joint, out[f.joint]))
            out[f.joint] = f.open + self.closed_fraction(opening) * (f.closed - f.open)
            if note and note not in notes:
                notes.append(note)
        return out, notes

    def start_targets(self, ee: EndEffector) -> dict[str, float]:
        return self.finger_targets(ee, self.preshape)

    def grip_targets(self, ee: EndEffector, close: bool) -> dict[str, float]:
        return self.finger_targets(ee, self.min_closing if close else self.preshape)

    def grip_note(self, ee: EndEffector, close: bool) -> str:
        if close:
            return f"closing stops at {_mm(self.min_closing)} mm (task limit)" if self.min_closing > REPORT_TOLERANCE else ""
        if self.preshape < self.hardware_opening - REPORT_TOLERANCE:
            return f"opening clamped to {_mm(self.preshape)} mm (task limit)"
        return ""

    def describe(self) -> str:
        text = (f"the {self.arm} gripper opens at most {_mm(self.max_opening)} mm and closes to no less than "
                f"{_mm(self.min_closing)} mm; an open command opens it to {_mm(self.preshape)} mm")
        if self.max_force is not None:
            text += f"; each jaw pushes with at most {self.max_force:g} N"
        return text

    def to_json(self) -> dict[str, Any]:
        out = {"type": "parallel_gripper", "max_opening_mm": _mm(self.max_opening),
               "min_closing_mm": _mm(self.min_closing), "preshape_mm": _mm(self.preshape)}
        if self.max_force is not None:
            out["max_force_n"] = self.max_force
        return out


PARALLEL_FIELDS = ("max_opening_mm", "min_closing_mm", "preshape_mm", "max_force_n")


def parse_parallel_jaw(d: Mapping[str, Any], ee: EndEffector, arm: str, where: str) -> ParallelJawLimits:
    unknown = set(d) - set(PARALLEL_FIELDS)
    if unknown:
        raise ValueError(f"{where}: unknown parallel_gripper limit(s) {sorted(unknown)}; known: {list(PARALLEL_FIELDS)}")
    hw = ee.max_opening
    if hw <= 0 or not ee.fingers:
        raise ValueError(f"{where}: end effector {ee.name} declares no fingers or max_opening, so it has no opening to limit")
    max_open = float(d.get("max_opening_mm", hw * 1000)) / 1000
    min_close = float(d.get("min_closing_mm", 0.0)) / 1000
    preshape = float(d.get("preshape_mm", max_open * 1000)) / 1000
    errors = []
    if not 0 < max_open <= hw + 1e-9:
        errors.append(f"max_opening_mm {_mm(max_open)} outside the hardware range 0..{_mm(hw)} mm")
    if not 0 <= min_close < max_open:
        errors.append(f"min_closing_mm {_mm(min_close)} must be at least 0 and below max_opening_mm {_mm(max_open)}")
    if not min_close <= preshape <= max_open:
        errors.append(f"preshape_mm {_mm(preshape)} outside {_mm(min_close)}..{_mm(max_open)} mm")
    force = d.get("max_force_n")
    if force is not None:
        force = float(force)
        if not 0 < force <= ee.force:
            errors.append(f"max_force_n {force:g} outside the hardware range 0..{ee.force:g} N")
    if errors:
        raise ValueError(f"{where} ({arm} arm, {ee.name}): " + "; ".join(errors))
    return ParallelJawLimits(arm, hw, max_open, min_close, preshape, force)


register_limit_type("parallel_gripper", parse_parallel_jaw)


def parse_task_limits(block: Mapping[str, Any] | None, robot: RobotSpec, where: str = "gripper") -> dict[str, EndEffectorLimits]:
    """Per-arm limits of a task's ``gripper:`` block; empty without a block."""
    if not block:
        return {}
    if not isinstance(block, Mapping):
        raise ValueError(f"{where}: expected a mapping, got {type(block).__name__}")
    shared = {k: v for k, v in block.items() if k != "arms"}
    per_arm = block.get("arms") or {}
    unknown_arms = set(per_arm) - set(robot.arms)
    if unknown_arms:
        raise ValueError(f"{where}.arms: unknown arm(s) {sorted(unknown_arms)}; arms: {list(robot.arms)}")
    out: dict[str, EndEffectorLimits] = {}
    for arm in robot.arms:
        ee = robot.hand_of(arm)
        parser = LIMIT_TYPES.get(ee.type)
        if parser is None:
            raise ValueError(f"{where}: no limit schema registered for end-effector type {ee.type!r} "
                             f"({arm} arm); registered: {sorted(LIMIT_TYPES)}")
        out[arm] = parser({**shared, **(per_arm.get(arm) or {})}, ee, arm, where)
    return out


def describe_limits(limits: Mapping[str, EndEffectorLimits]) -> str:
    """The task's limits as one sentence for a model, or '' without limits."""
    if not limits:
        return ""
    return "End-effector limits of this task: " + "; ".join(lim.describe() for lim in limits.values()) + "."


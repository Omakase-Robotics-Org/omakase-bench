"""Tasks with several robots: who they are, how they take turns, and the referee interface.

A task declares its robots in a ``robots:`` list instead of ``scene.robot``::

    robots:
      - id: a                      # names the robot's joints, cameras and contacts ("a/...")
        preset: d1
        base: [0, 0, 0]            # x, y (m), yaw (deg) in the world
        start_pose: home           # named pose of the preset at reset (default: its home)
        idle_pose: side_idle       # held while another robot acts (default: the start pose)
        return_via: ee_start       # back to the idle pose through the ee start pose (default: direct)
        lift: 0.15                 # optional per robot: lift, neck, gripper, ee, cameras,
        agent: {model: scripted/oracle, interface: scripted}     # instruction, agent
      - id: b
        preset: d1
        base: [0.95, 0, 180]
    schedule:
      mode: turns                  # turns | simultaneous
      max_turns: 4
      turn: {sim_seconds: 40, model_calls: 30, api_calls: 60}   # caps per turn
    referee: omakase_bench.tasks.multi_robot.block_relay.relay:RelayReferee
    faults: {inter_robot_collision: fail}                      # fail (default) | report

Each robot is driven by its own subject through its own session: it sees its
own joints and cameras only, and receives the shared observations the referee
gives it (``Ruling.messages``) as part of its instruction. A referee is a
class named by ``module:Class``; one instance is built per episode with the
episode's values and the task's ``params``.

    turns         the robot the referee names acts while the others hold their
                  idle pose. The harness moves the acting robot from its idle
                  pose to its start pose before the turn and back afterwards
                  (transfers, checked for contacts). A turn ends when its subject
                  returns, when a per-turn cap is reached, or when
                  ``Referee.turn_over`` says so; the referee then rules on it
                  (``after_turn``) and names the next robot, or ends the game.
    simultaneous  every robot's subject runs at once in lockstep: simulated time
                  advances one control tick when every running subject waits for
                  it, so a robot whose model is thinking holds the world. Idle
                  poses are not used.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from .backend import OracleSnapshot

SCHEDULE_MODES = ("turns", "simultaneous")
FAULT_POLICIES = ("fail", "report")
#: fault categories a task may set a policy for
FAULTS = ("inter_robot_collision", "transfer_collision")


@dataclass(frozen=True)
class TurnBudget:
    """Caps of one turn; ``None`` means only the episode's budget applies."""
    sim_seconds: float | None = None
    model_calls: int | None = None
    api_calls: int | None = None


@dataclass(frozen=True)
class Schedule:
    mode: str = "turns"
    max_turns: int = 0              # 0: one turn per robot
    turn: TurnBudget = field(default_factory=TurnBudget)


@dataclass(frozen=True)
class RobotSlot:
    """One robot of a task: its preset and the per-robot overrides of the task's blocks."""
    id: str
    preset: str
    gripper: Mapping[str, Any] = field(default_factory=dict)
    ee: Mapping[str, Any] = field(default_factory=dict)
    cameras: tuple[str, ...] | None = None
    instruction: str = ""
    agent: Mapping[str, str] = field(default_factory=dict)
    #: named poses: at reset ('' = the preset's home) and while another robot acts ('' = start)
    start_pose: str = ""
    idle_pose: str = ""
    #: "ee_start": after its turn the robot's arms pass through the task's ee start pose on the way to
    #: the idle pose (the reverse of the move that brought them from the idle pose); '' = direct
    return_via: str = ""


@dataclass(frozen=True)
class TeamSpec:
    robots: tuple[RobotSlot, ...]
    schedule: Schedule = field(default_factory=Schedule)
    referee: str = ""
    faults: Mapping[str, str] = field(default_factory=dict)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(r.id for r in self.robots)

    def slot(self, rid: str) -> RobotSlot:
        return next(r for r in self.robots if r.id == rid)

    def fault_policy(self, kind: str) -> str:
        return self.faults.get(kind, "fail")


def parse_team(raw: Mapping[str, Any], where: str) -> TeamSpec:
    """The ``robots``, ``schedule``, ``referee`` and ``faults`` blocks of a task file."""
    slots = []
    for d in raw["robots"]:
        rid = str(d["id"])
        if not rid or not all(c.isalnum() or c == "_" for c in rid):
            raise ValueError(f"{where}: robot id {rid!r} must be letters, digits or _")
        cams = d.get("cameras")
        slots.append(RobotSlot(rid, d.get("preset", "d1"), d.get("gripper") or {}, d.get("ee") or {},
                               tuple(cams) if cams is not None else None, str(d.get("instruction", "")).strip(),
                               dict(d.get("agent") or {}), str(d.get("start_pose") or d.get("home") or ""),
                               str(d.get("idle_pose") or ""), str(d.get("return_via") or "")))
        if slots[-1].return_via not in ("", "ee_start"):
            raise ValueError(f"{where}: robot {rid}: return_via must be 'ee_start' or omitted")
    ids = [s.id for s in slots]
    if len(set(ids)) != len(ids) or len(ids) < 1:
        raise ValueError(f"{where}: robot ids must be unique, got {ids}")
    sch = raw.get("schedule") or {}
    mode = sch.get("mode", "turns")
    if mode not in SCHEDULE_MODES:
        raise ValueError(f"{where}: schedule.mode must be one of {SCHEDULE_MODES}, got {mode!r}")
    turn = sch.get("turn") or {}
    unknown = set(turn) - {"sim_seconds", "model_calls", "api_calls"}
    if unknown:
        raise ValueError(f"{where}: unknown schedule.turn field(s) {sorted(unknown)}")
    unknown = set(sch) - {"mode", "max_turns", "turn"}
    if unknown:
        raise ValueError(f"{where}: unknown schedule field(s) {sorted(unknown)}")
    schedule = Schedule(mode, int(sch.get("max_turns", 0)),
                        TurnBudget(turn.get("sim_seconds"), turn.get("model_calls"), turn.get("api_calls")))
    faults = dict(raw.get("faults") or {})
    for k, v in faults.items():
        if k not in FAULTS or v not in FAULT_POLICIES:
            raise ValueError(f"{where}: faults.{k}: {v!r}; categories {FAULTS}, policies {FAULT_POLICIES}")
    return TeamSpec(tuple(slots), schedule, str(raw.get("referee", "")), faults)


def placement_base(d: Mapping[str, Any]) -> tuple[float, float, float]:
    b = d.get("base", [0, 0, 0])
    return (float(b[0]), float(b[1]), math.radians(float(b[2])))


# --- inter-robot contacts -----------------------------------------------------------------------

def robot_of(part: str, ids: Sequence[str]) -> str:
    """The robot a contact part such as ``b/left_pad0`` belongs to, or '' for scene elements."""
    head, sep, _ = part.partition("/")
    return head if sep and head in ids else ""


def inter_robot_contacts(snap: OracleSnapshot, ids: Sequence[str]) -> list[tuple[str, str]]:
    """Pairs of parts of two different robots that touch now or have touched since reset (each
    pair once)."""
    out = set()
    for a, b in snap.contacts | snap.contacts_ever:
        ra, rb = robot_of(a, ids), robot_of(b, ids)
        if ra and rb and ra != rb:
            out.add(tuple(sorted((a, b))))
    return sorted(out)


# --- referee interface --------------------------------------------------------------------------

@dataclass(frozen=True)
class TurnReport:
    """What happened in one turn, for the referee."""
    turn: int
    robot: str
    ended: str                      # finished | turn_end:<why> | budget_exceeded:<what> | error:<what>
    before: OracleSnapshot
    after: OracleSnapshot
    sim_start: float
    sim_end: float


@dataclass(frozen=True)
class Ruling:
    """A referee's decision at the start and after every turn.

    ``next_robot``: who acts next (None ends the game). ``messages``: robot -> text added to that
    robot's instruction from its next turn on (the shared observations, e.g. the move history).
    ``restore``: object -> (position, quaternion w x y z) put back before the next turn, logged.
    ``valid``/``note``: the referee's verdict on the turn just played, recorded per turn.
    ``stop``: a non-empty exit reason ends the episode as a harness stop (the turn was cut short by a
    budget or an error, not by the game); it becomes the episode's ``exit_reason``."""
    next_robot: str | None = None
    messages: Mapping[str, str] = field(default_factory=dict)
    restore: Mapping[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = \
        field(default_factory=dict)
    valid: bool = True
    note: str = ""
    stop: str = ""


@dataclass(frozen=True)
class GameResult:
    """The referee's final verdict. ``robots``: per robot, the referee's own metrics."""
    success: bool
    evidence: str
    outcome: str = ""
    robots: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


@dataclass
class GameView:
    """What a referee may read: the robots, the episode's values and params, the turns so far and
    the robots that can no longer act (their own budget ran out)."""
    robots: tuple[str, ...]
    values: Mapping[str, Any]
    params: Mapping[str, Any]
    history: list[tuple[TurnReport, Ruling]] = field(default_factory=list)
    out: set[str] = field(default_factory=set)
    #: live readings for clocks: "sim_time" (callable), "usage" (robot -> its Usage, model_seconds = the
    #: model calls' wall-clock latency), "turn_start" (robot -> (sim time, model seconds) when its
    #: current or last turn began: when its subject called ``session.ready()`` after the move to the
    #: task's start pose, else when the subject was started); "idle_error" (callable: robot -> (largest
    #: arm joint error in rad, lift error in m) from its idle pose now)
    live: dict[str, Any] = field(default_factory=dict)

    @property
    def turn(self) -> int:
        return len(self.history)


class Referee(Protocol):
    """Task-provided rules of a turn-based task (``referee: module:Class`` in the task YAML).
    Built per episode as ``Class(values, params)``; it reads only oracle snapshots."""

    def start(self, game: GameView, snap: OracleSnapshot) -> Ruling: ...

    def turn_over(self, game: GameView, robot: str, snap: OracleSnapshot) -> bool:
        """Polled after every robot API call or ee step of the acting robot: its turn has ended."""
        ...

    # optional: def hear(self, game, robot, keyword, text, snap) -> str
    #   a line an agent addressed to the referee (ee task block ``say:``, e.g. ``move: e4``); the
    #   answer is shown to that agent in its next prompt. turn_over is polled right after.

    def after_turn(self, game: GameView, report: TurnReport) -> Ruling: ...

    # optional: def progress(self) -> dict
    #   the game so far, written to the episode's progress.json after every ruled turn (with the turns
    #   and each robot's usage), so that an episode that dies can be resumed from its last complete turn.

    def result(self, game: GameView, snap: OracleSnapshot) -> GameResult: ...

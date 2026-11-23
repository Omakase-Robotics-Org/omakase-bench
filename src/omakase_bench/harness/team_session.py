"""One episode of a task with several robots: the shared world, a session per robot, the schedule.

    world = World(backend, ...)                 # one simulation holding every robot
    view = RobotView(world, "a", robot)         # robot a's window: its joints, cameras, contacts
    session = RobotSession(world, "a", ...)     # a Session over that view: budgets, API, limits
    run_team_episode(...)                       # turns or simultaneous, referee, faults, scores

Every joint target of robot ``a`` still leaves through ``session.command_joints``
(its gripper limits apply) and reaches the backend with ``a/`` prefixed names.
In simultaneous mode each subject runs in its own thread; every backend call is
carried out on the thread that runs the episode, and simulated time advances
one control tick when every running subject waits for it.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..core.backend import SIMULATOR, BackendInfo, OracleSnapshot
from .session import Recorder, RunBudget, Session, StopEpisode, Usage, _short
from ..core.task import Episode
from ..core.team import GameResult, GameView, Ruling, TeamSpec, TurnReport, inter_robot_contacts

#: exit reasons that end the whole episode, not only one robot's turn or participation
EPISODE_ENDS = ("success", "budget_exceeded:sim_seconds", "budget_exceeded:run_model_calls",
                "budget_exceeded:run_usd", "fault:inter_robot_collision")


class World:
    """The shared simulation of a multi-robot episode: time, recording, faults, the stop flag."""

    def __init__(self, backend, team: TeamSpec, recorder: Recorder, sim_seconds: float):
        self.backend = backend
        self.team = team
        self.recorder = recorder
        self.sim_seconds = sim_seconds
        self.info: BackendInfo = backend.info()
        self.control_dt = self.info.control_dt
        self.trace: list[dict[str, Any]] = []
        self.t0_wall = time.monotonic()
        self.stop: StopEpisode | None = None
        self.faults: list[dict[str, Any]] = []
        self._seen_contacts: set[tuple[str, str]] = set()
        self._main = threading.get_ident()
        self._lockstep: _Lockstep | None = None
        #: the robot whose turn it is (turns schedule)
        self.acting: str | None = None
        self._trace_fh = None

    def stream_trace(self, path) -> None:
        """Append every trace entry to ``path`` as it is logged, so that a run that dies keeps its trace.
        The file is flushed on every entry and synced to disk at the end of every turn."""
        self._trace_fh = open(path, "w")

    def sync_trace(self, close: bool = False) -> None:
        if self._trace_fh is not None:
            self._trace_fh.flush()
            os.fsync(self._trace_fh.fileno())
            if close:
                self._trace_fh.close()
                self._trace_fh = None

    # --- calls into the backend --------------------------------------------------------------
    def call(self, fn: Callable, *args):
        """Run ``fn(*args)`` on the episode's thread (the backend is not thread-safe)."""
        if self._lockstep is None or threading.get_ident() == self._main:
            return fn(*args)
        return self._lockstep.request(fn, args)

    def tick(self, robot: str) -> None:
        """One control tick for ``robot``; in lockstep it waits until every running robot asks."""
        if self.stop is not None:
            raise StopEpisode(self.stop.reason, self.stop.detail)
        if self._lockstep is None or threading.get_ident() == self._main:
            self._world_tick()
        else:
            self._lockstep.arrive(robot)
        if self.stop is not None:
            raise StopEpisode(self.stop.reason, self.stop.detail)

    def _world_tick(self) -> None:
        self.backend.step(self.control_dt)
        t = self.backend.sim_time()
        self.recorder.maybe_capture(self.backend, t)
        if t > self.sim_seconds and self.stop is None:
            self.stop = StopEpisode("budget_exceeded:sim_seconds", f"{t:.1f} s simulated")

    # --- faults ----------------------------------------------------------------------------------
    def check_faults(self) -> None:
        """Record inter-robot contacts new since the last check; under the ``fail`` policy the
        first one ends the episode."""
        snap = self.call(self.backend.oracle)
        new = [p for p in inter_robot_contacts(snap, self.team.ids) if p not in self._seen_contacts]
        if not new:
            return
        self._seen_contacts.update(new)
        self.fault("inter_robot_collision", sorted({p.split("/")[0] for pair in new for p in pair}), new)

    def fault(self, kind: str, robots: Sequence[str], parts: Sequence[tuple[str, str]]) -> None:
        """Record a fault; under the ``fail`` policy of its kind the episode ends."""
        t = self.call(self.backend.sim_time)
        self.faults.append({"kind": kind, "sim_time": round(t, 3), "robots": sorted(robots),
                            "parts": [list(p) for p in parts]})
        self.log("fault", None, fault=kind, robots=sorted(robots), parts=[list(p) for p in parts])
        if self.team.fault_policy(kind) == "fail" and self.stop is None:
            self.stop = StopEpisode(f"fault:{kind}", ", ".join(f"{a} touched {b}" for a, b in parts))

    # --- trace -----------------------------------------------------------------------------------
    def log(self, kind: str, robot: str | None, **data) -> None:
        entry = {"kind": kind, "sim_time": round(self.call(self.backend.sim_time), 3),
                 "wall": round(time.monotonic() - self.t0_wall, 2)}
        if robot is not None:
            entry["robot"] = robot
        self.trace.append({**entry, **data})
        if self._trace_fh is not None:
            self._trace_fh.write(json.dumps(self.trace[-1], default=str) + "\n")
            self._trace_fh.flush()


class _Lockstep:
    """Simultaneous mode: subjects run in threads; backend calls and ticks are served here."""

    def __init__(self, world: World, robots: Sequence[str]):
        self.world = world
        self.queue: queue.Queue = queue.Queue()
        self.live = set(robots)
        self.waiting: dict[str, threading.Event] = {}

    def request(self, fn, args):
        done, box = threading.Event(), {}
        self.queue.put(("call", (fn, args, done, box)))
        done.wait()
        if "error" in box:
            raise box["error"]
        return box["value"]

    def arrive(self, robot: str) -> None:
        go = threading.Event()
        self.queue.put(("tick", (robot, go)))
        go.wait()

    def leave(self, robot: str) -> None:
        self.queue.put(("done", robot))

    def serve(self) -> None:
        """Serve calls and ticks until every subject has returned."""
        while self.live:
            kind, item = self.queue.get()
            if kind == "call":
                fn, args, done, box = item
                try:
                    box["value"] = fn(*args)
                except BaseException as exc:     # handed to the calling thread
                    box["error"] = exc
                done.set()
            elif kind == "tick":
                robot, go = item
                self.waiting[robot] = go
            else:
                self.live.discard(item)
                self.waiting.pop(item, None)
            if self.live and set(self.waiting) >= self.live:
                if self.world.stop is None:
                    self.world._world_tick()
                for go in self.waiting.values():
                    go.set()
                self.waiting.clear()


class RobotView:
    """One robot's view of a multi-robot backend under the Backend Protocol: its own joints and
    cameras with their unprefixed names, and an oracle whose hand, base and contact names are its
    own (other robots' parts keep their ``<id>/`` prefix)."""

    def __init__(self, world: World, robot_id: str, robot, cameras: Sequence[str] | None = None):
        self.world = world
        self.id = robot_id
        self.prefix = f"{robot_id}/"
        self.robot = robot
        self._cameras = tuple(cameras if cameras is not None else robot.cameras)
        unknown = set(self._cameras) - set(robot.cameras)
        if unknown:
            raise ValueError(f"robot {robot_id}: cameras {sorted(unknown)} are not cameras of preset {robot.name}")
        b = world.backend
        if "depth_camera" in world.info.capabilities:
            self.render_depth = lambda camera, w, h: world.call(b.render_depth, self._camera(camera), w, h)
        if "gripper_force_limit" in world.info.capabilities:
            self.set_gripper_force = lambda arm, n: world.call(b.set_gripper_force, self.prefix + arm, n)
        if "pose_clearance" in world.info.capabilities:
            self.pose_clearance = lambda joints, cap=0.05: world.call(b.pose_clearance, self.prefix, joints, cap)

    def info(self) -> BackendInfo:
        i = self.world.info
        caps = frozenset(c for c in i.capabilities
                         if c in SIMULATOR or c in ("depth_camera", "gripper_force_limit", "pose_clearance"))
        return replace(i, robot=self.robot.name, robot_sha256=self.robot.sha256(), capabilities=caps)

    def load(self, robot, scene) -> None:
        raise RuntimeError("a robot view is loaded with the world")

    def reset(self, scene, *, seed: int) -> None:
        raise RuntimeError("a robot view is reset with the world")

    def joint_positions(self) -> dict[str, float]:
        q = self.world.call(self.world.backend.joint_positions)
        n = len(self.prefix)
        return {k[n:]: v for k, v in q.items() if k.startswith(self.prefix)}

    def command_joints(self, targets: Mapping[str, float]) -> None:
        self.world.call(self.world.backend.command_joints, {self.prefix + k: v for k, v in targets.items()})

    def step(self, seconds: float) -> None:
        for _ in range(max(1, int(round(seconds / self.world.control_dt)))):
            self.world.tick(self.id)

    def sim_time(self) -> float:
        return self.world.call(self.world.backend.sim_time)

    def cameras(self) -> Sequence[str]:
        return list(self._cameras)

    def _camera(self, camera: str) -> str:
        if camera not in self._cameras:
            raise KeyError(f"robot {self.id} has no camera {camera!r}; cameras: {', '.join(self._cameras)}")
        return self.prefix + camera

    def render(self, camera: str, width: int, height: int) -> np.ndarray:
        return self.world.call(self.world.backend.render, self._camera(camera), width, height)

    def oracle(self) -> OracleSnapshot | None:
        snap = self.world.call(self.world.backend.oracle)
        return None if snap is None else own_snapshot(snap, self.id)

    def close(self) -> None:
        pass


def own_snapshot(snap: OracleSnapshot, robot_id: str) -> OracleSnapshot:
    """``snap`` as robot ``robot_id`` sees it: its hands, openings, base and contact parts under
    their single-robot names; other robots' parts keep their prefix."""
    p = f"{robot_id}/"

    def part(name: str) -> str:
        return name[len(p):] if name.startswith(p) else name

    def own(d: Mapping[str, Any]) -> dict[str, Any]:
        return {k[len(p):]: v for k, v in d.items() if k.startswith(p)}

    return replace(snap, tcp=own(snap.tcp), gripper_opening=own(snap.gripper_opening),
                   base=snap.bases.get(robot_id, snap.base),
                   contacts=frozenset((part(a), part(b)) for a, b in snap.contacts),
                   contacts_ever=frozenset((part(a), part(b)) for a, b in snap.contacts_ever),
                   held_by={o: part(arm) for o, arm in snap.held_by.items()})


class RobotSession(Session):
    """The session of one robot of a multi-robot episode. Budgets are per robot; time, recording,
    success and faults are the world's; the turn caps of the schedule apply on top."""

    def __init__(self, world: World, robot_id: str, episode: Episode, view: RobotView, robot, *,
                 run_budget: RunBudget, time_mode: str, on_action: Callable[[str], None],
                 on_say: Callable[[str, str, str], str] | None = None,
                 on_ready: Callable[[str], None] | None = None):
        self.world = world
        self.robot_id = robot_id
        self.on_action = on_action
        self.on_say = on_say
        self.on_ready = on_ready
        self.turn: int | None = None
        self._turn_start: tuple[Usage, float] | None = None
        self.turn_budget = world.team.schedule.turn
        #: model replies whose actions are all carried out (``before_model_call``)
        self.replies_done = 0
        slot = world.team.slot(robot_id)
        placement = episode.scene.placement(robot_id)
        #: named poses of the robot ('' = its preset's home) and the lift it acts at
        self.start_pose = slot.start_pose
        self.idle_pose = slot.idle_pose or slot.start_pose
        self.start_lift = placement.lift if placement.lift is not None else robot.pose_lift(slot.start_pose)
        super().__init__(episode, view, robot, recorder=Recorder(enabled=False), run_budget=run_budget,
                         time_mode=time_mode,
                         footprint=robot.mobile_base.footprint if robot.mobile_base else None)

    # --- turns -------------------------------------------------------------------------------------
    def begin_turn(self, turn: int, instruction: str) -> None:
        self.turn = turn
        self.episode = replace(self.episode, instruction=instruction)
        self._turn_start = (replace(self.usage), self.backend.sim_time())
        self.log("turn_start", turn=turn, instruction=instruction[:2000])

    def _turn_cap(self) -> str:
        if self._turn_start is None:
            return ""
        u0, t0 = self._turn_start
        b = self.turn_budget
        if b.sim_seconds is not None and self.backend.sim_time() - t0 >= float(b.sim_seconds):
            return "sim_seconds"
        if b.model_calls is not None and self.usage.model_calls - u0.model_calls >= int(b.model_calls):
            return "model_calls"
        if b.api_calls is not None and self.usage.api_calls - u0.api_calls >= int(b.api_calls):
            return "api_calls"
        return ""

    def _check_world(self) -> None:
        stop = self.world.stop
        if stop is not None:
            raise StopEpisode(stop.reason, stop.detail)
        cap = self._turn_cap()
        if cap:
            raise StopEpisode(f"turn_end:{cap}", "turn cap reached")

    # --- overrides -----------------------------------------------------------------------------------
    def advance(self, seconds: float, on_tick=None) -> None:
        self._check_world()
        super().advance(seconds, on_tick)

    def before_model_call(self) -> None:
        self._check_world()
        super().before_model_call()
        # Between two replies: every action of the replies so far is carried out. A referee that counts in
        # replies is asked here, so that a turn ends before another reply is paid for, not inside one.
        self.replies_done = self.usage.model_calls
        self.on_action(self.robot_id)

    def before_api_call(self, name: str, arguments: Mapping[str, Any]) -> None:
        self._check_world()
        super().before_api_call(name, arguments)

    def after_api_call(self, name: str, arguments: Mapping[str, Any], result: Any, refused: str = "") -> None:
        if refused:
            self.usage.refusals += 1
        self.log("api", name=name, arguments=dict(arguments), refused=refused, result=_short(result))
        self.on_action(self.robot_id)

    def ready(self) -> None:
        self.log("turn_ready", turn=self.turn)
        if self.on_ready is not None:
            self.on_ready(self.robot_id)

    def say(self, keyword: str, text: str) -> str:
        """Passes the line to the referee (``Referee.hear``), logs both, then polls whether the turn is
        over, as after an action."""
        self._check_world()
        answer = self.on_say(self.robot_id, keyword, text) if self.on_say is not None else ""
        self.log("say", keyword=keyword, text=text[:500], answer=answer)
        self.on_action(self.robot_id)
        return answer

    def success(self):
        raise RuntimeError("a robot of a multi-robot task has no success of its own; the world scores it")

    def log(self, kind: str, **data) -> None:
        self.world.log(kind, self.robot_id, **data)


# --- the episode ----------------------------------------------------------------------------------

@dataclass
class Member:
    """A robot's seat in the episode: its session, its subject and what it did."""
    id: str
    session: RobotSession
    subject: Any
    exits: list[str]
    model_ids: set[str]
    artifacts: dict[str, str]


def _run_member(m: Member) -> str:
    s = m.session
    try:
        reason = m.subject.run(s) or "finished"
    except StopEpisode as stop:
        reason = stop.reason
        if stop.detail:
            s.log("stop", reason=stop.reason, detail=stop.detail)
    except Exception as exc:     # a subject crash ends that robot's part, not the episode
        reason = "error:subject"
        s.log("error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-4000:])
    m.exits.append(reason)
    m.model_ids |= set(s.model_ids)
    tag = f"turn{s.turn:02d}_" if s.turn is not None else ""
    for name, content in (s.artifacts or {}).items():
        m.artifacts[f"{m.id}_{tag}{name}"] = content
    s.artifacts = {}
    return reason


def transfer(world: World, session: RobotSession, pose: str | None, lift: float | None = None,
             via: Mapping[str, Sequence[float]] | None = None) -> None:
    """Move a robot to a named pose of its preset ('' = its start pose; None = arms stay, only the torso
    moves to ``lift``): a harness move between turns, outside the turn caps, logged. Each TCP first
    rises 5 cm where that is reachable; then the arms move in joint space. The joints in
    ``leave_home_first`` move first when the robot leaves its start pose and last when it returns to
    it; the torso rises before the arms and lowers after them. Any contact of the robot's parts with
    anything but the floor during the move is a ``transfer_collision`` fault."""
    from .api import Refused
    if world.stop is not None:
        return
    spec, api = session.robot, session.api
    start_pose = session.start_pose
    q = session.backend.joint_positions()
    if pose is None:
        targets = {j: q[j] for a in spec.arms.values() for j in a.joints}
    elif pose:
        targets = spec.pose_arm_targets(pose)
    else:
        targets = {j: v for a in spec.arms.values() for j, v in zip(a.joints, a.home)}
    if lift is None and pose:
        lift = spec.pose_lift(pose)
    if pose is None and (lift is None or not spec.lift_joint or abs(q[spec.lift_joint] - lift) < 1e-3):
        return
    saved, session._turn_start = session._turn_start, None
    before = _own_contacts(world, session.robot_id)
    session.log("transfer_begin", pose=(pose or start_pose or "home") if pose is not None else "(arms stay)", lift=lift)
    try:
        q = session.backend.joint_positions()
        at_start = all(abs(q[j] - v) < 0.02 for a in spec.arms.values() for j, v in zip(a.joints, a.home))
        if not at_start and pose is not None:
            for arm in spec.arms:
                api._q()
                p, rot = api.kin.tcp_pose(arm)
                try:
                    api._cartesian(arm, p + np.array([0.0, 0.0, 0.05]), [rot])
                except Refused:
                    pass
        lift_j = spec.lift_joint
        lift_now = session.backend.joint_positions()[lift_j] if lift_j else None
        if lift is not None and lift_j and lift > lift_now:
            api._joint_move([lift_j], [lift], spec.lift_speed)
        to_start = pose == start_pose
        for a in (spec.arms.values() if pose is not None else ()):
            q = session.backend.joint_positions()
            now, end = np.array([q[j] for j in a.joints]), np.array([targets[j] for j in a.joints])
            staged = [i for i, j in enumerate(a.joints) if j in a.leave_home_first]
            path = [now, end]
            arm_name = next(n for n, x in spec.arms.items() if x is a)
            if via and arm_name in via:
                path = [now, np.asarray(via[arm_name], float), end]
            elif staged and (at_start or to_start):
                # leaving the start pose: staged joints first; returning to it: staged joints last
                mid = end.copy() if to_start else now.copy()
                mid[staged] = now[staged] if to_start else end[staged]
                path = [now, mid, end]
            durations = [max(0.2, float(np.max(np.abs(b - c))) / a.max_joint_speed) for c, b in zip(path, path[1:])]
            api._run_joint_path(list(a.joints), path, durations)
        if lift is not None and lift_j and lift < lift_now:
            api._joint_move([lift_j], [lift], spec.lift_speed)
    except StopEpisode:
        return
    finally:
        session._turn_start = saved
    new = sorted(_own_contacts(world, session.robot_id) - before)
    session.log("transfer", pose=(pose or start_pose or "home") if pose is not None else "(arms stay)", lift=lift,
                contacts=[list(c) for c in new])
    if new:
        world.fault("transfer_collision", [session.robot_id], new)


def _ee_start_joints(session: RobotSession) -> dict[str, tuple[float, ...]]:
    """Per arm the joints of the task's ee start pose nearest the current ones (``ee.start``)."""
    from .ee_control import EEController
    from .kinematics.ik import solve_near
    start = (session.task.raw.get("ee") or {}).get("start") or {}
    ctl = EEController(session)
    q = session.backend.joint_positions()
    out = {}
    for arm, (x, y, z, *yaw) in start.items():
        res = solve_near(session.api.ik, ctl.kin, arm, (x, y, z), ctl._rotation(arm, yaw[0] if yaw else 0.0), q)
        if res.ok:
            out[arm] = res.q
    return out


def _own_lift(session: RobotSession) -> bool:
    """The robot's idle pose sets a torso lift of its own (robot.yaml ``poses.<pose>.lift``)."""
    pose = session.idle_pose
    return bool(pose) and session.robot.pose_lift(pose) is not None


def _own_contacts(world: World, robot_id: str) -> set[tuple[str, str]]:
    """Contacts since reset between a part of ``robot_id`` and anything but itself or the floor."""
    snap = world.call(world.backend.oracle)
    p = f"{robot_id}/"
    return {(a, b) for a, b in snap.contacts | snap.contacts_ever
            if a.startswith(p) and not b.startswith(p) and b != "floor"}


def play_turns(world: World, members: Mapping[str, Member], referee, game: GameView,
               base_instructions: Mapping[str, str], on_turn: Callable[[], None] | None = None) -> None:
    """The turns schedule: the referee names who acts; the others hold."""
    team = world.team
    max_turns = team.schedule.max_turns or len(team.robots)
    messages: dict[str, str] = {}

    def apply(ruling: Ruling) -> str | None:
        for rid, text in ruling.messages.items():
            messages[rid] = text
        if ruling.restore:
            for name, (pos, quat) in ruling.restore.items():
                world.backend.set_object_pose(name, pos, quat)
            world.log("restore", None, objects={k: [list(map(float, p)), list(map(float, q))]
                                                for k, (p, q) in ruling.restore.items()})
        return ruling.next_robot

    if referee is not None:
        ruling = referee.start(game, world.backend.oracle())
        world.log("referee", None, turn=0, next=ruling.next_robot, note=ruling.note, messages=dict(ruling.messages))
        nxt = apply(ruling)
    else:
        nxt = team.ids[0]
    for m in members.values():
        if m.session.idle_pose != m.session.start_pose or _own_lift(m.session):
            transfer(world, m.session, m.session.idle_pose)
    while nxt is not None and game.turn < max_turns and world.stop is None:
        if nxt not in members:
            raise ValueError(f"referee named unknown robot {nxt!r}")
        m = members[nxt]
        if m.session.start_lift is not None:
            transfer(world, m.session, None, m.session.start_lift)
        before = world.backend.oracle()
        t0 = world.backend.sim_time()
        text = base_instructions[nxt] + (f"\n{messages[nxt]}" if messages.get(nxt) else "")
        m.session.begin_turn(game.turn + 1, text)
        game.live.setdefault("turn_start", {})[nxt] = (world.backend.sim_time(), m.session.usage.model_seconds)
        world.acting = nxt
        reason = _run_member(m)
        world.acting = None
        m.session.log("turn_end", turn=game.turn + 1, ended=reason)
        if reason in EPISODE_ENDS or world.stop is not None:
            if world.stop is None:
                world.stop = StopEpisode(reason, f"in turn {game.turn + 1} of robot {nxt}")
            break
        if not reason.startswith("turn_end:") and reason not in ("finished", "horizon"):
            game.out.add(nxt)          # the robot's own budget ran out or its subject failed
        idle = m.session.idle_pose
        transfer(world, m.session, idle,
                 m.session.start_lift if idle == m.session.start_pose and not _own_lift(m.session) else None,
                 via=_ee_start_joints(m.session) if team.slot(nxt).return_via == "ee_start" else None)
        world.check_faults()
        report = TurnReport(game.turn + 1, nxt, reason, before, world.backend.oracle(), t0, world.backend.sim_time())
        if referee is None:
            game.history.append((report, Ruling()))
            ids = [r for r in team.ids if r not in game.out]
            nxt = ids[game.turn % len(ids)] if ids else None
            continue
        ruling = referee.after_turn(game, report)
        game.history.append((report, ruling))
        world.log("referee", nxt, turn=report.turn, valid=ruling.valid, note=ruling.note,
                  next=ruling.next_robot, messages=dict(ruling.messages))
        nxt = apply(ruling)
        if ruling.stop and world.stop is None:
            world.stop = StopEpisode(ruling.stop, ruling.note)
        if on_turn is not None:
            on_turn()


def play_simultaneous(world: World, members: Mapping[str, Member], referee, game: GameView,
                      base_instructions: Mapping[str, str]) -> None:
    """The simultaneous schedule: every subject in its own thread, lockstep time."""
    messages: dict[str, str] = {}
    if referee is not None:
        ruling = referee.start(game, world.backend.oracle())
        messages = dict(ruling.messages)
        world.log("referee", None, turn=0, note=ruling.note, messages=messages)
    for rid, m in members.items():
        text = base_instructions[rid] + (f"\n{messages[rid]}" if messages.get(rid) else "")
        m.session.begin_turn(1, text)
    lock = _Lockstep(world, list(members))
    world._lockstep = lock

    def body(m: Member) -> None:
        try:
            _run_member(m)
        finally:
            lock.leave(m.id)

    threads = [threading.Thread(target=body, args=(m,), name=f"robot-{m.id}", daemon=True) for m in members.values()]
    for t in threads:
        t.start()
    try:
        lock.serve()
    finally:
        world._lockstep = None
        for t in threads:
            t.join(timeout=5)
    for m in members.values():
        m.session.log("turn_end", turn=1, ended=m.exits[-1] if m.exits else "")


def score_team(world: World, task, episode: Episode, referee, game: GameView,
               context: Mapping[str, Any]) -> dict[str, Any]:
    """Success from the task's ``success`` predicate and the referee's result (both must hold
    where both exist), failed by any fault under the ``fail`` policy."""
    from ..core.predicates import evaluate
    world.check_faults()
    snap = world.backend.oracle()
    ok, evidence = True, []
    if task.success:
        v = evaluate(task.success, snap, context)
        ok &= v.ok
        evidence.append(v.evidence)
    result: GameResult | None = None
    if referee is not None:
        result = referee.result(game, snap)
        ok &= result.success
        evidence.append(result.evidence)
    failing = [f for f in world.faults if world.team.fault_policy(f["kind"]) == "fail"]
    if failing:
        ok = False
        evidence.append(f"fault: {failing[0]['kind']} ({', '.join(failing[0]['robots'])})")
    subgoals, credit, total = [], 0.0, sum(s.weight for s in task.subgoals) or 1.0
    for s in task.subgoals:
        v = evaluate(s.predicate, snap, context)
        subgoals.append({"name": s.name, "ok": v.ok, "evidence": v.evidence, "weight": s.weight})
        credit += s.weight * v.ok
    return {"success": bool(ok), "evidence": "; ".join(e for e in evidence if e), "subgoals": subgoals,
            "credit": 1.0 if ok else round(credit / total, 4), "final_state": snap.to_json(),
            "result": result}


def robot_task(task, slot):
    """The task as one robot of it sees it: its preset, its gripper limits, its ``ee`` block (the
    task's with the robot's own fields on top)."""
    raw = dict(task.raw)
    if slot.ee or raw.get("ee"):
        raw["ee"] = {**(raw.get("ee") or {}), **slot.ee}
    return replace(task, robot=slot.preset, gripper=task.team_gripper.get(slot.id, {}), raw=raw,
                   stop_on_success=False, team=None, team_gripper={})


def robot_instruction(task, slot, values: Mapping[str, Any]) -> str:
    template = slot.instruction or task.instruction
    names = {k.replace(".", "_"): (v if not isinstance(v, float) else round(v, 3)) for k, v in values.items()}
    return template.format(**names) if "{" in template else template


def run_team_episode(task, seed: int, team_subject, backend, cfg, episode_dir) -> dict[str, Any]:
    """One episode of a multi-robot task (see :func:`omakase_bench.core.runner.run_episode`)."""
    import json

    from ..core.task import resolve_ref, robot_for
    team: TeamSpec = task.team
    info = backend.info()
    if "multi_robot" not in info.capabilities:
        raise ValueError(f"task {task.id} has {len(team.robots)} robots; backend {info.name} holds one robot only "
                         "(no multi_robot capability)")
    missing = [r for r in team.ids if r not in team_subject.members]
    if missing:
        raise ValueError(f"no subject bound to robot(s) {missing} of {task.id}")
    episode = task.resolve(seed)
    params = task.raw.get("params", {})
    robots = {r.id: robot_for(task, episode.scene, r.id) for r in episode.scene.robots}
    backend.load_team(robots, episode.scene)
    backend.reset(episode.scene, seed=seed)
    initial = backend.oracle()
    video = task.raw.get("video") or {}
    tiles = tuple(video.get("tiles", [f"{rid}/head" for rid in team.ids if "head" in robots[rid].cameras]))
    recorder = Recorder(camera=video.get("camera", cfg.video_camera), fps=float(video.get("fps", cfg.video_fps)),
                        enabled=cfg.video, size=tuple(video.get("size", (640, 480))), tiles=tiles,
                        tile_layout=str(video.get("tile_layout", "bottom")),
                        tile_size=tuple(video.get("tile_size", (240, 180))),
                        tile_columns=int(video.get("tile_columns", 2)),
                        joints=tuple(f"{rid}/{j}" for rid in team.ids for a in robots[rid].arms.values()
                                     for j in a.joints) if video.get("record_joints") else (),
                        stream_to=(episode_dir / "raw.mp4") if (cfg.video and video.get("stream")) else None)
    episode_dir.mkdir(parents=True, exist_ok=True)
    recorder.maybe_capture(backend, backend.sim_time(), force=True)
    world = World(backend, team, recorder, task.budget.sim_seconds)
    world.stream_trace(episode_dir / "trace.jsonl")
    game = GameView(team.ids, episode.values, params)
    referee = resolve_ref(team.referee)(episode.values, params) if team.referee else None

    def on_action(rid: str) -> None:
        world.check_faults()
        if world.stop is not None:
            raise StopEpisode(world.stop.reason, world.stop.detail)
        if task.stop_on_success and task.success:
            from ..core.predicates import evaluate
            snap = world.call(backend.oracle)
            if evaluate(task.success, snap, context).ok:
                world.stop = StopEpisode("success", f"after an action of robot {rid}")
                raise StopEpisode("success")
        if (referee is not None and team.schedule.mode == "turns" and world.acting == rid
                and referee.turn_over(game, rid, world.call(backend.oracle))):
            raise StopEpisode("turn_end:referee", "the referee ended the turn")

    def on_say(rid: str, keyword: str, text: str) -> str:
        if referee is None or not hasattr(referee, "hear"):
            return ""
        return str(referee.hear(game, rid, keyword, text, world.call(backend.oracle)) or "")

    def on_ready(rid: str) -> None:
        # the move from the idle pose to the task's start pose is the harness's, not the robot's time
        game.live.setdefault("turn_start", {})[rid] = (world.call(backend.sim_time),
                                                       members[rid].session.usage.model_seconds)

    context = {"footprint": None, "values": episode.values, "params": params}
    members: dict[str, Member] = {}
    instructions: dict[str, str] = {}
    for slot in team.robots:
        rtask = robot_task(task, slot)
        own = (episode.values.get("robots") or {}).get(slot.id, {})
        values = {**episode.values, **own, "robot": slot.id}
        instructions[slot.id] = robot_instruction(task, slot, values)
        ep = Episode(rtask, seed, episode.scene, values, instructions[slot.id])
        view = RobotView(world, slot.id, robots[slot.id], slot.cameras)
        session = RobotSession(world, slot.id, ep, view, robots[slot.id], run_budget=cfg.run_budget,
                               time_mode=cfg.time_mode, on_action=on_action, on_say=on_say, on_ready=on_ready)
        session.episode_dir = episode_dir
        members[slot.id] = Member(slot.id, session, team_subject.members[slot.id], [], set(), {})
    episode_dir.mkdir(parents=True, exist_ok=True)
    def idle_error(rid: str) -> tuple[float, float]:
        """How far robot ``rid`` is from its idle pose now: (largest arm joint error, rad; lift error, m)."""
        spec, s = robots[rid], members[rid].session
        q = world.call(backend.joint_positions)
        pose = s.idle_pose
        arm = (spec.pose_arm_targets(pose) if pose else
               {j: v for a in spec.arms.values() for j, v in zip(a.joints, a.home)})
        lift = spec.pose_lift(pose) if pose else None
        if lift is None:
            lift = s.start_lift
        lift_err = abs(q[f"{rid}/{spec.lift_joint}"] - lift) if spec.lift_joint and lift is not None else 0.0
        return max(abs(q[f"{rid}/{j}"] - v) for j, v in arm.items()), lift_err

    def save_progress() -> None:
        """After every ruled turn: the trace on disk, and the episode so far in progress.json (the turns,
        each robot's usage and what the referee reports of the game), replaced atomically."""
        world.sync_trace()
        state = {"task": task.id, "task_version": task.version, "seed": seed, "sim_seconds": round(backend.sim_time(), 2),
                 "turns": [{"turn": r.turn, "robot": r.robot, "ended": r.ended, "valid": ru.valid, "note": ru.note}
                           for r, ru in game.history],
                 "robots": {rid: {"subject": m.subject.name, "usage": m.session.usage_dict()} for rid, m in members.items()},
                 "game": referee.progress() if referee is not None and hasattr(referee, "progress") else None}
        tmp = episode_dir / "progress.json.tmp"
        with tmp.open("w") as fh:
            fh.write(json.dumps(state, indent=1, default=str))
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(episode_dir / "progress.json")

    game.live.update(sim_time=lambda: world.call(backend.sim_time),
                     usage={rid: m.session.usage for rid, m in members.items()}, idle_error=idle_error,
                     replies={rid: (lambda s=m.session: s.replies_done) for rid, m in members.items()})
    t0 = time.monotonic()
    exit_reason, detail = "finished", ""
    try:
        if team.schedule.mode == "turns":
            play_turns(world, members, referee, game, instructions, save_progress)
        else:
            play_simultaneous(world, members, referee, game, instructions)
    except StopEpisode as stop:
        world.stop = world.stop or stop
    except Exception as exc:     # a referee or harness error fails the episode, not the run
        exit_reason, detail = "error:harness", f"{type(exc).__name__}: {exc}"
        world.log("error", None, traceback=traceback.format_exc()[-4000:])
    if world.stop is not None and exit_reason == "finished":
        exit_reason, detail = world.stop.reason, world.stop.detail
    wall = time.monotonic() - t0
    score = score_team(world, task, episode, referee, game, context)
    result_obj: GameResult | None = score.pop("result")
    artifacts: dict[str, str] = {}
    world.sync_trace(close=True)                  # trace.jsonl was written entry by entry
    artifacts["trace"] = "trace.jsonl"
    summary = " | ".join(f"{rid}: {members[rid].subject.name}" for rid in team.ids)
    if cfg.video and recorder.stream_to is not None:
        recorder.maybe_capture(backend, backend.sim_time(), force=True)
        recorder.close()
        (episode_dir / "frames.json").write_text(json.dumps(
            {"times": [round(t, 3) for t in recorder.times], "joint_names": list(recorder.joints),
             "joints": recorder.extra.get("joints", []), "size": list(recorder.size),
             "tiles": list(recorder.tiles), "tile_size": list(recorder.tile_size),
             "tile_columns": recorder.tile_columns}))
        artifacts["video"] = "raw.mp4"
        artifacts["frames"] = "frames.json"
    if cfg.video and recorder.frames:
        from ..report.video import write_episode_video, write_raw_video
        recorder.maybe_capture(backend, backend.sim_time(), force=True)
        write_raw_video(episode_dir / "raw.mp4", recorder.frames, cfg.video_fps)
        write_episode_video(episode_dir / "episode.mp4", recorder.frames, recorder.times, cfg.video_fps,
                            title=summary, instruction=task.title, success=score["success"])
        artifacts["video"] = "episode.mp4"
    for m in members.values():
        for name, content in m.artifacts.items():
            (episode_dir / name).write_text(content)
            artifacts[name.rsplit(".", 1)[0]] = name
    per_robot = {}
    total = Usage()
    for rid, m in members.items():
        u = m.session.usage
        for f in total.__dataclass_fields__:
            setattr(total, f, getattr(total, f) + getattr(u, f))
        per_robot[rid] = {"subject": m.subject.name, "interface": m.subject.interface,
                          "model_ids": sorted(m.model_ids), "usage": m.session.usage_dict(), "exits": m.exits,
                          "turns": sum(1 for r, _ in game.history if r.robot == rid) or len(m.exits),
                          "metrics": dict(result_obj.robots.get(rid, {})) if result_obj else {},
                          "faults": [f for f in world.faults if rid in f["robots"]],
                          "joint_travel": m.session.travel.to_json()}
    usage = {**total.__dict__, "usd": round(total.usd, 5), "model_seconds": round(total.model_seconds, 2)}
    turns = [{"turn": r.turn, "robot": r.robot, "ended": r.ended, "valid": ru.valid, "note": ru.note,
              "sim_start": round(r.sim_start, 2), "sim_end": round(r.sim_end, 2)} for r, ru in game.history]
    result = {
        "subject": team_subject.name, "task": task.id, "task_version": task.version, "seed": seed,
        "instruction": "\n".join(f"{rid}: {text}" for rid, text in instructions.items()),
        "variation": {k: v for k, v in episode.values.items() if k != "robots"},
        "success": score["success"], "credit": score["credit"], "evidence": score["evidence"],
        "subgoals": score["subgoals"], "exit_reason": exit_reason, "exit_detail": detail,
        "model_claim": None, "model_ids": sorted({i for m in members.values() for i in m.model_ids}),
        "interface": team_subject.interface, "sim_seconds": round(backend.sim_time(), 2),
        "success_sim_seconds": None, "wall_seconds": round(wall, 2), "usage": usage,
        "initial_state": initial.to_json() if initial else None, "final_state": score["final_state"],
        "artifacts": {k: str((episode_dir / v).relative_to(cfg.out_dir)) for k, v in artifacts.items()},
        "schedule": team.schedule.mode, "robots": per_robot, "faults": world.faults, "turns": turns,
        "outcome": result_obj.outcome if result_obj else "",
    }
    (episode_dir / "result.json").write_text(json.dumps(result, indent=1, default=str))
    return result

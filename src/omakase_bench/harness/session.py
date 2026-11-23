"""One episode's window onto the world: budgets, time, recording and scoring.

Every subject reaches the robot through ``session.api`` (or, for the chunk
interface, :meth:`Session.execute_chunk`), and every joint target goes out
through :meth:`Session.command_joints`, which holds the end effectors inside
the task's limits (``task.gripper``). The session advances the backend,
records video frames, appends the trace, enforces the budgets and stops the
episode (:class:`StopEpisode`) on success, a spent budget or a time limit.
Success is read only from the backend's oracle.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from ..core.backend import Backend, BackendUnsupported, OracleSnapshot
from ..core import goal_predicates  # noqa: F401  (registers the ``goal`` predicate)
from ..core.predicates import Verdict, evaluate
from ..core.task import Episode

TIME_MODES = ("lockstep", "realtime")
#: the end effectors settle at the task's start shape before the subject starts: until every joint
#: is within START_TOLERANCE of its target (m or rad), for at most START_SECONDS of simulated time
#: characters of a call's prompt and thinking summary kept in the trace
TRANSCRIPT_CHARS = 8000
START_TOLERANCE = 3e-4
START_SECONDS = 3.0


class StopEpisode(Exception):
    """Ends an episode. ``reason``: success | budget_exceeded:<what> | finished | error:<what>."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass
class Usage:
    api_calls: int = 0
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    usd: float = 0.0
    model_seconds: float = 0.0
    refusals: int = 0

    def add_model(self, *, input_tokens=0, output_tokens=0, reasoning_tokens=0, cached_tokens=0,
                  usd=0.0, seconds=0.0) -> None:
        self.model_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.reasoning_tokens += reasoning_tokens
        self.cached_tokens += cached_tokens
        self.usd += usd
        self.model_seconds += seconds


@dataclass
class RunBudget:
    """Caps shared by every episode of a run. ``None`` means no cap."""
    max_model_calls: int | None = 400
    max_usd: float | None = 10.0
    model_calls: int = 0
    usd: float = 0.0

    def exceeded(self) -> str:
        if self.max_model_calls is not None and self.model_calls >= self.max_model_calls:
            return "run_model_calls"
        if self.max_usd is not None and self.usd >= self.max_usd:
            return "run_usd"
        return ""


@dataclass
class Recorder:
    camera: str = "showcase"
    fps: float = 10.0
    size: tuple[int, int] = (640, 480)
    frames: list[np.ndarray] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    extra: Mapping[str, list] = field(default_factory=dict)
    enabled: bool = True
    #: cameras shown as small tiles along the bottom of every frame (e.g. each robot's head camera)
    tiles: tuple[str, ...] = ()
    #: "bottom" (tiles over the bottom edge of the main view) or "right" (a grid of ``tile_columns``
    #: columns of ``tile_size`` tiles beside the main view; '' leaves a slot empty)
    tile_layout: str = "bottom"
    tile_size: tuple[int, int] = (240, 180)
    tile_columns: int = 2
    #: joints whose positions are recorded with every frame (``extra["joints"]``)
    joints: tuple[str, ...] = ()
    #: when set, frames are written to this mp4 as they are captured and not kept in memory
    stream_to: Any = None
    _next: float = 0.0
    _writer: Any = None
    last: Any = None

    def maybe_capture(self, backend: Backend, t: float, force: bool = False) -> None:
        if not self.enabled or (t + 1e-9 < self._next and not force):
            return
        try:
            frame = backend.render(self.camera, *self.size)
        except BackendUnsupported:        # a backend without that camera records no video
            self.enabled = False
            return
        if self.tiles:
            frame = self._tile_grid(backend, frame) if self.tile_layout == "right" else self._with_tiles(backend, frame)
        if self.joints:
            q = backend.joint_positions()
            self.extra.setdefault("joints", []).append([round(float(q[j]), 4) for j in self.joints])
        if self.stream_to is not None:
            if self._writer is None:
                import imageio.v2 as imageio
                self._writer = imageio.get_writer(str(self.stream_to), fps=self.fps, codec="libx264", quality=8,
                                                  macro_block_size=16, pixelformat="yuv420p")
            self._writer.append_data(frame)
            self.last = frame
        else:
            self.frames.append(frame)
        self.times.append(t)
        self._next = t + 1.0 / self.fps

    def close(self) -> None:
        """Finish a streamed video."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def _tile_grid(self, backend: Backend, frame: np.ndarray) -> np.ndarray:
        tw, th = self.tile_size
        cols = max(1, self.tile_columns)
        rows = (len(self.tiles) + cols - 1) // cols
        h = max(frame.shape[0], rows * th)
        out = np.zeros((h, frame.shape[1] + cols * tw, 3), np.uint8)
        out[:frame.shape[0], :frame.shape[1]] = frame
        for k, cam in enumerate(self.tiles):
            if not cam:
                continue
            r, c = divmod(k, cols)
            out[r * th:(r + 1) * th, frame.shape[1] + c * tw:frame.shape[1] + (c + 1) * tw] = backend.render(cam, tw, th)
        return out

    def _with_tiles(self, backend: Backend, frame: np.ndarray) -> np.ndarray:
        from PIL import Image, ImageDraw
        w, h = self.size
        tw = min(w // 4, (w - 8) // len(self.tiles) - 6)
        th = tw * 3 // 4
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        for k, cam in enumerate(self.tiles):
            x, y = 6 + k * (tw + 6), h - th - 6
            draw.rectangle((x - 2, y - 2, x + tw + 1, y + th + 1), fill=(20, 20, 20))
            img.paste(Image.fromarray(backend.render(cam, tw, th)), (x, y))
            draw.rectangle((x, y, x + 8 + 7 * len(cam), y + 13), fill=(20, 20, 20))
            draw.text((x + 4, y + 1), cam, fill=(240, 240, 240))
        return np.asarray(img)


class JointTravel:
    """Per joint: the total distance travelled and the number of swings, one-way motions between two
    reversals, larger than ``swing`` (rad). Sampled at ``period`` seconds of simulated time; a
    reversal counts once the joint has come back by more than ``hysteresis``."""

    def __init__(self, joints, swing: float = math.radians(90), hysteresis: float = math.radians(2),
                 period: float = 0.1):
        self.joints = list(joints)
        self.swing, self.hysteresis, self.period = swing, hysteresis, period
        self.total = {j: 0.0 for j in self.joints}
        self.swings = {j: 0 for j in self.joints}
        self.largest = {j: 0.0 for j in self.joints}
        self._last: dict[str, float] = {}
        self._run: dict[str, list[float]] = {}        # joint -> [start, extreme, direction]
        self._next = 0.0

    def maybe_update(self, t: float, q: Callable[[], Mapping[str, float]]) -> None:
        if t + 1e-9 >= self._next:
            self.update(q())
            self._next = t + self.period

    def update(self, q: Mapping[str, float]) -> None:
        for j in self.joints:
            v = float(q[j])
            if j not in self._last:
                self._last[j], self._run[j] = v, [v, v, 0.0]
                continue
            self.total[j] += abs(v - self._last[j])
            self._last[j] = v
            start, extreme, direction = self._run[j]
            if direction == 0.0:
                if abs(v - start) > self.hysteresis:
                    self._run[j] = [start, v, math.copysign(1.0, v - start)]
            elif (v - extreme) * direction > 0:
                self._run[j][1] = v
            elif (extreme - v) * direction > self.hysteresis:
                self._close(j)
                self._run[j] = [extreme, v, -direction]

    def _close(self, j: str) -> None:
        start, extreme, _ = self._run[j]
        size = abs(extreme - start)
        self.largest[j] = max(self.largest[j], size)
        if size > self.swing:
            self.swings[j] += 1

    def to_json(self) -> dict[str, Any]:
        for j in self._run:
            self._close(j)
            self._run[j] = [self._run[j][1], self._run[j][1], 0.0]
        return {j: {"travel_deg": round(math.degrees(self.total[j]), 1),
                    "largest_swing_deg": round(math.degrees(self.largest[j]), 1),
                    "swings_over_90": self.swings[j]} for j in self.joints}


class Session:
    def __init__(self, episode: Episode, backend: Backend, robot, *, recorder: Recorder | None = None,
                 run_budget: RunBudget | None = None, time_mode: str = "lockstep",
                 footprint=None):
        from .api import RobotAPI   # the API needs the session; keep the import local
        if time_mode not in TIME_MODES:
            raise ValueError(f"time_mode must be one of {TIME_MODES}")
        self.episode = episode
        self.task = episode.task
        self.backend = backend
        self.robot = robot
        self.recorder = recorder or Recorder(enabled=False)
        self.run_budget = run_budget or RunBudget(None, None)
        self.time_mode = time_mode
        self.footprint = footprint
        self.usage = Usage()
        self.trace: list[dict[str, Any]] = []
        self.t0_wall = time.monotonic()
        self.model_claim: str | None = None
        self.model_ids: set[str] = set()
        self.artifacts: dict[str, str] = {}
        self.success_at: float | None = None
        info = backend.info()
        self.control_dt = info.control_dt
        #: False on a backend without ground truth: nothing is judged, during the episode or after it
        self.judged = "ground_truth" in info.capabilities
        #: arm -> end-effector limits of the task; empty = the hardware range
        self.gripper = dict(self.task.gripper)
        self._limited_arms = [(self.robot.hand_of(arm), lim) for arm, lim in self.gripper.items()]
        self.api = RobotAPI(self)
        #: arm joint travel of the episode (``JointTravel``), recorded in the results
        self.travel = JointTravel(j for a in robot.arms.values() for j in a.joints)
        self.travel.update(backend.joint_positions())
        self.recorder.maybe_capture(backend, backend.sim_time(), force=True)
        if self.gripper:
            self._apply_gripper_limits(info)

    # --- joint commands ----------------------------------------------------------
    def command_joints(self, targets: Mapping[str, float]) -> list[str]:
        """Send joint targets with every end-effector joint held inside the task's limits.
        Returns a note per clamp, such as ``left: opening clamped to 40 mm``."""
        if not self._limited_arms:
            self.backend.command_joints(targets)
            return []
        out, notes = dict(targets), []
        for ee, lim in self._limited_arms:
            out, clamped = lim.clamp_joint_targets(ee, out)
            notes += [f"{lim.arm}: {n}" for n in clamped]
        self.backend.command_joints(out)
        return notes

    def _apply_gripper_limits(self, info) -> None:
        forced = {arm: lim.max_force for arm, lim in self.gripper.items() if lim.max_force is not None}
        if forced:
            if "gripper_force_limit" not in info.capabilities:
                raise ValueError(f"task {self.task.id} caps the gripper force, which backend {info.name} cannot "
                                 "apply (no gripper_force_limit capability)")
            for arm, newtons in forced.items():
                self.backend.set_gripper_force(arm, newtons)
        start: dict[str, float] = {}
        for ee, lim in self._limited_arms:
            start.update(lim.start_targets(ee))
        if start:
            self.command_joints(start)
            for _ in range(int(START_SECONDS / 0.1)):
                self.advance(0.1)
                q = self.backend.joint_positions()
                if all(abs(q[j] - v) < START_TOLERANCE for j, v in start.items()):
                    break
        self.log("gripper_limits", limits={arm: lim.to_json() for arm, lim in self.gripper.items()})

    # --- time -----------------------------------------------------------------
    def advance(self, seconds: float, on_tick: Callable[[float], None] | None = None) -> None:
        """Advance simulated time, calling ``on_tick(elapsed)`` before every control tick."""
        n = max(1, int(round(seconds / self.control_dt)))
        for i in range(n):
            if on_tick is not None:
                on_tick(i * self.control_dt)
            self.backend.step(self.control_dt)
            t = self.backend.sim_time()
            self.recorder.maybe_capture(self.backend, t)
            self.travel.maybe_update(t, self.backend.joint_positions)
            if t > self.task.budget.sim_seconds:
                raise StopEpisode("budget_exceeded:sim_seconds", f"{t:.1f} s simulated")

    def wall_elapsed(self) -> float:
        return time.monotonic() - self.t0_wall

    def model_latency(self, seconds: float) -> None:
        """Realtime mode: the world keeps running (holding the last targets) while the model thinks."""
        if self.time_mode == "realtime" and seconds > 0:
            self.advance(min(seconds, self.task.budget.sim_seconds))

    # --- budgets ----------------------------------------------------------------
    def check_wall(self) -> None:
        if self.wall_elapsed() > self.task.budget.wall_seconds:
            raise StopEpisode("budget_exceeded:wall_seconds", f"{self.wall_elapsed():.0f} s")

    def before_model_call(self) -> None:
        """Called by an agent before each language-model request."""
        self.check_wall()
        b = self.task.budget
        if self.usage.model_calls >= b.model_calls:
            raise StopEpisode("budget_exceeded:model_calls", f"{self.usage.model_calls} calls")
        if b.usd is not None and self.usage.usd >= b.usd:
            raise StopEpisode("budget_exceeded:usd", f"${self.usage.usd:.3f}")
        run = self.run_budget.exceeded()
        if run:
            raise StopEpisode(f"budget_exceeded:{run}", "run-wide cap reached")

    def after_model_call(self, *, input_tokens=0, output_tokens=0, reasoning_tokens=0,
                         cached_tokens=0, usd=0.0, seconds=0.0, stop: str = "", text: str = "",
                         thinking: str = "", prompt: str = "") -> None:
        """Account one model call and append it to the trace. ``thinking`` (the provider's summary of the
        reasoning) and ``prompt`` (the part of the prompt that changes per call) are recorded only when
        given: a task opts into a transcript (``ee.transcript``)."""
        self.usage.add_model(input_tokens=input_tokens, output_tokens=output_tokens,
                             reasoning_tokens=reasoning_tokens, cached_tokens=cached_tokens, usd=usd,
                             seconds=seconds)
        self.run_budget.model_calls += 1
        self.run_budget.usd += usd
        extra = {k: v[:TRANSCRIPT_CHARS] for k, v in (("thinking", thinking), ("prompt", prompt)) if v}
        self.log("model", stop=stop, input_tokens=input_tokens, output_tokens=output_tokens,
                 usd=round(usd, 5), seconds=round(seconds, 2), text=text[:2000], **extra)
        self.model_latency(seconds)

    def ready(self) -> None:
        """Called by a subject when its preparation is over (the arms at the task's start pose). A
        multi-robot session starts the robot's turn clock here; a single-robot session has none."""

    def say(self, keyword: str, text: str) -> str:
        """A line of the subject addressed to the referee (``move: e4``); returns the referee's answer,
        '' when the episode has no referee that listens."""
        return ""

    def before_api_call(self, name: str, arguments: Mapping[str, Any]) -> None:
        self.check_wall()
        if self.usage.api_calls >= self.task.budget.api_calls:
            raise StopEpisode("budget_exceeded:api_calls", f"{self.usage.api_calls} calls")
        self.usage.api_calls += 1

    def after_api_call(self, name: str, arguments: Mapping[str, Any], result: Any, refused: str = "") -> None:
        if refused:
            self.usage.refusals += 1
        self.log("api", name=name, arguments=dict(arguments), refused=refused,
                 result=_short(result))
        if self.judged and self.task.stop_on_success and self.success().ok:
            self.success_at = self.backend.sim_time()
            raise StopEpisode("success", f"after {name}")

    # --- chunk interface ---------------------------------------------------------
    def execute_chunk(self, actions, layout: list[str], hz: float) -> dict[str, Any]:
        """Apply joint-position action rows ``actions[t][i]`` for joints ``layout`` at ``hz``.
        Lockstep: the world waits for the next chunk; realtime: see :meth:`model_latency`."""
        rows = np.asarray(actions, float)
        if rows.ndim != 2 or rows.shape[1] != len(layout):
            raise ValueError(f"chunk must be (T, {len(layout)}), got {rows.shape}")
        self.before_api_call("execute_chunk", {"rows": int(rows.shape[0])})
        period = 1.0 / hz
        notes: list[str] = []
        for row in rows:
            notes += [n for n in self.command_joints(dict(zip(layout, row))) if n not in notes]
            self.advance(period)
        result: dict[str, Any] = {"executed": int(rows.shape[0])}
        if notes:
            result["gripper"] = notes
        self.after_api_call("execute_chunk", {"rows": int(rows.shape[0])}, result)
        return result

    # --- scoring -----------------------------------------------------------------
    def oracle(self) -> OracleSnapshot | None:
        """Privileged ground truth. Scorers and oracle-informed scripted subjects only."""
        return self.backend.oracle() if self.judged else None

    def context(self) -> dict[str, Any]:
        return {"footprint": self.footprint, "values": self.episode.values,
                "params": self.task.raw.get("params", {})}

    def success(self) -> Verdict:
        snap = self.backend.oracle()
        return evaluate(self.task.success, snap, self.context())

    def score(self) -> dict[str, Any]:
        if not self.judged:
            return {"success": False, "evidence": "not judged: the backend has no ground truth", "success_at": None,
                    "subgoals": [], "credit": 0.0, "final_state": None}
        snap = self.backend.oracle()
        ctx = self.context()
        final = evaluate(self.task.success, snap, ctx)
        ok = final.ok or self.success_at is not None
        subgoals = []
        total = sum(s.weight for s in self.task.subgoals) or 1.0
        credit = 0.0
        for s in self.task.subgoals:
            v = evaluate(s.predicate, snap, ctx)
            subgoals.append({"name": s.name, "ok": v.ok, "evidence": v.evidence, "weight": s.weight})
            credit += s.weight * v.ok
        return {"success": ok, "evidence": final.evidence, "success_at": self.success_at,
                "subgoals": subgoals, "credit": 1.0 if ok else round(credit / total, 4),
                "final_state": snap.to_json() if snap else None}

    # --- trace ---------------------------------------------------------------------
    def log(self, kind: str, **data) -> None:
        self.trace.append({"kind": kind, "sim_time": round(self.backend.sim_time(), 3),
                           "wall": round(self.wall_elapsed(), 2), **data})

    def usage_dict(self) -> dict[str, Any]:
        d = asdict(self.usage)
        d["usd"] = round(d["usd"], 5)
        d["model_seconds"] = round(d["model_seconds"], 2)
        return d


def _short(result: Any, limit: int = 1500) -> Any:
    if isinstance(result, (dict, list)):
        import json
        text = json.dumps(result, default=str)
        return result if len(text) <= limit else text[:limit] + "..."
    if isinstance(result, str):
        return result[:limit]
    return repr(result)[:limit]

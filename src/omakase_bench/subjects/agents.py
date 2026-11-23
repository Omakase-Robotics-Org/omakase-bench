"""Language-model subjects: the ``tools`` agent and the ``code`` agent.

Both see the same robot API (generated docs), the task instruction and a head
camera photo, and both have their every model request counted against the
episode and run budgets before it is sent. Neither decides success: a model's
"done" is recorded as its claim, and the oracle scores the episode.
"""
from __future__ import annotations

import json
import re
import textwrap
from typing import Any

from ..harness.session import Session, StopEpisode
from ..models.base import Image, Message, ModelAdapter, Reply, Request, Text, ToolResult, ToolSpec
from ..harness.api import ImageResult, Refused, encode_png
from ..harness.docgen import coerce_arguments, python_stubs, tool_specs
from ..sandbox import run_program, sandbox_report
from .base import SubjectCard, price_detail

FINISH = ToolSpec(
    "finish",
    "End the episode when you believe the task is complete or cannot be completed. The simulator then "
    "measures the result; your statement is recorded but decides nothing.",
    {"type": "object", "properties": {"note": {"type": "string", "description": "what you believe you achieved"}},
     "required": ["note"], "additionalProperties": False})

FRAME_TEXT = """Coordinates are metres in the robot's base frame: x forward, y to the robot's left,
z up from the floor under the robot. Angles are degrees. The robot has two arms ("left",
"right"), each ending in a two-finger parallel gripper; the TCP is the point midway between
the fingertip pads, and the fingertips reach about 1 cm past it along the direction the jaws
point. Objects are reported by `list_objects` with their centre position, size and yaw."""

_COUNT = {1: "one arm", 2: "two arms"}


def frame_text(robot=None) -> str:
    """The frame and arm description for ``robot``. A robot the published text describes word for word
    (two arms, ``left`` and ``right``, fingertips about 1 cm past the TCP) gets it with its line breaks."""
    if robot is None:
        return FRAME_TEXT
    arms = list(robot.arms)
    reach = round(100 * max((robot.hand_of(a).tip_reach for a in arms), default=0.0))
    text = (f"Coordinates are metres in the robot's base frame: x forward, y to the robot's left, z up from the "
            f"floor under the robot. Angles are degrees. The robot has {_COUNT.get(len(arms), f'{len(arms)} arms')} "
            f"({', '.join(chr(34) + a + chr(34) for a in arms)}), each ending in a two-finger parallel gripper; the "
            f"TCP is the point midway between the fingertip pads, and the fingertips reach about {reach} cm past it "
            "along the direction the jaws point. Objects are reported by `list_objects` with their centre "
            "position, size and yaw.")
    return FRAME_TEXT if text == " ".join(FRAME_TEXT.split()) else textwrap.fill(text, 96)


def _kind(robot=None) -> str:
    return "mobile manipulator robot" if robot is None or robot.mobile_base else "robot arm"


_TOOLS = """You control a {kind} in a physics simulation through function calls.
{frame}

Call exactly one function per turn. Each call returns what was measured; a refused call
did nothing and says why. Plan from measurements, not assumptions. When the task is done
(or cannot be done), call `finish`. The simulator decides whether the task succeeded."""

_CODE = """You control a {kind} in a physics simulation by writing Python programs.
{frame}

Write one complete program in a single ```python fenced block. A ready object `robot`
exposes the functions below (call them as `robot.move_ee(...)`); they block until the motion
is done and return plain dicts and lists. A refused or invalid call raises `RobotError`
(nothing moved); you may catch it. `math`, `numpy` (as `np`), `json` and `random` are
available; there is no file, network or process access. Use `print` to log what you need.
The program has a wall-clock limit of {{timeout}} s.

Robot API:

{{stubs}}"""


def tools_system(robot=None) -> str:
    return _TOOLS.format(kind=_kind(robot), frame=frame_text(robot))


def code_system(robot=None) -> str:
    """The code interface's system prompt, with ``{timeout}`` and ``{stubs}`` still to fill in."""
    return _CODE.format(kind=_kind(robot), frame=frame_text(robot))


#: the prompts for a robot the published frame text describes
TOOLS_SYSTEM = tools_system()
CODE_SYSTEM = code_system()

CODE_ITERATE = """After each run you get the program's output, any exception, the robot state and a head
camera photo. Then either send an improved program (it starts from the current world state;
the world is not reset), or reply with the single word DONE and no code if the task is complete."""


def _first_view(session: Session) -> list[Any]:
    img = session.api.camera_image.__wrapped__(session.api, "head")
    return [Image(img.png, "image/png", label="Head camera photo at the start:")]


def _account(session: Session, reply: Reply, prompt: str = "") -> None:
    u = reply.usage
    session.after_model_call(input_tokens=u.input_tokens + u.cached_tokens + u.cache_write_tokens,
                             output_tokens=u.output_tokens, reasoning_tokens=u.reasoning_tokens,
                             cached_tokens=u.cached_tokens, usd=u.usd, seconds=u.seconds, stop=reply.stop,
                             text=reply.text, thinking=reply.thinking, prompt=prompt)
    if reply.model:
        session.model_ids.add(reply.model)


def _model_failure(reply: Reply) -> StopEpisode | None:
    if reply.stop in ("error", "refusal", "max_tokens"):
        return StopEpisode(f"error:model_{reply.stop}", reply.error or reply.text[:300])
    return None


class ToolsAgent:
    kind = "agent"
    interface = "tools"

    def __init__(self, model: ModelAdapter, spec: str):
        self.model, self.spec = model, spec
        self.name = f"{spec}:tools"

    def card(self) -> SubjectCard:
        return SubjectCard(self.name, self.kind, self.interface, model=self.spec,
                           detail={"tools": [t.name for t in tool_specs()] + ["finish"], "max_calls_per_turn": 1,
                                   **price_detail(self.model)})

    def run(self, session: Session) -> str:
        session.model_ids = set()
        tools = tool_specs(session.robot, session.gripper) + [FINISH]
        conversation: list[Message] = [Message("user", (
            Text(f"TASK: {session.episode.instruction}"), *_first_view(session)))]
        nudged = False
        while True:
            session.before_model_call()
            reply = self.model.respond(Request(tools_system(session.robot), list(conversation), tools, max_calls=1))
            _account(session, reply)
            failure = _model_failure(reply)
            if failure:
                raise failure
            conversation.append(reply.as_message())
            if not reply.calls:
                if nudged:
                    session.model_claim = reply.text[:1000]
                    return "model_stopped"
                nudged = True
                conversation.append(Message("user", (Text(
                    "Call a function to act, or call `finish` if you are done."),)))
                continue
            results = []
            for call in reply.calls[:1]:
                if call.name == FINISH.name:
                    session.model_claim = str(call.arguments.get("note", ""))[:1000]
                    session.log("finish", note=session.model_claim)
                    return "finished"
                results.append(self._execute(session, call))
            for extra in reply.calls[1:]:
                results.append(ToolResult(extra.id, (Text("not run: one function call per turn"),), True))
            conversation.append(Message("user", results=tuple(results)))

    def _execute(self, session: Session, call) -> ToolResult:
        try:
            args = coerce_arguments(call.name, dict(call.arguments), session.robot)
        except ValueError as exc:
            session.log("invalid_call", name=call.name, arguments=dict(call.arguments), error=str(exc))
            return ToolResult(call.id, (Text(f"invalid call: {exc}"),), True)
        try:
            result = getattr(session.api, call.name)(**args)
        except Refused as exc:
            return ToolResult(call.id, (Text(f"refused: {exc}"),), True)
        if isinstance(result, ImageResult):
            return ToolResult(call.id, (Text(json.dumps(result.to_json())),
                                        Image(result.png, "image/png", label=f"{result.camera} camera:")))
        return ToolResult(call.id, (Text(json.dumps(result)),))


_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_program(text: str) -> str | None:
    blocks = _CODE_BLOCK.findall(text or "")
    return max(blocks, key=len).strip() if blocks else None


class CodeAgent:
    kind = "agent"
    interface = "code"

    def __init__(self, model: ModelAdapter, spec: str, *, iterative: bool = False, timeout: float = 120.0):
        self.model, self.spec, self.iterative, self.timeout = model, spec, iterative, timeout
        self.name = f"{spec}:code-{'iterative' if iterative else 'oneshot'}"

    def card(self) -> SubjectCard:
        return SubjectCard(self.name, self.kind, self.interface, model=self.spec,
                           detail={"mode": "iterative" if self.iterative else "one-shot",
                                   "program_timeout_s": self.timeout, "sandbox": sandbox_report(),
                                   **price_detail(self.model)})

    def system(self, robot=None, gripper=None) -> str:
        text = code_system(robot).replace("{timeout}", f"{self.timeout:.0f}").replace("{stubs}", python_stubs(robot, gripper))
        return text + ("\n\n" + CODE_ITERATE if self.iterative else "")

    def run(self, session: Session) -> str:
        session.model_ids = set()
        session.artifacts = {}
        conversation: list[Message] = [Message("user", (
            Text(f"TASK: {session.episode.instruction}"), *_first_view(session)))]
        iterations = session.task.budget.code_iterations if self.iterative else 1
        programs: list[str] = []
        for k in range(iterations):
            session.before_model_call()
            reply = self.model.respond(Request(self.system(session.robot, session.gripper), list(conversation), (), max_calls=0))
            _account(session, reply)
            failure = _model_failure(reply)
            if failure:
                raise failure
            conversation.append(reply.as_message())
            program = extract_program(reply.text)
            if program is None:
                if self.iterative and k > 0 and "DONE" in reply.text.upper():
                    session.model_claim = reply.text[:1000]
                    return "finished"
                conversation.append(Message("user", (Text("No ```python block found. Send one program."),)))
                continue
            programs.append(program)
            session.artifacts["program.py"] = _join_programs(programs)
            result = self._run(session, program, k)
            if result.stop is not None:
                raise result.stop
            if not self.iterative:
                session.model_claim = (result.stdout or "")[-500:]
                return "program_finished" if result.ok else ("program_timeout" if result.timed_out else "program_error")
            conversation.append(Message("user", tuple(self._feedback(session, result))))
        return "iterations_exhausted" if self.iterative else "no_program"

    def _run(self, session: Session, program: str, k: int):
        api = session.api

        def dispatch(fn: str, args: list, kwargs: dict):
            from ..harness.api import exposed_functions
            params = list(__import__("inspect").signature(exposed_functions().get(fn, lambda self: None)).parameters)[1:]
            if fn not in exposed_functions():
                raise ValueError(f"robot has no function {fn!r}")
            if len(args) > len(params):
                raise ValueError(f"{fn}: too many positional arguments")
            merged = dict(zip(params, args), **kwargs)
            merged = coerce_arguments(fn, merged, session.robot)
            return getattr(api, fn)(**merged)

        session.log("program", index=k, code=program)
        result = run_program(program, dispatch, timeout=self.timeout)
        session.log("program_result", index=k, exit_code=result.exit_code, timed_out=result.timed_out,
                    calls=result.calls, stdout=result.stdout[-3000:], stderr=result.stderr[-3000:],
                    stop=getattr(result.stop, "reason", None))
        return result

    def _feedback(self, session: Session, result) -> list[Any]:
        status = ("timed out" if result.timed_out else
                  "finished" if result.exit_code == 0 else f"failed (exit {result.exit_code})")
        parts: list[Any] = [Text(f"Program {status} after {result.calls} robot calls.\n"
                                 f"stdout (tail):\n{result.stdout[-2500:] or '(empty)'}\n"
                                 f"stderr (tail):\n{result.stderr[-2500:] or '(empty)'}")]
        try:
            state = session.api.get_state.__wrapped__(session.api)
            parts.append(Text("Robot state now: " + json.dumps(state)))
        except Refused:
            pass
        for img in result.images[-2:]:
            parts.append(Image(img.png, "image/png", label=f"{img.camera} photo taken by the program:"))
        head = session.api.camera_image.__wrapped__(session.api, "head")
        parts.append(Image(head.png, "image/png", label="Head camera now:"))
        parts.append(Text("Send the next program, or DONE if the task is complete."))
        return parts


def _join_programs(programs: list[str]) -> str:
    if len(programs) == 1:
        return programs[0] + "\n"
    return "\n\n".join(f"# --- program {i + 1} ---\n{p}" for i, p in enumerate(programs)) + "\n"

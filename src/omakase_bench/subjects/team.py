"""A team: one subject per robot of a multi-robot task.

A run binds robots to agents with ``--bind a=openai/gpt-6.2-astra:ee``; a robot
without a binding takes the run's ``--model``, and without one the agent its
task names (``robots[].agent: {model, interface}``). Each robot gets its
own subject instance, so agents running at the same time share no state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .base import SubjectCard


@dataclass
class TeamSubject:
    members: Mapping[str, Any]
    kind: str = "team"
    interface: str = "team"
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = " | ".join(f"{rid}={s.name}" for rid, s in self.members.items())

    def card(self) -> SubjectCard:
        return SubjectCard(self.name, self.kind, self.interface,
                           detail={"members": {rid: {"name": s.name, "kind": s.kind, "interface": s.interface,
                                                     **{k: v for k, v in s.card().detail.items()
                                                        if k == "price_usd_per_mtok"}}
                                               for rid, s in self.members.items()}})

    def run(self, session) -> str:
        raise RuntimeError("a team drives a multi-robot episode through the runner, not one session")


def parse_binding(text: str) -> tuple[str, str, str | None]:
    """``a=openai/gpt-6.2-astra:ee`` -> ("a", "openai/gpt-6.2-astra", "ee")."""
    rid, sep, rest = text.partition("=")
    if not sep or not rid or not rest:
        raise ValueError(f"binding {text!r}: expected <robot>=<provider/model>[:<interface>]")
    spec, _, interface = rest.partition(":")
    return rid, spec, interface or None


def build_team(task, make: Callable[[str, str | None, str], Any], bindings: Mapping[str, tuple[str, str | None]],
               default: tuple[str, str | None] | None = None) -> TeamSubject:
    """The subjects of ``task``'s robots. Precedence per robot: ``bindings`` (the run's ``--bind``),
    ``default`` (the run's ``--model``), the task's ``agent``. ``make(spec, interface, robot_id)``
    builds one subject."""
    unknown = set(bindings) - set(task.team.ids)
    if unknown:
        raise ValueError(f"task {task.id} has no robot(s) {sorted(unknown)}; robots: {', '.join(task.team.ids)}")
    members = {}
    for slot in task.team.robots:
        if slot.id in bindings:
            spec, interface = bindings[slot.id]
        elif default is not None:
            spec, interface = default
        elif slot.agent.get("model"):
            spec, interface = slot.agent["model"], slot.agent.get("interface")
        else:
            raise ValueError(f"robot {slot.id} of {task.id} has no agent: bind it with --bind {slot.id}=<model>")
        members[slot.id] = make(spec, interface or slot.agent.get("interface"), slot.id)
    return TeamSubject(members)

"""What a benchmark subject is: something that drives one episode through a session.

``interface`` says which control surface it consumes: ``tools`` (one robot API
call per model turn), ``code`` (a written program against the robot API),
``chunks`` (joint action chunks, the VLA interface), ``ee`` (one row of
fingertip targets per step from camera images, no object list) or ``scripted``
(a hand-written program, scored like any other subject). ``kind`` is a display
label (agent, vla, scripted).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from ..harness.session import Session

INTERFACES = ("tools", "code", "chunks", "ee", "scripted")


@dataclass(frozen=True)
class SubjectCard:
    name: str
    kind: str
    interface: str
    model: str = ""
    detail: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Subject(Protocol):
    name: str
    kind: str
    interface: str

    def card(self) -> SubjectCard: ...

    def run(self, session: Session) -> str:
        """Drive the episode; return how it ended from the subject's side
        (e.g. ``finished``). The session may stop it earlier with ``StopEpisode``."""
        ...


def price_detail(model) -> dict[str, Any]:
    """The price a model's calls are costed with, for a subject card: USD per million tokens, and whether
    it is a published list price or an assumption (``models/pricing.py``). {} for a model without one."""
    p = getattr(model, "price", None)
    if p is None:
        return {}
    return {"price_usd_per_mtok": {"input": p.input, "output": p.output, "cached_input": p.cached_input,
                                   "cache_write": p.cache_write, "assumed": bool(p.assumed)}}

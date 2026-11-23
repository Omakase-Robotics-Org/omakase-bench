"""Subject registry: a ``--model`` spec plus an interface becomes a subject.

    scripted/oracle               hand-written, oracle-informed programs (upper bound)
    stub/hold                     chunk-interface stub that holds still
    <provider>/<model>            a language model through the tools, code or ee interface
"""
from __future__ import annotations

from typing import Any

from .base import INTERFACES, Subject, SubjectCard


def build_subject(spec: str, interface: str | None = None, *, code_mode: str = "one-shot",
                  program_timeout: float = 120.0, model_options: dict[str, Any] | None = None) -> Subject:
    if spec.startswith("scripted/"):
        from .scripted import ScriptedSubject
        if spec != "scripted/oracle":
            # any other name would run the reference solutions under a label that reads like a model
            raise ValueError(f"unknown scripted subject {spec!r}; known: scripted/oracle")
        return ScriptedSubject("oracle")
    if spec.startswith("stub/"):
        from .chunks import ChunkSubject, HoldPolicy
        if spec != "stub/hold":
            raise ValueError(f"unknown stub {spec!r}")
        return ChunkSubject(HoldPolicy())
    interface = interface or "tools"
    from ..models import build
    model = build(spec, **(model_options or {}))
    if interface == "tools":
        from .agents import ToolsAgent
        return ToolsAgent(model, spec)
    if interface == "code":
        from .agents import CodeAgent
        if code_mode not in ("one-shot", "iterative"):
            raise ValueError("code mode must be one-shot or iterative")
        return CodeAgent(model, spec, iterative=code_mode == "iterative", timeout=program_timeout)
    if interface == "ee":
        from .ee_agent import EEAgent
        return EEAgent(model, spec)
    raise ValueError(f"interface {interface!r} is not available for {spec}; a model drives tools, code or ee")


__all__ = ["INTERFACES", "Subject", "SubjectCard", "build_subject"]

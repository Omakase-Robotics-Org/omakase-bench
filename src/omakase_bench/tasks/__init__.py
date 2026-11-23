"""Task packages and their registry.

    tasks/<family>/<task>/__init__.py      PACKAGE = TaskPackage(...)

Families: ``manipulation`` (a fixed base and its arms), ``mobile_manipulation`` (the base moves as part of
the task), ``multi_robot`` (several robots, one agent each), ``games`` (multi-robot tasks ruled move by
move). The registry imports every task package once; the scripted subject, the CLI and task lookup ask it,
so adding a task is adding a package.
"""
from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path

from .base import Program, TaskPackage

_ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def packages() -> tuple[TaskPackage, ...]:
    """Every task package, ordered by name."""
    found = []
    for init in sorted(_ROOT.glob("*/*/__init__.py")):
        family, task = init.parts[-3], init.parts[-2]
        module = importlib.import_module(f"{__name__}.{family}.{task}")
        package = getattr(module, "PACKAGE", None)
        if package is None:
            raise ImportError(f"{module.__name__} defines no PACKAGE")
        if package.name != f"{family}/{task}":
            raise ImportError(f"{module.__name__}: PACKAGE.name is {package.name!r}, expected '{family}/{task}'")
        found.append(package)
    return tuple(found)


def package_for(task_id: str) -> TaskPackage | None:
    """The package that registers ``task_id``."""
    return next((p for p in packages() if task_id in p.scripted), None)


def scripted_program(task_id: str) -> Program | None:
    package = package_for(task_id)
    return package.scripted[task_id] if package else None


def programs() -> dict[str, Program]:
    """Task id -> reference solution, over every package."""
    return {task: program for p in packages() for task, program in p.scripted.items()}


def definitions() -> list[Path]:
    """The ``task.yaml`` of every package that has one."""
    return [p.definition for p in packages() if p.definition is not None]

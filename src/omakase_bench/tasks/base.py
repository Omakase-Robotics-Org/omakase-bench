"""What a task package registers.

A task is a package ``tasks/<family>/<task>/``. Its definition is YAML (``task.yaml`` in the package, or
the copy a published suite holds): scene, variation, instruction, success, budget, and the names of the
package's own hooks (``generator``, ``goal``, ``referee``). What YAML cannot name is registered here: the
reference solution of each task id and the package's command-line tools.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

if TYPE_CHECKING:
    from ..harness.api import RobotAPI
    from ..harness.session import Session

#: a reference solution: drives one episode (or one turn of a multi-robot task) through the robot API
Program = Callable[["RobotAPI", "Session"], None]


@dataclass(frozen=True)
class TaskPackage:
    #: ``<family>/<task>``, the package's path under ``tasks/``
    name: str
    summary: str
    #: task id -> its reference solution, run by the ``scripted/oracle`` subject
    scripted: Mapping[str, Program] = field(default_factory=dict)
    #: task id -> ``moves(snapshot, values, params)``: the pick-and-place moves of a reference solution that
    #: acts through end-effector rows (``subjects/scripted_ee.py`` turns them into waypoints)
    moves: Mapping[str, Callable[..., list]] = field(default_factory=dict)
    #: adds the package's commands to the CLI: ``commands(subparsers)``; each parser sets ``fn``
    commands: Callable[[Any], None] | None = None
    #: the package directory; ``task.yaml`` there defines a task that no published suite holds yet
    directory: Path = Path(".")

    @property
    def family(self) -> str:
        return self.name.split("/")[0]

    @property
    def definition(self) -> Path | None:
        path = self.directory / "task.yaml"
        return path if path.exists() else None

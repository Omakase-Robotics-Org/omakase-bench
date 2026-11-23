"""Pick one cylinder out of a tight grid: the task has to limit the gripper opening.

Definition: ``suites/grip/v1/packed_grid_pick.yaml``. ``generators.py`` holds the scene generator and the
moves of the reference solution; it is part of the hash of ``grip@1``.
"""
from __future__ import annotations

from pathlib import Path

from ....subjects.scripted_ee import solve
from ...base import TaskPackage
from . import generators

PACKAGE = TaskPackage(
    name="manipulation/packed_grid",
    summary="lift the named cylinder out of a 3 x 3 grid under a task-level gripper limit",
    scripted={task: solve for task in generators.MOVES},
    moves=generators.MOVES,
    directory=Path(__file__).parent,
)

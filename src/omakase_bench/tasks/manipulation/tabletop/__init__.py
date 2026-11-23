"""Tabletop manipulation with a goal per episode: chess_move, stack, sort, clutter_pick.

Definitions: ``suites/manip/v1/*.yaml``. ``generators.py`` holds the scene generators, the ``goal`` checks
and the pick-and-place moves of the reference solutions; it is part of the hash of every suite whose tasks
name it.
"""
from __future__ import annotations

from pathlib import Path

from ....subjects.scripted_ee import solve
from ...base import TaskPackage
from . import generators

PACKAGE = TaskPackage(
    name="manipulation/tabletop",
    summary="move a chess piece, stack blocks, sort by colour, pick from clutter",
    scripted={task: solve for task in generators.MOVES},
    moves=generators.MOVES,
    directory=Path(__file__).parent,
)

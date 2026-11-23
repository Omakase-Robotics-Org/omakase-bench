"""Two robots pass blocks across a shared zone, taking turns under a referee.

Definition: ``suites/team/v1/block_relay.yaml``. ``relay.py`` holds the generator, ``RelayReferee`` and
the moves of the reference solution; it is part of the hash of ``team@1``.
"""
from __future__ import annotations

from pathlib import Path

from ....subjects.scripted_ee import solve_turn
from ...base import TaskPackage
from . import relay

PACKAGE = TaskPackage(
    name="multi_robot/block_relay",
    summary="two robots relay blocks through a shared zone, one turn each",
    scripted={task: solve_turn for task in relay.MOVES},
    moves=relay.MOVES,
    directory=Path(__file__).parent,
)

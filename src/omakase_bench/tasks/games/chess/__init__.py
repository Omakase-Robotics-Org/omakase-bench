"""A game of chess between two robots facing each other across a table, one model per robot.

    chess   two D1s, White (robot a) and Black (robot b), a club-size Staunton set on a
            45 mm board between them. In its turn a robot names its move to the referee
            (``move: Nf3``), moves the pieces with its own arms and says ``done``.

The referee (:class:`~.referee.ChessReferee`) owns the rules and the logical position
(:mod:`.rules`). It answers every named move: an illegal one is refused with the
reason (``max_illegal`` attempts per turn, then the game is lost), a legal one is
accepted with one line. The physical moves it takes (the captured piece to a free
capture slot of the mover, the king and the rook when castling, the pawn taken en
passant from its own square) and their order are the player's; the referee only
checks where the pieces end up. After the turn it compares every piece
with the position within ``tolerance_m`` (``slot_tolerance_m`` in the capture
slots), upright; a robot that says ``done`` too early gets the list of
differences and ``correction_replies`` more replies to put them right. What still differs then is a
manipulation fault of the mover: the referee puts those pieces where the
position has them (logged, and shown in the trace). A promoted pawn is
exchanged by the referee for a spare piece of the chosen kind once it stands on
its square. The game ends by the rules (checkmate, stalemate, insufficient
material, draws that can be claimed), by a forfeit, on time, or by a contact. With
``max_moves`` set, a game is cut off after that many moves per side: adjudicated
by material (``max_moves_action: adjudicate``, the default) or stopped without a
result as a harness stop (``stop``, what ``games/chess`` uses as a backstop). A contact
between the two robots loses the game for the mover, or for the waiting robot when
it was not in its waiting position (:meth:`~.referee.ChessReferee._not_waiting`). A turn
that a budget or an error cuts short (``TurnReport.ended`` starting with
``budget_exceeded:`` or ``error:``) ends the episode as a harness stop: no
result, the clocks as before that turn.

Players see their own cameras, the move list in SAN, the position as FEN and
the side to move; never piece poses. The scripted player (:func:`~.player.scripted_turn`)
reads ground truth and plays the moves of ``params.scripted_game``.

Definition: ``suites/games/v1/chess.yaml``. Modules: ``rules`` (legal moves, SAN, FEN, PGN, endings), ``layout``
(board geometry, reach tables), ``scene`` (generator and the text each player reads), ``referee``,
``player`` (scripted), and the tools behind ``omakase-bench chess``: ``tour`` (reach check), ``resume``
(task file that continues a stopped game), ``cpl`` (centipawn loss from a UCI engine), ``video`` (cuts of a
recorded game).
"""
from __future__ import annotations

from pathlib import Path

from ...base import TaskPackage
from .cli import commands
from .player import scripted_turn

PACKAGE = TaskPackage(
    name="games/chess",
    summary="two robots play a game of chess on a physical board, one model per robot",
    scripted={"games/chess": scripted_turn},
    commands=commands,
    directory=Path(__file__).parent,
)

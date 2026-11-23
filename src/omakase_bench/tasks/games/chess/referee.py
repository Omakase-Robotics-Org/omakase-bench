"""``ChessReferee``: the rules of a turn, the clocks, faults, contacts and the result (see the package
docstring)."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ....core.backend import OracleSnapshot
from ....core.task import SceneSpec, Shape
from ....core.team import GameResult, GameView, Ruling, TurnReport, inter_robot_contacts
from . import rules as chess
from .layout import (COLOUR_NAME, HARNESS_ENDS, KIND_NAME, KIND_OF, Layout, _clock, _mm, _quat_yaw, _upright_cos,
                     _yaw_of)

_UCI = re.compile(r"[a-h][1-8][a-h][1-8][qrbnQRBN]?")


@dataclass
class SubMove:
    """One pick and place: ``piece`` from ``src`` to ``dst`` (square names, ``slot:<colour>:<k>``)."""
    piece: str
    src: str
    dst: str


@dataclass
class TurnState:
    move: Any = None                     # the accepted chess.Move
    san: str = ""
    plan: list[SubMove] = field(default_factory=list)
    illegal: int = 0
    finished: bool = False
    correcting: bool = False
    #: the number of the reply in which the robot was first told what differs; the replies after it correct
    correction_from: int = 0
    forfeit: str = ""
    handover: bool = False
    holder: dict = field(default_factory=dict)          # piece -> the arm that last held it
    adjourned: bool = False
    flagged: bool = False                                # the side's clock ran out in this turn
    contact: dict = field(default_factory=dict)         # the ruling on a contact between the robots
    last_sim: float | None = None                        # simulated time of the last poll of the turn


class ChessReferee:
    """Rules, logical position, physical checks and restores of a game between two robots."""

    #: what it needs from a backend beyond ground truth: exact contacts between the robots, and putting a
    #: displaced piece back
    requires = ("contacts", "object_pose")

    def __init__(self, values: Mapping[str, Any], params: Mapping[str, Any]):
        self.v, self.p = values, params
        self.board = chess.Board()
        self.where = {n: d["at"] for n, d in values["pieces"].items()}
        self.kind = {n: d["kind"] for n, d in values["pieces"].items()}
        self.colour = {n: d["colour"] == "white" for n, d in values["pieces"].items()}
        self.side = {chess.WHITE: str(params["sides"]["white"]), chess.BLACK: str(params["sides"]["black"])}
        self.colour_of = {r: c for c, r in self.side.items()}
        self.tol = float(params.get("tolerance_m", 0.010))
        self.slot_tol = float(params.get("slot_tolerance_m", 0.015))
        self.max_illegal = int(params.get("max_illegal", 3))
        self.correction_replies = int(params.get("correction_replies", 4))
        #: moves per side after which the game is cut off (None: never), and what the cut-off is:
        #: "adjudicate" (a result by material) or "stop" (a harness stop without a result, resumable)
        self.max_moves = None if params.get("max_moves", 40) is None else int(params.get("max_moves", 40))
        self.max_moves_action = str(params.get("max_moves_action", "adjudicate"))
        if self.max_moves_action not in ("adjudicate", "stop"):
            raise ValueError(f"params.max_moves_action must be adjudicate or stop, got {self.max_moves_action!r}")
        self.values = {KIND_OF[k]: float(v) for k, v in params.get("values", {"pawn": 1, "knight": 3, "bishop": 3,
                                                                              "rook": 5, "queen": 9}).items()}
        self.layout: Layout | None = None
        self.turn = TurnState()
        self.san: list[str] = []
        self.ending = ""
        self.result_code = "*"
        #: a manipulation fault (the referee had to restore the position) costs this much clock, and the
        #: ``fault_limit``-th fault of one side in a game loses it; None: no limit
        self.fault_penalty_s = float(params.get("fault_clock_penalty_s", 0.0))
        self.fault_limit = params.get("fault_limit")
        self.metrics = {r: {"moves": 0, "illegal_attempts": 0, "manipulation_faults": 0, "knocked_over": 0,
                            "restored_pieces": 0, "corrections": 0, "handover_turns": 0, "captures": 0,
                            "contacts": 0, "fault_penalty_s": 0.0, "fault_limit": self.fault_limit}
                        for r in self.side.values()}
        #: robot -> manipulation faults in the game so far (a resumed game starts with those it had)
        self.faults = {r: int((params.get("start_faults") or {}).get(r, 0)) for r in self.side.values()}
        #: robot -> what the referee tells it at the start of its next turn about its own last one
        self.pending: dict[str, str] = {}
        #: waiting position: arm joints and lift this close to the idle pose (and the base at its station)
        self.idle_tol_rad = math.radians(float(params.get("idle_tolerance_deg", 5)))
        self.idle_lift_tol = float(params.get("idle_lift_tolerance_m", 0.01))
        self._contacts_seen: set[tuple[str, str]] = set()
        self.events: list[dict[str, Any]] = []
        clock = params.get("clock") or {}
        self.base_s = float(clock.get("base_s", 3600.0))
        self.increment_s = float(clock.get("increment_s", 30.0))
        #: robot -> seconds left on its clock (Fischer: base, plus the increment after each completed move)
        self.clock = {r: self.base_s for r in self.side.values()}
        # a game resumed from a recorded position: its moves so far and its clocks
        for san in params.get("start_moves") or []:
            move = self.board.parse_san(san)
            self.san.append(self.board.san(move))
            self.board.push(move)
        self.clock.update({r: float(v) for r, v in (params.get("start_clocks") or {}).items()})
        #: per completed turn: robot, SAN, model (think) and simulated (execute) seconds, clock after
        self.move_log: list[dict[str, Any]] = []

    # --- geometry ----------------------------------------------------------------------------------
    def _layout(self, snap: OracleSnapshot) -> Layout:
        if self.layout is None:
            table = snap.objects["table"]
            top = table.pos[2] + table.size[2] / 2
            # the cell is fixed to the robots' stations, not to where their bases stand now
            stations = self.v.get("stations") or {r: snap.bases[r] for r in self.side.values()}
            fake = SceneSpec(robot="", fixtures=(Shape("table", "box", table.size, (0, 0, top - table.size[2] / 2)),),
                             robots=tuple(_Placement(r, tuple(stations[r])) for r in self.side.values()))
            self.layout = Layout.from_params(fake, self.p)
        return self.layout

    def location_xyz(self, snap: OracleSnapshot, name: str, where: str | None = None) -> np.ndarray:
        L = self._layout(snap)
        where = where if where is not None else self.where[name]
        h = L.height(self.kind[name])
        if where.startswith("slot:"):
            _, c, k = where.split(":")
            if k == "*":
                k = self.free_slot(c == "w").split(":")[2]
            xy, bottom = L.slot_xy(c == "w", int(k)), L.table_top
        elif where.startswith("reserve:"):
            xy, bottom = L.reserve_xy(self.colour[name], where.split(":")[1]), L.table_top
        else:
            xy, bottom = L.square_xy(chess.parse_square(where)), L.board_top
        return np.array([xy[0], xy[1], bottom + h / 2])

    def mismatches(self, snap: OracleSnapshot, where: Mapping[str, str] | None = None) -> list[tuple[str, str]]:
        """Pieces that are not where the position has them: (piece, reason)."""
        where = where or self.where
        out = []
        for name, at in where.items():
            o = snap.objects[name]
            if at.endswith(":*"):
                at = self._slot_of(snap, name, at, where) or at
                if at.endswith(":*"):
                    reason = "knocked over" if _upright_cos(o.quat) < float(self.p.get("upright_cos", 0.95)) else (
                        "still held" if name in snap.held_by else "not in a free capture slot of its captor")
                    out.append((name, reason))
                    continue
            exp = self.location_xyz(snap, name, at)
            tol = self.slot_tol if at.startswith(("slot:", "reserve:")) else self.tol
            if _upright_cos(o.quat) < float(self.p.get("upright_cos", 0.95)):
                out.append((name, "knocked over"))
            elif name in snap.held_by:
                out.append((name, "still held"))
            elif float(np.linalg.norm(np.array(o.pos[:2]) - exp[:2])) > tol:
                out.append((name, f"{_mm(float(np.linalg.norm(np.array(o.pos[:2]) - exp[:2])))} mm from the centre of "
                                  f"{self._place_text(at)}"))
            elif abs(o.pos[2] - exp[2]) > 0.006:
                out.append((name, f"not standing on {self._place_text(at)}"))
        return out

    def _slot_of(self, snap: OracleSnapshot, name: str, at: str, where: Mapping[str, str]) -> str | None:
        """The free capture slot (of the colour in ``at`` = 'slot:<c>:*') a piece stands in, if any."""
        L = self._layout(snap)
        c = at.split(":")[1]
        used = {w for n, w in where.items() if n != name}
        xy = np.array(snap.objects[name].pos[:2])
        for k in range(L.n_slots):
            slot = f"slot:{c}:{k}"
            if slot not in used and float(np.linalg.norm(L.slot_xy(c == "w", k) - xy)) <= self.slot_tol:
                return slot
        return None

    @staticmethod
    def _place_text(at: str) -> str:
        if at.startswith("slot:"):
            k = at.split(":")[2]
            return "a free capture slot" if k == "*" else f"capture slot {int(k) + 1}"
        if at.startswith("reserve:"):
            return "its reserve spot"
        return at

    def _describe(self, name: str) -> str:
        return f"{COLOUR_NAME[self.colour[name]]} {self.kind[name]}"

    def piece_on(self, square: str) -> str:
        return next(n for n, w in self.where.items() if w == square)

    def free_slot(self, colour: bool, where: Mapping[str, str] | None = None) -> str:
        used = set((where or self.where).values())
        c = "w" if colour else "b"
        n = len(self.p["capture_slots"]["y"]) * int(self.p["capture_slots"]["per_row"])
        return next(f"slot:{c}:{k}" for k in range(n) if f"slot:{c}:{k}" not in used)

    # --- rules -------------------------------------------------------------------------------------
    def parse(self, text: str):
        """The legal move named by ``text`` (SAN or UCI); ValueError with the reason otherwise."""
        t = text.strip().split()[0] if text.strip() else ""
        t = t.strip(".,;!?").replace("0-0-0", "O-O-O").replace("0-0", "O-O")
        if not t:
            raise ValueError("no move given")
        if _UCI.fullmatch(t):
            # square to square also reads as a pawn move in SAN (g1f3: a pawn from g1), so UCI is tried first
            move = chess.Move.from_uci(t.lower())
            if move in self.board.legal_moves:
                return move
        try:
            return self.board.parse_san(t)
        except chess.IllegalMoveError:
            raise ValueError(f"{t} is not legal in this position") from None
        except chess.AmbiguousMoveError:
            raise ValueError(f"{t} is ambiguous here; name the file or rank of the piece") from None
        except chess.InvalidMoveError:
            pass
        try:
            move = chess.Move.from_uci(t.lower())
        except ValueError:
            raise ValueError(f"'{t}' is not a move in SAN or UCI notation") from None
        if move not in self.board.legal_moves:
            raise ValueError(f"{t} is not legal in this position")
        return move

    def plan(self, move) -> list[SubMove]:
        """Where the pieces ``move`` touches end up: the captured piece in any free capture slot of the
        mover, the moving piece on its square, the rook of a castling king on its square. The order and
        the way the robot carries them out are its own."""
        b, out = self.board, []
        src, dst = chess.square_name(move.from_square), chess.square_name(move.to_square)
        if b.is_capture(move):
            vsq = move.to_square
            if b.is_en_passant(move):
                vsq = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
            victim = self.piece_on(chess.square_name(vsq))
            out.append(SubMove(victim, chess.square_name(vsq), f"slot:{'w' if b.turn else 'b'}:*"))
        out.append(SubMove(self.piece_on(src), src, dst))
        if b.is_castling(move):
            rank = chess.square_rank(move.from_square)
            r_from, r_to = (7, 5) if chess.square_file(move.to_square) == 6 else (0, 3)
            rs, rd = chess.square_name(chess.square(r_from, rank)), chess.square_name(chess.square(r_to, rank))
            out.append(SubMove(self.piece_on(rs), rs, rd))
        return out

    def where_after(self, plan: Sequence[SubMove]) -> dict[str, str]:
        w = dict(self.where)
        for s in plan:
            w[s.piece] = s.dst
        return w

    def material(self) -> dict[bool, float]:
        return {c: sum(self.values.get(pc.piece_type, 0.0) for pc in self.board.piece_map().values() if pc.color == c)
                for c in (chess.WHITE, chess.BLACK)}

    def movetext(self) -> str:
        out = []
        for i, s in enumerate(self.san):
            out.append(f"{i // 2 + 1}. {s}" if i % 2 == 0 else s)
        return " ".join(out)

    # --- messages ----------------------------------------------------------------------------------
    def _message(self, robot: str, note: str = "") -> str:
        colour = self.colour_of[robot]
        b = self.board
        hist = self.movetext() or "none yet"
        mat = self.material()
        text = (f"Referee: you play {COLOUR_NAME[colour].capitalize()}. Moves so far (SAN): {hist}. "
                f"Position (FEN): {b.fen()}. Side to move: {COLOUR_NAME[b.turn].capitalize()}, move {b.fullmove_number}. "
                f"Material (pawn 1, knight 3, bishop 3, rook 5, queen 9): White {mat[chess.WHITE]:g}, "
                f"Black {mat[chess.BLACK]:g}.")
        other = self.side[not colour]
        text += (f" Clocks: you {_clock(self.clock[robot])}, the other robot {_clock(self.clock[other])} "
                 f"(+{self.increment_s:g} s per completed move).")
        if b.is_check():
            text += " You are in check."
        if note:
            text += " " + note
        if robot in self.pending:
            text += " " + self.pending.pop(robot)
        return text

    # --- referee interface -------------------------------------------------------------------------------
    def start(self, game: GameView, snap: OracleSnapshot) -> Ruling:
        first = self.side[self.board.turn]
        self._layout(snap)
        return Ruling(next_robot=first, messages={first: self._message(first)},
                      note=f"{COLOUR_NAME[self.board.turn].capitalize()} to move")

    def _used(self, game: GameView, robot: str) -> tuple[float, float]:
        """(model seconds, simulated seconds) of the robot's current turn so far."""
        live = game.live
        start = (live.get("turn_start") or {}).get(robot)
        if start is None or "usage" not in live:
            return 0.0, 0.0
        sim0, ms0 = start
        sim = self.turn.last_sim if self.turn.last_sim is not None else live["sim_time"]()
        return max(0.0, live["usage"][robot].model_seconds - ms0), max(0.0, sim - sim0)

    def _flag(self, game: GameView, robot: str) -> bool:
        think, execute = self._used(game, robot)
        if think + execute >= self.clock[robot]:
            self.turn.flagged = True
        return self.turn.flagged

    def hear(self, game: GameView, robot: str, keyword: str, text: str, snap: OracleSnapshot) -> str:
        t = self.turn
        colour = self.colour_of.get(robot)
        if colour is None or colour != self.board.turn:
            return "It is not your turn."
        if "sim_time" in game.live:
            t.last_sim = game.live["sim_time"]()
        if self._flag(game, robot):
            return "Your clock has run out."
        if t.forfeit:
            return f"The game is over: {t.forfeit}."
        if keyword == "move":
            if t.move is not None:
                return f"Your move {t.san} is already accepted; complete it and write 'done'."
            try:
                move = self.parse(text)
            except ValueError as exc:
                t.illegal += 1
                self.metrics[robot]["illegal_attempts"] += 1
                self.events.append({"kind": "illegal", "robot": robot, "text": text[:40], "reason": str(exc)})
                if t.illegal >= self.max_illegal:
                    t.forfeit = f"{COLOUR_NAME[colour]} named no legal move in {self.max_illegal} attempts"
                    return f"Illegal: {exc}. That was attempt {t.illegal} of {self.max_illegal}; you lose the game."
                return f"Illegal: {exc}. Attempt {t.illegal} of {self.max_illegal}; name another move."
            t.move, t.san, t.plan = move, self.board.san(move), self.plan(move)
            self.events.append({"kind": "accepted", "robot": robot, "san": t.san})
            return f"Accepted: {t.san}."
        if keyword == "adjourn":            # the scripted player at the end of its line; not offered to models
            t.adjourned = True
            return "Adjourned."
        if keyword == "done":
            if t.move is None:
                return "Name your move first: 'move: <SAN>'."
            return self._check_done(game, robot, snap)
        return f"Unknown referee line '{keyword}'."

    def _station_error(self, robot: str, snap: OracleSnapshot) -> tuple[float, float] | None:
        """(distance in m, heading error in deg) of a robot's base from its station; None when unknown."""
        station = (self.v.get("stations") or {}).get(robot)
        if station is None or robot not in snap.bases:
            return None
        x, y, yaw = snap.bases[robot]
        return (math.hypot(x - station[0], y - station[1]),
                abs((math.degrees(yaw - station[2]) + 180) % 360 - 180))

    def _not_waiting(self, game: GameView, robot: str, snap: OracleSnapshot) -> str:
        """Why ``robot`` is not in its waiting position ('' when it is): its arms in the idle pose within
        ``idle_tolerance_deg``, its lift within ``idle_lift_tolerance_m`` and its base within the station
        tolerance."""
        why = []
        err = self._station_error(robot, snap)
        tol, tol_deg = float(self.p.get("station_tolerance_m", 0.03)), float(self.p.get("station_tolerance_deg", 3))
        if err is not None and (err[0] > tol or err[1] > tol_deg):
            why.append(f"its base was {_mm(err[0])} mm and {err[1]:.0f} deg from its station")
        if "idle_error" in game.live:
            arm, lift = game.live["idle_error"](robot)
            if arm > self.idle_tol_rad:
                why.append(f"an arm joint was {math.degrees(arm):.0f} deg from the idle pose")
            if lift > self.idle_lift_tol:
                why.append(f"its lift was {_mm(lift)} mm from the idle height")
        return "; ".join(why)

    def _contact(self, game: GameView, mover: str, snap: OracleSnapshot) -> bool:
        """Rule on a contact between the two robots that is new in ``snap``: the mover loses, unless the
        waiting robot was not in its waiting position, in which case the waiting robot loses."""
        if self.turn.contact:
            return True
        new = [p for p in inter_robot_contacts(snap, list(self.side.values())) if p not in self._contacts_seen]
        if not new:
            return False
        self._contacts_seen.update(new)
        waiting = self.side[not self.colour_of[mover]]
        off = self._not_waiting(game, waiting, snap)
        loser = waiting if off else mover
        parts = [list(p) for p in new]
        touched = ", ".join(f"{a} against {b}" for a, b in new)
        why = (f"robot {waiting} was touched while not in its waiting position ({off})" if off else
               f"robot {mover}, whose turn it was, touched robot {waiting}, which was in its waiting position")
        self.turn.contact = {"kind": "contact", "robot": loser, "mover": mover, "hit": waiting, "parts": parts,
                             "ruled_against": loser, "why": why, "sim_time": round(float(snap.time), 2)}
        for r in self.side.values():
            self.metrics[r]["contacts"] += 1
        self.events.append(self.turn.contact)
        self.turn.forfeit = f"{COLOUR_NAME[self.colour_of[loser]]} lost by contact: {why} ({touched})"
        return True

    def _hands_ok(self, robot: str, snap: OracleSnapshot) -> list[str]:
        L = self._layout(snap)
        prefix = f"{robot}/"
        clear = L.board_top + float(self.p.get("hands_clear_z", 0.11))
        out = []
        for k, v in snap.gripper_opening.items():
            if k.startswith(prefix) and v < float(self.p.get("open_min", 0.02)):
                out.append(f"the {k[len(prefix):]} gripper is closed")
        for k, pos in snap.tcp.items():
            if k.startswith(prefix) and pos[2] < clear:
                out.append(f"the {k[len(prefix):]} hand is below z {_mm(clear)}")
        err = self._station_error(robot, snap)
        if err is not None:
            d, dyaw = err
            tol, tol_deg = float(self.p.get("station_tolerance_m", 0.03)), float(self.p.get("station_tolerance_deg", 3))
            if d > tol or dyaw > tol_deg:
                out.append(f"your base is {_mm(d)} mm and {dyaw:.0f} deg from its station (at most {_mm(tol)} mm and "
                           f"{tol_deg:g} deg)")
        return out

    def _check_done(self, game: GameView, robot: str, snap: OracleSnapshot) -> str:
        t = self.turn
        wrong = self.mismatches(snap, self.where_after(t.plan))
        hands = self._hands_ok(robot, snap)
        if not wrong and not hands:
            t.finished = True
            return "Position confirmed."
        current = self._replies_done(game, robot) + 1          # the reply this 'done' is part of
        if not t.correcting:
            t.correcting, t.correction_from = True, current
            self.metrics[robot]["corrections"] += 1
        left = max(0, self.correction_replies - (current - t.correction_from))
        items = [f"{self._describe(n)}: {why}" for n, why in wrong] + hands
        return (f"Not yet: {'; '.join(items)}. You have {left} more replies to correct this; then the referee restores "
                "the position and records a manipulation fault.")

    @staticmethod
    def _replies_done(game: GameView, robot: str) -> int:
        """Model replies of the robot whose actions are all carried out (0 for a subject that is no model)."""
        count = (game.live.get("replies") or {}).get(robot)
        return int(count()) if count is not None else 0

    def turn_over(self, game: GameView, robot: str, snap: OracleSnapshot) -> bool:
        t = self.turn
        t.last_sim = game.live["sim_time"]() if "sim_time" in game.live else snap.time
        if self._contact(game, robot, snap):
            return True
        if t.forfeit or t.finished or t.adjourned or self._flag(game, robot):
            return True
        for piece, arm in snap.held_by.items():
            if piece in self.where:
                if t.holder.get(piece, arm) != arm:
                    t.handover = True
                t.holder[piece] = arm
        if t.move is None:
            return False
        if t.correcting and self._replies_done(game, robot) - t.correction_from >= self.correction_replies:
            return True          # asked between two replies: the last correcting reply was carried out in full
        if not self.mismatches(snap, self.where_after(t.plan)) and not self._hands_ok(robot, snap):
            t.finished = True
            return True
        return False

    def _restore(self, snap: OracleSnapshot, names: Sequence[str]) -> dict:
        out = {}
        for n in names:
            pos = self.location_xyz(snap, n)
            o = snap.objects[n]
            yaw = _yaw_of(o.quat) if _upright_cos(o.quat) > 0.95 else self._layout(snap).bases[
                self.side[self.colour[n]]][2]
            out[n] = (tuple(float(v) for v in pos), _quat_yaw(yaw))
        return out

    def after_turn(self, game: GameView, report: TurnReport) -> Ruling:
        robot, snap, t = report.robot, report.after, self.turn
        colour = self.colour_of[robot]
        other = self.side[not colour]
        think, execute = self._used(game, robot)
        self.clock[robot] -= think + execute
        if self.clock[robot] <= 0.0 and not report.ended.startswith(HARNESS_ENDS):
            # the clock ran out in a model call after which the turn was not polled again
            t.flagged = True
        if self._contact(game, robot, snap):          # also one made on the way back to the idle pose
            c = t.contact
            loser = self.colour_of[c["ruled_against"]]
            self.ending, self.result_code = t.forfeit, ("0-1" if loser == chess.WHITE else "1-0")
            self.move_log.append({"robot": robot, "san": t.san or "(contact)", "think_s": round(think, 1),
                                  "exec_s": round(execute, 1), "clock_s": round(self.clock[robot], 1)})
            self.turn = TurnState()
            return Ruling(next_robot=None, valid=False, note=f"turn {report.turn}: {self.ending}; {self.result_code}")
        if t.flagged:
            self._flag_fall(robot)
            self.move_log.append({"robot": robot, "san": t.san or "(flag)", "think_s": round(think, 1),
                                  "exec_s": round(execute, 1), "clock_s": 0.0})
            self.turn = TurnState()
            return Ruling(next_robot=None, valid=False, note=f"turn {report.turn}: {self.ending}; {self.result_code}")
        if t.adjourned:
            self._adjudicate(f"adjourned after {len(self.san)} half-moves")
            self.turn = TurnState()
            return Ruling(next_robot=None, note=f"turn {report.turn}: game adjourned; {self.result_code} ({self.ending})")
        if report.ended.startswith(HARNESS_ENDS) and not t.forfeit:
            # the turn was cut short by a budget or an error, not by the player: the game stops unfinished,
            # the clocks as they stood before the turn, so that it can be resumed from the recorded moves
            self.clock[robot] += think + execute
            self.ending = (f"stopped by the harness in {COLOUR_NAME[colour]}'s turn ({report.ended}) after "
                           f"{len(self.san)} half-moves")
            self.events.append({"kind": "harness_stop", "robot": robot, "reason": report.ended,
                                "accepted": t.san})
            self.turn = TurnState()
            return Ruling(next_robot=None, valid=False, stop=report.ended, note=f"turn {report.turn}: {self.ending}")
        if t.forfeit or t.move is None:
            reason = t.forfeit or f"{COLOUR_NAME[colour]} named no legal move in its turn ({report.ended})"
            self.ending, self.result_code = reason, ("0-1" if colour == chess.WHITE else "1-0")
            self.turn = TurnState()
            return Ruling(next_robot=None, valid=False, note=f"turn {report.turn}: {reason}; {self.result_code}")
        after = self.where_after(t.plan)
        for n, w in list(after.items()):
            if w.endswith(":*"):
                after[n] = self._slot_of(snap, n, w, after) or w
        wrong = self.mismatches(snap, after)
        for n, w in list(after.items()):
            if w.endswith(":*"):
                after[n] = self.free_slot(w.split(":")[1] == "w", after)
        self.where = after
        if self.board.is_capture(t.move):
            self.metrics[robot]["captures"] += 1
        self.board.push(t.move)
        self.san.append(t.san)
        self.metrics[robot]["moves"] += 1
        self.clock[robot] += self.increment_s
        self.move_log.append({"robot": robot, "san": t.san, "think_s": round(think, 1), "exec_s": round(execute, 1),
                              "clock_s": round(self.clock[robot], 1)})
        restore = self._restore(snap, [n for n, _ in wrong])
        notes = []
        fault_ends = ""
        if wrong:
            self.metrics[robot]["manipulation_faults"] += 1
            self.metrics[robot]["knocked_over"] += sum(1 for _, why in wrong if why == "knocked over")
            self.metrics[robot]["restored_pieces"] += len(wrong)
            self.faults[robot] += 1
            self.clock[robot] -= self.fault_penalty_s
            self.metrics[robot]["fault_penalty_s"] += self.fault_penalty_s
            self.move_log[-1]["clock_s"] = round(max(0.0, self.clock[robot]), 1)
            count = f"fault {self.faults[robot]}" + (f" of {self.fault_limit}" if self.fault_limit else "")
            penalty = (f", {_clock(self.fault_penalty_s)} taken from {COLOUR_NAME[colour]}'s clock, "
                       f"{_clock(max(0.0, self.clock[robot]))} left") if self.fault_penalty_s else ""
            notes.append(f"manipulation fault, {count}{penalty}: the referee restored " + ", ".join(
                f"{self._describe(n)} ({why})" for n, why in wrong))
            self.pending[robot] = (f"In your last turn the referee had to restore the position: manipulation {count}"
                                   + (f", {_clock(self.fault_penalty_s)} taken from your clock" if self.fault_penalty_s else "")
                                   + ".")
            self.events.append({"kind": "fault", "robot": robot, "san": t.san, "pieces": [n for n, _ in wrong],
                                "fault": self.faults[robot], "penalty_s": self.fault_penalty_s,
                                "clock_s": round(max(0.0, self.clock[robot]), 1)})
            if self.fault_limit and self.faults[robot] >= int(self.fault_limit):
                fault_ends = "limit"
            elif self.clock[robot] <= 0.0:
                fault_ends = "flag"
        if t.move.promotion:
            restore.update(self._promote(snap, t.move))
            notes.append(f"the referee exchanged the pawn on {chess.square_name(t.move.to_square)} for a "
                         f"{KIND_NAME[t.move.promotion]}")
        if t.handover:
            self.metrics[robot]["handover_turns"] += 1
        note = f"turn {report.turn}: {COLOUR_NAME[colour]} played {t.san}" + (f" ({'; '.join(notes)})" if notes else "")
        if t.handover:                  # in the trace only: the opponent is told the move, not how it was made
            note += "; hand-over between the arms"
        self.turn = TurnState()
        nxt, stop = other, ""
        b = self.board
        if b.is_game_over(claim_draw=True):       # a move that ends the game ends it, whatever its execution cost
            out = b.outcome(claim_draw=True)
            self.result_code = out.result()
            self.ending = out.termination.replace("_", " ")
            nxt = None
        elif fault_ends == "limit":
            self.result_code = "0-1" if colour == chess.WHITE else "1-0"
            self.ending = (f"loss by manipulation faults ({COLOUR_NAME[colour]}, fault {self.faults[robot]} of "
                           f"{self.fault_limit})")
            nxt = None
        elif fault_ends == "flag":
            self._flag_fall(robot, " after a manipulation fault penalty")
            nxt = None
        elif self.max_moves is not None and len(self.san) >= 2 * self.max_moves:
            nxt = None
            if self.max_moves_action == "stop":       # a backstop of the harness: it never decides a result
                stop = "budget_exceeded:max_moves"
                self.ending = f"stopped by the harness after {self.max_moves} moves per side ({stop})"
            else:
                self._adjudicate(f"after {self.max_moves} moves per side")
        if nxt is None and not stop:
            note += f"; game over: {self.result_code} ({self.ending})"
        elif stop:
            note += f"; {self.ending}"
        last = f"{COLOUR_NAME[colour].capitalize()} played {self.san[-1]}" + (
            f" ({'; '.join(notes)})" if notes else "") + "."
        return Ruling(next_robot=nxt, messages={nxt: self._message(nxt, last)} if nxt else {}, restore=restore,
                      valid=not wrong, note=note, stop=stop)

    def _flag_fall(self, robot: str, why: str = "") -> None:
        """The robot's clock has run out: it loses, unless the other side has no material to mate with."""
        colour = self.colour_of[robot]
        self.clock[robot] = 0.0
        mate_possible = any(p.piece_type != chess.KING for p in self.board.piece_map().values()
                            if p.color != colour) and not self._only_minor(not colour)
        self.ending = (f"{COLOUR_NAME[colour]} lost on time{why}" if mate_possible else
                       f"{COLOUR_NAME[colour]}'s clock ran out{why} but {COLOUR_NAME[not colour]} cannot mate")
        self.result_code = ("0-1" if colour == chess.WHITE else "1-0") if mate_possible else "1/2-1/2"

    def _only_minor(self, colour: bool) -> bool:
        """``colour`` has a king and at most one knight or bishop: it cannot mate."""
        rest = [p for p in self.board.piece_map().values() if p.color == colour and p.piece_type != chess.KING]
        return len(rest) <= 1 and all(p.piece_type in (chess.KNIGHT, chess.BISHOP) for p in rest)

    def _adjudicate(self, why: str) -> None:
        mat = self.material()
        d = mat[chess.WHITE] - mat[chess.BLACK]
        self.result_code = "1-0" if d > 0 else "0-1" if d < 0 else "1/2-1/2"
        self.ending = f"{why}, adjudicated by material (White {mat[chess.WHITE]:g}, Black {mat[chess.BLACK]:g})"

    def exchange_promoted(self, move) -> list[str]:
        """Exchange the pawn that ``move`` promoted for a spare piece (or a captured one) of the chosen kind, in
        the record of where the pieces stand; the two pieces, or [] when no such piece is off the board."""
        sq = chess.square_name(move.to_square)
        pawn = self.piece_on(sq)
        colour = self.colour[pawn]
        kind = KIND_NAME[move.promotion]
        pool = [n for n, w in self.where.items() if self.colour[n] == colour and self.kind[n] == kind
                and w.startswith(("reserve:", "slot:"))]
        pool.sort(key=lambda n: not self.where[n].startswith("reserve:"))
        if not pool:            # no piece of that kind off the board: the pawn stays and stands for it
            return []
        new = pool[0]
        self.where[pawn], self.where[new] = self.where[new], sq
        return [pawn, new]

    def _promote(self, snap: OracleSnapshot, move) -> dict:
        """Put the pieces of a promotion where :meth:`exchange_promoted` records them."""
        return self._restore(snap, self.exchange_promoted(move))

    def pgn(self, white: str, black: str) -> str:
        return chess.to_pgn(self.board, {"Event": "omakase-bench games/chess (MuJoCo)", "White": white, "Black": black,
                                         "Termination": self.ending or "unfinished"}, self.result_code)

    def result(self, game: GameView, snap: OracleSnapshot) -> GameResult:
        self.ending = self.ending or f"unfinished after {len(self.san)} half-moves"
        ending = self.ending
        if self.result_code == "*":         # no result is made up for a game that did not end
            mat = self.material()
            ending += f"; no result, material White {mat[chess.WHITE]:g}, Black {mat[chess.BLACK]:g}"
        per = {r: {**m, "colour": COLOUR_NAME[self.colour_of[r]]} for r, m in self.metrics.items()}
        for r in per:
            mine = [m for m in self.move_log if m["robot"] == r]
            per[r]["think_s"] = round(sum(m["think_s"] for m in mine), 1)
            per[r]["exec_s"] = round(sum(m["exec_s"] for m in mine), 1)
            per[r]["clock_left_s"] = round(self.clock[r], 1)
            per[r]["faults_in_game"] = self.faults[r]          # with those of the part a resumed game continues
            per[r]["move_times"] = mine
            per[r]["pgn_moves"] = self.movetext()
            per[r]["events"] = [e for e in self.events if e.get("robot") == r]
        evidence = f"{self.result_code} ({ending}): {self.movetext()}"
        done = self.result_code != "*"
        return GameResult(done, evidence, self.result_code, per)

    def progress(self) -> dict[str, Any]:
        """The game so far, for the episode's progress file: enough to resume it after a crash."""
        return {"moves": list(self.san), "movetext": self.movetext(), "clocks": {r: round(c, 1) for r, c in self.clock.items()},
                "fen": self.board.fen(), "result": self.result_code, "ending": self.ending,
                "move_times": list(self.move_log), "faults": dict(self.faults)}


@dataclass(frozen=True)
class _Placement:
    id: str
    base: tuple[float, float, float]

"""The chess scene and what each player is told about it: the generator ``chess_game``, the start
position, the layout text and the reach text."""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from ....core.frames import to_base
from ....core.task import SceneSpec, Shape
from . import rules as chess
from .layout import COLOUR_NAME, KIND_NAME, Layout, Values, _mm, piece_name, reach_tilt
from .referee import ChessReferee


def chess_game(scene: SceneSpec, rng: np.random.Generator, p: Mapping[str, Any]) -> tuple[SceneSpec, Values]:
    """The board, the 32 pieces in the starting position, one spare piece of each kind a pawn can promote
    to per side, capture-slot marks, and per robot its goal and the layout paragraph in its own base
    frame. The position is always the starting position; per seed each piece stands up to
    ``params.jitter_m`` from its square's centre and a knight turns up to ``params.jitter_deg``, as on a
    board set up by hand."""
    L = Layout.from_params(scene, p)
    b = p["board"]
    colours = {chess.WHITE: tuple(p["colours"]["white"]), chess.BLACK: tuple(p["colours"]["black"])}
    border = float(b["border"])
    size = 8 * L.square + 2 * border
    fixtures = [Shape("board", "box", (size, size, float(b["thickness"])),
                      (L.centre[0], L.centre[1], L.table_top + float(b["thickness"]) / 2),
                      color=tuple(b["frame_color"]), label="chessboard")]
    for sq in chess.SQUARES:
        x, y = L.square_xy(sq)
        dark = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 0
        fixtures.append(Shape(f"sq_{chess.square_name(sq)}", "box", (L.square, L.square, 0.0006),
                              (float(x), float(y), L.board_top + 0.0003),
                              color=tuple(b["dark"] if dark else b["light"]), label="square"))
    for colour in (chess.WHITE, chess.BLACK):
        for k in range(L.n_slots):
            x, y = L.slot_xy(colour, k)
            fixtures.append(Shape(f"slot_{'w' if colour else 'b'}{k + 1}", "box", (0.034, 0.034, 0.0006),
                                  (float(x), float(y), L.table_top + 0.0003), color=tuple(p["slot_color"]),
                                  label="capture slot"))
    objects, pieces = [], {}
    yaw0 = {c: L.bases[L.side[c]][2] for c in colours}           # knights face the opponent

    jitter, jitter_yaw = float(p.get("jitter_m", 0.0)), math.radians(float(p.get("jitter_deg", 0.0)))
    placed = {}

    def add(colour: bool, kind: str, name: str, xy, bottom: float, where: str) -> None:
        h, d, m = L.dims[kind]
        xy = np.asarray(xy, float) + rng.uniform(-jitter, jitter, 2)
        yaw = yaw0[colour] + (rng.uniform(-jitter_yaw, jitter_yaw) if kind == "knight" else 0.0)
        placed[name] = [round(float(xy[0]), 4), round(float(xy[1]), 4), round(math.degrees(yaw), 1)]
        objects.append(Shape(name, "staunton", (h, d), (float(xy[0]), float(xy[1]), bottom + h / 2), yaw=yaw,
                             color=colours[colour], mass=m, friction=1.0, label=f"{COLOUR_NAME[colour]} {kind}",
                             variant=kind))
        pieces[name] = {"colour": COLOUR_NAME[colour], "kind": kind, "at": where}

    start = _start_position(p)
    for name, at in start.items():
        colour, kind = name[0] == "w", name.split("_")[1]
        if at.startswith("slot:"):
            add(colour, kind, name, L.slot_xy(at.split(":")[1] == "w", int(at.split(":")[2])), L.table_top, at)
        elif at.startswith("reserve:"):
            add(colour, kind, name, L.reserve_xy(colour, at.split(":")[1]), L.table_top, at)
        else:
            add(colour, kind, name, L.square_xy(chess.parse_square(at)), L.board_top, at)
    robots = {L.side[c]: {"goal": _goal(c), "layout": _layout_text(L, c, p), "colour": COLOUR_NAME[c]}
              for c in colours}
    return (replace(scene, fixtures=(*scene.fixtures, *fixtures), objects=(*scene.objects, *objects)),
            {"pieces": pieces, "robots": robots, "placed_at": placed,
             "stations": {r.id: [round(float(v), 6) for v in r.base] for r in scene.robots}})


def _start_position(p: Mapping[str, Any]) -> dict[str, str]:
    """Where every piece and spare stands after ``params.start_moves`` (SAN, from the initial position; a
    game resumed from a recorded position): piece name -> square, capture slot or reserve spot. A captured
    piece takes its captor's first free slot. A promoted pawn is exchanged as the referee does it in play:
    a spare (or captured) piece of the chosen kind stands on the promotion square and the pawn where that
    piece stood."""
    board = chess.Board()
    where = {piece_name(pc.color, KIND_NAME[pc.piece_type], chess.square_name(sq)): chess.square_name(sq)
             for sq, pc in sorted(board.piece_map().items())}
    for colour in (chess.WHITE, chess.BLACK):
        where.update({piece_name(colour, kind, "spare"): f"reserve:{kind}" for kind in p["reserve"]})
    moves = list(p.get("start_moves") or [])
    if not moves:
        return where
    pieces = {n: {"colour": "white" if n[0] == "w" else "black", "kind": n.split("_")[1], "at": w} for n, w in where.items()}
    ref = ChessReferee({"pieces": pieces}, {**p, "start_moves": [], "start_clocks": {}})
    for san in moves:
        move = ref.board.parse_san(san)
        after = ref.where_after(ref.plan(move))
        for n, w in list(after.items()):
            if w.endswith(":*"):
                after[n] = ref.free_slot(w.split(":")[1] == "w", after)
        ref.where = after
        ref.board.push(move)
        if move.promotion:
            ref.exchange_promoted(move)
    return ref.where


def _goal(colour: bool) -> str:
    me, other = COLOUR_NAME[colour].capitalize(), COLOUR_NAME[not colour].capitalize()
    return (f"You play {me} in a game of chess against the robot facing you across the table, which plays {other}. "
            "Choose your moves to win.")


def _frame_xy(L: Layout, colour: bool, world_xy) -> tuple[int, int]:
    b = to_base(L.bases[L.side[colour]], np.asarray(world_xy, float))
    return _mm(round(float(b[0]), 6)), _mm(round(float(b[1]), 6))     # both robots read the same half millimetres


def _layout_text(L: Layout, colour: bool, p: Mapping[str, Any]) -> str:
    """The cell as an installation would describe it, in the robot's own base frame."""
    near, far = ("1", "8") if colour == chess.WHITE else ("8", "1")
    a1 = _frame_xy(L, colour, L.square_xy(chess.A1))
    h8 = _frame_xy(L, colour, L.square_xy(chess.H8))
    files = "a to h" if colour == chess.WHITE else "h to a"
    lines = [
        f"Cell layout in your base frame at your station: the table top is at z={_mm(L.table_top)} and the top of the chessboard at "
        f"z={_mm(L.board_top)}. The board lies in front of you with {_mm(L.square)} mm squares; rank {near} is the rank "
        f"nearest you and rank {far} the farthest, files run {files} from your left to your right. Square centres: "
        f"a1 at x {a1[0]} y {a1[1]}, h8 at x {h8[0]} y {h8[1]}; every other square follows on the {_mm(L.square)} mm grid "
        f"(one rank further from you = +{_mm(L.square)} mm in x for you, one file to your right = -{_mm(L.square)} mm "
        "in y). The other robot faces you from beyond rank "
        f"{far}.",
        "Pieces (Staunton, " + "; ".join(
            f"{k}: {_mm(h)} mm tall, base {_mm(d)} mm, head {_mm(_head(k, d))} mm wide"
            for k, (h, d, _) in L.dims.items()) + "; the head is the part 10-18 mm below the top).",
        f"Your capture slots: {L.n_slots} marked spots on the table on your right beside the board, in "
        f"{len(L.slot_rows(colour))} rows of {len(L.slot_x)} at x {_mm(L.slot_x[0])} to {_mm(L.slot_x[-1])} "
        f"({_mm(L.slot_x[1] - L.slot_x[0])} mm apart), y {', '.join(str(_mm(y)) for y in L.slot_rows(colour))}; pieces you "
        "capture go there, one piece per slot.",
    ]
    if p.get("reach"):
        lines.append(_reach_text(p, colour))
    return " ".join(lines)


def _reach_text(p: Mapping[str, Any], colour: bool) -> str:
    """The reach table in the player's words: squares only one arm reaches, the tilts the far squares need."""
    names = {arm: {} for arm in ("left", "right")}
    for sq in chess.SQUARES:
        for arm in names:
            t = reach_tilt(p, colour, arm, chess.square_name(sq))
            if t is not None:
                names[arm][chess.square_name(sq)] = t
    only = {arm: [s for s in names[arm] if s not in names["right" if arm == "left" else "left"]] for arm in names}
    parts = []
    for arm in names:
        by_tilt: dict[int, list[str]] = {}
        for s, t in sorted(names[arm].items()):
            if t:
                by_tilt.setdefault(int(t), []).append(s)
        parts.append(f"{arm} arm " + ", ".join(f"t{t} on {' '.join(v)}" for t, v in sorted(by_tilt.items())))
    n_slots = len(p["capture_slots"]["y"]) * int(p["capture_slots"]["per_row"])
    slots = [k + 1 for k in range(n_slots) if reach_tilt(p, colour, "right", f"slot:x:{k}") is not None]
    tilt_text = "; ".join(parts)
    station = ""
    if p.get("reach_station"):
        st = dict(p, reach=p["reach_station"])
        missing = [chess.square_name(sq) for sq in chess.SQUARES
                   if all(reach_tilt(st, colour, arm, chess.square_name(sq)) is None for arm in ("left", "right"))]
        slots_missing = [k + 1 for k in range(n_slots) if reach_tilt(st, colour, "right", f"slot:x:{k}") is None]
        station = (f"From your station your arms do not reach {', '.join(missing) or 'no square'}"
                   + (f" or capture slots {', '.join(map(str, slots_missing))}" if slots_missing else "")
                   + f". With your base {_mm(float(p.get('reach_forward_m', 0.0)))} mm forward of its station: ")
    return (station + "Reach of your arms (measured for this cell, jaws at yaw "
            f"{p['grasp_yaw']['left']} for the left arm and {p['grasp_yaw']['right']} for the right): only your left "
            f"arm reaches {', '.join(sorted(only['left']))}; only your right arm reaches {', '.join(sorted(only['right']))}; "
            f"both reach every other square. The right arm reaches capture slots {slots[0]}-{slots[-1]}. Measured tilts at "
            f"which the hand stays clear of the robot's own body and of the other robot: {tilt_text}; 0 elsewhere.")


def _head(kind: str, base_d: float) -> float:
    from ....objects import staunton
    return staunton.head_width(kind, base_d)

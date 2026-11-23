"""Board geometry of the chess table, piece names and the reach tables of the task's YAML."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ....core.frames import to_world
from ....core.task import SceneSpec
from . import rules as chess

Values = dict[str, Any]
KIND_NAME = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop", chess.ROOK: "rook",
             chess.QUEEN: "queen", chess.KING: "king"}
KIND_OF = {v: k for k, v in KIND_NAME.items()}
COLOUR_NAME = {chess.WHITE: "white", chess.BLACK: "black"}
#: turn endings that are the harness's, not the player's: a spent budget, a model or subject error
HARNESS_ENDS = ("budget_exceeded:", "error:")


def _clock(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    return f"{s // 60}:{s % 60:02d}"


def _mm(v: float) -> int:
    return int(round(v * 1000))


@dataclass(frozen=True)
class Layout:
    """The cell in world coordinates, from the task's ``params`` and the robots' bases."""
    square: float
    centre: tuple[float, float]
    table_top: float
    board_top: float
    bases: Mapping[str, tuple[float, float, float]]      # robot -> x, y, yaw (rad)
    side: Mapping[bool, str]                             # colour -> robot
    dims: Mapping[str, tuple[float, float, float]]       # kind -> height, base diameter, mass
    slot_x: tuple[float, ...]                            # capture slots, in the mover's frame
    slot_y: tuple[float, ...]
    reserve: Mapping[str, tuple[float, float]]           # spare kind -> x, y in the owner's frame
    slot_y_black: tuple[float, ...] = ()                 # Black's slot rows when they differ from White's

    @classmethod
    def from_params(cls, scene: SceneSpec, p: Mapping[str, Any]) -> "Layout":
        table = next(f for f in scene.fixtures if f.name == "table")
        top = table.pos[2] + table.size[2] / 2
        b = p["board"]
        side = {chess.WHITE: str(p["sides"]["white"]), chess.BLACK: str(p["sides"]["black"])}
        slots = p["capture_slots"]
        sq = float(p["square"])
        return cls(sq, tuple(float(v) for v in b["centre"]), top, top + float(b["thickness"]),
                   {r.id: tuple(r.base) for r in scene.robots}, side,
                   {k: tuple(float(x) for x in v) for k, v in p["pieces"].items()},
                   tuple(float(slots["x0"]) + i * float(slots["pitch"]) for i in range(int(slots["per_row"]))),
                   tuple(float(y) for y in slots["y"]),
                   {k: tuple(float(x) for x in v) for k, v in p["reserve"].items()},
                   tuple(float(y) for y in slots.get("y_black", ())))

    def square_xy(self, sq: int) -> np.ndarray:
        """World xy of a square's centre. Rank 1 is nearest White's robot, file a on its left."""
        cx, cy = self.centre
        white = self.bases[self.side[chess.WHITE]]
        local = ((chess.square_rank(sq) - 3.5) * self.square, (3.5 - chess.square_file(sq)) * self.square)
        c, s = math.cos(white[2]), math.sin(white[2])
        return np.array([cx + c * local[0] - s * local[1], cy + s * local[0] + c * local[1]])

    def slot_xy(self, colour: bool, k: int) -> np.ndarray:
        """World xy of capture slot ``k`` of ``colour``: rows along x beside the board on the mover's right,
        the inner row first."""
        row, i = divmod(k, len(self.slot_x))
        return to_world(self.bases[self.side[colour]], (self.slot_x[i], self.slot_rows(colour)[row]))

    def slot_rows(self, colour: bool) -> tuple[float, ...]:
        return self.slot_y_black if colour == chess.BLACK and self.slot_y_black else self.slot_y

    @property
    def n_slots(self) -> int:
        return len(self.slot_x) * len(self.slot_y)

    def reserve_xy(self, colour: bool, kind: str) -> np.ndarray:
        return to_world(self.bases[self.side[colour]], self.reserve[kind])

    def height(self, kind: str) -> float:
        return self.dims[kind][0]


def piece_name(colour: bool, kind: str, where: str) -> str:
    return f"{'w' if colour else 'b'}_{kind}_{where}"


def _quat_yaw(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def _yaw_of(quat) -> float:
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _upright_cos(quat) -> float:
    _, x, y, _ = quat
    return 1 - 2 * (x * x + y * y)


def _mover_index(colour: bool, sq: int) -> tuple[int, int]:
    """(rows from the mover's side, files from the mover's left) of a square."""
    r, f = chess.square_rank(sq), chess.square_file(sq)
    return (r, f) if colour == chess.WHITE else (7 - r, 7 - f)


def reach_tilt(p: Mapping[str, Any], colour: bool, arm: str, where: str) -> float | None:
    """The tilt (deg) the reach table gives ``arm`` for a square or capture slot, None if it does not reach."""
    table = p.get("reach") or {}
    if where.startswith("slot:"):
        v = (table.get("slots") or {}).get(arm, [None] * 16)[int(where.split(":")[2])]
    else:
        r, f = _mover_index(colour, chess.parse_square(where))
        v = table[arm][r][f] if arm in table else 0
    return None if v is None else float(v)

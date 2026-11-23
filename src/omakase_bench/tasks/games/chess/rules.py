"""The rules of chess: legal moves, SAN and UCI, FEN and PGN, game endings.

    board = Board()                       # or Board(fen)
    move = board.parse_san("Nf3")         # SAN or, with parse_uci, UCI; errors say why
    san = board.san(move); board.push(move)
    board.is_game_over(claim_draw=True); board.outcome(claim_draw=True).result()

Squares are numbered 0 (a1) to 63 (h8). Move generation is pseudo-legal moves
filtered by making each move and testing whether the mover's king is attacked.
Endings: checkmate, stalemate, insufficient material (no possible mate: K v K,
K and one minor piece v K, kings and bishops all on squares of one colour),
the seventy-five-move rule and fivefold repetition automatically, and with
``claim_draw`` also the fifty-move rule and threefold repetition (FIDE Laws of
Chess, articles 5 and 9). A position repeats when the placement, the side to
move, the castling rights and the en passant capture that is actually legal
are the same.

Correctness is checked by perft against the standard positions and by
replaying recorded games (``tests/tasks/games/chess/test_rules.py``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

WHITE, BLACK = True, False
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6
PIECE_SYMBOLS = (None, "p", "n", "b", "r", "q", "k")
PIECE_NAMES = (None, "pawn", "knight", "bishop", "rook", "queen", "king")
FILE_NAMES = "abcdefgh"
RANK_NAMES = "12345678"
SQUARES = range(64)
SQUARE_NAMES = [f + r for r in RANK_NAMES for f in FILE_NAMES]
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
A1, C1, D1, E1, F1, G1, H1 = 0, 2, 3, 4, 5, 6, 7
A8, C8, D8, E8, F8, G8, H8 = 56, 58, 59, 60, 61, 62, 63


class InvalidMoveError(ValueError):
    """The text is not a move in the notation."""


class IllegalMoveError(ValueError):
    """A well-formed move that is not legal in the position."""


class AmbiguousMoveError(ValueError):
    """SAN that matches more than one legal move."""


def square(file: int, rank: int) -> int:
    return rank * 8 + file


def square_file(sq: int) -> int:
    return sq & 7


def square_rank(sq: int) -> int:
    return sq >> 3


def square_name(sq: int) -> str:
    return SQUARE_NAMES[sq]


def parse_square(name: str) -> int:
    try:
        return SQUARE_NAMES.index(name)
    except ValueError:
        raise ValueError(f"not a square: {name!r}") from None


def piece_name(kind: int) -> str:
    return PIECE_NAMES[kind]


def piece_symbol(kind: int) -> str:
    return PIECE_SYMBOLS[kind]


@dataclass(frozen=True)
class Piece:
    piece_type: int
    color: bool

    def symbol(self) -> str:
        s = PIECE_SYMBOLS[self.piece_type]
        return s.upper() if self.color else s


@dataclass(frozen=True)
class Move:
    from_square: int
    to_square: int
    promotion: int | None = None

    def uci(self) -> str:
        p = PIECE_SYMBOLS[self.promotion] if self.promotion else ""
        return SQUARE_NAMES[self.from_square] + SQUARE_NAMES[self.to_square] + p

    @classmethod
    def from_uci(cls, text: str) -> "Move":
        m = re.fullmatch(r"([a-h][1-8])([a-h][1-8])([qrbn]?)", text)
        if not m:
            raise InvalidMoveError(f"not a UCI move: {text!r}")
        promo = PIECE_SYMBOLS.index(m.group(3)) if m.group(3) else None
        return cls(parse_square(m.group(1)), parse_square(m.group(2)), promo)

    def __str__(self) -> str:
        return self.uci()


# --- attack tables ---------------------------------------------------------------------------------------

def _steps(deltas) -> list[list[int]]:
    out = []
    for sq in SQUARES:
        f, r = square_file(sq), square_rank(sq)
        out.append([square(f + df, r + dr) for df, dr in deltas if 0 <= f + df < 8 and 0 <= r + dr < 8])
    return out


def _rays(deltas) -> list[list[list[int]]]:
    out = []
    for sq in SQUARES:
        f, r = square_file(sq), square_rank(sq)
        rays = []
        for df, dr in deltas:
            ray, ff, rr = [], f + df, r + dr
            while 0 <= ff < 8 and 0 <= rr < 8:
                ray.append(square(ff, rr))
                ff, rr = ff + df, rr + dr
            rays.append(ray)
        out.append(rays)
    return out


KNIGHT_STEPS = _steps([(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)])
KING_STEPS = _steps([(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)])
ROOK_RAYS = _rays([(1, 0), (-1, 0), (0, 1), (0, -1)])
BISHOP_RAYS = _rays([(1, 1), (1, -1), (-1, 1), (-1, -1)])
#: squares a pawn of each colour on a square attacks
PAWN_ATTACKS = {WHITE: _steps([(-1, 1), (1, 1)]), BLACK: _steps([(-1, -1), (1, -1)])}
CASTLING = {  # right: (king from, king to, rook from, rook to, squares that must be empty, squares not attacked)
    "K": (E1, G1, H1, F1, (F1, G1), (E1, F1, G1)),
    "Q": (E1, C1, A1, D1, (1, C1, D1), (E1, D1, C1)),
    "k": (E8, G8, H8, F8, (F8, G8), (E8, F8, G8)),
    "q": (E8, C8, A8, D8, (57, C8, D8), (E8, D8, C8)),
}


@dataclass(frozen=True)
class Outcome:
    termination: str            # checkmate | stalemate | insufficient_material | fifty_moves | threefold_repetition | ...
    winner: bool | None

    def result(self) -> str:
        return "1-0" if self.winner is WHITE else "0-1" if self.winner is BLACK else "1/2-1/2"


class Board:
    """A position and the moves that led to it (``move_stack``), with push and pop."""

    def __init__(self, fen: str = STARTING_FEN):
        self.set_fen(fen)

    # --- FEN -------------------------------------------------------------------------------------------
    def set_fen(self, fen: str) -> None:
        parts = fen.split()
        if len(parts) == 4:
            parts += ["0", "1"]
        if len(parts) != 6:
            raise ValueError(f"FEN needs 6 fields: {fen!r}")
        rows = parts[0].split("/")
        if len(rows) != 8:
            raise ValueError(f"FEN placement needs 8 ranks: {fen!r}")
        self.squares: list[Piece | None] = [None] * 64
        for i, row in enumerate(rows):
            f = 0
            for c in row:
                if c.isdigit():
                    f += int(c)
                elif c.lower() in "pnbrqk":
                    if f > 7:
                        raise ValueError(f"FEN rank too long: {row!r}")
                    self.squares[square(f, 7 - i)] = Piece(PIECE_SYMBOLS.index(c.lower()), c.isupper())
                    f += 1
                else:
                    raise ValueError(f"bad FEN character {c!r}")
            if f != 8:
                raise ValueError(f"FEN rank of wrong length: {row!r}")
        if parts[1] not in ("w", "b"):
            raise ValueError(f"side to move must be w or b: {fen!r}")
        self.turn = parts[1] == "w"
        self.castling = "".join(c for c in "KQkq" if c in parts[2])
        self.ep_square = None if parts[3] == "-" else parse_square(parts[3])
        self.halfmove_clock = int(parts[4])
        self.fullmove_number = int(parts[5])
        self.move_stack: list[Move] = []
        self._undo: list[tuple] = []
        self._keys: list[str] = [self._key()]

    def board_fen(self) -> str:
        rows = []
        for r in range(7, -1, -1):
            row, empty = "", 0
            for f in range(8):
                p = self.squares[square(f, r)]
                if p is None:
                    empty += 1
                    continue
                if empty:
                    row += str(empty)
                    empty = 0
                row += p.symbol()
            rows.append(row + (str(empty) if empty else ""))
        return "/".join(rows)

    def fen(self) -> str:
        ep = SQUARE_NAMES[self.ep_square] if self.ep_square is not None and self._ep_legal() else "-"
        return (f"{self.board_fen()} {'w' if self.turn else 'b'} {self.castling or '-'} {ep} "
                f"{self.halfmove_clock} {self.fullmove_number}")

    def copy(self) -> "Board":
        b = Board.__new__(Board)
        b.squares = list(self.squares)
        b.turn, b.castling, b.ep_square = self.turn, self.castling, self.ep_square
        b.halfmove_clock, b.fullmove_number = self.halfmove_clock, self.fullmove_number
        b.move_stack, b._undo, b._keys = list(self.move_stack), list(self._undo), list(self._keys)
        return b

    # --- queries -------------------------------------------------------------------------------------------
    def piece_at(self, sq: int) -> Piece | None:
        return self.squares[sq]

    def piece_map(self) -> dict[int, Piece]:
        return {sq: p for sq, p in enumerate(self.squares) if p is not None}

    def king(self, color: bool) -> int | None:
        for sq, p in enumerate(self.squares):
            if p is not None and p.piece_type == KING and p.color == color:
                return sq
        return None

    def is_attacked_by(self, color: bool, sq: int) -> bool:
        s = self.squares
        for t in PAWN_ATTACKS[not color][sq]:
            p = s[t]
            if p is not None and p.color == color and p.piece_type == PAWN:
                return True
        for t in KNIGHT_STEPS[sq]:
            p = s[t]
            if p is not None and p.color == color and p.piece_type == KNIGHT:
                return True
        for t in KING_STEPS[sq]:
            p = s[t]
            if p is not None and p.color == color and p.piece_type == KING:
                return True
        for rays, kinds in ((ROOK_RAYS[sq], (ROOK, QUEEN)), (BISHOP_RAYS[sq], (BISHOP, QUEEN))):
            for ray in rays:
                for t in ray:
                    p = s[t]
                    if p is None:
                        continue
                    if p.color == color and p.piece_type in kinds:
                        return True
                    break
        return False

    def is_check(self) -> bool:
        k = self.king(self.turn)
        return k is not None and self.is_attacked_by(not self.turn, k)

    # --- move generation -------------------------------------------------------------------------------------
    def _pseudo(self) -> Iterator[Move]:
        s, me = self.squares, self.turn
        for sq in SQUARES:
            p = s[sq]
            if p is None or p.color != me:
                continue
            k = p.piece_type
            if k == PAWN:
                yield from self._pawn_moves(sq)
            elif k == KNIGHT or k == KING:
                for t in (KNIGHT_STEPS if k == KNIGHT else KING_STEPS)[sq]:
                    q = s[t]
                    if q is None or q.color != me:
                        yield Move(sq, t)
            else:
                rays = (ROOK_RAYS[sq] if k == ROOK else BISHOP_RAYS[sq] if k == BISHOP
                        else ROOK_RAYS[sq] + BISHOP_RAYS[sq])
                for ray in rays:
                    for t in ray:
                        q = s[t]
                        if q is None:
                            yield Move(sq, t)
                            continue
                        if q.color != me:
                            yield Move(sq, t)
                        break
        yield from self._castling_moves()

    def _pawn_moves(self, sq: int) -> Iterator[Move]:
        s, me = self.squares, self.turn
        step = 8 if me else -8
        start_rank, last_rank = (1, 7) if me else (6, 0)
        r = square_rank(sq)

        def emit(t: int) -> Iterator[Move]:
            if square_rank(t) == last_rank:
                for k in (QUEEN, ROOK, BISHOP, KNIGHT):
                    yield Move(sq, t, k)
            else:
                yield Move(sq, t)

        t = sq + step
        if 0 <= t < 64 and s[t] is None:
            yield from emit(t)
            t2 = t + step
            if r == start_rank and s[t2] is None:
                yield Move(sq, t2)
        for t in PAWN_ATTACKS[me][sq]:
            q = s[t]
            if (q is not None and q.color != me) or t == self.ep_square:
                yield from emit(t)

    def _castling_moves(self) -> Iterator[Move]:
        s, me = self.squares, self.turn
        for right in (("K", "Q") if me else ("k", "q")):
            if right not in self.castling:
                continue
            kf, kt, rf, _, empty, safe = CASTLING[right]
            king, rook = s[kf], s[rf]
            if (king is None or king.piece_type != KING or king.color != me or rook is None
                    or rook.piece_type != ROOK or rook.color != me):
                continue
            if any(s[e] is not None for e in empty):
                continue
            if any(self.is_attacked_by(not me, x) for x in safe):
                continue
            yield Move(kf, kt)

    @property
    def legal_moves(self) -> list[Move]:
        out = []
        me = self.turn
        for m in self._pseudo():
            self._make(m)
            k = self.king(me)
            if k is None or not self.is_attacked_by(not me, k):
                out.append(m)
            self._unmake()
        return out

    def is_legal(self, move: Move) -> bool:
        return move in self.legal_moves

    def _ep_legal(self) -> bool:
        return self.ep_square is not None and any(m.to_square == self.ep_square and self.is_en_passant(m)
                                                  for m in self.legal_moves)

    # --- move kinds ---------------------------------------------------------------------------------------------
    def is_en_passant(self, move: Move) -> bool:
        p = self.squares[move.from_square]
        return (p is not None and p.piece_type == PAWN and move.to_square == self.ep_square
                and square_file(move.from_square) != square_file(move.to_square) and self.squares[move.to_square] is None)

    def is_capture(self, move: Move) -> bool:
        q = self.squares[move.to_square]
        return (q is not None and q.color != self.turn) or self.is_en_passant(move)

    def is_castling(self, move: Move) -> bool:
        p = self.squares[move.from_square]
        return p is not None and p.piece_type == KING and abs(square_file(move.from_square) - square_file(move.to_square)) == 2

    # --- making moves -------------------------------------------------------------------------------------------
    def _make(self, m: Move) -> None:
        s = self.squares
        p = s[m.from_square]
        captured, cap_sq = s[m.to_square], m.to_square
        undo = (m, captured, cap_sq, self.castling, self.ep_square, self.halfmove_clock, self.fullmove_number)
        if p.piece_type == PAWN and m.to_square == self.ep_square and captured is None \
                and square_file(m.from_square) != square_file(m.to_square):
            cap_sq = m.to_square + (-8 if p.color else 8)
            captured = s[cap_sq]
            s[cap_sq] = None
            undo = (m, captured, cap_sq, *undo[3:])
        s[m.to_square] = Piece(m.promotion, p.color) if m.promotion else p
        s[m.from_square] = None
        if p.piece_type == KING and abs(square_file(m.from_square) - square_file(m.to_square)) == 2:
            for right, (kf, kt, rf, rt, _, _) in CASTLING.items():
                if kf == m.from_square and kt == m.to_square:
                    s[rt], s[rf] = s[rf], None
        rights = self.castling
        for sq, lost in ((E1, "KQ"), (E8, "kq"), (H1, "K"), (A1, "Q"), (H8, "k"), (A8, "q")):
            if m.from_square == sq or m.to_square == sq:
                rights = "".join(c for c in rights if c not in lost)
        self.castling = rights
        self.ep_square = None
        if p.piece_type == PAWN and abs(m.to_square - m.from_square) == 16:
            self.ep_square = (m.from_square + m.to_square) // 2
        self.halfmove_clock = 0 if p.piece_type == PAWN or captured is not None else self.halfmove_clock + 1
        if not p.color:
            self.fullmove_number += 1
        self.turn = not self.turn
        self._undo.append(undo)

    def _unmake(self) -> None:
        m, captured, cap_sq, self.castling, self.ep_square, self.halfmove_clock, self.fullmove_number = self._undo.pop()
        s = self.squares
        self.turn = not self.turn
        moved = s[m.to_square]
        s[m.from_square] = Piece(PAWN, moved.color) if m.promotion else moved
        s[m.to_square] = None
        s[cap_sq] = captured
        if moved.piece_type == KING and abs(square_file(m.from_square) - square_file(m.to_square)) == 2:
            for right, (kf, kt, rf, rt, _, _) in CASTLING.items():
                if kf == m.from_square and kt == m.to_square:
                    s[rf], s[rt] = s[rt], None

    def push(self, move: Move) -> None:
        self._make(move)
        self.move_stack.append(move)
        self._keys.append(self._key())

    def pop(self) -> Move:
        self._unmake()
        self._keys.pop()
        return self.move_stack.pop()

    def push_san(self, san: str) -> Move:
        m = self.parse_san(san)
        self.push(m)
        return m

    def _key(self) -> str:
        ep = self.ep_square if self.ep_square is not None and self._ep_legal() else None
        return f"{self.board_fen()} {self.turn} {self.castling} {ep}"

    # --- SAN ------------------------------------------------------------------------------------------------------
    def san(self, move: Move) -> str:
        """Standard algebraic notation of a legal move, with + or #."""
        text = self._san_body(move)
        self._make(move)
        try:
            if self.is_check():
                text += "#" if not self.legal_moves else "+"
        finally:
            self._unmake()
        return text

    def _san_body(self, move: Move) -> str:
        p = self.squares[move.from_square]
        if self.is_castling(move):
            return "O-O" if square_file(move.to_square) == 6 else "O-O-O"
        to = SQUARE_NAMES[move.to_square]
        capture = self.is_capture(move)
        if p.piece_type == PAWN:
            text = (FILE_NAMES[square_file(move.from_square)] + "x" if capture else "") + to
            if move.promotion:
                text += "=" + PIECE_SYMBOLS[move.promotion].upper()
            return text
        others = [m for m in self.legal_moves if m.to_square == move.to_square and m.from_square != move.from_square
                  and self.squares[m.from_square] == p]
        dis = ""
        if others:
            same_file = any(square_file(m.from_square) == square_file(move.from_square) for m in others)
            same_rank = any(square_rank(m.from_square) == square_rank(move.from_square) for m in others)
            if not same_file:
                dis = FILE_NAMES[square_file(move.from_square)]
            elif not same_rank:
                dis = RANK_NAMES[square_rank(move.from_square)]
            else:
                dis = SQUARE_NAMES[move.from_square]
        return PIECE_SYMBOLS[p.piece_type].upper() + dis + ("x" if capture else "") + to

    _SAN = re.compile(r"^([NBRQK])?([a-h])?([1-8])?(x)?([a-h][1-8])(?:=?([NBRQnbrq]))?$")

    def parse_san(self, text: str) -> Move:
        """The legal move a SAN string names. Lenient about check marks, annotations, '0-0', 'e8Q' and missing
        or extra disambiguation; raises InvalidMoveError, IllegalMoveError or AmbiguousMoveError."""
        t = text.strip().rstrip("+#!?").replace("0-0-0", "O-O-O").replace("0-0", "O-O")
        legal = self.legal_moves
        if t in ("O-O", "O-O-O"):
            want = 6 if t == "O-O" else 2
            for m in legal:
                if self.is_castling(m) and square_file(m.to_square) == want:
                    return m
            raise IllegalMoveError(f"illegal san: {text!r} in {self.fen()}")
        m = self._SAN.match(t)
        if not m:
            raise InvalidMoveError(f"invalid san: {text!r}")
        piece, ffile, frank, _, to, promo = m.groups()
        kind = PIECE_SYMBOLS.index(piece.lower()) if piece else PAWN
        to_sq = parse_square(to)
        promo_kind = PIECE_SYMBOLS.index(promo.lower()) if promo else None
        cands = []
        for mv in legal:
            p = self.squares[mv.from_square]
            if p.piece_type != kind or mv.to_square != to_sq or self.is_castling(mv):
                continue
            if ffile and FILE_NAMES[square_file(mv.from_square)] != ffile:
                continue
            if kind == PAWN and not ffile and square_file(mv.from_square) != square_file(to_sq):
                continue                    # a pawn capture names its file ("dxe4"); "e4" is a push
            if frank and RANK_NAMES[square_rank(mv.from_square)] != frank:
                continue
            if mv.promotion != promo_kind:
                continue
            cands.append(mv)
        if not cands:
            raise IllegalMoveError(f"illegal san: {text!r} in {self.fen()}")
        if len(cands) > 1:
            raise AmbiguousMoveError(f"ambiguous san: {text!r} in {self.fen()}")
        return cands[0]

    def parse_uci(self, text: str) -> Move:
        m = Move.from_uci(text)
        if m not in self.legal_moves:
            raise IllegalMoveError(f"illegal uci: {text!r} in {self.fen()}")
        return m

    # --- endings --------------------------------------------------------------------------------------------------
    def is_checkmate(self) -> bool:
        return self.is_check() and not self.legal_moves

    def is_stalemate(self) -> bool:
        return not self.is_check() and not self.legal_moves

    def is_insufficient_material(self) -> bool:
        pieces = [(sq, p) for sq, p in self.piece_map().items() if p.piece_type != KING]
        if any(p.piece_type in (PAWN, ROOK, QUEEN) for _, p in pieces):
            return False
        if len(pieces) <= 1:
            return True
        if all(p.piece_type == BISHOP for _, p in pieces):
            colours = {(square_file(sq) + square_rank(sq)) % 2 for sq, _ in pieces}
            return len(colours) == 1
        return False

    def _repetitions(self) -> int:
        return self._keys.count(self._keys[-1])

    def outcome(self, claim_draw: bool = False) -> Outcome | None:
        if not self.legal_moves:
            return Outcome("checkmate", not self.turn) if self.is_check() else Outcome("stalemate", None)
        if self.is_insufficient_material():
            return Outcome("insufficient_material", None)
        if self.halfmove_clock >= 150:
            return Outcome("seventyfive_moves", None)
        if self._repetitions() >= 5:
            return Outcome("fivefold_repetition", None)
        if claim_draw:
            if self.halfmove_clock >= 100:
                return Outcome("fifty_moves", None)
            if self._repetitions() >= 3:
                return Outcome("threefold_repetition", None)
        return None

    def is_game_over(self, claim_draw: bool = False) -> bool:
        return self.outcome(claim_draw) is not None


def perft(board: Board, depth: int) -> int:
    """Number of leaf positions of the legal move tree (the standard test of a move generator)."""
    if depth == 0:
        return 1
    moves = board.legal_moves
    if depth == 1:
        return len(moves)
    n = 0
    for m in moves:
        board._make(m)
        n += perft(board, depth - 1)
        board._unmake()
    return n


# --- PGN --------------------------------------------------------------------------------------------------

def movetext(board: Board, start_fen: str = STARTING_FEN) -> str:
    """The moves of ``board.move_stack`` in SAN with move numbers."""
    b = Board(start_fen)
    out = []
    for m in board.move_stack:
        if b.turn == WHITE:
            out.append(f"{b.fullmove_number}. {b.san(m)}")
        else:
            out.append((f"{b.fullmove_number}... " if not out else "") + b.san(m))
        b.push(m)
    return " ".join(out)


def to_pgn(board: Board, headers: dict[str, str], result: str = "*", start_fen: str = STARTING_FEN) -> str:
    tags = {"Event": "?", "Site": "?", "Date": "????.??.??", "Round": "?", "White": "?", "Black": "?",
            "Result": result, **headers}
    if start_fen != STARTING_FEN:
        tags.update({"SetUp": "1", "FEN": start_fen})
    head = "\n".join(f'[{k} "{v}"]' for k, v in tags.items())
    return f"{head}\n\n{movetext(board, start_fen)} {tags['Result']}\n"


def read_pgn(text: str) -> tuple[dict[str, str], Board]:
    """Headers and the game of one PGN; comments, variations, NAGs and move numbers are skipped."""
    headers = dict(re.findall(r'\[(\w+)\s+"([^"]*)"\]', text))
    body = re.sub(r"\[[^\]]*\]", " ", text)
    body = re.sub(r"\{[^}]*\}", " ", body)
    while "(" in body:
        body = re.sub(r"\([^()]*\)", " ", body)
    body = re.sub(r"\$\d+", " ", body)
    board = Board(headers.get("FEN", STARTING_FEN))
    for tok in body.split():
        if re.fullmatch(r"\d+\.+", tok) or tok in ("1-0", "0-1", "1/2-1/2", "*"):
            continue
        tok = re.sub(r"^\d+\.+", "", tok)
        if tok:
            board.push_san(tok)
    return headers, board

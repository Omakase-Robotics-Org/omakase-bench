"""Average centipawn loss per side of a game, from any UCI engine found on the machine (not a dependency).

    omakase-bench chess cpl GAME.pgn --engine /path/to/uci-engine [--depth 18] [--moves] [--worst 5]

For every move: the engine's evaluation of the position before it (from the mover's side) minus its
evaluation after it (negated, from the opponent's side), at least 0. Evaluations are held within
+-1000 cp and a mate counts as +-1000 cp, so a loss is at most 2000 cp and a won position does not
inflate the average. Prints one JSON object: per side the number of moves, the average loss, the
moves that lost 50-99 cp (inaccuracies), 100-199 (mistakes) and 200 or more (blunders), and the
``--worst`` worst moves with the engine's preferred move; ``--moves`` adds every move.

The engine is any UCI program, for example an official Stockfish release binary
(github.com/official-stockfish/Stockfish/releases; check the archive against the sha256 the release
lists): ``--engine /path/to/stockfish``. It needs no network. No engine ships with the bench.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from .rules import Board, read_pgn

MATE_CP = 1000


class Engine:
    def __init__(self, path: str):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self._send("uci")
        self._until("uciok")

    def _send(self, line: str) -> None:
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def _until(self, word: str) -> list[str]:
        lines = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("the engine stopped")
            lines.append(line.strip())
            if line.startswith(word):
                return lines

    def score(self, fen: str, depth: int) -> int:
        """Centipawns for the side to move."""
        return self.evaluate(fen, depth)[0]

    def evaluate(self, fen: str, depth: int) -> tuple[int, str]:
        """(centipawns for the side to move, held within +-MATE_CP; the engine's best move in UCI)."""
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")
        cp, best = 0, ""
        for line in self._until("bestmove"):
            t = line.split()
            if t and t[0] == "bestmove" and len(t) > 1:
                best = t[1]
            if "score" in t:
                i = t.index("score")
                kind, value = t[i + 1], int(t[i + 2])
                cp = value if kind == "cp" else (MATE_CP if value > 0 else -MATE_CP)
        return max(-MATE_CP, min(MATE_CP, cp)), best

    def close(self) -> None:
        self._send("quit")
        self.p.wait(timeout=10)


def acpl(board_moves: Board, engine: Engine, depth: int, start_fen: str | None = None) -> dict:
    b = Board(start_fen) if start_fen else Board()
    loss = {"white": [], "black": []}
    for move in board_moves.move_stack:
        side = "white" if b.turn else "black"
        if not b.legal_moves:
            break
        before = engine.score(b.fen(), depth)
        b.push(move)
        after = MATE_CP if b.is_checkmate() else (0 if b.outcome() else -engine.score(b.fen(), depth))
        loss[side].append(min(MATE_CP, max(0, before - after)))
    return {s: {"moves": len(v), "acpl": round(sum(v) / len(v), 1) if v else None} for s, v in loss.items()}


#: centipawns lost by a move from which it counts as an inaccuracy, a mistake, a blunder
GRADES = (("inaccuracies", 50), ("mistakes", 100), ("blunders", 200))


def analyse(board_moves: Board, engine: Engine, depth: int, start_fen: str | None = None, worst: int = 5) -> dict:
    """Per side: moves, average loss, graded moves and the worst ones; and every move with its loss."""
    b = Board(start_fen) if start_fen else Board()
    moves = []
    for move in board_moves.move_stack:
        if not b.legal_moves:
            break
        side = "white" if b.turn else "black"
        number, san = b.fullmove_number, b.san(move)
        before, best = engine.evaluate(b.fen(), depth)
        try:
            best_san = b.san(b.parse_uci(best)) if best else ""
        except ValueError:
            best_san = best
        b.push(move)
        after = MATE_CP if b.is_checkmate() else (0 if b.outcome() else -engine.evaluate(b.fen(), depth)[0])
        moves.append({"side": side, "move": f"{number}{'.' if side == 'white' else '...'} {san}", "eval_before": before,
                      "eval_after": after, "loss": max(0, before - after), "engine_move": best_san})
    out = {}
    for side in ("white", "black"):
        mine = [m for m in moves if m["side"] == side]
        losses = [m["loss"] for m in mine]
        grade = {}
        for k, (name, low) in enumerate(GRADES):
            high = GRADES[k + 1][1] if k + 1 < len(GRADES) else math.inf
            grade[name] = sum(1 for v in losses if low <= v < high)
        out[side] = {"moves": len(mine), "acpl": round(sum(losses) / len(losses), 1) if losses else None, **grade,
                     "worst": sorted((m for m in mine if m["loss"] >= GRADES[0][1]), key=lambda m: -m["loss"])[:worst]}
    out["moves"] = moves
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="omakase-bench chess cpl", description=__doc__.split("\n\n")[0])
    ap.add_argument("pgn", type=Path)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--depth", type=int, default=18)
    ap.add_argument("--moves", action="store_true", help="also print every move with its evaluations and loss")
    ap.add_argument("--worst", type=int, default=5, help="worst moves listed per side")
    a = ap.parse_args(argv)
    headers, board = read_pgn(a.pgn.read_text())
    eng = Engine(a.engine)
    try:
        report = analyse(board, eng, a.depth, headers.get("FEN"), a.worst)
    finally:
        eng.close()
    if not a.moves:
        del report["moves"]
    print(json.dumps(report, indent=1))
    return 0


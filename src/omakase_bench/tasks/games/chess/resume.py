"""Write the task file that resumes a stopped game of games/chess.

    omakase-bench chess resume RUN_DIR [--out resume_chess.yaml]

Reads the game's ``result.json`` under RUN_DIR (or, for a run that died before it was written, the
``progress.json`` the harness replaces after every ruled turn) and writes a copy of the task with ``params.start_moves``
(the completed moves), ``params.start_clocks`` (each robot's clock as it stood before the interrupted
turn) and ``params.start_faults`` (each robot's manipulation faults so far, which count towards the limit). Run the copy with ``omakase-bench run --task <the file> --seeds <the same seed> ...``: the side to
move acts first. The physical position is rebuilt from the moves, so a piece that stood off-centre stands
on its square's centre again, and a promoted pawn is exchanged for a spare piece as the referee did it in
play. A game that has a result is refused.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

from ....core.suite import find_task

TASK = "games/chess"


def resume_task(result: dict) -> dict:
    """The raw task of the game recorded in ``result``, set up at its last completed move."""
    if result.get("task") != TASK:
        raise ValueError(f"not a {TASK} episode: {result.get('task')!r}")
    if result.get("outcome") != "*":
        raise ValueError(f"the game has a result ({result.get('outcome')}); there is nothing to resume")
    robots = result["robots"]
    text = next(iter(robots.values()))["metrics"].get("pgn_moves", "")
    moves = [t for t in text.split() if not re.fullmatch(r"\d+\.+", t)]
    raw = dict(find_task(TASK).raw)
    earlier = list(raw["params"].get("start_moves") or [])
    if moves[:len(earlier)] != earlier:
        moves = earlier + moves
    raw["params"] = {**raw["params"], "start_moves": moves,
                     "start_clocks": {rid: float(r["metrics"]["clock_left_s"]) for rid, r in robots.items()},
                     "start_faults": {rid: int(r["metrics"].get("faults_in_game", 0)) for rid, r in robots.items()}}
    return raw


def from_progress(state: dict) -> dict:
    """An episode record in the shape of result.json from a progress.json (a result.json passes through)."""
    game = state.get("game")
    if "robots" in state and game is None:
        return state
    clocks = (game or {}).get("clocks", {})
    return {"task": state.get("task"), "seed": state.get("seed"), "outcome": (game or {}).get("result", "*"),
            "exit_reason": "died after turn %d" % len(state.get("turns", [])),
            "robots": {rid: {"metrics": {"pgn_moves": (game or {}).get("movetext", ""), "clock_left_s": clocks[rid],
                                         "faults_in_game": ((game or {}).get("faults") or {}).get(rid, 0)}}
                       for rid in clocks}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="omakase-bench chess resume", description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path, help="run directory, or the game's result.json")
    ap.add_argument("--out", type=Path, default=Path("resume_chess.yaml"))
    a = ap.parse_args(argv)
    files = [a.run] if a.run.is_file() else sorted(a.run.glob("episodes/*/*/*/result.json"))
    if not files and a.run.is_dir():        # a run that died: the progress file of its last complete turn
        files = sorted(a.run.glob("episodes/*/*/*/progress.json"))
    games = [r for r in (from_progress(json.loads(f.read_text())) for f in files) if r.get("task") == TASK]
    if len(games) != 1:
        raise SystemExit(f"{a.run}: expected one {TASK} episode, found {len(games)}")
    try:
        raw = resume_task(games[0])
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    a.out.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
    p = raw["params"]
    print(f"{a.out}: {len(p['start_moves'])} half-moves, clocks {p['start_clocks']}, stopped by "
          f"{games[0].get('exit_reason')}; resume with --task {a.out} --seeds {games[0]['seed']}")
    return 0


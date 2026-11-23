"""``omakase-bench chess <tool> ...``: the tools of ``games/chess``."""
from __future__ import annotations

import argparse
import importlib

TOOLS = {
    "tour": "reach check: one robot carries a king to every square and capture slot",
    "resume": "write the task file that resumes a stopped game",
    "cpl": "average centipawn loss per side of a PGN, from a UCI engine",
    "video": "game, highlights and review cuts of a recorded game",
}


def commands(subparsers) -> None:
    p = subparsers.add_parser("chess", help="tools of games/chess: " + " | ".join(TOOLS),
                              description="; ".join(f"{k}: {v}" for k, v in TOOLS.items()))
    p.add_argument("tool", choices=sorted(TOOLS))
    p.add_argument("args", nargs=argparse.REMAINDER, help="the tool's own arguments (try: <tool> --help)")
    p.set_defaults(fn=run)


def run(args) -> int:
    return importlib.import_module(f"{__package__}.{args.tool}").main(args.args)

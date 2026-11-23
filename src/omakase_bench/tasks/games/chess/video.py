"""Videos of a games/chess episode from its streamed frames, trace and result.

    omakase-bench chess video RUN/EPISODE_DIR --mode game --out game.mp4 [--max-seconds 210]
    omakase-bench chess video RUN/EPISODE_DIR --mode highlights --out highlights.mp4 [--max-seconds 60]
    omakase-bench chess video RUN/EPISODE_DIR --mode review --out review.mp4 [--speed 2] [--turns 3-6]
    omakase-bench chess video RUN/EPISODE_DIR --still 12.5 --out frame.png      (output second 12.5 of the cut)

The episode must have been recorded with ``video: {stream: true, tile_layout: right}`` (``raw.mp4`` and
``frames.json``: the side camera and, as tiles, each robot's head and wrist cameras and the board from
above, all of the same instants) and, for the text panel, ``ee.transcript: true`` (per model call its
changing prompt text, the thinking summary, the reply and the call's wall-clock seconds in the trace).

Each output frame (1280 x 720): the side camera on the left; on the right the acting robot's head and
wrist cameras under its model and colour, the board from above and a think timer; below them a text panel
with the acting model's observation, thinking summary and reply; a footer with the move, both clocks,
material, the counters and each move's think and execution time.

Time. Simulated time stands still while a model is called, so the recording has no frames of the
thinking. The cut puts them back: during a call the picture holds and the timer runs for the call's real
seconds. ``review`` plays everything at ``--speed``. ``game`` and ``highlights`` compress to
``--max-seconds``, execution least, the harness's moves between turns and the model calls most; the timer
and the per-move "think / exec" figures always show the real seconds, however fast they pass. In a
compressed cut the text panel shows one key line of thinking per move instead of every call.

Highlights prefer the end of the game, moves that swing an engine's evaluation by 200 cp or more
(``--cpl``, the report of ``omakase-bench chess cpl --moves``), queens moved onto an attacked square, faults
and refused moves, then captures, checks, hand-overs and castling.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from ....report.video import font
from .rules import Board

W, H = 1280, 720
HEADER = 34
MAIN = (720, 540)                    # the side camera
COL_X = MAIN[0]                      # the right column starts here
CAM_Y = HEADER + 30                  # below the acting robot's label
HEAD = (280, 210)
WRIST = (140, 105)
PANEL_Y = CAM_Y + HEAD[1] + 6
FOOT_Y = HEADER + MAIN[1]
VALUES = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9}
#: share of real time a second of each kind gets in a compressed cut, relative to execution
WEIGHT = {"exec": 1.0, "transfer": 0.3, "think": 0.25}
FULL_DETAIL_SPEED = 4.0              # slower than this, the panel shows every call; faster, one line per move
END_HOLD_S = 3.0
SIDE = {"white": (236, 228, 205), "black": (150, 170, 215)}
BG, PANEL, INK, DIM = (16, 17, 21), (26, 28, 34), (236, 236, 236), (150, 152, 160)
CHIPS = {"CAPTURE": (190, 120, 30), "CHECK": (60, 120, 200), "CHECKMATE": (40, 150, 80), "CASTLING": (90, 90, 160),
         "HAND-OVER": (30, 140, 140), "ILLEGAL": (200, 110, 20), "FAULT": (200, 50, 40), "RESTORE": (120, 60, 180),
         "CONTACT": (200, 30, 30), "ON TIME": (200, 30, 30), "QUEEN OFFER": (170, 60, 130),
         "EVAL SWING": (150, 40, 40)}
SWING_CP = 200                       # a move that loses this much by the engine's figures is a highlight


@dataclass
class Call:
    robot: str
    turn: int
    sim: float
    seconds: float
    thinking: str = ""
    prompt: str = ""
    text: str = ""


@dataclass
class Turn:
    n: int
    robot: str
    t0: float
    ready: float = math.nan
    t1: float = math.inf
    san: str = ""
    accepted_at: float = math.inf
    note: str = ""
    illegal: list[tuple[float, str]] = field(default_factory=list)
    restored: list[str] = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)
    move_call: Call | None = None
    think_s: float = 0.0
    exec_s: float = 0.0
    clock_before: float = 0.0
    clock_after: float = 0.0
    fen_before: str = ""
    fen_after: str = ""
    queen_offer: bool = False        # a queen moved onto a square the opponent attacks
    swing: int = 0                   # centipawns the move lost by an engine's figures (``omakase-bench chess cpl``)

    @property
    def tags(self) -> list[str]:
        out = []
        if "x" in self.san:
            out.append("CAPTURE")
        if "#" in self.san:
            out.append("CHECKMATE")
        elif "+" in self.san:
            out.append("CHECK")
        if "O-O" in self.san:
            out.append("CASTLING")
        if "hand-over" in self.note:
            out.append("HAND-OVER")
        if self.illegal:
            out.append("ILLEGAL")
        if "manipulation fault" in self.note:
            out.append("FAULT")
        if self.restored:
            out.append("RESTORE")
        if self.queen_offer:
            out.append("QUEEN OFFER")
        if self.swing >= SWING_CP:
            out.append("EVAL SWING")
        if "contact" in self.note:
            out.append("CONTACT")
        if "on time" in self.note or "clock ran out" in self.note:
            out.append("ON TIME")
        return out


@dataclass
class Segment:
    kind: str                # exec | transfer | think
    a: float                 # simulated time at its start
    b: float                 # and end (a == b while a model is called)
    dur: float               # real seconds
    turn: Turn | None
    call: Call | None = None
    out: float = 0.0         # seconds of the cut


def load(ep: Path):
    frames = json.loads((ep / "frames.json").read_text())
    trace = [json.loads(line) for line in (ep / "trace.jsonl").read_text().splitlines() if line.strip()]
    result = json.loads((ep / "result.json").read_text())
    return frames, trace, result


def timeline(trace: list[dict], result: dict, cpl: dict | None = None) -> list[Turn]:
    """The turns of the episode with their move, model calls, markers, clocks and positions. ``cpl`` is the
    report of ``omakase-bench chess cpl`` for the game (with ``--moves`` every move's loss, else the worst ones)."""
    losses = {}
    for m in [*((cpl or {}).get("moves") or []), *(w for side in ("white", "black") for w in ((cpl or {}).get(side) or {}).get("worst", []))]:
        losses[m["move"]] = int(m["loss"])
    turns: list[Turn] = []
    for e in trace:
        t, kind, rid = e["sim_time"], e["kind"], e.get("robot", "")
        cur = turns[-1] if turns else None
        if kind == "turn_start":
            turns.append(Turn(e["turn"], rid, t))
        elif cur is None:
            continue
        elif kind == "turn_ready":
            cur.ready = t
        elif kind == "model" and t <= cur.t1:
            cur.calls.append(Call(rid, cur.n, t, float(e.get("seconds", 0.0)), e.get("thinking", ""),
                                  e.get("prompt", ""), e.get("text", "")))
        elif kind == "say" and e.get("keyword") == "move":
            m = re.match(r"Accepted: (\S+)\.", e.get("answer", ""))
            if m:
                cur.san, cur.accepted_at = m.group(1), t
                cur.move_call = cur.calls[-1] if cur.calls else None
            elif e.get("answer", "").startswith("Illegal"):
                cur.illegal.append((t, e.get("text", "")))
        elif kind == "turn_end":
            cur.t1 = t
        elif kind == "referee" and e.get("turn"):
            cur.note = e.get("note", "")
        elif kind == "restore":
            cur.restored += list(e.get("objects", {}))
    params_clock = 3600.0
    robots = result.get("robots", {})
    played = {rid: list((r.get("metrics") or {}).get("move_times", [])) for rid, r in robots.items()}
    clock = {rid: None for rid in robots}
    board = Board()
    for tr in turns:
        if math.isnan(tr.ready):
            tr.ready = tr.t0
        rec = played.get(tr.robot, [])
        rec = rec.pop(0) if rec else {}
        tr.think_s = float(rec.get("think_s", sum(c.seconds for c in tr.calls)))
        tr.exec_s = float(rec.get("exec_s", max(0.0, min(tr.t1, 1e12) - tr.ready) if tr.t1 < math.inf else 0.0))
        if clock[tr.robot] is None:        # the first clock is the one after the move, less its increment, plus its use
            clock[tr.robot] = params_clock if not rec else None
        tr.clock_after = float(rec.get("clock_s", clock[tr.robot] or params_clock))
        tr.clock_before = clock[tr.robot] if clock[tr.robot] is not None else min(
            params_clock, tr.clock_after + tr.think_s + tr.exec_s - (30.0 if tr.san and "(flag)" not in tr.san else 0.0))
        clock[tr.robot] = tr.clock_after
        tr.fen_before = board.board_fen()
        if tr.san and " played " in tr.note:
            try:
                label = f"{board.fullmove_number}{'.' if board.turn else '...'} {tr.san}"
                move = board.parse_san(tr.san)
                queen = board.piece_at(move.from_square).symbol().lower() == "q"
                mover = board.turn
                board.push(move)
                tr.queen_offer = queen and board.is_attacked_by(not mover, move.to_square)
                tr.swing = losses.get(label, 0)
            except ValueError:
                pass
        tr.fen_after = board.board_fen()
    return turns


def segments(turns: list[Turn], t_first: float, t_last: float) -> list[Segment]:
    """The episode in order as stretches of simulated time (a turn's execution, the harness's moves between
    turns) and model calls, which take real time and no simulated time."""
    cuts = sorted({t_first, t_last, *(x for tr in turns for x in (tr.t0, tr.ready, min(tr.t1, t_last)))})
    out: list[Segment] = []

    def owner(t: float) -> tuple[Turn | None, str]:
        k = bisect_right([tr.t0 for tr in turns], t) - 1
        if k < 0:
            return (turns[0] if turns else None), "transfer"
        tr = turns[k]
        return tr, ("exec" if tr.ready <= t < tr.t1 else "transfer")

    calls = sorted((c for tr in turns for c in tr.calls), key=lambda c: c.sim)
    by_turn = {tr.n: tr for tr in turns}
    marks = sorted({*cuts, *(c.sim for c in calls if t_first <= c.sim <= t_last)})
    k = 0
    for a, b in zip(marks, marks[1:]):
        while k < len(calls) and calls[k].sim <= a + 1e-9:
            out.append(Segment("think", calls[k].sim, calls[k].sim, calls[k].seconds, by_turn[calls[k].turn], calls[k]))
            k += 1
        tr, kind = owner((a + b) / 2)
        out.append(Segment(kind, a, b, b - a, tr))
    for c in calls[k:]:
        out.append(Segment("think", c.sim, c.sim, c.seconds, by_turn[c.turn], c))
    return [s for s in out if s.dur > 0]


def choose_highlights(turns: list[Turn], most: int) -> set[int]:
    """Turn numbers of a highlights cut: the end, large swings of an engine's evaluation and queens moved
    into attack, faults and refused moves, captures, checks, hand-overs, castling, in that order of
    preference, at most ``most``."""
    rank = {}
    for k, tr in enumerate(turns):
        tags = tr.tags
        score = (8 if k >= len(turns) - 2 else 0) + 6 * bool({"FAULT", "ILLEGAL", "CONTACT", "RESTORE", "ON TIME"} & set(tags)) \
            + 7 * ("EVAL SWING" in tags) + 7 * ("QUEEN OFFER" in tags) \
            + 5 * ("CHECKMATE" in tags) + 4 * ("CAPTURE" in tags) + 3 * ("CHECK" in tags) + 3 * ("HAND-OVER" in tags) \
            + 2 * ("CASTLING" in tags) + (1 if k < 2 else 0)
        if score:
            rank[tr.n] = score
    keep = sorted(rank, key=lambda n: (-rank[n], n))[:most]
    return set(keep)


def plan(segs: list[Segment], mode: str, max_seconds: float, speed: float, keep: set[int] | None) -> float:
    """Give every segment its seconds in the cut; returns the speed-up of execution."""
    for s in segs:
        s.out = 0.0
    use = [s for s in segs if keep is None or (s.turn is not None and s.turn.n in keep and s.kind != "transfer")]
    if mode == "review":
        for s in use:
            s.out = s.dur / speed
        return speed
    total = sum(WEIGHT[s.kind] * s.dur for s in use)
    k = max(1.0, total / max(1e-6, max_seconds))
    for s in use:
        s.out = WEIGHT[s.kind] * s.dur / k
    return k


def material(fen_board: str) -> tuple[int, int]:
    return (sum(VALUES.get(c.lower(), 0) for c in fen_board if c.isupper()),
            sum(VALUES.get(c, 0) for c in fen_board if c.islower()))


def clock_text(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def secs(seconds: float) -> str:
    return f"{seconds:.0f} s" if seconds < 100 else f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


_HEADER = re.compile(r"^\s*\*\*(.+?)\*\*\s*$", re.M)


def summary_parts(text: str) -> tuple[str, str]:
    """(header, body) of a thinking summary: a leading ``**header**`` line apart, and the markdown emphasis
    marks and later header lines out of the body."""
    text = text.strip()
    header = ""
    m = _HEADER.match(text)
    if m:
        header, text = m.group(1).strip(), text[m.end():]
    text = _HEADER.sub(lambda h: h.group(1).strip() + ":", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.S).replace("**", "")
    text = re.sub(r"(?m)^\s*#+\s*", "", text)
    return header, text.strip()


def key_line(tr: Turn, limit: int = 190) -> str:
    """One line of the turn's thinking: the start of the summary of the call that named the move."""
    call = tr.move_call or (tr.calls[0] if tr.calls else None)
    text = " ".join(summary_parts(call.thinking if call else "")[1].split())
    if not text and call:
        text = " ".join(line for line in call.text.splitlines() if not re.match(r"\s*(a|move|done|locate|base|lift|head|"
                                                                               r"joints?|gripper)\s*:", line, re.I))
    if len(text) <= limit:
        return text
    cut = text[:limit]
    stop = max(cut.rfind(". "), cut.rfind("; "))
    return cut[:stop + 1] if stop > limit // 2 else cut.rsplit(" ", 1)[0] + " …"


def model_name(subject: str) -> str:
    return subject.split(":")[0].split("/")[-1]


class Composer:
    def __init__(self, frames: dict, result: dict, turns: list[Turn], title: str = ""):
        self.size, self.tiles = frames["size"], frames["tiles"]
        self.tw, self.th = frames["tile_size"]
        self.cols = frames["tile_columns"]
        self.turns = turns
        robots = result["robots"]
        self.model = {rid: model_name(r["subject"]) for rid, r in robots.items()}
        self.colour = {rid: (r.get("metrics") or {}).get("colour", "") for rid, r in robots.items()}
        self.by_colour = {c: rid for rid, c in self.colour.items()}
        self.result = result
        limit = next((r.get("metrics", {}).get("fault_limit") for r in robots.values()), None)
        #: "/3" after a fault count when the game was played with a fault limit
        self.fault_limit = f"/{limit}" if limit else ""
        self.title = title or (f"Chess between two robots:  {self.model.get(self.by_colour.get('white', 'a'), '')} (White)"
                               f"  vs  {self.model.get(self.by_colour.get('black', 'b'), '')} (Black)")
        self.f_title, self.f_big, self.f_mid = font(19), font(26), font(17)
        self.f_small, self.f_text, self.f_mono = font(13), font(17), font(13, mono=True)

    def tile(self, raw: np.ndarray, name: str, size: tuple[int, int]) -> Image.Image | None:
        if name not in self.tiles:
            return None
        r, c = divmod(self.tiles.index(name), self.cols)
        x0 = self.size[0] + c * self.tw
        return Image.fromarray(raw[r * self.th:(r + 1) * self.th, x0:x0 + self.tw]).resize(size)

    @staticmethod
    def label(d: ImageDraw.ImageDraw, xy, text: str, f) -> None:
        d.text(xy, text, font=f, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))

    def wrap(self, text: str, f, width: int) -> list[str]:
        per = max(12, int(width / (f.size * 0.52)))
        out = []
        for para in text.splitlines() or [""]:
            out += textwrap.wrap(para, per) or [""]
        return out

    def panel_lines(self, tr: Turn, seg: Segment, frac: float, detail: str, sim: float) -> list[tuple]:
        """(text, colour, font) lines of the text panel for the turn at this instant, the newest last."""
        width = W - COL_X - 24
        lines: list[tuple] = []

        def block(head: str, body: str, colour, f=None, reveal: float = 1.0, title: str = ""):
            if not body.strip():
                return
            lines.append((head, DIM, self.f_small))
            for ln in self.wrap(title, f or self.f_text, width) if title else ():
                lines.append((ln, colour, f or self.f_text, True))            # the summary's header, in bold
            shown = body[:max(1, int(len(body) * reveal))] if reveal < 1.0 else body
            for ln in self.wrap(shown, f or self.f_text, width):
                lines.append((ln, colour, f or self.f_text))
            lines.append(("", DIM, self.f_small))

        def header_of(tr_: Turn) -> str:
            call = tr_.move_call or (tr_.calls[0] if tr_.calls else None)
            return summary_parts(call.thinking if call else "")[0]

        side = SIDE.get(self.colour.get(tr.robot, ""), INK)
        if detail == "key":
            earlier = [t for t in self.turns if t.n < tr.n and t.san][-2:]
            for t in earlier:
                c = self.colour.get(t.robot, "")
                block(f"{(t.n + 1) // 2}{'.' if c == 'white' else '...'} {t.san}   {self.model.get(t.robot, '')}",
                      key_line(t, 120), DIM, self.f_small)
            started = tr.move_call is not None and (sim > tr.move_call.sim or (seg.call is tr.move_call) or sim >= tr.accepted_at)
            if started:
                thinking_now = seg.kind == "think" and seg.call is tr.move_call
                block("THINKING (summary)", key_line(tr) or "(the model returned no reasoning summary for this call)",
                      side, reveal=frac if thinking_now else 1.0, title=header_of(tr))
                if not thinking_now:
                    block("REPLY", f"move: {tr.san}" if tr.san else tr.move_call.text.strip()[:80], INK, self.f_mono)
            else:
                block("OBSERVATION", "reading the board from its cameras and the referee's message", DIM)
            return lines
        done = [c for c in tr.calls if c.sim < sim or (c.sim == sim and seg.call is not c and seg.kind != "think")]
        if seg.kind == "think":
            done = tr.calls[:tr.calls.index(seg.call)] if seg.call in tr.calls else done
        for c in done[-3:]:
            first = next((ln for ln in c.text.strip().splitlines() if ln.strip()), "")
            lines.append((f"> {first[:86]}", DIM, self.f_mono))
        if done:
            lines.append(("", DIM, self.f_small))
        cur = seg.call if seg.kind == "think" else (done[-1] if done else None)
        if cur is not None:
            obs = [ln for ln in cur.prompt.strip().splitlines() if ln.strip()]
            block("OBSERVATION (prompt excerpt)", "\n".join(ln[:230] for ln in obs[-2:]), DIM, self.f_mono)
            head, body = summary_parts(cur.thinking)
            block("THINKING (summary)", body or "(no summary returned)", side,
                  reveal=frac if seg.kind == "think" else 1.0, title=head)
            if seg.kind != "think":
                block("REPLY", cur.text.strip()[:400], INK, self.f_mono)
        return lines

    def compose(self, raw: np.ndarray, seg: Segment, frac: float, detail: str, speed: float, over: str = "") -> np.ndarray:
        tr = seg.turn
        sim = seg.a + frac * (seg.b - seg.a)
        acting = tr.robot if tr else "a"
        side = self.colour.get(acting, "")
        img = Image.new("RGB", (W, H), BG)
        img.paste(Image.fromarray(raw[:self.size[1], :self.size[0]]).resize(MAIN), (0, HEADER))
        d = ImageDraw.Draw(img, "RGBA")
        d.text((12, 6), self.title, font=self.f_title, fill=INK)
        note = f"x{speed:.0f} of real time" if speed >= 1.5 else "real time"
        d.text((W - 12 - d.textlength(note, font=self.f_small), 10), note, font=self.f_small, fill=DIM)
        self.label(d, (8, HEADER + 6), "side camera", self.f_small)
        # the acting robot: label, head and wrist cameras, board from above, think timer
        d.rectangle((COL_X, HEADER, W, CAM_Y), fill=PANEL)
        chip = SIDE.get(side, INK)
        d.rounded_rectangle((COL_X + 8, HEADER + 5, COL_X + 28, HEADER + 25), radius=4, fill=chip, outline=(90, 90, 90))
        d.text((COL_X + 36, HEADER + 5), f"{side.upper()} to move · {self.model.get(acting, '')} · robot {acting}",
               font=self.f_mid, fill=INK)
        x = COL_X
        for name, size, text, pos in ((f"{acting}/head", HEAD, "head camera", (x, CAM_Y)),
                                      (f"{acting}/left_wrist", WRIST, "left wrist", (x + HEAD[0], CAM_Y)),
                                      (f"{acting}/right_wrist", WRIST, "right wrist", (x + HEAD[0], CAM_Y + WRIST[1])),
                                      ("board", WRIST, "board from above", (x + HEAD[0] + WRIST[0], CAM_Y))):
            t = self.tile(raw, name, size)
            if t is not None:
                img.paste(t, pos)
                self.label(d, (pos[0] + 5, pos[1] + 3), text, self.f_small)
        think_turn = exec_turn = call_elapsed = 0.0
        if tr:
            before = [c for c in tr.calls if c.sim < sim or (seg.kind != "think" and c.sim <= sim)]
            if seg.kind == "think" and seg.call in tr.calls:
                before = tr.calls[:tr.calls.index(seg.call)]
                call_elapsed = frac * seg.call.seconds
            think_turn = sum(c.seconds for c in before) + call_elapsed
            exec_turn = max(0.0, min(sim, tr.t1) - tr.ready)
        bx, by = COL_X + HEAD[0] + WRIST[0], CAM_Y + WRIST[1]
        thinking = seg.kind == "think"
        d.rectangle((bx, by, W, by + WRIST[1]), fill=(70, 46, 12) if thinking else (22, 44, 30))
        state = "THINKING" if thinking else ("moving" if seg.kind == "exec" else "between turns")
        d.text((bx + 8, by + 5), state, font=self.f_mid, fill=(255, 205, 120) if thinking else (150, 220, 170))
        if thinking:
            d.text((bx + 8, by + 27), secs(call_elapsed), font=self.f_big, fill=(255, 235, 200))
            d.text((bx + 8, by + 60), f"this call: {secs(seg.call.seconds)}", font=self.f_small, fill=(230, 200, 160))
        else:
            d.text((bx + 8, by + 30), f"exec {secs(exec_turn)}", font=self.f_mid, fill=INK)
        d.text((bx + 8, by + 80), f"turn: think {secs(think_turn)}", font=self.f_small, fill=INK)
        # text panel
        d.rectangle((COL_X, PANEL_Y, W, H), fill=PANEL)
        if tr:
            lines = self.panel_lines(tr, seg, frac, detail, sim)
            heights = [ln[2].size + 5 for ln in lines]
            room, used, first = H - PANEL_Y - 16, 0, len(lines)
            while first > 0 and used + heights[first - 1] <= room:
                first -= 1
                used += heights[first]
            y = PANEL_Y + 10
            for ln, h in zip(lines[first:], heights[first:]):
                text, colour, f = ln[:3]
                bold = len(ln) > 3
                d.text((COL_X + 12, y), text, font=f, fill=colour, stroke_width=1 if bold else 0, stroke_fill=colour)
                y += h
        # footer
        d.rectangle((0, FOOT_Y, COL_X, H), fill=(10, 10, 14))
        if tr:
            mv = (tr.n + 1) // 2
            accepted = sim >= tr.accepted_at
            san = tr.san if accepted and tr.san else "choosing …"
            head = f"Move {mv}{'' if side == 'white' else ' …'}   {side.capitalize()}   {san}"
            d.text((12, FOOT_Y + 6), head, font=self.f_big, fill=SIDE.get(side, INK))
            mx = 12 + d.textlength(head, font=self.f_big) + 14
            tags = [t for t in tr.tags if t not in ("FAULT", "RESTORE", "HAND-OVER", "EVAL SWING") or sim >= tr.t1 - 0.5]
            tags = [t for t in tags if accepted or t == "ILLEGAL"]
            for t in tags:
                w = d.textlength(t, font=self.f_small)
                if mx + w + 16 > COL_X - 6:
                    break
                d.rounded_rectangle((mx, FOOT_Y + 11, mx + w + 14, FOOT_Y + 33), radius=6, fill=(*CHIPS[t], 240))
                d.text((mx + 7, FOOT_Y + 14), t, font=self.f_small, fill=(255, 255, 255))
                mx += w + 22
            live = max(0.0, tr.clock_before - think_turn - exec_turn) if sim < tr.t1 else tr.clock_after
            clocks = {tr.robot: live}
            for other in self.model:
                if other != tr.robot:
                    last = [t for t in self.turns if t.robot == other and t.n < tr.n]
                    clocks[other] = last[-1].clock_after if last else 3600.0
            fen = tr.fen_after if sim >= tr.t1 else tr.fen_before
            mw, mb = material(fen)
            wr, br = self.by_colour.get("white", "a"), self.by_colour.get("black", "b")
            d.text((12, FOOT_Y + 44), f"clock   White {clock_text(clocks.get(wr, 0))}    Black {clock_text(clocks.get(br, 0))}"
                                      f"        material   White {mw}   Black {mb}", font=self.f_mid, fill=INK)
            tally = {}
            for rid in self.model:
                past = [t for t in self.turns if t.robot == rid and (t.t1 <= sim or t is tr)]
                tally[rid] = (sum(sum(1 for a, _ in t.illegal if a <= sim) for t in past),
                              sum("manipulation fault" in t.note for t in past if t.t1 <= sim),
                              sum(len(t.restored) for t in past if t.t1 <= sim))
            d.text((12, FOOT_Y + 70), "    ".join(
                f"{self.colour.get(rid, rid).capitalize()}: illegal {v[0]} · faults {v[1]}{self.fault_limit} · restored {v[2]}"
                for rid, v in sorted(tally.items(), key=lambda kv: self.colour.get(kv[0], "") != "white")),
                font=self.f_small, fill=DIM)
            recent = [t for t in self.turns if t.t1 <= sim and t.san][-3:]
            text = "     ".join(f"{(t.n + 1) // 2}{'.' if self.colour.get(t.robot) == 'white' else '…'} {t.san}  "
                                f"think {secs(t.think_s)} / exec {secs(t.exec_s)}" for t in recent)
            d.text((12, FOOT_Y + 94), text or "think and execution time of each move appear here", font=self.f_small,
                   fill=INK if recent else DIM)
            d.text((12, FOOT_Y + 118), f"this move so far:  think {secs(think_turn)} / exec {secs(exec_turn)}",
                   font=self.f_small, fill=(255, 205, 120) if thinking else DIM)
        if over:
            w = max(d.textlength(ln, font=self.f_big) for ln in over.splitlines()) + 60
            d.rounded_rectangle(((MAIN[0] - w) / 2, HEADER + 200, (MAIN[0] + w) / 2, HEADER + 210 + 40 * len(over.splitlines())),
                                radius=12, fill=(10, 10, 14, 225), outline=(255, 255, 255, 120))
            for i, ln in enumerate(over.splitlines()):
                d.text(((MAIN[0] - d.textlength(ln, font=self.f_big)) / 2, HEADER + 208 + 40 * i), ln, font=self.f_big,
                       fill=(255, 255, 255))
        return np.asarray(img)


def render(ep: Path, mode: str, out: Path | None, max_seconds: float, speed: float, fps: float, title: str = "",
           turn_range: tuple[int, int] | None = None, detail: str = "auto", stills: dict[float, Path] | None = None,
           crf: int = 24, cpl: dict | None = None) -> dict:
    frames, trace, result = load(ep)
    times = frames["times"]
    turns = timeline(trace, result, cpl)
    segs = segments(turns, times[0], times[-1])
    keep = None
    if turn_range:
        keep = {t.n for t in turns if turn_range[0] <= t.n <= turn_range[1]}
    elif mode == "highlights":
        keep = choose_highlights(turns, max(4, int(max_seconds // 4)))
    k = plan(segs, mode, max_seconds - (END_HOLD_S if mode != "review" else 0.0), speed, keep)
    if detail == "auto":
        detail = "full" if k <= FULL_DETAIL_SPEED else "key"
    comp = Composer(frames, result, turns, title)
    use = [s for s in segs if s.out > 0]
    edges, acc = [], 0.0
    for s in use:
        edges.append(acc)
        acc += s.out
    n_out = int(acc * fps)
    last_turn = turns[-1] if turns else None
    ending = ""
    if keep is None or (last_turn and last_turn.n in keep):
        ev = result.get("evidence", "")
        why = ev[ev.find("(") + 1:ev.find("):")] if "(" in ev else result.get("exit_reason", "")
        ending = f"{result.get('outcome') or '*'}\n" + "\n".join(textwrap.wrap(why, 44)[:3])
    reader = imageio.get_reader(str(ep / "raw.mp4"))
    writer = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=None, macro_block_size=16, pixelformat="yuv420p",
                                output_params=["-crf", str(crf), "-preset", "slow", "-movflags", "+faststart"]) if out and out.suffix == ".mp4" else None
    want = sorted((int(t * fps), p) for t, p in (stills or {}).items())
    raw, at, frame = None, -1, None
    for n in range(n_out + (int(END_HOLD_S * fps) if ending and mode != "review" else 0)):
        tau = min(n, n_out - 1) / fps
        j = min(len(use) - 1, bisect_right(edges, tau) - 1)
        seg = use[j]
        frac = min(1.0, max(0.0, (tau - edges[j]) / seg.out))
        sim = seg.a + frac * (seg.b - seg.a)
        i = max(0, min(len(times) - 1, bisect_right(times, sim + 1e-6) - 1))
        if not writer and not any(abs(w - n) < 1 for w, _ in want):
            continue
        if i < at:                      # frames are read forwards; a still asked for out of order reopens the file
            reader.close()
            reader, at = imageio.get_reader(str(ep / "raw.mp4")), -1
        while at < i:
            raw = reader.get_next_data()
            at += 1
        frame = comp.compose(raw, seg, frac, detail, k if seg.kind != "think" else k / WEIGHT["think"] * (1 if mode != "review" else WEIGHT["think"]),
                             ending if n >= n_out else "")
        if writer:
            writer.append_data(frame)
        for w, path in want:
            if w == n:
                Image.fromarray(frame).save(path)
    if writer:
        writer.close()
    return {"turns": len(turns), "kept_turns": sorted(keep) if keep else "all", "speed": round(k, 1), "detail": detail,
            "seconds": round(acc + (END_HOLD_S if ending and mode != "review" else 0.0), 1), "frames": n_out,
            "think_real_s": round(sum(s.dur for s in use if s.kind == "think"), 1),
            "exec_real_s": round(sum(s.dur for s in use if s.kind == "exec"), 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="omakase-bench chess video", description=__doc__.split("\n\n")[0])
    ap.add_argument("episode", type=Path)
    ap.add_argument("--mode", choices=("game", "highlights", "review"), default="game")
    ap.add_argument("--out", type=Path, help="mp4 to write")
    ap.add_argument("--max-seconds", type=float, default=210.0)
    ap.add_argument("--speed", type=float, default=2.0, help="review: speed of everything")
    ap.add_argument("--turns", help="only these turns, e.g. 3-6")
    ap.add_argument("--detail", choices=("auto", "key", "full"), default="auto",
                    help="text panel: every call (full) or one key line of thinking per move (key)")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--crf", type=int, default=24)
    ap.add_argument("--title", default="")
    ap.add_argument("--cpl", type=Path, help="report of omakase-bench chess cpl (best with --moves): moves that lose "
                                             f"{SWING_CP} cp or more are tagged and preferred in highlights")
    ap.add_argument("--still", action="append", default=[], metavar="SECOND=PNG",
                    help="also write the frame at this second of the cut to a PNG (repeatable)")
    a = ap.parse_args(argv)
    rng = tuple(int(x) for x in a.turns.split("-")) if a.turns else None
    stills = {float(s.split("=")[0]): Path(s.split("=")[1]) for s in a.still}
    info = render(a.episode, a.mode, a.out, a.max_seconds, a.speed, a.fps, a.title,
                  (rng[0], rng[-1]) if rng else None, a.detail, stills, a.crf,
                  json.loads(a.cpl.read_text()) if a.cpl else None)
    if a.out and a.out.exists():
        info["mb"] = round(a.out.stat().st_size / 1e6, 1)
    print(json.dumps(info))
    return 0


"""Two robots across one table take turns moving their own blocks into a shared middle zone.

    block_relay   each robot has blocks on its side of the table; in its turn it moves one of
                  them into its half of the middle zone. The referee alternates the robots,
                  checks every turn (one own block placed, nothing else moved), puts back what
                  a turn knocked over and ends the game when every block is in the zone.

The generator places the blocks in each robot's own base frame, so the same
parameters describe the cell from either side, and returns per robot the goal
and the layout paragraph in that robot's frame (``values["robots"][id]``).
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from ....core.backend import OracleSnapshot
from ....core.frames import to_base, to_world
from ....core.task import Region, SceneSpec, Shape
from ....core.team import GameResult, GameView, Ruling, TurnReport

Values = dict[str, Any]


def _mm(v: float) -> int:
    return int(round(v * 1000))


def _table_top(scene: SceneSpec) -> float:
    t = next(f for f in scene.fixtures if f.name == "table")
    return t.pos[2] + t.size[2] / 2


def block_relay(scene: SceneSpec, rng: np.random.Generator, p: Mapping[str, Any]) -> tuple[SceneSpec, Values]:
    top, size = _table_top(scene), float(p["block"])
    colours: list[str] = list(p["colours"])
    zone_x, half = float(p["zone_x"]), float(p["zone_half_x"])
    objects, blocks, slots, robots = [], {}, {}, {}
    for r in scene.robots:
        names = []
        for k, (x_rng, y_rng) in enumerate(zip(p["source_x"], p["source_y"])):
            name = f"{r.id}_{colours[k]}_block"
            xy = to_world(r.base, (rng.uniform(*x_rng), rng.uniform(*y_rng)))
            yaw = r.base[2] + math.radians(rng.uniform(-10, 10))
            objects.append(Shape(name, "box", (size, size, size), (float(xy[0]), float(xy[1]), top + size / 2), yaw=yaw,
                                 color=colours[k], mass=0.05, friction=1.0, label=f"{colours[k]} block"))
            slots[name] = tuple(float(v) for v in to_world(r.base, (zone_x, float(p["slot_y"][k]))))
            names.append(name)
        blocks[r.id] = names
    for r in scene.robots:
        lo, hi = float(p["slot_y"][0]) - size, float(p["slot_y"][-1]) + size
        mine = ", ".join(f"{n.split('_')[1]}" for n in blocks[r.id])
        layout = (f"Cell layout in your base frame: the table top is at z={_mm(top)}. Another robot faces you across "
                  f"the table and moves its own blocks in its own turns. Your {len(blocks[r.id])} blocks ({mine}, "
                  f"{_mm(size)} mm cubes) stand on your left near you, in x {_mm(p['source_x'][0][0])}.."
                  f"{_mm(p['source_x'][-1][1])}, y {_mm(p['source_y'][0][0])}..{_mm(p['source_y'][-1][1])}. The "
                  f"middle zone is x {_mm(zone_x - half)}..{_mm(zone_x + half)}; your half of it is y {_mm(lo)}.."
                  f"{_mm(hi)} (the other half belongs to the other robot). Block positions are not given; find them "
                  "in the camera image.")
        robots[r.id] = {"goal": f"Move one of your blocks ({mine}) into your half of the middle zone.",
                        "layout": layout}
    regions = {**scene.regions, "middle_zone": _xy_box(scene, zone_x, half, top)}
    return replace(scene, objects=(*scene.objects, *objects), regions=regions), \
        {"blocks": blocks, "slots": slots, "robots": robots,
         "placed_at": {o.name: [round(o.pos[0], 4), round(o.pos[1], 4)] for o in objects}}


def _xy_box(scene: SceneSpec, zone_x: float, half: float, top: float) -> Region:
    """The middle zone as a world box: ``zone_x ± half`` from the first robot, 0.5 m across."""
    a = to_world(scene.robots[0].base, (zone_x - half, -0.25))
    b = to_world(scene.robots[0].base, (zone_x + half, 0.25))
    return Region("middle_zone", (float(min(a[0], b[0])), float(min(a[1], b[1])), top),
                  (float(max(a[0], b[0])), float(max(a[1], b[1])), top + 0.2))


# --- checks ------------------------------------------------------------------------------------------

def _upright(snap: OracleSnapshot, name: str) -> bool:
    _, x, y, _ = snap.objects[name].quat
    return 1 - 2 * (x * x + y * y) > 0.9


def in_zone(snap: OracleSnapshot, name: str, p: Mapping[str, Any]) -> bool:
    """Resting upright on the table inside the middle zone, not held."""
    o, r = snap.objects[name], snap.regions["middle_zone"]
    size = float(p["block"])
    return (r.lo[0] + size / 2 <= o.pos[0] <= r.hi[0] - size / 2 and r.lo[1] <= o.pos[1] <= r.hi[1]
            and o.pos[2] < r.lo[2] + size / 2 + 0.006 and _upright(snap, name) and name not in snap.held_by
            and float(np.linalg.norm(o.lin_vel)) < 0.02)


def _moved(a: OracleSnapshot, b: OracleSnapshot, name: str) -> float:
    return float(np.linalg.norm(np.array(a.objects[name].pos) - np.array(b.objects[name].pos)))


def _off_pad(snap: OracleSnapshot, name: str) -> bool:
    return any(x == name and y.endswith("_offpad") for x, y in snap.contacts_ever)


class RelayReferee:
    """Alternates the robots, one block per turn; restores what a turn displaced."""

    def __init__(self, values: Mapping[str, Any], params: Mapping[str, Any]):
        self.blocks: Mapping[str, list[str]] = values["blocks"]
        self.p = params
        self.tolerance = float(params.get("displaced_m", 0.02))
        self.placed: dict[str, list[str]] = {r: [] for r in self.blocks}
        self.invalid: dict[str, int] = {r: 0 for r in self.blocks}
        self.lines: list[str] = []
        self._zone_at_start: set[str] = set()

    def _zone(self, snap: OracleSnapshot) -> set[str]:
        return {n for names in self.blocks.values() for n in names if in_zone(snap, n, self.p)}

    def _remaining(self, robot: str, snap: OracleSnapshot) -> list[str]:
        return [n for n in self.blocks[robot] if not in_zone(snap, n, self.p)]

    def _messages(self, game: GameView, nxt: str | None) -> dict[str, str]:
        history = ("Turns so far: " + " ".join(self.lines)) if self.lines else "No turns played yet."
        return {r: f"Referee: {history} It is your turn {len(self.lines) + 1}." for r in game.robots if r == nxt}

    def start(self, game: GameView, snap: OracleSnapshot) -> Ruling:
        first = str(self.p.get("first", game.robots[0]))
        self._zone_at_start = self._zone(snap)
        return Ruling(next_robot=first, messages=self._messages(game, first), note=f"{first} plays first")

    def turn_over(self, game: GameView, robot: str, snap: OracleSnapshot) -> bool:
        new = [n for n in self.blocks[robot] if n not in self._zone_at_start and in_zone(snap, n, self.p)]
        if not new:
            return False
        prefix = f"{robot}/"
        open_min = float(self.p.get("open_min", 0.06))
        clear = snap.regions["middle_zone"].lo[2] + float(self.p["block"]) + float(self.p.get("hands_clear_m", 0.05))
        return (all(v >= open_min for k, v in snap.gripper_opening.items() if k.startswith(prefix))
                and all(p[2] >= clear for k, p in snap.tcp.items() if k.startswith(prefix)))

    def after_turn(self, game: GameView, report: TurnReport) -> Ruling:
        before, after, robot = report.before, report.after, report.robot
        zone = self._zone(after)
        new = [n for n in self.blocks[robot] if n in zone and n not in self._zone_at_start]
        displaced = sorted(n for names in self.blocks.values() for n in names
                           if n not in new and _moved(before, after, n) > self.tolerance)
        valid = len(new) == 1 and not displaced
        restore = {n: (before.objects[n].pos, before.objects[n].quat) for n in displaced}
        # a turn that placed several blocks keeps the first; the others go back
        for n in new[1:]:
            restore[n] = (before.objects[n].pos, before.objects[n].quat)
        if new:
            self.placed[robot].append(new[0])
        if not valid:
            self.invalid[robot] += 1
        what = f"put its {new[0].split('_')[1]} block in the middle zone" if new else "placed no block"
        if displaced:
            what += f" and displaced {', '.join(displaced)} (put back)"
        self.lines.append(f"turn {report.turn}: robot {robot} {what}.")
        # what the referee puts back is where it was before the turn: a block already placed stays placed
        put_back = set(new[1:]) | set(displaced)
        self._zone_at_start = (zone - put_back) | (self._zone_at_start & put_back)
        others = [r for r in game.robots if r != robot] + [robot]
        nxt = next((r for r in others if r not in game.out and self._remaining(r, after)), None)
        if all(not self._remaining(r, after) for r in game.robots):
            nxt = None
        return Ruling(next_robot=nxt, messages=self._messages(game, nxt), restore=restore, valid=valid,
                      note=self.lines[-1])

    def result(self, game: GameView, snap: OracleSnapshot) -> GameResult:
        names = [n for ns in self.blocks.values() for n in ns]
        done = [n for n in names if in_zone(snap, n, self.p)]
        off = [n for n in names if _off_pad(snap, n)]
        ok = len(done) == len(names) and not off
        per = {r: {"placed": sum(n in done for n in self.blocks[r]), "blocks": len(self.blocks[r]),
                   "invalid_turns": self.invalid[r],
                   "off_pad": [n for n in self.blocks[r] if n in off]} for r in self.blocks}
        evidence = f"{len(done)} of {len(names)} blocks in the middle zone" + (f"; touched off the pads: {', '.join(off)}"
                                                                               if off else "")
        return GameResult(ok, evidence, "complete" if ok else "incomplete", per)


def relay_moves(snap: OracleSnapshot, v: Mapping[str, Any], p: Mapping[str, Any]) -> list[dict]:
    """The scripted solver's move for one turn: the robot's next block not yet in the zone, to its
    slot. ``snap`` is the robot's own view (``snap.base`` is its base pose); positions come back in
    its base frame."""
    rid = v["robot"]
    name = next(n for n in v["blocks"][rid] if not in_zone(snap, n, p))
    size = float(p["block"])
    top = snap.regions["middle_zone"].lo[2]
    o = snap.objects[name]
    w, x, y, z = o.quat
    yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)) - snap.base[2])
    # a cube grips alike every quarter turn: the yaw nearest the start yaw of the task's ee block
    start_yaw = float((p.get("start_yaw") or 0.0))
    yaw = start_yaw + (yaw - start_yaw + 45) % 90 - 45
    return [{"arm": "left", "top": to_base(snap.base, np.array(o.pos)) + [0, 0, size / 2], "height": size,
             "yaw": round(yaw), "place": to_base(snap.base, np.array([*v["slots"][name], 0.0]))[:2],
             "support_z": top, "clear_z": top + size}]


MOVES = {"team/block_relay": relay_moves}

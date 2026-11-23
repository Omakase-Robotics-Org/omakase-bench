"""Tabletop tasks with several named objects and a goal that changes per episode.

Each task is a YAML file whose ``generator`` names a function here. The
generator receives the static scene, the episode's random generator and the
YAML ``params`` (every dimension of the cell), and returns the episode's scene
plus its values: the goal sentence, a ``layout`` paragraph stating what an
installation would know about the cell (never object positions), and what the
success check and the scripted solver need. ``success`` names the matching
check through the ``goal`` predicate.

Rules shared by the checks: an object the task asks to move must have touched
a gripper only on the fingertip pads within the preset's pad band, and both
grippers must be open at the end.

    chess_move     move the named piece to the named square of a board section
    stack          build a tower of cubes in the named order
    sort           put every cylinder into the tray of its colour
    clutter_pick   lift the named cylinder out of a tightly packed row
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from ....core.backend import OracleSnapshot
from ....core.predicates import Verdict
from ....core.task import Region, SceneSpec, Shape

Values = dict[str, Any]


def _mm(v: float) -> int:
    return int(round(v * 1000))


def _fixture(scene: SceneSpec, name: str) -> Shape:
    return next(f for f in scene.fixtures if f.name == name)


def _table_text(scene: SceneSpec) -> str:
    t = _fixture(scene, "table")
    (sx, sy, sz), (cx, cy, cz) = t.size, t.pos
    return (f"Cell layout: the table top is at z={_mm(cz + sz / 2)} and spans x {_mm(cx - sx / 2)}..{_mm(cx + sx / 2)}, "
            f"y {_mm(cy - sy / 2)}..{_mm(cy + sy / 2)}. ")


def _table_top(scene: SceneSpec) -> float:
    t = _fixture(scene, "table")
    return t.pos[2] + t.size[2] / 2


def _cylinder(name: str, radius: float, height: float, xy, bottom: float, color, label: str = "") -> Shape:
    return Shape(name, "cylinder", (float(radius), float(height)), (float(xy[0]), float(xy[1]), bottom + height / 2),
                 color=color, mass=0.05, friction=1.0, label=label or name.replace("_", " "))


# --- checks shared by the tasks ----------------------------------------------------------------

def _upright(snap: OracleSnapshot, name: str) -> bool:
    _, x, y, _ = snap.objects[name].quat
    return 1 - 2 * (x * x + y * y) > 0.9


def _moved(snap: OracleSnapshot, name: str) -> float:
    return float(np.linalg.norm(np.array(snap.objects[name].pos[:2]) - np.array(snap.initial[name][:2])))


def _still(snap: OracleSnapshot, name: str) -> bool:
    return float(np.linalg.norm(snap.objects[name].lin_vel)) < 0.02


def _touched(snap: OracleSnapshot, name: str) -> bool:
    return any(a == name and b.endswith("_hand") for a, b in snap.contacts_ever)


def pad_grasp(snap: OracleSnapshot, name: str) -> bool:
    """The object touched a gripper, and only on the fingertip pads within the pad band."""
    return _touched(snap, name) and not any(a == name and b.endswith("_offpad") for a, b in snap.contacts_ever)


def _released(snap: OracleSnapshot, params: Mapping[str, Any]) -> bool:
    return all(v >= float(params.get("open_min", 0.06)) for v in snap.gripper_opening.values())


def _others_ok(snap: OracleSnapshot, moved: tuple[str, ...], limit: float) -> list[str]:
    """Names of the bystanders that were moved beyond ``limit`` or tipped over."""
    return sorted(k for k, o in snap.objects.items() if o.kind == "object" and k not in moved
                  and (_moved(snap, k) > limit or not _upright(snap, k)))


def _verdict(checks: Mapping[str, bool], note: str = "") -> Verdict:
    failed = [k for k, ok in checks.items() if not ok]
    return Verdict(not failed, (("failed: " + ", ".join(failed)) if failed else "all checks hold")
                   + (f"; {note}" if note else ""))


# --- chess_move ----------------------------------------------------------------------------------

def _square_xy(p: Mapping[str, Any], sq: str) -> np.ndarray:
    """Centre of a square such as 'g1': file a on the robot's left (+y), rank 1 nearest the robot."""
    files, ranks, s = p["files"], str(p["ranks"]), float(p["square"])
    return np.array([float(p["rank1_x"]) + s * ranks.index(sq[1]), s * ((len(files) - 1) / 2 - files.index(sq[0]))])


def chess_move(scene: SceneSpec, rng: np.random.Generator, p: Mapping[str, Any]) -> tuple[SceneSpec, Values]:
    files, ranks, s = p["files"], str(p["ranks"]), float(p["square"])
    top, thick = _table_top(scene), float(p["board_thickness"])
    board_top = top + thick
    cx = float(p["rank1_x"]) + s * (len(ranks) - 1) / 2
    fixtures = [Shape("board", "box", (s * len(ranks), s * len(files), thick), (cx, 0.0, top + thick / 2),
                      color=tuple(p["board_light"]), label="board")]
    proud = 0.0004     # dark squares stand proud of the light base so the two colours do not share a surface
    for f in files:
        for r in ranks:
            if (files.index(f) + ranks.index(r)) % 2 == 0:      # a1 is a dark square
                x, y = _square_xy(p, f + r)
                fixtures.append(Shape(f"sq_{f}{r}", "box", (s, s, thick + proud), (x, y, top + (thick + proud) / 2),
                                      color=tuple(p["board_dark"]), label="square"))
    pieces: Mapping[str, list] = p["pieces"]
    names = list(pieces)
    squares = [f + r for f in files for r in ranks]
    arm_files: Mapping[str, str] = p["arm_files"]

    def free(sq: str, taken: set[str]) -> bool:
        # the open jaws straddle a piece along its rank and need the neighbouring files clear
        i = files.index(sq[0])
        return not any(files[j] + sq[1] in taken for j in (i - 1, i, i + 1) if 0 <= j < len(files))

    while True:
        placed = dict(zip(names, (squares[i] for i in rng.permutation(len(squares))[:len(names)])))
        mover = names[int(rng.integers(len(names)))]
        src = placed[mover]
        others = set(placed.values()) - {src}
        arms = [a for a in arm_files if src[0] in arm_files[a]]
        targets = [q for q in squares if q != src and free(q, others) and any(q[0] in arm_files[a] for a in arms)]
        if free(src, others) and targets:
            dst = targets[int(rng.integers(len(targets)))]
            break
    radius = float(p["piece_radius"])
    objects = tuple(_cylinder(k, radius, float(pieces[k][1]), _square_xy(p, sq), board_top, pieces[k][0])
                    for k, sq in placed.items())
    tx, ty = _square_xy(p, dst)
    region = Region("target_square", (tx - s / 2, ty - s / 2, board_top), (tx + s / 2, ty + s / 2, board_top + 0.2))
    file_y = ", ".join(f"{f} y={_mm(_square_xy(p, f + ranks[0])[1])}" for f in files)
    rank_x = ", ".join(f"{r} x={_mm(_square_xy(p, files[0] + r)[0])}" for r in ranks)
    kinds: dict[str, float] = {}
    for k in names:
        kinds.setdefault(k.split("_", 1)[1], float(pieces[k][1]))
    heights = ", ".join(f"{kind} {_mm(h)} mm{' tall' if i == 0 else ''}" for i, (kind, h) in enumerate(kinds.items()))
    reach = " and ".join(f"the {a} arm reaches files {fs[0]}..{fs[-1]}" for a, fs in arm_files.items())
    layout = (_table_text(scene)
              + f"A chessboard section lies on the table: files {files[0]}..{files[-1]} and ranks {ranks[0]}..{ranks[-1]} "
              f"as White sees them from the robot's side (file a on the robot's left, rank 1 nearest the robot), "
              f"{_mm(s)} mm squares, a1 dark, board surface at z={_mm(board_top)}. Square centres: files {file_y}; "
              f"ranks {rank_x}. The pieces are cylinders of radius {_mm(radius)} mm standing on square centres: "
              f"{heights}; white pieces are white, black pieces are black. On the board: "
              + ", ".join(k.replace("_", " ") for k in names) + f". {reach[0].upper() + reach[1:]}. Only the squares "
              "named in the goal are given; find the other pieces in the image.")
    values = {"goal": f"Move the {mover.replace('_', ' ')} from {src} to {dst}.", "layout": layout,
              "piece": mover, "from": src, "to": dst, "squares": placed}
    return replace(scene, fixtures=(*scene.fixtures, *fixtures), objects=objects,
                   regions={**scene.regions, region.name: region}), values


def chess_move_ok(snap: OracleSnapshot, v: Mapping[str, Any], p: Mapping[str, Any]) -> Verdict:
    k = v["piece"]
    o = snap.objects[k]
    r = snap.regions["target_square"]
    disturbed = _others_ok(snap, (k,), float(p["max_other_displacement"]))
    err = float(np.linalg.norm(np.array(o.pos[:2]) - _square_xy(p, v["to"])))
    return _verdict({
        "piece inside the target square": r.lo[0] < o.pos[0] < r.hi[0] and r.lo[1] < o.pos[1] < r.hi[1],
        "piece resting on the board": o.pos[2] < r.lo[2] + o.size[1] / 2 + 0.004,
        "piece upright": _upright(snap, k),
        "grippers open": _released(snap, p),
        "grasp on the pads": pad_grasp(snap, k),
        "other pieces undisturbed": not disturbed,
    }, f"{k} {err * 1000:.0f} mm from the centre of {v['to']}" + (f", disturbed: {', '.join(disturbed)}" if disturbed else ""))


def chess_move_moves(snap: OracleSnapshot, v: Mapping[str, Any], p: Mapping[str, Any]) -> list[dict]:
    k, src, dst = v["piece"], v["from"], v["to"]
    arm_files: Mapping[str, str] = p["arm_files"]
    arm = next(a for a in reversed(list(arm_files)) if src[0] in arm_files[a] and dst[0] in arm_files[a])
    h = float(p["pieces"][k][1])
    board_top = snap.regions["target_square"].lo[2]
    return [{"arm": arm, "top": np.array([*_square_xy(p, src), board_top + h]), "height": h,
             "place": _square_xy(p, dst), "support_z": board_top + 0.0004,
             "clear_z": board_top + max(float(x[1]) for x in p["pieces"].values())}]


# --- stack -----------------------------------------------------------------------------------------

def stack(scene: SceneSpec, rng: np.random.Generator, p: Mapping[str, Any]) -> tuple[SceneSpec, Values]:
    colours: list[str] = list(p["colours"])
    block, top = float(p["block"]), _table_top(scene)
    n = int(rng.integers(2, len(colours) + 1))
    order = [f"{colours[i]}_block" for i in rng.permutation(len(colours))[:n]]
    side = 1.0 if rng.random() < 0.5 else -1.0
    while True:
        pts = [np.array([rng.uniform(*p["x"]), side * rng.uniform(*p["y_abs"])]) for _ in order]
        if all(np.linalg.norm(a - b) >= float(p["separation"]) for i, a in enumerate(pts) for b in pts[:i]):
            break
    objects = tuple(Shape(k, "box", (block, block, block), (float(q[0]), float(q[1]), top + block / 2),
                          color=k.split("_")[0], mass=0.05, friction=1.0, label=k.replace("_", " "))
                    for k, q in zip(order, pts))
    names = [k.split("_")[0] for k in order]
    layout = (_table_text(scene)
              + f"Two or three {_mm(block)} mm cubes ({', '.join(colours)}) stand on the table on one side, in "
              f"x {_mm(p['x'][0])}..{_mm(p['x'][1])}, |y| {_mm(p['y_abs'][0])}..{_mm(p['y_abs'][1])}, with their faces "
              f"parallel to the x and y axes. A block resting on the table has its top at z={_mm(top + block)}; each "
              f"block added to the tower raises the top by {_mm(block)} mm. Their positions are not given; find them "
              "in the camera image.")
    values = {"goal": f"Stack order from the bottom: {', '.join(names)}. The {names[0]} block stays where it is.",
              "layout": layout, "order": order}
    return replace(scene, objects=objects), values


def _stacked(snap: OracleSnapshot, order: list[str], p: Mapping[str, Any]) -> int:
    """Number of blocks correctly stacked, counting the bottom one."""
    block, n = float(p["block"]), 1
    for lower, upper in zip(order, order[1:]):
        d = np.array(snap.objects[upper].pos) - np.array(snap.objects[lower].pos)
        if not (np.linalg.norm(d[:2]) <= float(p["align_tol"]) and abs(d[2] - block) < 0.006 and _upright(snap, upper)):
            break
        n += 1
    return n


def stack_ok(snap: OracleSnapshot, v: Mapping[str, Any], p: Mapping[str, Any]) -> Verdict:
    order = list(v["order"])
    n = _stacked(snap, order, p)
    return _verdict({
        "tower complete": n == len(order),
        "grippers open": _released(snap, p),
        "bottom block in place": _moved(snap, order[0]) <= 0.02,
        "tower at rest": all(_still(snap, k) for k in order),
        "grasps on the pads": all(pad_grasp(snap, k) for k in order[1:]),
        "bottom block untouched": not _touched(snap, order[0]),
    }, f"{n} of {len(order)} blocks stacked")


def stack_moves(snap: OracleSnapshot, v: Mapping[str, Any], p: Mapping[str, Any]) -> list[dict]:
    order, block = list(v["order"]), float(p["block"])
    base = np.array(snap.initial[order[0]])
    top = base[2] - block / 2
    return [{"arm": "left" if base[1] > 0 else "right", "top": np.array(snap.initial[k]) + [0, 0, block / 2],
             "height": block, "place": base[:2], "support_z": top + block * (i + 1), "clear_z": top + block * (i + 1),
             "approach_z": top + block * (i + 1) + 0.03}      # the open jaws pass above the tower
            for i, k in enumerate(order[1:])]


# --- sort ------------------------------------------------------------------------------------------

def sort(scene: SceneSpec, rng: np.random.Generator, p: Mapping[str, Any]) -> tuple[SceneSpec, Values]:
    top = _table_top(scene)
    trays: Mapping[str, str] = p["trays"]                      # colour -> tray fixture
    colours_all = list(trays)
    n = int(rng.integers(2, int(p["max_objects"]) + 1))
    keys = [f"item_{i + 1}" for i in range(n)]
    while True:
        pts = [np.array([rng.uniform(*p["x"]), rng.uniform(*p["y"])]) for _ in keys]
        if all(abs(a[0] - b[0]) >= float(p["separation"]) for i, a in enumerate(pts) for b in pts[:i]):
            break
    colours = colours_all[:2] + [colours_all[int(rng.integers(len(colours_all)))] for _ in keys[2:]]
    colours = [colours[i] for i in rng.permutation(n)]
    objects = []
    for k, q, c in zip(keys, pts, colours):
        radius, height = rng.uniform(*p["radius"]), rng.uniform(*p["height"])
        objects.append(_cylinder(k, radius, height, q, top, c, label=f"{c} cylinder"))
    side = {c: ("left" if _fixture(scene, t).pos[1] > 0 else "right") for c, t in trays.items()}
    tray = _fixture(scene, next(iter(trays.values())))
    tr = "; ".join(f"{side[c]} tray centre x={_mm(_fixture(scene, t).pos[0])} y={_mm(_fixture(scene, t).pos[1])}"
                   for c, t in trays.items())
    layout = (_table_text(scene)
              + f"Two white trays are fixed on the table: {tr}; each {_mm(tray.size[0])} mm square, "
              f"{_mm(tray.size[2])} mm high, walls and floor {_mm(tray.wall)} mm thick. Two or three "
              f"{' or '.join(colours_all)} cylinders (radius {_mm(p['radius'][0])}..{_mm(p['radius'][1])} mm, height "
              f"{_mm(p['height'][0])}..{_mm(p['height'][1])} mm) stand between the trays in x {_mm(p['x'][0])}.."
              f"{_mm(p['x'][1])}, y {_mm(p['y'][0])}..{_mm(p['y'][1])}, at least {_mm(p['separation'])} mm apart in x; "
              "both arms reach this strip. Their positions are not given; find them in the camera image.")
    goal = ", ".join(f"{c} cylinders go to the {side[c]} tray" for c in colours_all)
    values = {"goal": goal[0].upper() + goal[1:] + ".", "layout": layout, "colours": dict(zip(keys, colours))}
    return replace(scene, objects=tuple(objects)), values


def _in_tray(snap: OracleSnapshot, name: str, tray: str, wall: float) -> bool:
    o, t = snap.objects[name], snap.objects[tray]
    inner = t.size[0] / 2 - wall
    bottom = t.pos[2] - t.size[2] / 2
    return bool(np.all(np.abs(np.array(o.pos[:2]) - np.array(t.pos[:2])) < inner - o.size[0] * 0.5)
                and o.pos[2] < bottom + wall + o.size[1] / 2 + 0.015)


def sort_ok(snap: OracleSnapshot, v: Mapping[str, Any], p: Mapping[str, Any]) -> Verdict:
    colours: Mapping[str, str] = v["colours"]
    done = [k for k, c in colours.items() if _in_tray(snap, k, p["trays"][c], float(p["tray_wall"]))]
    return _verdict({
        "every cylinder in its tray": len(done) == len(colours),
        "grippers open": _released(snap, p),
        "grasps on the pads": all(pad_grasp(snap, k) for k in colours),
    }, f"{len(done)} of {len(colours)} sorted")


def sort_moves(snap: OracleSnapshot, v: Mapping[str, Any], p: Mapping[str, Any]) -> list[dict]:
    moves, placed = [], {c: 0 for c in p["trays"]}
    for k, c in v["colours"].items():
        o, t = snap.objects[k], snap.objects[p["trays"][c]]
        height = o.size[1]
        # cylinders of one colour go side by side along x, not on top of each other
        spot = np.array(t.pos[:2]) + [(-0.03, 0.03)[placed[c] % 2], 0.0]
        placed[c] += 1
        floor = t.pos[2] - t.size[2] / 2 + float(p["tray_wall"])
        moves.append({"arm": "left" if t.pos[1] > 0 else "right",
                      "top": np.array(snap.initial[k]) + [0, 0, height / 2], "height": height, "place": spot,
                      "support_z": floor, "clear_z": t.pos[2] + t.size[2] / 2 + 0.025})
    return moves


# --- clutter_pick ----------------------------------------------------------------------------------

def clutter_pick(scene: SceneSpec, rng: np.random.Generator, p: Mapping[str, Any]) -> tuple[SceneSpec, Values]:
    top = _table_top(scene)
    colours: list[str] = list(p["colours"])
    n = int(rng.integers(int(p["count"][0]), int(p["count"][1]) + 1))
    keys = [f"{colours[i]}_cylinder" for i in rng.permutation(len(colours))[:n]]
    side = 1.0 if rng.random() < 0.5 else -1.0
    x0 = rng.uniform(*p["x"])
    sizes = [(rng.uniform(*p["radius"]), rng.uniform(*p["height"])) for _ in keys]
    gaps = [rng.uniform(*p["gap"]) for _ in keys[1:]]
    ys = [0.0]
    for i, g in enumerate(gaps):
        ys.append(ys[-1] + sizes[i][0] + g + sizes[i + 1][0])
    ys = np.array(ys) - (ys[0] + ys[-1]) / 2 + side * float(p["y_abs"])
    target = keys[int(rng.integers(n))]
    objects = tuple(_cylinder(k, r, h, (x0 + rng.uniform(-0.003, 0.003), y), top, k.split("_")[0])
                    for k, y, (r, h) in zip(keys, ys, sizes))
    lift = float(p["lift"])
    layout = (_table_text(scene)
              + f"Four to six cylinders of different colours ({', '.join(colours)}; radius {_mm(p['radius'][0])}.."
              f"{_mm(p['radius'][1])} mm, height {_mm(p['height'][0])}..{_mm(p['height'][1])} mm) stand on the table on "
              f"one side in a row parallel to the y axis, at x {_mm(p['x'][0])}..{_mm(p['x'][1])}, centred near "
              f"|y|={_mm(p['y_abs'])}, with gaps of {_mm(p['gap'][0])}..{_mm(p['gap'][1])} mm between neighbours. The "
              "jaws do not fit between neighbours. Their positions are not given; find them in the camera image.")
    values = {"goal": f"Pick the {target.split('_')[0]} cylinder and lift it {_mm(lift)} mm.", "layout": layout,
              "target": target}
    return replace(scene, objects=objects), values


def clutter_pick_ok(snap: OracleSnapshot, v: Mapping[str, Any], p: Mapping[str, Any]) -> Verdict:
    k = v["target"]
    lift = snap.objects[k].pos[2] - snap.initial[k][2]
    disturbed = _others_ok(snap, (k,), float(p["max_other_displacement"]))
    return _verdict({
        "cylinder lifted": lift >= float(p["lift"]),
        "grasp on the pads": pad_grasp(snap, k),
        "neighbours undisturbed": not disturbed,
    }, f"lifted {lift * 1000:.0f} mm" + (f", disturbed: {', '.join(disturbed)}" if disturbed else ""))


def clutter_pick_moves(snap: OracleSnapshot, v: Mapping[str, Any], p: Mapping[str, Any]) -> list[dict]:
    k = v["target"]
    pos, height = np.array(snap.initial[k]), snap.objects[k].size[1]
    # each arm reaches the quarter turn towards its own side
    return [{"arm": "left" if pos[1] > 0 else "right", "top": pos + [0, 0, height / 2], "height": height,
             "lift": float(p["lift"]) + 0.03, "yaw": 90 if pos[1] < 0 else -90}]


#: task id -> pick-and-place moves for the scripted solver
MOVES = {"manip/chess_move": chess_move_moves, "manip/stack": stack_moves, "manip/sort": sort_moves,
         "manip/clutter_pick": clutter_pick_moves}

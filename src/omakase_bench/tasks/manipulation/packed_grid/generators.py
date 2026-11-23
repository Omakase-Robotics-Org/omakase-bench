"""Tabletop tasks whose objects stand so close that the task has to limit the gripper opening.

    packed_grid_pick   lift the named cylinder out of a 3 x 3 grid; neighbours along the jaw
                       axis stand closer than a fully open jaw reaches, so the task's
                       ``gripper:`` block limits the opening

The success check is the clutter_pick check of :mod:`tabletop` (named cylinder lifted, pads only,
neighbours undisturbed). Kept apart from :mod:`tabletop` so that ``manip@0`` keeps its hash.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from ....core.backend import OracleSnapshot
from ....core.task import SceneSpec
from ..tabletop.generators import Values, _cylinder, _mm, _table_text, _table_top


def packed_grid_pick(scene: SceneSpec, rng: np.random.Generator, p: Mapping[str, Any]) -> tuple[SceneSpec, Values]:
    top = _table_top(scene)
    rows, cols = int(p["rows"]), int(p["cols"])
    sx, sy = float(p["spacing_x"]), rng.uniform(*p["spacing_y"])
    radius, height = rng.uniform(*p["radius"]), rng.uniform(*p["height"])
    side = 1.0 if rng.random() < 0.5 else -1.0
    cx, cy = rng.uniform(*p["x"]), side * float(p["y_abs"])
    colours: list[str] = list(p["colours"])
    colour = colours[int(rng.integers(len(colours)))]
    target_cell = int(rng.integers(rows * cols))
    objects, target = [], ""
    for i in range(rows):
        for j in range(cols):
            k = i * cols + j
            xy = (cx + (i - (rows - 1) / 2) * sx + rng.uniform(-0.002, 0.002),
                  cy + (j - (cols - 1) / 2) * sy + rng.uniform(-0.002, 0.002))
            if k == target_cell:
                target = f"{colour}_cylinder"
                objects.append(_cylinder(target, radius, height, xy, top, colour))
            else:
                objects.append(_cylinder(f"cylinder_{k}", radius, height, xy, top, p["others"], label="cylinder"))
    lift = float(p["lift"])
    layout = (_table_text(scene)
              + f"{rows * cols} cylinders of equal size (radius {_mm(p['radius'][0])}..{_mm(p['radius'][1])} mm, height "
              f"{_mm(p['height'][0])}..{_mm(p['height'][1])} mm) stand in a {rows} x {cols} grid on one side of the table, "
              f"centred at x {_mm(p['x'][0])}..{_mm(p['x'][1])} and |y|={_mm(p['y_abs'])}: rows {_mm(sx)} mm apart along x, "
              f"columns {_mm(p['spacing_y'][0])}..{_mm(p['spacing_y'][1])} mm apart along y. One cylinder is "
              f"{colour}, the others are {p['others']}. Fully open jaws would land on the neighbours. Their positions "
              "are not given; find them in the camera image.")
    values = {"goal": f"Pick the {colour} cylinder and lift it {_mm(lift)} mm.", "layout": layout, "target": target}
    return replace(scene, objects=tuple(objects)), values


def packed_grid_pick_moves(snap: OracleSnapshot, v: Mapping[str, Any], p: Mapping[str, Any]) -> list[dict]:
    """Grasp at yaw 0 (jaws closing along y, across the tighter spacing)."""
    k = v["target"]
    pos, height = np.array(snap.initial[k]), snap.objects[k].size[1]
    return [{"arm": "left" if pos[1] > 0 else "right", "top": pos + [0, 0, height / 2], "height": height,
             "lift": float(p["lift"]) + 0.03, "yaw": 0}]


#: task id -> pick-and-place moves for the scripted solver
MOVES = {"grip/packed_grid_pick": packed_grid_pick_moves}

"""Reach check of the chess cell through the harness: one robot carries a king, the tallest piece, to every
square and every capture slot in turn.

    omakase-bench chess tour ROBOT OUT.json [--targets e4,slot:w:7]

The robot drives ``params.reach_forward_m`` forward through the base command a model uses and moves the king
with the scripted player's pick and place (rows through the ee controller, the reach table's arm and tilt
per square, a hand-over where no arm reaches both ends); the other robot waits in its idle pose at its
station. A placement counts when the king stands upright within the referee's tolerance of the target,
is not held and no waypoint was given up. Prints one line per target and writes the placements, the faults
and the contacts between the two robots to OUT.json. Exit status 1 when a placement failed or the robots
touched.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from ....backends.mujoco.backend import MujocoBackend
from ....core.frames import to_base
from ....core.suite import find_task
from ....core.task import Episode, robot_for
from ....core.team import inter_robot_contacts
from ....harness.ee_control import EEController, EELimits, WholeBody
from ....harness.session import Recorder, RunBudget
from ....harness.team_session import RobotSession, RobotView, World, robot_task, transfer
from . import rules as chess
from .layout import Layout, _upright_cos, reach_tilt
from .player import _bottom_xy, _Follower, _legs, _pick_place

TASK = "games/chess"
MAX_STEPS = 900          # ee steps allowed for one placement


def tour(rid: str, targets: list[str] | None = None, progress=print) -> dict:
    task = find_task(TASK)
    ep = task.resolve(0)
    p, ee = task.raw["params"], task.raw["ee"]
    colour = rid == p["sides"]["white"]
    king = next(o for o in ep.scene.objects if o.name == ("w_king_e1" if colour else "b_king_e8"))
    scene = replace(ep.scene, objects=(king,))
    robots = {r.id: robot_for(task, scene, r.id) for r in scene.robots}
    be = MujocoBackend()
    be.load_team(robots, scene)
    be.reset(scene, seed=0)
    world = World(be, task.team, Recorder(enabled=False), 1e9)
    sessions = {}
    for r in task.team.ids:
        slot = task.team.slot(r)
        episode = Episode(robot_task(task, slot), 0, scene, {**ep.values, "robot": r}, ep.instruction)
        sessions[r] = RobotSession(world, r, episode, RobotView(world, r, robots[r], slot.cameras), robots[r],
                                   run_budget=RunBudget(None, None), time_mode="lockstep", on_action=lambda x: None)
    for r in (*[x for x in task.team.ids if x != rid], rid):
        transfer(world, sessions[r], sessions[r].idle_pose)
    s = sessions[rid]
    transfer(world, s, None, s.start_lift)
    L = Layout.from_params(scene, p)
    ctl = EEController(s, EELimits.from_mapping(ee.get("limits")))
    ctl.max_tilt_deg = float(ee["tilt"]["max_deg"])
    ctl.move_to_start(ee["start"])
    ctl.body = WholeBody.from_mapping(ee.get("whole_body"), ee.get("base_noise"))
    progress("drive: " + ctl.base_move(float(p["reach_forward_m"])))
    park = {a: tuple(ee["start"][a]) for a in ctl.arms}
    me = s.oracle().base
    c = "w" if colour else "b"
    order = []                       # a snake over the board from the mover's near rank, then the slots
    for rank in (range(8) if colour else range(7, -1, -1)):
        files = "abcdefgh" if (rank % 2 == 0) == colour else "hgfedcba"
        order += [f"{f}{rank + 1}" for f in files]
    order += [f"slot:{c}:{k}" for k in range(L.n_slots)]
    here = "e1" if colour else "e8"

    def world_xy(where: str) -> tuple[np.ndarray, float]:
        if where.startswith("slot:"):
            return L.slot_xy(colour, int(where.split(":")[2])), L.table_top
        return L.square_xy(chess.parse_square(where)), L.board_top

    def measure(arm: str) -> np.ndarray:
        snap = s.oracle()
        if snap.held_by.get(king.name) != arm:
            return np.zeros(2)
        return _bottom_xy(snap, king.name, me) - ctl.tip_pose(arm)[0][:2]

    def locate(xy) -> np.ndarray:
        found = _bottom_xy(s.oracle(), king.name, me)
        return found if np.linalg.norm(found - xy) < 0.03 else np.asarray(xy, float)

    results = []
    for target in targets or order:
        if target == here:
            continue
        snap = s.oracle()
        legs = _legs(p, colour, here, target, set(), L)
        waypoints = []
        for arm, a, b in legs:
            (sxy, sz), (txy, tz) = world_xy(a), world_xy(b)
            if a == here:
                sxy = np.array(snap.objects[king.name].pos[:2])
            waypoints += _pick_place(L, p, colour, arm, "king",
                                     (to_base(me, np.array([*sxy, 0]))[:2], sz, reach_tilt(p, colour, arm, a)),
                                     (to_base(me, np.array([*txy, 0]))[:2], tz, reach_tilt(p, colour, arm, b)), park[arm])
        follower = _Follower(ctl, waypoints, measure, locate)
        unreachable = 0
        for _ in range(MAX_STEPS):
            row = follower.row()
            if row is None:
                break
            unreachable += len(ctl.step(row))
        s.advance(0.5)
        snap = s.oracle()
        o = snap.objects[king.name]
        xy, support = world_xy(target)
        err = float(np.linalg.norm(np.array(o.pos[:2]) - xy))
        tol = p["slot_tolerance_m"] if target.startswith("slot") else p["tolerance_m"]
        ok = bool(err <= tol and _upright_cos(o.quat) > float(p.get("upright_cos", 0.95))
                  and king.name not in snap.held_by and follower.given_up == 0)
        results.append({"target": target, "arms": [leg[0] for leg in legs], "ok": ok, "err_mm": round(err * 1000, 1),
                        "upright": round(_upright_cos(o.quat), 3), "unreachable_steps": unreachable,
                        "gave_up": follower.gave_up})
        progress(f"{target} {'+'.join(leg[0] for leg in legs)} {'OK' if ok else 'FAIL'} {err * 1000:.1f} mm "
                 f"unreachable {unreachable}")
        if not ok:                   # stand the king on the target so that the tour goes on
            be.set_object_pose(king.name, (*xy, support + L.height("king") / 2), (1, 0, 0, 0))
        here = target
    return {"robot": rid, "results": results, "faults": list(world.faults),
            "inter_robot": [list(pair) for pair in inter_robot_contacts(be.oracle(), task.team.ids)]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="omakase-bench chess tour", description=__doc__.split("\n\n")[0])
    ap.add_argument("robot", help="robot id of games/chess (a or b)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--targets", help="comma list of squares and slots (slot:<w|b>:<k>) instead of the whole tour")
    a = ap.parse_args(argv)
    report = tour(a.robot, a.targets.split(",") if a.targets else None)
    a.out.write_text(json.dumps(report, indent=1))
    ok = sum(r["ok"] for r in report["results"])
    worst = max((r["err_mm"] for r in report["results"]), default=0.0)
    print(f"robot {a.robot}: {ok}/{len(report['results'])} placements ok, largest error {worst} mm, "
          f"faults {len(report['faults'])}, contacts between the robots {len(report['inter_robot'])}")
    return 0 if ok == len(report["results"]) and not report["inter_robot"] else 1


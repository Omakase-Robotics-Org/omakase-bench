"""The scripted chess player: reads ground truth and plays the moves of ``params.scripted_games`` with
the pick and place a model would have to find (reach table, tilt, hand-overs)."""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from ....core.backend import OracleSnapshot
from ....core.frames import to_base
from ....core.task import SceneSpec, Shape
from . import rules as chess
from .layout import Layout, reach_tilt
from .referee import _Placement


GRASP_INSERTION = 0.014     # m, fingertip centre below the top of a piece when the jaws close
CARRY_CLEAR = 0.020         # m, carried piece's bottom above the tallest piece on the board
RELEASE_GAP = 0.003         # m, piece bottom above its support when the jaws open
FINE_STEP = 0.015           # m, vertical increments between standing pieces, each awaited
PATH_STEP = 0.03            # m, waypoints along a carry path; the tilt changes gradually along it
STUCK_STEPS = 20            # steps on one waypoint before the scripted player gives it up
ARRIVE_M, PASS_M, ARRIVE_DEG = 0.003, 0.012, 3.0
HEAD_MARGIN = 0.010         # m, jaw opening beyond the head width


def _fen_of(instruction: str) -> str:
    key = "Position (FEN): "
    i = instruction.rfind(key)
    if i < 0:
        return chess.STARTING_FEN
    return instruction[i + len(key):].split(". Side to move")[0].strip()


def scripted_moves(p: Mapping[str, Any], seed: int) -> list[str]:
    games = p.get("scripted_games") or [{"name": "game", "moves": p.get("scripted_game", [])}]
    return list(games[seed % len(games)]["moves"])


class _Follower:
    """Rows of fingertip waypoints [arm, tip (m), yaw, tilt, gripper (0, 1 or 'w<mm>'), hold steps, tolerance]."""

    def __init__(self, controller, waypoints: list[list], measure=None, locate=None):
        self.c, self.wps = controller, waypoints
        #: xy (robot frame) -> where the base of the piece standing nearest to it is
        self.locate = locate
        #: arm -> xy offset (m, robot frame) of the held piece from the fingertip centre
        self.measure = measure
        self.stuck, self._last, self.given_up = 0, None, 0
        self.gave_up: list = []

    def _arrived(self) -> bool:
        arm, tip, yaw, tilt, _, _, tol = self.wps[0][:7]
        pos, cur = self.c.tip_pose(arm)
        ok = np.linalg.norm(tip - pos) < tol and abs((yaw - cur + 180) % 360 - 180) < ARRIVE_DEG
        if ok and self.c.max_tilt_deg:
            ok = abs(tilt - self.c.tilt(arm)) < ARRIVE_DEG
        return bool(ok)

    def done(self) -> bool:
        return not self.wps

    def row(self):
        from ....harness.ee_control import ArmTarget
        if self.wps and self.wps[0][7:] == ["pick"] and self.locate is not None:
            first = self.wps[0]
            if getattr(self, "_picked", None) is not first:
                self._picked = first
                at = self.locate(first[1][:2])
                shift = np.array([at[0] - first[1][0], at[1] - first[1][1], 0.0])
                for wp in self.wps:
                    if wp[7:] != ["pick"]:
                        break
                    wp[1] = wp[1] + shift
        while self.wps and self.wps[0][5] <= 0 and self._arrived():
            done = self.wps.pop(0)
            if done[7:8] == ["settle"] and self.measure is not None:
                residual = self.c.tip_pose(done[0])[0][:2] + self.measure(done[0]) - done[8]
                if float(np.linalg.norm(residual)) < 0.015:
                    for wp in self.wps:
                        if wp[7:] != ["place"]:
                            break
                        wp[1] = wp[1] - np.array([residual[0], residual[1], 0.0])
            if done[7:] == ["measure"] and self.measure is not None:
                offset = self.measure(done[0])
                seen = False
                for wp in self.wps:
                    if wp[7:] == ["place"]:
                        seen = True
                        wp[1] = wp[1] - np.array([offset[0], offset[1], 0.0])
                    elif seen:
                        break
        if not self.wps:
            return None
        arm, tip, yaw, tilt, g = self.wps[0][:5]
        if self._arrived():
            self.wps[0][5] -= 1
        else:
            self.stuck = self.stuck + 1 if self.wps[0] is self._last else 0
            self._last = self.wps[0]
            if self.stuck >= STUCK_STEPS:
                self.given_up += 1
                self.gave_up.append([arm, [round(float(v), 3) for v in tip], round(yaw), round(tilt)])
                self.wps.pop(0)
                self.stuck = 0
                return self.row()
        width = int(g[1:]) if isinstance(g, str) else None
        row = {a: None for a in self.c.arms}
        row[arm] = ArmTarget(tuple(int(v) for v in np.round(tip * 1000)), int(round(yaw)), 0 if width else int(g),
                             width, int(round(tilt)))
        return row


def _pick_place(L: Layout, p, colour: bool, arm: str, kind: str, src: tuple, dst: tuple, park) -> list[list]:
    """Waypoints of one pick and place by ``arm``: ``src``/``dst`` are (xy in the robot frame, support z, tilt)."""
    h, d, _ = L.dims[kind]
    yaw = float(p["grasp_yaw"][arm])
    from ....objects import staunton
    w = f"w{int(round((staunton.head_width(kind, d) + HEAD_MARGIN) * 1000))}"
    tallest = max(v[0] for v in L.dims.values())
    carry = L.board_top + tallest + CARRY_CLEAR + (h - GRASP_INSERTION)
    (sxy, sz, st), (dxy, dz, dt) = src, dst
    grasp = sz + h - GRASP_INSERTION
    release = dz + RELEASE_GAP + h - GRASP_INSERTION

    def column(xy, z0, z1, tilt, g):
        n = max(1, int(math.ceil(abs(z1 - z0) / FINE_STEP)))
        return [[arm, np.array([xy[0], xy[1], z0 + (z1 - z0) * (i + 1) / n]), yaw, tilt, g, 0, ARRIVE_M]
                for i in range(n)]

    def path(a, b, ta, tb, g, yaw_a=yaw, yaw_b=yaw):
        """Waypoints from ``a`` to ``b`` (excluding ``a``), the tilt and yaw changing in proportion."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / PATH_STEP)))
        return [[arm, a + (b - a) * (i + 1) / n, yaw_a + (yaw_b - yaw_a) * (i + 1) / n, ta + (tb - ta) * (i + 1) / n,
                 g, 0, PASS_M] for i in range(n - 1)]

    px, py, pz, pyaw = park
    start = np.array([px, py, pz])
    above_src, above_dst = np.array([sxy[0], sxy[1], carry]), np.array([dxy[0], dxy[1], carry])
    out = path(start, above_src, 0.0, st, w, pyaw, yaw)
    pick = [[arm, above_src, yaw, st, w, 0, ARRIVE_M]]
    pick += column(sxy, carry, grasp, st, w)
    pick += [[arm, np.array([sxy[0], sxy[1], grasp]), yaw, st, 1, 2, ARRIVE_M]]
    for wp in pick:
        wp.append("pick")                   # re-centred on the piece where it stands when the leg begins
    out += pick
    out += column(sxy, grasp, carry, st, 1)
    out[-1].append("measure")              # the piece's offset in the jaws is measured here
    out += path(above_src, above_dst, st, dt, 1)
    place = [[arm, above_dst, yaw, dt, 1, 0, ARRIVE_M]]
    place += column(dxy, carry, release, dt, 1)
    for wp in place:
        wp.append("place")                  # shifted by that offset so that the piece lands centred
    place[-1][5] = 1                        # held a step at the release height; the offset is measured again
    place[-1][7:] = ["settle", np.asarray(dxy, float)]
    place += [[arm, np.array([dxy[0], dxy[1], release]), yaw, dt, w, 1, ARRIVE_M, "place"]]
    out += place
    out += column(dxy, release, carry, dt, w)
    out += path(above_dst, start, dt, 0.0, 0, yaw, pyaw)
    out += [[arm, start, pyaw, 0.0, 0, 0, PASS_M]]
    return out


def scripted_turn(api, session) -> None:
    """One turn of the scripted player: names the next move of its game, plans the physical moves from
    ground truth and the reach table (a hand-over over an empty square both arms reach when no arm
    reaches both ends), drives them through ee rows and says ``done``. When the task lets the base drive
    (``ee.whole_body``) it drives ``params.reach_forward_m`` forward first, through the same command a
    model uses, and back to its station before ``done``."""
    from ....harness.ee_control import EEController, EELimits, WholeBody
    p = session.task.raw["params"]
    ee = session.task.raw.get("ee", {})
    controller = EEController(session, EELimits.from_mapping(ee.get("limits")))
    controller.max_tilt_deg = float((ee.get("tilt") or {}).get("max_deg", 0.0))
    controller.body = WholeBody.from_mapping(ee.get("whole_body"), ee.get("base_noise"))
    controller.move_to_start(ee.get("start", {}), ee.get("start_path", "joint") == "staged")
    session.ready()
    forward = float(p.get("reach_forward_m", 0.0)) if controller.body.base else 0.0
    board = chess.Board(_fen_of(session.episode.instruction))
    colour = board.turn
    ply = (board.fullmove_number - 1) * 2 + (0 if colour == chess.WHITE else 1)
    moves = scripted_moves(p, session.episode.seed)
    if ply >= len(moves):
        session.say("adjourn", "")
        return
    answer = session.say("move", moves[ply])
    if not answer.startswith("Accepted"):
        return
    move = board.parse_san(moves[ply])
    if forward:
        _drive(controller, forward)
    snap = session.oracle()
    scene_like = SceneSpec(robot="", fixtures=(Shape("table", "box", snap.objects["table"].size,
                                                      snap.objects["table"].pos),),
                           robots=tuple(_Placement(r, tuple(b)) for r, b in session.episode.values["stations"].items()))
    L = Layout.from_params(scene_like, p)
    me = snap.base
    names = [n for n in snap.objects if n[:2] in ("w_", "b_")]

    def piece_at(world_xy) -> str:
        return min(names, key=lambda n: np.linalg.norm(np.array(snap.objects[n].pos[:2]) - world_xy))

    def free_slot() -> str:
        for k in range(L.n_slots):
            xy = L.slot_xy(colour, k)
            if all(np.linalg.norm(np.array(snap.objects[n].pos[:2]) - xy) > 0.02 for n in names):
                return f"slot:{'w' if colour else 'b'}:{k}"
        raise RuntimeError("no free capture slot")

    def world(where: str) -> tuple[np.ndarray, float]:
        if where.startswith("slot:"):
            return L.slot_xy(colour, int(where.split(":")[2])), L.table_top
        return L.square_xy(chess.parse_square(where)), L.board_top

    subs: list[tuple[str, str, str]] = []           # (piece, src, dst)
    src, dst = chess.square_name(move.from_square), chess.square_name(move.to_square)
    if board.is_capture(move):
        vsq = move.to_square
        if board.is_en_passant(move):
            vsq = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
        v = chess.square_name(vsq)
        subs.append((piece_at(world(v)[0]), v, free_slot()))
    subs.append((piece_at(world(src)[0]), src, dst))
    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        r_from, r_to = (7, 5) if chess.square_file(move.to_square) == 6 else (0, 3)
        rs, rd = chess.square_name(chess.square(r_from, rank)), chess.square_name(chess.square(r_to, rank))
        subs.append((piece_at(world(rs)[0]), rs, rd))
    occupied = {chess.square_name(s) for s in board.piece_map()}
    park = {arm: tuple(ee["start"][arm]) for arm in controller.arms}
    waypoints: list[list] = []
    for piece, a, b in subs:
        kind = piece.split("_")[1]
        legs = _legs(p, colour, a, b, occupied, L)
        for arm, s, t in legs:
            (sxy, sz), (txy, tz) = world(s), world(t)
            waypoints += _pick_place(L, p, colour, arm, kind,
                                     (to_base(me, np.array([*sxy, 0]))[:2], sz, reach_tilt(p, colour, arm, s)),
                                     (to_base(me, np.array([*txy, 0]))[:2], tz, reach_tilt(p, colour, arm, t)),
                                     park[arm])
        occupied.discard(a)
        occupied.add(b)
    def measure(arm: str) -> np.ndarray:
        s2 = session.oracle()
        held = [n for n, a in s2.held_by.items() if a == arm]
        if not held:
            return np.zeros(2)
        return _bottom_xy(s2, held[0], me) - controller.tip_pose(arm)[0][:2]

    def locate(xy) -> np.ndarray:
        s2 = session.oracle()
        near = min((n for n in s2.objects if n[:2] in ("w_", "b_")),
                   key=lambda n: np.linalg.norm(_bottom_xy(s2, n, me) - xy))
        found = _bottom_xy(s2, near, me)
        return found if np.linalg.norm(found - xy) < 0.03 else np.asarray(xy, float)

    follower = _Follower(controller, waypoints, measure, locate)
    for _ in range(int(ee.get("scripted_steps", 600))):
        row = follower.row()
        if row is None:
            break
        controller.step(row)
    if forward:
        _drive(controller, -forward)
    session.advance(0.5)
    session.say("done", "")


def _drive(controller, metres: float) -> list[str]:
    """Drive the base through the robot API, the same request lines a model sends."""
    from ....subjects.ee_agent import EEAgent
    return EEAgent.body_commands(controller, f"base: drive {metres:.4f}")


def _bottom_xy(snap: OracleSnapshot, name: str, base) -> np.ndarray:
    """xy (robot frame) of the centre of a piece's base, where it will stand when it is put down upright."""
    o = snap.objects[name]
    w, x, y, z = o.quat
    axis = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    h = float(o.size[0])
    return to_base(base, np.array(o.pos) - axis * h / 2)[:2]


def _legs(p, colour: bool, a: str, b: str, occupied: set[str], L: Layout) -> list[tuple[str, str, str]]:
    """(arm, from, to) legs of one physical move: one arm when one reaches both ends (the smaller sum of
    tilts), else a hand-over over the empty square both arms reach nearest to the middle of the two."""
    best = None
    for arm in ("left", "right"):
        ta, tb = reach_tilt(p, colour, arm, a), reach_tilt(p, colour, arm, b)
        if ta is not None and tb is not None:
            cost = abs(ta) + abs(tb)
            if best is None or cost < best[0]:
                best = (cost, arm)
    if best is not None:
        return [(best[1], a, b)]
    first = next(arm for arm in ("left", "right") if reach_tilt(p, colour, arm, a) is not None)
    second = "right" if first == "left" else "left"
    mid = (np.asarray(L.square_xy(chess.parse_square(a))) + np.asarray(
        L.square_xy(chess.parse_square(b)) if not b.startswith("slot:") else L.slot_xy(colour, int(b.split(":")[2])))) / 2
    spots = [chess.square_name(s) for s in chess.SQUARES if chess.square_name(s) not in occupied
             and reach_tilt(p, colour, "left", chess.square_name(s)) is not None
             and reach_tilt(p, colour, "right", chess.square_name(s)) is not None]
    spot = min(spots, key=lambda s: np.linalg.norm(L.square_xy(chess.parse_square(s)) - mid))
    return [(first, a, spot), (second, spot, b)]

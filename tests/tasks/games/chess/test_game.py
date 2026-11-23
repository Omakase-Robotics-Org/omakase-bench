"""games/chess: Staunton pieces, tilted ee rows, the chess referee, whole-body commands and a scripted opening."""
import dataclasses
import json
import math

import numpy as np
import pytest
import yaml

from omakase_bench.backends.mujoco.backend import MujocoBackend  # noqa: E402
from omakase_bench.core.runner import RunConfig, run  # noqa: E402
from omakase_bench.harness.session import RunBudget  # noqa: E402
from omakase_bench.core.suite import find_task, load_suite
from omakase_bench.core.task import load_task, robot_for  # noqa: E402
from omakase_bench.core.team import GameView, TurnReport  # noqa: E402
from omakase_bench.objects import staunton  # noqa: E402
from omakase_bench.harness.ee_control import (ArmTarget, EEController, ParseError, format_row, lean_axis,  # noqa: E402
                                            parse_row)
from omakase_bench.harness.kinematics.model import Kinematics  # noqa: E402
from omakase_bench.robots.spec import load_robot  # noqa: E402
from omakase_bench.subjects import build_subject  # noqa: E402
from omakase_bench.tasks.games.chess import rules as chess  # noqa: E402
from omakase_bench.tasks.games.chess.layout import reach_tilt  # noqa: E402
from omakase_bench.tasks.games.chess.referee import ChessReferee  # noqa: E402

TASK = "games/chess"
ARMS = ("left", "right")


# --- pieces -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", staunton.KINDS)
def test_staunton_pieces_fit_their_size_with_collision_up_to_the_top(kind):
    h, d = 0.077, 0.031
    verts, faces = staunton.mesh(kind, h, d)
    assert faces.min() >= 0 and faces.max() < len(verts)
    assert verts[:, 2].min() == pytest.approx(-h / 2) and verts[:, 2].max() <= h / 2 + 1e-9
    assert np.hypot(verts[:, 0], verts[:, 1]).max() <= d / 2 * (1.05 if kind == "knight" else 1.0) + 1e-9
    prims = staunton.collision(kind, h, d)
    assert prims and all(p.pos[2] - (p.size[-1] if p.type != "sphere" else p.size[0]) >= -h / 2 - 1e-9 for p in prims)
    top = max(p.pos[2] + (p.size[0] if p.type == "sphere" else p.size[-1]) for p in prims)
    assert top == pytest.approx(h / 2, abs=0.002)
    # the jaws close on the head: its width is the documented fraction of the base
    assert staunton.head_width(kind, d) == pytest.approx(staunton.HEAD_WIDTH[kind] * d)


def test_pieces_stand_still_in_mujoco():
    ep = find_task(TASK).resolve(0)
    task = find_task(TASK)
    be = MujocoBackend()
    be.load_team({r.id: robot_for(task, ep.scene, r.id) for r in ep.scene.robots}, ep.scene)
    be.reset(ep.scene, seed=0)
    before = be.oracle()
    be.step(1.0)
    after = be.oracle()
    pieces = [n for n in after.objects if n[:2] in ("w_", "b_")]
    assert len(pieces) == 40           # 32 and a spare queen, rook, bishop and knight per side
    assert max(np.linalg.norm(np.subtract(after.objects[n].pos, before.objects[n].pos)) for n in pieces) < 1e-3


# --- tilted rows --------------------------------------------------------------------------------------------

def test_tilt_values_parse_only_when_the_task_allows_them():
    row = parse_row("L 520 -100 820 -45 t25 w27 | R hold", ARMS, widths=True, tilt=True)
    assert row["left"] == ArmTarget((520, -100, 820), -45, 0, 27, 25) and row["right"] is None
    assert format_row(row, ARMS) == "L 520 -100 820 -45 t25 w27 | R hold"
    assert parse_row("L 520 -100 820 -45 1", ARMS, tilt=True)["left"].tilt_deg == 0
    with pytest.raises(ParseError):
        parse_row("L 520 -100 820 -45 t25 1", ARMS)
    # without a tilt the canonical text is the one rows always had
    assert format_row(parse_row("L 1 2 3 4 1", ARMS, tilt=True), ARMS) == "L 1 2 3 4 1 | R hold"


def test_tilted_rotation_leans_away_from_the_robot_and_keeps_the_jaws_level():
    robot = load_robot("d1")
    kin = Kinematics(robot)

    class _C:
        spec, kin = robot, None
    _C.kin = kin
    for arm, yaw in (("left", -45), ("right", 45)):
        flat = EEController._rotation(_C, arm, yaw)
        assert np.allclose(EEController._rotation(_C, arm, yaw, 0.0), flat)
        rot = EEController._rotation(_C, arm, yaw, 25.0)
        hand = robot.hand_of(arm)
        approach = rot @ np.array(hand.approach_axis, float)
        closing = rot @ kin._closing_local[hand.name]
        assert math.degrees(math.acos(-approach[2])) == pytest.approx(25.0, abs=1e-6)
        assert abs(closing[2]) < 1e-9                           # the jaws stay level
        assert np.dot(approach, lean_axis(closing)) > 0 and approach[0] > 0


# --- referee --------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def game():
    task = find_task(TASK)
    ep = task.resolve(0)
    be = MujocoBackend()
    be.load_team({r.id: robot_for(task, ep.scene, r.id) for r in ep.scene.robots}, ep.scene)
    be.reset(ep.scene, seed=0)
    return task, ep, be


def _referee(game):
    task, ep, be = game
    ref = ChessReferee(ep.values, task.raw["params"])
    view = GameView(task.team.ids, ep.values, task.raw["params"])
    ref.start(view, be.oracle())
    return ref, view


def _play(ref, view, be, robot, san, turn):
    """Name ``san``, carry out its physical moves by setting poses, and end the turn."""
    snap = be.oracle()
    ans = ref.hear(view, robot, "move", san, snap)
    assert ans.startswith("Accepted"), ans
    for sub in ref.turn.plan:
        pos = ref.location_xyz(snap, sub.piece, sub.dst)
        be.set_object_pose(sub.piece, pos, snap.objects[sub.piece].quat)
    after = be.oracle()
    assert ref.turn_over(view, robot, after) is False or ref.turn.finished
    ruling = ref.after_turn(view, TurnReport(turn, robot, "turn_end:referee", snap, after, 0.0, 1.0))
    for name, (pos, quat) in ruling.restore.items():
        be.set_object_pose(name, pos, quat)
    return ruling


def test_illegal_moves_are_refused_with_the_reason_and_three_lose_the_game(game):
    task, ep, be = game
    ref, view = _referee(game)
    snap = be.oracle()
    assert ref.hear(view, "b", "move", "e5", snap) == "It is not your turn."
    assert ref.hear(view, "a", "move", "e5", snap).startswith("Illegal: e5 is not legal")
    assert ref.hear(view, "a", "move", "xyz", snap).startswith("Illegal: 'xyz' is not a move")
    ans = ref.hear(view, "a", "move", "Ke2", snap)
    assert "you lose the game" in ans and ref.turn_over(view, "a", snap)
    ruling = ref.after_turn(view, TurnReport(1, "a", "turn_end:referee", snap, snap, 0.0, 1.0))
    assert ruling.next_robot is None and ref.result_code == "0-1"
    assert ref.metrics["a"]["illegal_attempts"] == 3


def test_captures_castling_en_passant_and_promotion_follow_the_physical_moves(game):
    task, ep, be = game
    be.reset(ep.scene, seed=0)
    ref, view = _referee(game)
    line = "e4 Nf6 e5 d5 exd6 Bf5 dxc7 e6 cxb8=Q Rxb8 Nf3 Be7 Be2 O-O O-O".split()
    for k, san in enumerate(line):
        robot = "a" if k % 2 == 0 else "b"
        before = dict(ref.where)
        ruling = _play(ref, view, be, robot, san, k + 1)
        assert ruling.valid, ruling.note
        if san == "exd6":                  # en passant: the pawn on d5 goes to White's first capture slot
            assert ref.where[[n for n, w in before.items() if w == "d5"][0]] == "slot:w:0"
        if san == "cxb8=Q":                # the knight on b8 to a slot, the pawn exchanged for the spare queen
            assert ref.where["w_queen_spare"] == "b8" and ref.where["w_pawn_e2"].startswith("reserve:")
            assert set(ruling.restore) == {"w_queen_spare", "w_pawn_e2"}
    assert ref.where["w_king_e1"] == "g1" and ref.where["w_rook_h1"] == "f1"
    assert ref.where["b_king_e8"] == "g8" and ref.where["b_rook_h8"] == "f8"
    assert ref.board.fen() == "1r1q1rk1/pp2bppp/4pn2/5b2/8/5N2/PPPPBPPP/RNBQ1RK1 b - - 5 8"
    assert ref.material() == {chess.WHITE: 38, chess.BLACK: 34}


def test_a_displaced_piece_is_a_fault_and_is_restored(game):
    task, ep, be = game
    be.reset(ep.scene, seed=0)
    ref, view = _referee(game)
    snap = be.oracle()
    ref.hear(view, "a", "move", "e4", snap)
    be.set_object_pose("w_pawn_e2", ref.location_xyz(snap, "w_pawn_e2", "e4"), (1, 0, 0, 0))
    knocked = (math.cos(0.8), math.sin(0.8), 0.0, 0.0)
    be.set_object_pose("w_pawn_d2", np.add(snap.objects["w_pawn_d2"].pos, (0, 0, 0.002)), knocked)
    after = be.oracle()
    assert "knocked over" in ref.hear(view, "a", "done", "", after)
    ruling = ref.after_turn(view, TurnReport(1, "a", "turn_end:referee", snap, after, 0.0, 1.0))
    assert not ruling.valid and set(ruling.restore) == {"w_pawn_d2"}
    assert ref.metrics["a"]["manipulation_faults"] == 1 and ref.metrics["a"]["knocked_over"] == 1
    pos, quat = ruling.restore["w_pawn_d2"]
    assert np.allclose(pos, ref.location_xyz(snap, "w_pawn_d2", "d2"), atol=1e-6) and quat[1:3] == (0.0, 0.0)
    assert ref.san == ["e4"] and ruling.next_robot == "b"


def _turn_with_a_fallen_pawn(ref, view, be, robot, san, turn, pawn, fix=False):
    """Name ``san`` and carry it out, but leave ``pawn`` knocked over; with ``fix`` the robot stands it up
    again after the referee has listed it. Returns the ruling, with what the referee restored put back."""
    snap = be.oracle()
    assert ref.hear(view, robot, "move", san, snap).startswith("Accepted")
    for sub in ref.turn.plan:
        be.set_object_pose(sub.piece, ref.location_xyz(snap, sub.piece, sub.dst), snap.objects[sub.piece].quat)
    upright = (snap.objects[pawn].pos, snap.objects[pawn].quat)
    be.set_object_pose(pawn, np.add(upright[0], (0, 0, 0.002)), (math.cos(0.8), math.sin(0.8), 0.0, 0.0))
    assert "knocked over" in ref.hear(view, robot, "done", "", be.oracle())
    if fix:
        be.set_object_pose(pawn, *upright)
        assert ref.hear(view, robot, "done", "", be.oracle()) == "Position confirmed."
    ruling = ref.after_turn(view, TurnReport(turn, robot, "turn_end:referee", snap, be.oracle(), 0.0, 1.0))
    for name, (pos, quat) in ruling.restore.items():
        be.set_object_pose(name, pos, quat)
    return ruling


def test_what_the_mover_puts_right_itself_costs_nothing_and_a_restore_costs_a_minute(game):
    task, ep, be = game
    p = task.raw["params"]
    assert (p["fault_clock_penalty_s"], p["fault_limit"], p["correction_replies"]) == (60, None, 4)
    rules = " ".join(task.source.read_text().split())
    assert "what you put right yourself in those replies costs nothing" in rules
    assert "Each manipulation fault takes 1 minute from your clock. Your clock runs" in rules and "third manipulation fault" not in rules
    be.reset(ep.scene, seed=0)
    ref, view = _referee(game)
    ruling = _turn_with_a_fallen_pawn(ref, view, be, "a", "e4", 1, "w_pawn_d2", fix=True)
    assert ruling.valid and ruling.restore == {} and ref.clock["a"] == 3630.0
    assert ref.metrics["a"]["corrections"] == 1 and ref.metrics["a"]["manipulation_faults"] == 0
    assert ref.metrics["a"]["fault_penalty_s"] == 0.0 and ref.faults["a"] == 0
    # Black leaves a pawn lying: the referee restores it, and that costs a minute
    ruling = _turn_with_a_fallen_pawn(ref, view, be, "b", "e5", 2, "b_pawn_d7")
    assert not ruling.valid and set(ruling.restore) == {"b_pawn_d7"} and ruling.next_robot == "a"
    assert ref.clock["b"] == 3600.0 + 30.0 - 60.0 and ref.metrics["b"]["fault_penalty_s"] == 60.0
    assert ref.move_log[-1]["clock_s"] == 3570.0 and ref.result_code == "*"
    told = ruling.messages["a"]                 # what the other side reads
    assert "manipulation fault, fault 1, 1:00 taken from black's clock, 59:30 left: the referee restored" in told
    ruling = _play(ref, view, be, "a", "Nf3", 3)
    mine = ruling.messages["b"]                 # and what the mover reads at its next turn
    assert "In your last turn the referee had to restore the position: manipulation fault 1, 1:00 taken from your clock." in mine
    assert "Clocks: you 59:30" in mine
    result = ref.result(view, be.oracle())
    assert result.robots["b"]["manipulation_faults"] == 1 and result.robots["b"]["fault_limit"] is None
    assert result.robots["b"]["faults_in_game"] == 1 and result.robots["a"]["faults_in_game"] == 0


def test_no_number_of_faults_ends_a_chess_game_and_a_task_may_set_a_limit(game):
    task, ep, be = game
    # chess as published: faults are counted and cost clock time, the game goes on
    be.reset(ep.scene, seed=0)
    ref, view = _referee(game)
    for turn, (robot, san) in enumerate((("a", "e4"), ("b", "e5"), ("a", "Nf3"), ("b", "Nc6"), ("a", "Bc4"), ("b", "Bc5"), ("a", "d3")), 1):
        ruling = _turn_with_a_fallen_pawn(ref, view, be, "a", san, turn, "w_pawn_a2") if robot == "a" else _play(ref, view, be, "b", san, turn)
        assert ruling.next_robot is not None and ref.result_code == "*"
    assert ref.faults["a"] == 4 and ref.metrics["a"]["fault_penalty_s"] == 240.0 and "fault 4, 1:00 taken" in ruling.messages["b"]
    # with a limit, a side's fault of that number loses the game and the one before it does not
    be.reset(ep.scene, seed=0)
    params = {**task.raw["params"], "fault_limit": 3, "fault_clock_penalty_s": 120}
    ref = ChessReferee(ep.values, params)
    view = GameView(task.team.ids, ep.values, params)
    ref.start(view, be.oracle())
    assert _turn_with_a_fallen_pawn(ref, view, be, "a", "e4", 1, "w_pawn_d2").next_robot == "b"
    _play(ref, view, be, "b", "e5", 2)
    second = _turn_with_a_fallen_pawn(ref, view, be, "a", "Nf3", 3, "w_pawn_d2")
    assert second.next_robot == "b" and ref.result_code == "*" and ref.faults["a"] == 2
    assert "fault 2 of 3" in second.messages["b"] and ref.clock["a"] == 3600.0 + 2 * 30.0 - 2 * 120.0
    _play(ref, view, be, "b", "Nc6", 4)
    third = _turn_with_a_fallen_pawn(ref, view, be, "a", "Bc4", 5, "w_pawn_d2")
    assert third.next_robot is None and ref.result_code == "0-1"
    assert ref.ending == "loss by manipulation faults (white, fault 3 of 3)" and "game over: 0-1" in third.note
    assert ref.san[-1] == "Bc4" and ref.metrics["a"]["manipulation_faults"] == 3


def test_a_fault_penalty_that_empties_the_clock_is_a_flag_fall_and_faults_carry_over_a_resume(game):
    task, ep, be = game
    be.reset(ep.scene, seed=0)
    params = {**task.raw["params"], "start_clocks": {"a": 20.0}}
    ref = ChessReferee(ep.values, params)
    view = GameView(task.team.ids, ep.values, params)
    ref.start(view, be.oracle())
    ruling = _turn_with_a_fallen_pawn(ref, view, be, "a", "e4", 1, "w_pawn_d2")       # 20 + 30 - 60 s
    assert ruling.next_robot is None and ref.result_code == "0-1" and ref.clock["a"] == 0.0
    assert ref.ending == "white lost on time after a manipulation fault penalty"
    # a resumed game starts with the faults it had: one more is the third
    be.reset(ep.scene, seed=0)
    params = {**task.raw["params"], "fault_limit": 3, "start_faults": {"a": 2, "b": 0}}
    ref = ChessReferee(ep.values, params)
    view = GameView(task.team.ids, ep.values, params)
    ref.start(view, be.oracle())
    assert _turn_with_a_fallen_pawn(ref, view, be, "a", "e4", 1, "w_pawn_d2").next_robot is None
    assert ref.ending.startswith("loss by manipulation faults") and ref.metrics["a"]["manipulation_faults"] == 1
    from omakase_bench.tasks.games.chess.resume import resume_task
    stopped = {"task": TASK, "outcome": "*", "robots": {
        "a": {"metrics": {"pgn_moves": "1. e4 e5", "clock_left_s": 3000.0, "faults_in_game": 2}},
        "b": {"metrics": {"pgn_moves": "1. e4 e5", "clock_left_s": 3100.0, "faults_in_game": 1}}}}
    assert resume_task(stopped)["params"]["start_faults"] == {"a": 2, "b": 1}


def test_the_move_backstop_stops_a_game_without_a_result_and_adjudication_is_opt_in(game):
    task, ep, be = game
    be.reset(ep.scene, seed=0)
    ref, view = _referee(game)
    ref.max_moves = 1
    _play(ref, view, be, "a", "e4", 1)
    ruling = _play(ref, view, be, "b", "d5", 2)
    assert ruling.next_robot is None and ruling.stop == "budget_exceeded:max_moves" and ref.result_code == "*"
    assert "stopped by the harness after 1 moves per side" in ref.ending
    res = ref.result(view, be.oracle())
    assert not res.success and res.outcome == "*" and "no result" in res.evidence
    assert ref.progress()["moves"] == ["e4", "d5"] and set(ref.progress()["clocks"]) == {"a", "b"}
    be.reset(ep.scene, seed=0)
    old = ChessReferee(ep.values, {**task.raw["params"], "max_moves": 1, "max_moves_action": "adjudicate"})
    old.start(view, be.oracle())
    _play(old, view, be, "a", "e4", 1)
    ruling = _play(old, view, be, "b", "d5", 2)
    assert ruling.next_robot is None and not ruling.stop and old.result_code == "1/2-1/2"
    assert "adjudicated by material" in old.ending


def test_reach_table_covers_every_square_and_slot_for_both_colours():
    p = find_task(TASK).raw["params"]
    for colour in (chess.WHITE, chess.BLACK):
        for sq in chess.SQUARE_NAMES:
            assert any(reach_tilt(p, colour, arm, sq) is not None for arm in ARMS), sq
        assert all(reach_tilt(p, colour, "right", f"slot:x:{k}") is not None for k in range(16))
    # a1 and h8 belong to one arm each: the table is in the mover's own frame
    assert reach_tilt(p, chess.WHITE, "right", "a1") is None and reach_tilt(p, chess.BLACK, "left", "a1") is None


def test_layout_text_states_no_piece_poses():
    ep = find_task(TASK).resolve(0)
    for rid in ("a", "b"):
        text = ep.values["robots"][rid]["layout"]
        assert "a1 at x" in text and "capture slots" in text and "only your left arm reaches" in text
        assert "w_pawn" not in text and "b_pawn" not in text


# --- a scripted opening through the real harness ------------------------------------------------------------------

def test_scripted_players_play_an_en_passant_opening_without_faults(tmp_path):
    raw = dict(find_task(TASK).raw)
    raw["schedule"] = {**raw["schedule"], "max_turns": 5}
    raw["params"] = {**raw["params"], "scripted_games": [{"name": "ep", "moves": ["e4", "Nf6", "e5", "d5", "exd6"]}]}
    path = tmp_path / "chess.yaml"
    path.write_text(yaml.safe_dump(raw))
    task = load_task(path)
    cfg = RunConfig(out_dir=tmp_path / "run", video=False, run_budget=RunBudget(None, None))
    ep = run([build_subject("scripted/oracle")], [task], [0], cfg, progress=lambda s: None)["episodes"][0]
    assert ep["faults"] == [], ep["faults"]
    assert [t["valid"] for t in ep["turns"]] == [True] * 5, ep["turns"]
    final = ep["final_state"]["objects"]
    assert final["b_pawn_d7"]["pos"][1] < -0.2                         # taken en passant, in White's slot
    assert final["b_pawn_e7"]["pos"][0] == pytest.approx(0.5405, abs=0.003)   # untouched
    j5 = max(ep["robots"][r]["joint_travel"][f"{a}_arm_joint5"]["largest_swing_deg"] for r in ("a", "b") for a in ARMS)
    assert j5 < 90
    # the scripted players drove 60 mm forward for every move and back to their stations before 'done'
    trace = [json.loads(line) for line in (tmp_path / "run" / ep["artifacts"]["trace"]).read_text().splitlines()]
    drives = [e["arguments"]["forward_m"] for e in trace if e["kind"] == "api" and e.get("name") == "base_move"]
    assert drives == [0.06, -0.06] * 5
    for rid, station in (("a", 0.0), ("b", 0.856)):
        assert ep["final_state"]["bases"][rid][0] == pytest.approx(station, abs=0.01)
        assert abs(ep["final_state"]["bases"][rid][1]) < 0.008          # drift is noise, not a bias per turn
    # the clock starts at the working pose: the harness's move there from the idle pose is not charged
    first, turn = ep["robots"]["a"]["metrics"]["move_times"][0], ep["turns"][0]
    assert 5 < first["exec_s"] < turn["sim_end"] - turn["sim_start"] - 3.0


def test_the_games_suite_holds_the_chess_task():
    assert load_suite("games@1").tasks[0].id == TASK


def test_fischer_clock_counts_model_time_and_execution_and_flags(game):
    from omakase_bench.harness.session import Usage
    task, ep, be = game
    be.reset(ep.scene, seed=0)
    ref, view = _referee(game)
    usage = {"a": Usage(), "b": Usage()}
    now = {"t": 100.0}
    view.live.update(sim_time=lambda: now["t"], usage=usage, turn_start={"a": (100.0, 0.0)})
    usage["a"].model_seconds = 40.0               # the model thought for 40 s of wall-clock time
    now["t"] = 125.0                              # and the arms moved for 25 s of simulated time
    ruling = _play(ref, view, be, "a", "e4", 1)
    assert ref.move_log[-1] == {"robot": "a", "san": "e4", "think_s": 40.0, "exec_s": 25.0, "clock_s": 3600 - 65 + 30}
    assert "Clocks: you 60:00, the other robot 59:25 (+30 s per completed move)" in ruling.messages["b"]
    # Black thinks past its whole clock: the flag falls at the next poll and Black loses on time
    view.live["turn_start"]["b"] = (125.0, 0.0)
    usage["b"].model_seconds = 3601.0
    snap = be.oracle()
    assert ref.hear(view, "b", "move", "e5", snap) == "Your clock has run out."
    assert ref.turn_over(view, "b", snap)
    ruling = ref.after_turn(view, TurnReport(2, "b", "turn_end:referee", snap, snap, 125.0, 126.0))
    assert ruling.next_robot is None and ref.result_code == "1-0" and ref.ending == "black lost on time"


def test_a_flag_that_falls_in_the_last_model_call_of_a_turn_is_not_missed(game):
    """The turn ends without another poll (a third unusable reply): the time is charged when the turn is ruled."""
    from omakase_bench.harness.session import Usage
    task, ep, be = game
    be.reset(ep.scene, seed=0)
    ref, view = _referee(game)
    usage = {"a": Usage(), "b": Usage()}
    view.live.update(sim_time=lambda: 100.0, usage=usage, turn_start={"a": (100.0, 0.0)})
    snap = be.oracle()
    assert ref.hear(view, "a", "move", "e4", snap).startswith("Accepted")
    usage["a"].model_seconds = 3700.0
    ruling = ref.after_turn(view, TurnReport(1, "a", "parse_failures", snap, be.oracle(), 100.0, 101.0))
    assert ruling.next_robot is None and ref.result_code == "0-1" and ref.ending == "white lost on time"
    assert ref.clock["a"] == 0.0
    # a turn the harness cut short stops the game unfinished with the clocks as they stood, as before
    ref, view = _referee(game)
    view.live.update(sim_time=lambda: 100.0, usage=usage, turn_start={"a": (100.0, 0.0)})
    ruling = ref.after_turn(view, TurnReport(1, "a", "budget_exceeded:run_usd", snap, be.oracle(), 100.0, 101.0))
    assert ref.result_code == "*" and ref.clock["a"] == 3600.0


def test_moves_in_uci_are_accepted_for_every_piece_and_for_castling(game):
    task, ep, be = game
    ref, _ = _referee(game)
    assert [ref.parse(t).uci() for t in ("g1f3", "b1c3", "e2e4", "Nf3", "e4")] == ["g1f3", "b1c3", "e2e4", "g1f3", "e2e4"]
    with pytest.raises(ValueError, match="g1g3 is not legal in this position"):
        ref.parse("g1g3")
    castle = ChessReferee(ep.values, {**task.raw["params"], "start_moves": ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"]})
    assert castle.parse("e1g1") == castle.parse("O-O") and castle.parse("c4e2").uci() == "c4e2"
    promote = ChessReferee(ep.values, {**task.raw["params"], "start_moves": ["a4", "b5", "axb5", "a6", "bxa6", "Bb7", "axb7", "Nc6"]})
    assert promote.parse("b7a8q") == promote.parse("bxa8=Q")


def test_a_captured_piece_may_stand_in_any_free_capture_slot(game):
    task, ep, be = game
    be.reset(ep.scene, seed=0)
    ref, view = _referee(game)
    for k, san in enumerate("e4 d5".split()):
        _play(ref, view, be, "ab"[k % 2], san, k + 1)
    snap = be.oracle()
    assert ref.hear(view, "a", "move", "exd5", snap) == "Accepted: exd5."
    L = ref._layout(snap)
    be.set_object_pose("b_pawn_d7", (*L.slot_xy(chess.WHITE, 11), L.table_top + 0.019), (1, 0, 0, 0))
    be.set_object_pose("w_pawn_e2", ref.location_xyz(snap, "w_pawn_e2", "d5"), (1, 0, 0, 0))
    after = be.oracle()
    assert not ref.mismatches(after, ref.where_after(ref.turn.plan))
    ruling = ref.after_turn(view, TurnReport(3, "a", "turn_end:referee", snap, after, 0.0, 1.0))
    assert ruling.valid and ref.where["b_pawn_d7"] == "slot:w:11"


def _robot_session(task, ep, be, rid):
    from omakase_bench.harness.session import Recorder
    from omakase_bench.core.task import Episode
    from omakase_bench.harness.team_session import RobotSession, RobotView, World, robot_task
    world = World(be, task.team, Recorder(enabled=False), 1e9)
    slot = task.team.slot(rid)
    view = RobotView(world, rid, be_robots(task, ep)[rid], slot.cameras)
    e = Episode(robot_task(task, slot), 0, ep.scene, {**ep.values, "robot": rid}, ep.instruction)
    return RobotSession(world, rid, e, view, be_robots(task, ep)[rid], run_budget=RunBudget(None, None),
                        time_mode="lockstep", on_action=lambda r: None)


def be_robots(task, ep):
    return {r.id: robot_for(task, ep.scene, r.id) for r in ep.scene.robots}


def test_whole_body_commands_are_limited_checked_and_noisy(game):
    from omakase_bench.harness.ee_control import WholeBody
    from omakase_bench.subjects.ee_agent import EEAgent, EEConfig
    task, ep, be = game
    be.reset(ep.scene, seed=0)
    s = _robot_session(task, ep, be, "a")
    ctl = EEController(s)
    cfg = EEConfig.from_task(task.raw, s.robot)
    assert cfg.body.base and cfg.body.joints and cfg.body.lateral_per_m == 0.02
    ctl.body = cfg.body
    assert "at most 0.1 m" in ctl.base_move(0.12)
    assert "not both" in ctl.base_move(0.05, 5)
    assert "table" in ctl.base_move(0.10)                     # the lift pole would reach the table edge
    out = EEAgent.body_commands(ctl, "base: drive 0.06")
    assert out[:2] == ["POST /v1/chassis/cmd_vel -> ok", "POST /v1/chassis/cmd_vel -> ok"], out
    x, y, yaw = be.base_pose("a/")
    assert x == pytest.approx(0.06, abs=0.003) and abs(y) < 0.01 and ctl.odometry[0] == pytest.approx(0.06)
    out = EEAgent.body_commands(ctl, "lift: 0.13\nlift: 0.5\nhead: -0.3 0.2\nhead: 2 0\nbase: vel 0 0.9 0.2")
    assert out[0] == "POST /v1/slider/set_height -> ok"
    assert out[1].startswith("POST /v1/slider/set_height -> error 400 invalid: height 0.500 m is outside the travel")
    assert out[2] == "POST /v1/neck/cmd -> ok" and "clamped to pitch" in out[3]
    assert "clamped to vx 0 m/s, omega 0.6 rad/s" in out[4]
    q = be.joint_positions()
    out = EEAgent.body_commands(ctl, "joint: L 6 120\njoints: L 0 -90 0 0 0 0 0 vel=0.5\ngripper: R 0.25\n"
                                     "gripper: R 1\ngripper: R 1.2\njoints: L 1 2 3")
    assert out[0].startswith("POST /v1/arm/a/move_joint -> error 409 refused") and "outside its travel" in out[0]
    assert out[1] == "POST /v1/arm/a/mode -> ok" and out[2].startswith("POST /v1/arm/a/move_joints -> error 409 refused: at sample")
    assert out[3] == "POST /v1/end_effectors/b/set -> ok"
    assert out[4] == "POST /v1/end_effectors/b/set -> ok (opening clamped to 40 mm, a limit of this task)"
    assert out[5].startswith("POST /v1/end_effectors/b/set -> not sent") or "422 unprocessable" in out[5]
    assert out[6].startswith("joints: unusable line")
    assert be.joint_positions()["a/left_arm_joint1"] == pytest.approx(q["a/left_arm_joint1"], abs=0.01)
    ok = EEAgent.body_commands(ctl, "joint: L 3 -120 vel=1")
    assert ok == ["POST /v1/arm/a/mode -> ok", "POST /v1/arm/a/move_joint -> ok"]
    assert math.degrees(be.joint_positions()["a/left_arm_joint4"]) == pytest.approx(-120, abs=1)
    assert not WholeBody().any
    text = EEAgent._body_text(s, cfg)
    assert "never moves sideways" in text and "20 mm" in text and "POST /v1/arm/<side>/move_joints" in text


def test_without_whole_body_the_ee_prompt_is_unchanged():
    from omakase_bench.subjects.ee_agent import EEConfig
    from omakase_bench.robots.spec import load_robot
    cfg = EEConfig.from_task(find_task("manip/chess_move").raw, load_robot("d1"))
    assert not cfg.body.any and not cfg.say and cfg.tilt_max_deg == 0 and cfg.row_steps == 1


def test_a_game_resumes_from_recorded_moves():
    raw = dict(find_task(TASK).raw)
    params = {**raw["params"], "start_moves": ["e4", "d5", "exd5", "Qxd5", "Nc3"], "start_clocks": {"a": 3500, "b": 3400}}
    from omakase_bench.tasks.games.chess.scene import _start_position
    where = _start_position(params)
    assert where["w_pawn_e2"] == "d5" or where["b_queen_d8"] == "d5"
    assert where["b_pawn_d7"] == "slot:w:0" and where["w_pawn_e2"] == "slot:b:0" and where["w_knight_b1"] == "c3"
    ref = ChessReferee({"pieces": {n: {"colour": "white" if n[0] == "w" else "black", "kind": n.split("_")[1], "at": w}
                                   for n, w in where.items()}}, params)
    assert ref.board.fen().startswith("rnb1kbnr/ppp1pppp/8/3q4/8/2N5/PPPP1PPP/R1BQKBNR b KQkq")
    assert ref.san == ["e4", "d5", "exd5", "Qxd5", "Nc3"] and ref.clock == {"a": 3500.0, "b": 3400.0}


def test_base_noise_is_drawn_afresh_every_turn(game):
    task, ep, be = game
    be.reset(ep.scene, seed=0)
    s = _robot_session(task, ep, be, "a")
    draws = []
    for turn in (1, 2, 2):
        s.turn = turn
        draws.append(float(EEController(s)._noise.normal()))
    assert draws[0] != draws[1] and draws[1] == draws[2]


def test_base_lines_obey_the_command_limits_whatever_their_form(game):
    from omakase_bench.subjects.ee_agent import EEAgent, EEConfig
    task, ep, be = game
    be.reset(ep.scene, seed=0)
    s = _robot_session(task, ep, be, "a")
    ctl = EEController(s)
    ctl.body = EEConfig.from_task(task.raw, s.robot).body
    for line in ("base: drive -0.12", "base: vel -0.3 0 0.5", "base: turn 0.4", "base: vel 0 0.6 0.6"):
        out = EEAgent.body_commands(ctl, line)
        assert out[-1].startswith("base: refused, one command drives at most 0.1 m or turns at most 0.26 rad"), (line, out)
    assert be.base_pose("a/")[0] == pytest.approx(0.0, abs=1e-4) and not ctl.odometry.any()
    # a move within the limits is checked along its whole path, not only at its end
    out = EEAgent.body_commands(ctl, "base: drive 0.10")
    assert "refused at sample" in out[-1] and "table" in out[-1], out


def test_referee_lines_need_a_colon_or_stand_alone():
    from omakase_bench.subjects.ee_agent import EEAgent
    said = lambda text: EEAgent.said(text, ("move", "done"))
    assert said("move: Nf3\na: L hold | R hold") == [("move", "Nf3")]
    assert said("  Move : e4 \nDONE\ndone.\ndone:") == [("move", "e4"), ("done", ""), ("done", ""), ("done", "")]
    assert said("Move the knight to f3 first.\nDone with the lookups, now the pawn.\nmoved: e4\nmove e4") == []


def test_the_chess_prompt_states_the_turn_rules_and_nothing_that_contradicts_the_base_commands(game):
    from omakase_bench.subjects.ee_agent import EEAgent, EEConfig
    task, ep, be = game
    be.reset(ep.scene, seed=0)
    from omakase_bench.harness.team_session import robot_instruction
    texts = {}
    for rid in ("a", "b"):
        s = _robot_session(task, ep, be, rid)
        values = {**ep.values, **ep.values["robots"][rid], "robot": rid}      # as run_team_episode builds them
        s.episode = dataclasses.replace(s.episode, values=values,
                                        instruction=robot_instruction(task, task.team.slot(rid), values))
        texts[rid] = EEAgent(None, "x").prefix(s, EEConfig.from_task(s.task.raw, s.robot))
    a = texts["a"]
    assert "stays where it is" not in a and "At the first step the torso lift is at" in a
    assert "A turn lasts at most 60 steps." in a and "An episode lasts" not in a
    assert "your base stands at its station (within 30 mm and 3 deg" in a and "at most 60 steps and 80 replies" in a
    assert "One line drives at most 0.1 m and turns at most 0.26 rad" in a and "force_percent}}" not in a
    p, turn = task.raw["params"], task.team.schedule.turn
    assert (p["station_tolerance_m"], p["station_tolerance_deg"], task.raw["ee"]["horizon"], turn.model_calls) == (0.03, 3, 60, 80)
    # both players read the same text but for their colour and their own frame
    la, lb = a.splitlines(), texts["b"].splitlines()
    assert len(la) == len(lb) and [i for i, (x, y) in enumerate(zip(la, lb)) if x != y] == [0, 3]
    assert la[0].replace("White", "#").replace("Black", "White").replace("#", "Black") == lb[0]
    assert "a1 at x 270 y 158, h8 at x 586 y -158" in la[3] and "a1 at x 586 y -158, h8 at x 270 y 158" in lb[3]


# --- a contact between the two robots decides the game ----------------------------------------------------------

def _touch(snap, a="a/left_pad0", b="b/torso_link"):
    return dataclasses.replace(snap, contacts=frozenset({*snap.contacts, (a, b)}),
                               contacts_ever=frozenset({*snap.contacts_ever, (a, b)}))


def test_the_mover_loses_when_it_touches_a_robot_in_its_waiting_position(game):
    task, ep, be = game
    be.reset(ep.scene, seed=0)
    ref, view = _referee(game)
    view.live["idle_error"] = lambda rid: (math.radians(1.0), 0.002)       # the waiting robot holds its idle pose
    snap = be.oracle()
    assert not ref.turn_over(view, "a", snap) and ref.metrics["a"]["contacts"] == 0
    hit = _touch(snap)
    assert ref.turn_over(view, "a", hit)
    assert ref.hear(view, "a", "move", "e4", hit).startswith("The game is over: white lost by contact")
    ruling = ref.after_turn(view, TurnReport(1, "a", "turn_end:referee", snap, hit, 0.0, 1.0))
    assert ruling.next_robot is None and not ruling.valid and ref.result_code == "0-1"
    assert "robot a, whose turn it was, touched robot b, which was in its waiting position" in ref.ending
    assert "a/left_pad0 against b/torso_link" in ref.ending
    res = ref.result(view, hit)
    assert res.success and res.outcome == "0-1"
    event = res.robots["a"]["events"][-1]
    assert event["kind"] == "contact" and event["hit"] == "b" and event["ruled_against"] == "a"
    assert event["parts"] == [["a/left_pad0", "b/torso_link"]] and res.robots["b"]["contacts"] == 1


@pytest.mark.parametrize("why, live, shift", [
    ("its base was 50 mm and 0 deg from its station", (0.0, 0.0), (0.05, 0.0, 0.0)),
    ("its base was 0 mm and 4 deg from its station", (0.0, 0.0), (0.0, 0.0, math.radians(4))),
    ("an arm joint was 12 deg from the idle pose", (math.radians(12), 0.0), (0.0, 0.0, 0.0)),
    ("its lift was 30 mm from the idle height", (0.0, 0.03), (0.0, 0.0, 0.0))])
def test_a_robot_touched_outside_its_waiting_position_loses(game, why, live, shift):
    task, ep, be = game
    be.reset(ep.scene, seed=0)
    ref, view = _referee(game)
    view.live["idle_error"] = lambda rid: live
    snap = be.oracle()
    x, y, yaw = snap.bases["b"]
    moved = dataclasses.replace(_touch(snap), bases={**snap.bases, "b": (x + shift[0], y + shift[1], yaw + shift[2])})
    assert ref.turn_over(view, "a", moved)
    ref.after_turn(view, TurnReport(1, "a", "turn_end:referee", snap, moved, 0.0, 1.0))
    assert ref.result_code == "1-0" and f"robot b was touched while not in its waiting position ({why})" in ref.ending
    assert ref.events[-1]["ruled_against"] == "b" and ref.events[-1]["mover"] == "a" and ref.events[-1]["hit"] == "b"


def test_a_contact_on_the_way_back_to_the_idle_pose_is_ruled_after_the_turn(game):
    task, ep, be = game
    be.reset(ep.scene, seed=0)
    ref, view = _referee(game)
    view.live["idle_error"] = lambda rid: (0.0, 0.0)
    _play(ref, view, be, "a", "e4", 1)
    snap = be.oracle()
    ref.hear(view, "b", "move", "e5", snap)
    hit = _touch(snap, "a/head_link", "b/right_arm_link6")
    ruling = ref.after_turn(view, TurnReport(2, "b", "turn_end:referee", snap, hit, 0.0, 1.0))
    assert ruling.next_robot is None and ref.result_code == "1-0" and "black lost by contact" in ref.ending
    assert ref.san == ["e4"]                                   # the interrupted move is not played


def test_the_contact_rule_is_stated_and_the_waiting_robots_hold_their_idle_pose(tmp_path):
    task = find_task(TASK)
    assert task.team.fault_policy("inter_robot_collision") == "report"
    assert "the robot whose turn it is loses, unless the waiting robot was not in its waiting position" in \
        " ".join(task.raw["ee"]["rules"].split())
    p = task.raw["params"]
    assert (p["idle_tolerance_deg"], p["idle_lift_tolerance_m"]) == (5, 0.01)
    # through the harness: after a scripted turn each, both robots are within the tolerances by a wide margin
    raw = dict(task.raw)
    raw["schedule"] = {**raw["schedule"], "max_turns": 2}
    raw["params"] = {**raw["params"], "scripted_games": [{"name": "two", "moves": ["e4", "e5"]}]}
    path = tmp_path / "chess.yaml"
    path.write_text(yaml.safe_dump(raw))
    seen = {}
    from omakase_bench.tasks.games.chess import referee as chess_task
    original = chess_task.ChessReferee.after_turn

    def spy(self, game, report):
        seen[report.robot] = [game.live["idle_error"](r) for r in ("a", "b")]
        return original(self, game, report)

    chess_task.ChessReferee.after_turn = spy
    try:
        cfg = RunConfig(out_dir=tmp_path / "run", video=False, run_budget=RunBudget(None, None))
        ep = run([build_subject("scripted/oracle")], [load_task(path)], [0], cfg, progress=lambda s: None)["episodes"][0]
    finally:
        chess_task.ChessReferee.after_turn = original
    assert ep["faults"] == [] and [r["metrics"]["contacts"] for r in ep["robots"].values()] == [0, 0]
    for errors in seen.values():
        assert all(arm < math.radians(1.0) and lift < 0.003 for arm, lift in errors), seen

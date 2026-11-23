"""games/chess: a game cut short by a cap or an error is a harness stop, never a result; resuming a game."""
import dataclasses

import pytest

from omakase_bench.cli import _run_cap_shortfalls
from omakase_bench.core.runner import RunConfig, run
from omakase_bench.harness.session import RunBudget
from omakase_bench.core.suite import find_task
from omakase_bench.core.team import GameView
from omakase_bench.models.base import Capabilities, Reply, Usage
from omakase_bench.subjects import build_subject
from omakase_bench.subjects.ee_agent import EEAgent
from omakase_bench.subjects.team import TeamSubject
from omakase_bench.tasks.games.chess.referee import ChessReferee
from omakase_bench.tasks.games.chess.scene import _start_position

TASK = "games/chess"
HOLD = "a: L hold | R hold"


class TextModel:
    """Replies with the given texts in order, then holds; ``error`` replies stand for a provider failure."""
    capabilities = Capabilities(images=True)
    name = "test/text"

    def __init__(self, replies):
        self.replies = list(replies)

    def respond(self, request) -> Reply:
        self.asked_for_thinking = request.thinking_summary
        text = self.replies.pop(0) if self.replies else HOLD
        if text == "error":
            return Reply(stop="error", error="retryable: HTTP 529: overloaded")
        return Reply(text=text, stop="end", usage=Usage(input_tokens=100, output_tokens=20, usd=0.001, seconds=1.5),
                     model=self.name, thinking="I will hold.")


def _game(tmp_path, replies, budget=RunBudget(None, None), task=None):
    team = TeamSubject({"a": EEAgent(TextModel(replies), "test/text"), "b": build_subject("scripted/oracle")})
    task = task or find_task(TASK)
    cfg = RunConfig(out_dir=tmp_path / "run", video=False, run_budget=budget)
    return run([], [task], [0], cfg, progress=lambda s: None, teams={task.id: team})


def test_the_run_wide_call_cap_ends_a_game_as_a_harness_stop_without_a_result(tmp_path):
    res = _game(tmp_path, ["move: e4\n" + HOLD], RunBudget(max_model_calls=3))
    ep = res["episodes"][0]
    assert ep["exit_reason"] == "budget_exceeded:run_model_calls" and res["stopped"] == ep["exit_reason"]
    assert ep["outcome"] == "*" and not ep["success"] and "no result" in ep["evidence"]
    assert ep["usage"]["model_calls"] == 3
    # the transcript of the chess task: per call the changing prompt text, the thinking summary, the reply
    # and the call's wall-clock seconds, time-stamped
    import json
    trace = [json.loads(line) for line in (tmp_path / "run" / ep["artifacts"]["trace"]).read_text().splitlines()]
    calls = [e for e in trace if e["kind"] == "model"]
    assert len(calls) == 3 and all(e["thinking"] == "I will hold." and e["seconds"] == 1.5 for e in calls)
    assert calls[0]["prompt"].startswith("### current episode") and calls[0]["text"].startswith("move: e4")
    assert "Reply with" in calls[0]["prompt"] or "action line" in calls[0]["prompt"]
    assert all(set(e) >= {"sim_time", "wall", "robot"} for e in calls)


def test_a_robots_own_budget_ends_the_game_as_a_harness_stop_not_a_forfeit(tmp_path):
    task = find_task(TASK)
    task = dataclasses.replace(task, budget=dataclasses.replace(task.budget, model_calls=2))
    ep = _game(tmp_path, [], task=task)["episodes"][0]
    assert ep["exit_reason"] == "budget_exceeded:model_calls" and ep["outcome"] == "*" and not ep["success"]
    assert "stopped by the harness in white's turn" in ep["evidence"]
    assert ep["turns"][-1]["valid"] is False and ep["robots"]["a"]["metrics"]["moves"] == 0
    assert ep["robots"]["a"]["metrics"]["clock_left_s"] == 3600.0        # as before the interrupted turn


def test_a_model_error_ends_the_game_as_a_harness_stop_not_a_forfeit(tmp_path):
    ep = _game(tmp_path, ["move: e4\n" + HOLD, "error"])["episodes"][0]
    assert ep["exit_reason"] == "error:model_error" and ep["outcome"] == "*"
    assert "0-1" not in ep["evidence"] and "stopped by the harness" in ep["evidence"]
    events = ep["robots"]["a"]["metrics"]["events"]
    assert events[-1]["kind"] == "harness_stop" and events[-1]["accepted"] == "e4"


def test_a_robot_told_what_differs_gets_the_stated_number_of_replies_to_correct_it(tmp_path):
    """The prompt promises 4 more replies. Each reply is several actions (a row is repeated for up to 8 steps),
    and the allowance is counted in replies, not in actions."""
    import json
    task = find_task(TASK)
    allowed = task.raw["params"]["correction_replies"]
    assert allowed == 4 and f"you have {allowed} more replies to fix it" in " ".join(task.source.read_text().split())
    # White names its move, says done without having moved a piece, and then only holds
    res = _game(tmp_path, ["move: e4\n" + HOLD, "done\n" + HOLD], RunBudget(max_model_calls=2 + allowed + 3))
    ep = res["episodes"][0]
    trace = [json.loads(line) for line in (tmp_path / "run" / ep["artifacts"]["trace"]).read_text().splitlines()]
    first = next(i for i, e in enumerate(trace) if e["kind"] == "turn_end")
    turn = trace[:first + 1]
    said = [e for e in turn if e["kind"] == "say" and e["keyword"] == "done"]
    assert len(said) == 1 and f"You have {allowed} more replies to correct this" in said[0]["answer"]
    calls = [e for e in turn if e["kind"] == "model"]
    assert len(calls) == 2 + allowed                              # the move, the early done, and 4 replies to correct
    assert turn[-1]["ended"] == "turn_end:referee"
    steps_after_done = [e for e in turn[turn.index(said[0]):] if e["kind"] == "api"]
    assert len(steps_after_done) > allowed                        # more actions than replies: actions are not what is counted
    first_turn = ep["turns"][0]
    assert first_turn["valid"] is False and ep["robots"]["a"]["metrics"]["corrections"] == 1
    assert ep["robots"]["a"]["metrics"]["manipulation_faults"] == 1 and ep["robots"]["a"]["metrics"]["restored_pieces"] >= 1


def test_run_caps_below_a_tasks_run_budget_are_reported_before_the_run():
    task = find_task(TASK)
    model = TeamSubject({"a": EEAgent(TextModel([]), "test/text"), "b": build_subject("scripted/oracle")})
    scripted = TeamSubject({r: build_subject("scripted/oracle") for r in "ab"})
    short = _run_cap_shortfalls([task], {task.id: model}, [], 400, 10.0)
    assert len(short) == 1 and "32160 model calls" in short[0]
    assert _run_cap_shortfalls([task], {task.id: model}, [], 32160, 10.0) == []
    costly = dataclasses.replace(task, raw={**task.raw, "run_budget": {"model_calls": 100, "usd": 30}})
    assert "--max-usd is 10" in _run_cap_shortfalls([costly], {task.id: model}, [], 400, 10.0)[0]
    assert _run_cap_shortfalls([task], {task.id: model}, [], None, None) == []
    assert _run_cap_shortfalls([task], {task.id: scripted}, [], 400, 10.0) == []
    assert task.raw["run_budget"] == {"model_calls": task.team.schedule.max_turns * task.team.schedule.turn.model_calls}


@pytest.mark.parametrize("moves, robot, note", [(["e4", "d5", "exd5"], "b", "Black to move"),
                                                (["e4", "d5"], "a", "White to move")])
def test_a_resumed_game_starts_with_the_side_to_move(moves, robot, note):
    params = {**find_task(TASK).raw["params"], "start_moves": moves}
    where = _start_position(params)
    pieces = {n: {"colour": "white" if n[0] == "w" else "black", "kind": n.split("_")[1], "at": w}
              for n, w in where.items()}
    ref = ChessReferee({"pieces": pieces}, params)
    ref._layout = lambda snap: None
    ruling = ref.start(GameView(("a", "b"), {}, params), None)
    assert ruling.next_robot == robot and ruling.note == note and list(ruling.messages) == [robot]


def test_a_stopped_game_resumes_from_its_run_directory(tmp_path):
    import json

    from omakase_bench.core.task import load_task
    from omakase_bench.tasks.games.chess import resume as tool
    # White plays 1. e4 through the scripted player, then Black's model fails
    team = TeamSubject({"a": build_subject("scripted/oracle"), "b": EEAgent(TextModel(["error"]), "test/text")})
    cfg = RunConfig(out_dir=tmp_path / "run", video=False, run_budget=RunBudget(None, None))
    ep = run([], [find_task(TASK)], [0], cfg, progress=lambda s: None, teams={TASK: team})["episodes"][0]
    assert ep["exit_reason"] == "error:model_error" and ep["outcome"] == "*"
    out = tmp_path / "resume.yaml"
    assert tool.main([str(tmp_path / "run"), "--out", str(out)]) == 0
    task = load_task(out)
    p = task.raw["params"]
    assert p["start_moves"] == ["e4"] and p["start_clocks"]["b"] == 3600.0 and 3600 < p["start_clocks"]["a"] < 3630
    values = task.resolve(0).values
    assert values["pieces"]["w_pawn_e2"]["at"] == "e4"
    ref = ChessReferee(values, p)
    ref._layout = lambda snap: None
    assert ref.start(GameView(("a", "b"), values, p), None).next_robot == "b"
    with pytest.raises(ValueError):
        tool.resume_task({**ep, "outcome": "1-0"})


def test_a_game_stopped_after_a_promotion_resumes_with_the_promoted_piece_on_the_board(tmp_path):
    """The scripted line with 5. cxb8=Q is cut after the promotion; the resumed game starts from the same
    physical position (the spare queen on b8, the pawn on the queen's reserve spot) and goes on from there."""
    from omakase_bench.backends.mujoco.backend import MujocoBackend
    from omakase_bench.core.task import load_task, robot_for
    from omakase_bench.tasks.games.chess import resume as tool
    task = find_task(TASK)
    line = task.raw["params"]["scripted_games"][1]["moves"]
    cut = line.index("cxb8=Q") + 1
    short = dataclasses.replace(task, team=dataclasses.replace(
        task.team, schedule=dataclasses.replace(task.team.schedule, max_turns=cut)))
    scripted = TeamSubject({r: build_subject("scripted/oracle") for r in "ab"})
    cfg = RunConfig(out_dir=tmp_path / "run", video=False, run_budget=RunBudget(None, None))
    ep = run([], [short], [1], cfg, progress=lambda s: None, teams={TASK: scripted})["episodes"][0]
    assert ep["outcome"] == "*" and len(ep["turns"]) == cut and "cxb8=Q" in ep["robots"]["a"]["metrics"]["pgn_moves"]
    out = tmp_path / "resume.yaml"
    assert tool.main([str(tmp_path / "run"), "--out", str(out)]) == 0
    resumed = load_task(out)
    assert resumed.raw["params"]["start_moves"] == line[:cut]
    episode = resumed.resolve(1)
    pieces = episode.values["pieces"]
    assert pieces["w_queen_spare"]["at"] == "b8" and pieces["w_pawn_e2"]["at"] == "reserve:queen"
    assert pieces["b_knight_b8"]["at"].startswith("slot:w:")
    # the board the resumed game starts from is the board the stopped game left
    backend = MujocoBackend()
    try:
        backend.load_team({r.id: robot_for(resumed, episode.scene, r.id) for r in episode.scene.robots}, episode.scene)
        backend.reset(episode.scene, seed=1)
        start = backend.oracle().objects
    finally:
        backend.close()
    left = ep["final_state"]["objects"]
    assert set(pieces) <= set(start) and set(pieces) <= set(left)
    off = {n: max(abs(start[n].pos[0] - left[n]["pos"][0]), abs(start[n].pos[1] - left[n]["pos"][1])) for n in pieces}
    assert max(off.values()) < 0.02, sorted(off.items(), key=lambda kv: -kv[1])[:3]
    ref = ChessReferee(episode.values, resumed.raw["params"])
    assert ref.board.fen().split()[0] == "rQ1qkb1r/pp3ppp/4pn2/5b2/8/8/PPPP1PPP/RNBQKBNR" and not ref.board.turn
    # and it goes on: Black takes the new queen, White develops, every turn valid
    nxt = dataclasses.replace(resumed, team=dataclasses.replace(
        resumed.team, schedule=dataclasses.replace(resumed.team.schedule, max_turns=2)))
    cfg = RunConfig(out_dir=tmp_path / "rest", video=False, run_budget=RunBudget(None, None))
    rest = run([], [nxt], [1], cfg, progress=lambda s: None, teams={TASK: scripted})["episodes"][0]
    assert [t["valid"] for t in rest["turns"]] == [True, True] and not rest["faults"]
    assert rest["robots"]["a"]["metrics"]["pgn_moves"].split()[-2:] == ["6.", "Nf3"]


def test_no_fault_ends_a_game_without_a_result():
    team = find_task(TASK).team
    assert team.fault_policy("inter_robot_collision") == "report" and team.fault_policy("transfer_collision") == "report"


def test_a_killed_run_keeps_its_trace_and_resumes_from_its_last_complete_turn(tmp_path):
    import json
    import os
    import signal
    import subprocess
    import sys
    import time

    import yaml

    from omakase_bench.core.task import load_task
    raw = dict(find_task(TASK).raw)
    raw["params"] = {**raw["params"], "scripted_games": [{"name": "line", "moves": ["e4", "d5", "exd5", "Qxd5", "Nc3", "Qa5"]}]}
    path = tmp_path / "chess.yaml"
    path.write_text(yaml.safe_dump(raw))
    out = tmp_path / "run"
    proc = subprocess.Popen([sys.executable, "-m", "omakase_bench.cli", "run", "--task", str(path), "--seeds", "0",
                             "--no-video", "--out", str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            env={**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")})
    try:
        deadline, moves = time.time() + 240, []
        while time.time() < deadline and len(moves) < 3:
            files = list(out.glob("episodes/*/*/*/progress.json"))
            if files:
                moves = json.loads(files[0].read_text())["game"]["moves"]
            time.sleep(0.5)
        assert proc.poll() is None, "the run ended before it could be killed"
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait()
    episode = next(out.glob("episodes/*/*/*"))
    assert not (episode / "result.json").exists() and not (out / "results.json").exists()
    state = json.loads((episode / "progress.json").read_text())
    moves = state["game"]["moves"]
    assert moves[:3] == ["e4", "d5", "exd5"] and len(state["turns"]) == len(moves) and state["seed"] == 0
    assert set(state["game"]["clocks"]) == {"a", "b"} and state["robots"]["a"]["usage"]["api_calls"] > 0
    # every complete turn is in the trace on disk, each line a whole entry
    trace = [json.loads(line) for line in (episode / "trace.jsonl").read_text().splitlines()[:-1]]
    assert sum(1 for e in trace if e["kind"] == "referee" and e.get("turn")) >= len(moves)
    from omakase_bench.tasks.games.chess import resume as tool
    resumed = tmp_path / "resume.yaml"
    assert tool.main([str(out), "--out", str(resumed)]) == 0
    task = load_task(resumed)
    p = task.raw["params"]
    assert p["start_moves"] == moves and p["start_clocks"] == state["game"]["clocks"]
    values = task.resolve(0).values
    ref = ChessReferee(values, p)
    ref._layout = lambda snap: None
    assert ref.start(GameView(("a", "b"), values, p), None).next_robot == ("a" if len(moves) % 2 == 0 else "b")
    assert values["pieces"]["b_pawn_d7"]["at"] == "slot:w:0"


def test_chess_is_played_to_its_end_and_the_backstop_never_decides_a_result():
    task = find_task(TASK)
    p = task.raw["params"]
    assert (p["max_moves"], p["max_moves_action"]) == (200, "stop") and task.team.schedule.max_turns > 2 * p["max_moves"]
    assert task.budget.model_calls >= task.team.schedule.max_turns // 2 * task.team.schedule.turn.model_calls
    pieces = {n: {"colour": "white" if n[0] == "w" else "black", "kind": n.split("_")[1], "at": w}
              for n, w in _start_position(p).items()}
    ref = ChessReferee({"pieces": pieces}, {**p, "max_moves": 1})
    assert ref.max_moves == 1 and ref.max_moves_action == "stop"
    assert ChessReferee({"pieces": pieces}, {**p, "max_moves": None}).max_moves is None
    with pytest.raises(ValueError):
        ChessReferee({"pieces": pieces}, {**p, "max_moves_action": "draw"})

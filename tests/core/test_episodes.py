"""End-to-end episodes on MuJoCo with deterministic subjects (no network, no video)."""
import json
from pathlib import Path

import pytest

from omakase_bench.backends.mujoco.backend import MujocoBackend
from omakase_bench.core.runner import RunConfig, run
from omakase_bench.harness.session import RunBudget
from omakase_bench.core.suite import find_task
from omakase_bench.robots.spec import load_robot
from omakase_bench.subjects import build_subject


def initial_state(task_id: str, seed: int) -> dict:
    task = find_task(task_id)
    ep = task.resolve(seed)
    b = MujocoBackend()
    b.load(load_robot(task.robot), ep.scene)
    b.reset(ep.scene, seed=seed)
    snap = b.oracle().to_json()
    b.close()
    return snap


@pytest.mark.parametrize("task_id", ["core/touch_cube", "core/cube_to_tray", "core/press_button"])
def test_same_seed_same_initial_state(task_id):
    a, b = initial_state(task_id, 3), initial_state(task_id, 3)
    assert a == b
    assert initial_state(task_id, 4) != a


def _run(tmp_path: Path, subjects, task="core/press_button", seeds=(0,), budget=None):
    cfg = RunConfig(out_dir=tmp_path, video=False, run_budget=budget or RunBudget(None, None))
    return run(subjects, [find_task(task)], list(seeds), cfg, progress=lambda s: None)


@pytest.mark.parametrize("task", ["core/touch_cube", "core/cube_to_tray", "core/press_button"])
def test_scripted_subject_solves_phase1_tasks(tmp_path, task):
    res = _run(tmp_path, [build_subject("scripted/oracle")], task=task)
    assert res["episodes"][0]["success"], res["episodes"][0]["evidence"]


def test_scripted_subject_rides_the_elevator_without_sideways_moves(tmp_path):
    res = _run(tmp_path, [build_subject("scripted/oracle")], task="drafts/elevator_call")
    ep = res["episodes"][0]
    assert ep["success"], ep["evidence"]
    calls = [json.loads(line) for line in (tmp_path / ep["artifacts"]["trace"]).read_text().splitlines()]
    moves = [c for c in calls if c.get("name") == "move_base"]
    assert moves and all(not c["arguments"].get("left_m") and not c.get("refused") for c in moves)


def test_mock_tools_and_code_agents(tmp_path):
    subjects = [build_subject("mock/press", "tools"), build_subject("mock/press", "code"),
                build_subject("mock/press", "code", code_mode="iterative")]
    res = _run(tmp_path, subjects)
    assert [e["success"] for e in res["episodes"]] == [True, True, True]
    assert (tmp_path / "results.json").exists() and (tmp_path / "report.html").exists()
    code_ep = res["episodes"][1]
    assert "program" in code_ep["artifacts"]


def test_model_claim_never_scores(tmp_path):
    res = _run(tmp_path, [build_subject("mock/finish", "tools")])
    ep = res["episodes"][0]
    assert ep["exit_reason"] == "finished" and ep["model_claim"] == "done" and not ep["success"]


def test_episode_model_call_cap(tmp_path):
    res = _run(tmp_path, [build_subject("mock/loop", "tools")])
    ep = res["episodes"][0]
    assert ep["exit_reason"] == "budget_exceeded:model_calls"
    assert ep["usage"]["model_calls"] == find_task("core/press_button").budget.model_calls


def test_run_wide_call_cap_stops_the_run(tmp_path):
    res = _run(tmp_path, [build_subject("mock/loop", "tools")], seeds=(0, 1, 2), budget=RunBudget(max_model_calls=7))
    assert res["episodes"][0]["exit_reason"] == "budget_exceeded:run_model_calls"
    assert res["stopped"] == "budget_exceeded:run_model_calls"
    assert len(res["episodes"]) == 1
    assert res["provenance"]["run_budget"]["model_calls"] == 7


def test_run_wide_usd_cap(tmp_path):
    res = _run(tmp_path, [build_subject("mock/loop", "tools")], seeds=(0, 1), budget=RunBudget(None, max_usd=0.0045))
    assert res["episodes"][0]["exit_reason"] == "budget_exceeded:run_usd"
    assert res["episodes"][0]["usage"]["model_calls"] == 5


def test_code_agent_runaway_program_is_stopped(tmp_path):
    res = _run(tmp_path, [build_subject("mock/loop", "code")])
    assert res["episodes"][0]["exit_reason"] == "budget_exceeded:api_calls"


def test_chunk_stub(tmp_path):
    res = _run(tmp_path, [build_subject("stub/hold")])
    assert res["episodes"][0]["exit_reason"] == "chunks_exhausted" and not res["episodes"][0]["success"]

"""core/cube_to_tray, the template task: one robot, one subject, end to end."""
import pytest

from omakase_bench.backends.mujoco.backend import MujocoBackend
from omakase_bench.core.runner import RunConfig, run
from omakase_bench.core.suite import find_task
from omakase_bench.core.task import robot_for, with_robot
from omakase_bench.harness.session import RunBudget
from omakase_bench.subjects import build_subject
from omakase_bench.tasks import package_for

TASK = "core/cube_to_tray"


def _run(tmp_path, subject, task=None, seeds=(0,)):
    cfg = RunConfig(out_dir=tmp_path, video=False, run_budget=RunBudget(None, None))
    return run([subject], [task or find_task(TASK)], list(seeds), cfg, progress=lambda s: None)["episodes"]


def test_the_package_registers_the_task_and_its_reference_solution():
    package = package_for(TASK)
    assert package.name == "manipulation/cube_to_tray" and package.family == "manipulation"
    assert package.scripted[TASK].__name__ == "solve"
    task = find_task(TASK)
    assert task.team is None and {"tools", "code", "scripted"} <= task.interfaces


def test_the_same_seed_gives_the_same_scene_and_another_seed_another():
    def start(seed):
        task = find_task(TASK)
        episode = task.resolve(seed)
        backend = MujocoBackend()
        backend.load(robot_for(task, episode.scene), episode.scene)
        backend.reset(episode.scene, seed=seed)
        return episode.instruction, backend.oracle().to_json()

    assert start(2) == start(2) and start(2) != start(3)


def test_the_scripted_player_solves_it(tmp_path):
    episodes = _run(tmp_path, build_subject("scripted/oracle"), seeds=(0, 1, 2))
    assert [e["success"] for e in episodes] == [True] * 3, [e["evidence"] for e in episodes]
    assert all(e["credit"] == 1.0 and {g["name"] for g in e["subgoals"]} == {"grasped", "lifted", "placed"} for e in episodes)


@pytest.mark.parametrize("interface", ["tools", "code"])
def test_a_stand_in_model_solves_it_through_each_model_interface(tmp_path, interface):
    episode = _run(tmp_path, build_subject("mock/place", interface))[0]
    assert episode["success"], (episode["exit_reason"], episode["exit_detail"], episode["evidence"])
    assert episode["interface"] == interface and episode["usage"]["model_calls"] >= 1
    assert (tmp_path / episode["artifacts"]["trace"]).exists()


def test_a_model_that_only_claims_success_does_not_score(tmp_path):
    episode = _run(tmp_path, build_subject("mock/finish", "tools"))[0]
    assert not episode["success"] and episode["model_claim"] == "done" and episode["credit"] == 0.0


def test_the_stand_in_model_solves_it_on_another_robot(tmp_path):
    episode = _run(tmp_path, build_subject("mock/place", "tools"), task=with_robot(find_task(TASK), "arm6"))[0]
    assert episode["success"], (episode["exit_reason"], episode["evidence"])

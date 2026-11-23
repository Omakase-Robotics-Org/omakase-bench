import pytest

from omakase_bench.core.predicates import validate
from omakase_bench.core.suite import SUITES_DIR, find_task, list_suites, load_suite, suite_hash
from omakase_bench.harness.team_session import robot_instruction


def all_tasks():
    return [find_task(str(f)) for f in sorted(SUITES_DIR.glob("*/*/*.yaml")) if f.name != "suite.yaml"]


@pytest.mark.parametrize("task", all_tasks(), ids=lambda t: t.id)
def test_task_predicates_and_variation(task):
    if task.success or task.team is None:
        validate(task.success)
    for s in task.subgoals:
        validate(s.predicate)
    for seed in range(50):
        ep = task.resolve(seed)
        if task.team is None:
            assert "{" not in ep.instruction
        for slot in (task.team.robots if task.team else ()):
            own = {**ep.values, **ep.values["robots"][slot.id]} if "robots" in ep.values else ep.values
            assert "{" not in robot_instruction(task, slot, own)
    assert task.resolve(7).values == task.resolve(7).values
    assert task.resolve(7).values != task.resolve(8).values


def test_core_suite_frozen_shape():
    s = load_suite("core@1")
    assert [t.id for t in s.tasks] == ["core/touch_cube", "core/cube_to_tray", "core/press_button"]
    assert s.seeds == tuple(range(20))
    assert s.content_hash == suite_hash(s.path)
    assert len(list_suites()) >= 2

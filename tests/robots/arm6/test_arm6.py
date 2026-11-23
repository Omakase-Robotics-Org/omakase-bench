"""arm6: a second robot runs through the harness from its URDF and robot.yaml alone."""
import pytest

from omakase_bench.core.runner import RunConfig, run
from omakase_bench.core.suite import find_task
from omakase_bench.core.task import with_robot
from omakase_bench.harness.docgen import coerce_arguments, python_stubs, tool_specs
from omakase_bench.harness.session import RunBudget
from omakase_bench.robots.check import check
from omakase_bench.robots.spec import load_robot
from omakase_bench.subjects import build_subject
from omakase_bench.subjects.agents import FRAME_TEXT, frame_text, tools_system


def test_the_description_loads_and_passes_the_smoke_check(tmp_path):
    robot = load_robot("arm6")
    assert list(robot.arms) == ["arm"] and robot.mobile_base is None and not robot.lift_joint and not robot.neck
    assert robot.arm_for(0.3) == robot.arm_for(-0.3) == "arm"
    lines = []
    assert check(robot, tmp_path, echo=lines.append)
    assert "arm arm: 6 joints, side any" in lines[1]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["arm6_head.png", "arm6_outside.png", "arm6_wrist.png"]


def test_model_facing_text_names_the_robots_own_arm():
    robot = load_robot("arm6")
    move = next(t for t in tool_specs(robot) if t.name == "move_ee")
    assert move.parameters["properties"]["arm"]["enum"] == ["arm"]
    assert "arm: Literal['arm']" in python_stubs(robot)
    assert coerce_arguments("open_gripper", {"arm": "arm"}, robot) == {"arm": "arm"}
    with pytest.raises(ValueError, match="must be one of \\['arm'\\]"):
        coerce_arguments("open_gripper", {"arm": "left"}, robot)
    text = frame_text(robot)
    assert 'The robot has one arm ("arm")' in text and "about 1 cm" in text
    assert tools_system(robot).startswith("You control a robot arm in a physics simulation")
    # the D1 keeps the published text, line breaks included
    assert frame_text(load_robot("d1")) == FRAME_TEXT


@pytest.mark.parametrize("task_id", ["core/cube_to_tray", "core/press_button"])
def test_published_tasks_run_on_it_with_the_scripted_player(tmp_path, task_id):
    task = with_robot(find_task(task_id), "arm6")
    assert task.robot == "arm6" and task.scene.home_pose == "" and task.scene.lift is None and task.scene.neck == {}
    cfg = RunConfig(out_dir=tmp_path, video=False, run_budget=RunBudget(None, None))
    res = run([build_subject("scripted/oracle")], [task], [0, 1], cfg, progress=lambda s: None)
    assert [e["success"] for e in res["episodes"]] == [True, True], [e["evidence"] for e in res["episodes"]]
    assert res["provenance"]["backend"]["robot"] == "arm6"


def test_a_multi_robot_task_takes_no_robot_override():
    with pytest.raises(ValueError, match="places its robots itself"):
        with_robot(find_task("team/block_relay"), "arm6")

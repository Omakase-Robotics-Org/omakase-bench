"""Arm home: episodes start at the preset's home unless the task names another pose."""
import json

import numpy as np
import pytest

from omakase_bench.backends.mujoco.backend import MujocoBackend
from omakase_bench.harness.session import Session
from omakase_bench.core.suite import SUITES_DIR, find_task, load_suite
from omakase_bench.core.task import load_task, robot_for, scene_from_dict, scene_to_dict
from omakase_bench.robots.spec import load_robot
from omakase_bench.subjects.scripted_ee import make_controller

def _poses():
    robot = load_robot("d1")
    return json.loads((robot.preset_dir / "spec.json").read_text())["poses"], robot


def _arm_q(q, robot):
    return np.array([q[j] for a in robot.arms.values() for j in a.joints])


def _named(pose, robot):
    return np.array([pose[j] for a in robot.arms.values() for j in a.joints])


def test_preset_home_is_the_normal_home():
    poses, robot = _poses()
    assert np.allclose(_arm_q(robot.default_joint_positions(), robot), _named(poses["home"], robot))
    other = robot.with_home("manipulation_home")
    assert np.allclose(_arm_q(other.default_joint_positions(), other), _named(poses["manipulation_home"], robot))
    assert robot.with_home("") is robot
    with pytest.raises(ValueError, match="no pose"):
        robot.with_home("ready")


def test_every_published_task_starts_at_the_presets_home():
    for d in sorted(SUITES_DIR.glob("*/*/suite.yaml")):
        assert all(t.scene.home_pose == "" for t in load_suite(str(d.parent)).tasks), d


def test_a_task_may_name_another_start_pose_and_it_travels_in_the_wire_form(tmp_path):
    raw = (SUITES_DIR / "manip/v1/stack.yaml").read_text().replace("robot: {preset: d1,", "robot: {preset: d1, home: manipulation_home,", 1)
    assert "home: manipulation_home" in raw
    path = tmp_path / "stack.yaml"
    path.write_text(raw)
    scene = load_task(path).resolve(2).scene
    assert scene.home_pose == "manipulation_home" and scene_from_dict(scene_to_dict(scene)) == scene


@pytest.mark.parametrize("named, pose", [(False, "home"), (True, "manipulation_home")])
def test_episode_starts_in_the_scene_pose(tmp_path, named, pose):
    poses, robot = _poses()
    raw = (SUITES_DIR / "manip/v1/stack.yaml").read_text()
    path = tmp_path / "stack.yaml"
    path.write_text(raw.replace("robot: {preset: d1,", "robot: {preset: d1, home: manipulation_home,", 1) if named else raw)
    task = load_task(path)
    ep = task.resolve(0)
    backend = MujocoBackend()
    try:
        backend.load(robot_for(task, ep.scene), ep.scene)
        backend.reset(ep.scene, seed=0)
        assert np.allclose(_arm_q(backend.joint_positions(), robot), _named(poses[pose], robot), atol=0.02)
    finally:
        backend.close()


def test_staged_start_leaves_home_without_touching_the_board():
    """chess_move seed 1: a pawn stands under the open left jaw at home. One joint-space move to the
    start pose takes the TCP about 9 cm below its home height; the staged move does not go below it."""
    task = find_task("manip/chess_move")
    ep = task.resolve(1)
    robot = robot_for(task, ep.scene)
    backend = MujocoBackend()
    try:
        backend.load(robot, ep.scene)
        backend.reset(ep.scene, seed=1)
        session = Session(ep, backend, robot)
        home_z = min(p[2] for p in backend.oracle().tcp.values())
        lowest = [home_z]
        advance = session.advance

        def tracked(seconds, on_tick=None):
            def tick(t):
                if on_tick:
                    on_tick(t)
                lowest.append(min(p[2] for p in backend.oracle().tcp.values()))
            advance(seconds, tick)

        session.advance = tracked
        make_controller(session)
        snap = backend.oracle()
        assert min(lowest) > home_z - 0.002
        assert not [c for c in snap.contacts_ever if "_pad" in c[0] or c[0].endswith("_hand")]
        for arm, (x, y, z, _) in task.raw["ee"]["start"].items():
            assert np.linalg.norm(np.array(session.api.world_to_base(snap.tcp[arm])) - [x, y, z]) < 0.005
    finally:
        backend.close()

"""Mobile-base motion: a differential base follows the unicycle model, a holonomic one
translates in a straight line while turning."""
import dataclasses
import math
import shutil

import numpy as np
import pytest
import yaml

from omakase_bench.backends.mujoco.backend import MujocoBackend
from omakase_bench.harness.session import Session
from omakase_bench.core.suite import find_task
from omakase_bench.harness.api import BASE_SPEED_FRACTION, Refused
from omakase_bench.harness.docgen import python_stubs, tool_specs
from omakase_bench.robots.spec import PRESETS_DIR, load_robot


def _holonomic(robot):
    return dataclasses.replace(robot, mobile_base=dataclasses.replace(robot.mobile_base, drive="holonomic"))


def _session(robot):
    """The elevator lobby: open floor ahead, a wall 2 m in front."""
    ep = find_task("drafts/elevator_call").resolve(0)
    backend = MujocoBackend()
    backend.load(robot, ep.scene)
    backend.reset(ep.scene, seed=0)
    return Session(ep, backend, robot)


def _record(session):
    """Per control tick: the commanded and the measured base pose (x, y, yaw)."""
    mb = session.robot.mobile_base
    backend = session.backend
    commanded, measured = [], []
    command, step = backend.command_joints, backend.step

    def command_joints(targets):
        if mb.x in targets:
            commanded.append((targets[mb.x], targets[mb.y], targets[mb.yaw]))
        command(targets)

    def stepped(dt):
        step(dt)
        q = backend.joint_positions()
        measured.append((q[mb.x], q[mb.y], q[mb.yaw]))

    backend.command_joints, backend.step = command_joints, stepped
    return commanded, measured


def _lateral(poses) -> np.ndarray:
    """Displacement across the heading between consecutive poses, in the base frame."""
    p = np.array(poses)
    d = np.diff(p, axis=0)
    yaw = (p[:-1, 2] + p[1:, 2]) / 2
    return -np.sin(yaw) * d[:, 0] + np.cos(yaw) * d[:, 1]


def test_d1_preset_is_differential():
    mb = load_robot("d1").mobile_base
    assert mb.drive == "differential" and mb.differential


def test_differential_arc_never_moves_sideways():
    session = _session(load_robot("d1"))
    commanded, measured = _record(session)
    out = session.api.move_base(forward_m=1.0, turn_deg=60.0)
    assert len(commanded) > 100 and len(measured) > 100
    # the commanded path is the exact unicycle integral: no lateral component at any tick
    assert np.max(np.abs(_lateral(commanded))) < 1e-9
    # the simulated base follows it without slipping sideways
    lat = _lateral(measured)
    assert np.max(np.abs(lat)) < 1e-4, np.max(np.abs(lat))
    assert np.sum(np.abs(lat)) < 2e-3, np.sum(np.abs(lat))
    # forward plus turn is one arc of length 1 m: chord (r sin 60, r (1 - cos 60)) with r = 1 / (pi / 3)
    r = 1.0 / math.radians(60.0)
    assert out["moved_forward_m"] == pytest.approx(r * math.sin(math.radians(60)), abs=0.005)
    assert out["moved_left_m"] == pytest.approx(r * (1 - math.cos(math.radians(60))), abs=0.005)
    assert out["turned_deg"] == pytest.approx(60.0, abs=0.5)


def test_differential_respects_speed_limits():
    robot = load_robot("d1")
    mb = robot.mobile_base
    session = _session(robot)
    commanded, _ = _record(session)
    session.api.move_base(forward_m=0.5, turn_deg=90.0)
    p = np.array(commanded)
    dt = session.control_dt
    v = np.hypot(*np.diff(p[:, :2], axis=0).T) / dt
    w = np.abs(np.diff(p[:, 2])) / dt
    assert v.max() <= mb.max_speed * BASE_SPEED_FRACTION + 1e-6
    assert w.max() <= mb.max_yaw_rate * BASE_SPEED_FRACTION + 1e-6
    # v and w are constant over the same duration (the last tick ends part-way through)
    moving = np.flatnonzero(w > 1e-6)[:-1]
    assert len(moving) > 50
    assert np.ptp(v[moving]) < 1e-6 and np.ptp(w[moving]) < 1e-6


def test_left_m_refused_on_differential_base():
    session = _session(load_robot("d1"))
    before = session.api._base_pose()
    for left in (0.3, -0.05):
        with pytest.raises(Refused, match="differential drive"):
            session.api.move_base(forward_m=0.5, left_m=left)
    assert session.api._base_pose() == before
    # rounding noise is not a sideways request
    assert session.api.move_base(forward_m=0.2, left_m=1e-6)["moved_forward_m"] == pytest.approx(0.2, abs=0.005)


def test_differential_bump_backs_off_along_heading():
    session = _session(load_robot("d1"))
    session.api.move_base(turn_deg=25.0)    # toward the lobby wall left of the closed door
    out = session.api.move_base(forward_m=2.5)
    assert out["bumped_into"]
    assert out["moved_forward_m"] < 2.5
    assert abs(out["moved_left_m"]) < 0.005


def test_holonomic_base_unchanged():
    robot = _holonomic(load_robot("d1"))
    mb = robot.mobile_base
    session = _session(robot)
    x0, y0, yaw0 = session.api._base_pose()
    commanded, _ = _record(session)
    out = session.api.move_base(forward_m=0.4, left_m=0.3, turn_deg=30.0)
    # straight-line interpolation of (x, y, yaw), as before drive types existed
    c, s = math.cos(yaw0), math.sin(yaw0)
    start = np.array([x0, y0, yaw0])
    goal = np.array([x0 + 0.4 * c - 0.3 * s, y0 + 0.4 * s + 0.3 * c, yaw0 + math.radians(30.0)])
    dur = max(0.5 / (mb.max_speed * BASE_SPEED_FRACTION),
              math.radians(30.0) / (mb.max_yaw_rate * BASE_SPEED_FRACTION), 0.2)
    expected = [start + (goal - start) * min(1.0, i * session.control_dt / dur) for i in range(len(commanded))]
    np.testing.assert_allclose(np.array(commanded), np.array(expected), atol=1e-12)
    assert out["moved_forward_m"] == pytest.approx(0.4, abs=0.01)
    assert out["moved_left_m"] == pytest.approx(0.3, abs=0.01)
    assert out["turned_deg"] == pytest.approx(30.0, abs=0.5)
    # the MuJoCo base servos keep critical damping per joint
    act = session.backend.model.actuator(f"act_{mb.yaw}")
    assert -act.biasprm[2] / act.gainprm[0] < 0.05


def test_drive_defaults_to_holonomic_and_is_validated(tmp_path):
    preset = tmp_path / "bot"
    shutil.copytree(PRESETS_DIR / "d1", preset, ignore=shutil.ignore_patterns("meshes"))
    raw = yaml.safe_load((preset / "robot.yaml").read_text())
    del raw["mobile_base"]["drive"]
    (preset / "robot.yaml").write_text(yaml.safe_dump(raw))
    assert load_robot(preset).mobile_base.drive == "holonomic"
    raw["mobile_base"]["drive"] = "tracked"
    (preset / "robot.yaml").write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="mobile_base.drive"):
        load_robot(preset)


def test_docs_tell_the_model_about_the_drive():
    d1 = load_robot("d1")
    move_base = {s.name: s for s in tool_specs(d1)}["move_base"]
    assert "differential drive" in move_base.description and "left_m=0" in move_base.description
    assert "left_m" in move_base.parameters["properties"]
    assert "always pass left_m=0" in " ".join(python_stubs(d1).split())
    holonomic = {s.name: s for s in tool_specs(_holonomic(d1))}["move_base"]
    assert "holonomic" in holonomic.description and "always pass left_m=0" not in holonomic.description
    assert "always pass left_m=0" not in {s.name: s for s in tool_specs()}["move_base"].description

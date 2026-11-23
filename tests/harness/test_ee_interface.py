"""The end-effector interface: rows, step limits, IK feedback, depth lookup, tabletop tasks."""
import json
from pathlib import Path

import numpy as np
import pytest

from omakase_bench.backends.mujoco.backend import MujocoBackend
from omakase_bench.core.runner import RunConfig, run
from omakase_bench.harness.session import RunBudget, Session
from omakase_bench.core.suite import find_task, load_suite
from omakase_bench.models.base import Capabilities, Reply, Usage
from omakase_bench.harness import depth
from omakase_bench.harness.ee_control import (ArmTarget, EEController, EELimits, ParseError, format_row, parse_row,
                                            parse_rows, step_goal)
from omakase_bench.robots.spec import load_robot
from omakase_bench.subjects import build_subject
from omakase_bench.subjects.ee_agent import EEAgent, EEConfig
from omakase_bench.subjects.scripted_ee import make_follower

ARMS = ("left", "right")
TASKS = ["manip/chess_move", "manip/stack", "manip/sort", "manip/clutter_pick"]


def test_rows_round_trip_and_reject_malformed_input():
    row = parse_row("L 400 170 900 0 0 | R hold", ARMS)
    assert row == {"left": ArmTarget((400, 170, 900), 0, 0), "right": None}
    assert format_row(row, ARMS) == "L 400 170 900 0 0 | R hold"
    assert parse_rows("thinking...\nt03 | ... -> a: R 410 -200 800 -90 1", ARMS) == ["L hold | R 410 -200 800 -90 1"]
    for bad in ("L 1 2 3 | R hold", "L 1 2 3 0 2", "L 1 2 3 0 0 | L 1 2 3 0 0", "nothing"):
        with pytest.raises(ParseError):
            parse_row(bad, ARMS)
    with pytest.raises(ParseError):
        parse_rows("no action here", ARMS)


def test_step_goal_is_limited():
    lim = EELimits()
    goal, yaw = step_goal(np.array([0.4, 0.2, 1.0]), 0.0, ArmTarget((400, 200, 700), 170, 0), lim)
    assert np.allclose(goal, [0.4, 0.2, 1.0 - lim.max_step_m]) and yaw == lim.max_yaw_step_deg
    goal, yaw = step_goal(np.array([0.4, 0.2, 1.0]), 170.0, ArmTarget((410, 200, 1000), -170, 0), lim)
    assert np.allclose(goal, [0.41, 0.2, 1.0]) and yaw == pytest.approx(190.0)   # the short way round
    with pytest.raises(ValueError):
        EELimits.from_mapping({"max_step": 1})


@pytest.fixture()
def chess():
    task = find_task("manip/chess_move")
    robot = load_robot(task.robot)
    ep = task.resolve(0)
    backend = MujocoBackend()
    backend.load(robot, ep.scene)
    backend.reset(ep.scene, seed=0)
    session = Session(ep, backend, robot)
    cfg = EEConfig.from_task(task.raw, robot)
    controller = EEController(session, cfg.limits)
    controller.move_to_start(cfg.start)
    yield session, controller, cfg
    backend.close()


def test_steps_are_limited_and_an_unreachable_target_leaves_the_arm_in_place(chess):
    session, c, cfg = chess
    p0, yaw0 = c.tip_pose("left")
    assert np.allclose(p0, [0.40, 0.17, 1.0 - session.robot.hand_of("left").fingertip_offset[0]], atol=0.003)
    assert abs(yaw0) < 1.0
    right0 = c.tip_pose("right")[0]
    assert c.step({"left": ArmTarget((400, 170, 800), 60, 0), "right": None}) == []
    p1, yaw1 = c.tip_pose("left")
    assert 0.03 < np.linalg.norm(p1 - p0) <= cfg.limits.max_step_m + 0.004
    assert 20 < yaw1 <= cfg.limits.max_yaw_step_deg + 1.5
    assert np.allclose(c.tip_pose("right")[0], right0, atol=1e-3)
    # far beyond the arm's reach: steps succeed until the workspace ends, then the arm stays put
    seen = []
    for _ in range(16):
        seen.append(c.step({"left": ArmTarget((1200, 170, 900), 60, 1), "right": None}))
    assert seen[-1] == ["left"] and c.gripper_cmd["left"] == 1
    before = c.tip_pose("left")[0]
    c.step({"left": ArmTarget((1200, 170, 900), 60, 1), "right": None})
    assert np.linalg.norm(c.tip_pose("left")[0] - before) < 0.002
    assert c.unreachable_total >= 2


def test_depth_lookup_recovers_object_tops_without_ground_truth(chess):
    session, c, cfg = chess
    b, robot, kin = session.backend, session.robot, session.api.kin
    snap = b.oracle()
    w, h = cfg.view_width, cfg.view_height
    found = 0
    for name, o in snap.objects.items():
        if o.kind != "object":
            continue
        top = np.array(o.pos) + [0, 0, o.size[1] / 2]
        kin.set_q(b.joint_positions())
        pos, rot = depth.camera_pose(kin, robot, "head")
        ray = rot.T @ (top - pos)
        f = 0.5 * h / np.tan(np.radians(robot.cameras["head"].fovy) / 2)
        u, v = int(w / 2 + f * ray[0] / ray[2]), int(h / 2 + f * ray[1] / ray[2])
        if not (0 <= u < w and 0 <= v < h):
            continue
        r = depth.lookup(b, kin, robot, cfg.views, "head", u, v, w, h)
        assert r.point_mm is not None, r.text()
        assert np.linalg.norm(np.array(r.point_mm) / 1000 - top) < 0.006, (name, r.text(), top)
        found += 1
    assert found >= 3
    assert depth.lookup(b, kin, robot, cfg.views, "head", -1, 10, w, h).point_mm is None
    assert depth.parse_lookups("locate: head 10 20\nlocate: (bogus) 1 2\nlocate: right 5,6", cfg.views) == \
        [("head", 10, 20), ("right", 5, 6)]


def test_lookup_reads_no_object_state(chess):
    """The camera pose comes from a model that holds the robot only."""
    session, _, _ = chess
    kin = session.api.kin
    names = {kin.model.body(i).name for i in range(kin.model.nbody)}
    assert not names & {o.name for o in session.episode.scene.objects}


def _run(tmp_path: Path, subjects, task, seeds=(0,)):
    cfg = RunConfig(out_dir=tmp_path, video=False, run_budget=RunBudget(None, None))
    return run(subjects, [find_task(task)], list(seeds), cfg, progress=lambda s: None)


@pytest.mark.parametrize("task", TASKS)
def test_scripted_solver_solves_the_tabletop_tasks(tmp_path, task):
    res = _run(tmp_path, [build_subject("scripted/oracle")], task, seeds=(0, 1))
    assert [e["success"] for e in res["episodes"]] == [True, True], [e["evidence"] for e in res["episodes"]]


class _Model:
    """Stands in for a language model. ``oracle``: answers with the scripted solver's next row
    after one lookup-only reply; otherwise replies with ``text`` every time."""
    capabilities = Capabilities()
    name = "fake/model"

    def __init__(self, text: str | None = None):
        self.text, self.agent, self.follower, self.requests = text, None, None, []

    def respond(self, request):
        self.requests.append(request)
        usage = Usage(input_tokens=10, output_tokens=5, usd=0.001)
        if self.text is not None:
            return Reply(text=self.text, usage=usage)
        if len(self.requests) == 1:
            return Reply(text="locate: head 160 120", usage=usage)
        if self.follower is None:
            self.follower = make_follower(self.agent.session, self.agent.controller)
        return Reply(text="a: " + self.follower.row(), usage=usage)


def test_ee_agent_end_to_end_with_lookups(tmp_path):
    model = _Model()
    agent = EEAgent(model, "fake/model")
    model.agent = agent
    ep = _run(tmp_path, [agent], "manip/chess_move")["episodes"][0]
    assert ep["success"] and ep["exit_reason"] == "success", ep["evidence"]
    first, second = model.requests[0], model.requests[1]
    text = "".join(p.text for p in first.messages[0].parts if hasattr(p, "text"))
    assert "Task: Chess move" in text and "Move the" in text and "Depth lookup" in text and "Grasp depth" in text
    assert "a: L <x> <y> <z> <yaw> <g> | R <x> <y> <z> <yaw> <g>" in text
    assert "away from the robot (+x)" in text and "the robot's left (+y)" in text
    assert sum(hasattr(p, "data") for p in first.messages[0].parts) == 1
    assert "lookup head(160,120) ->" in "".join(p.text for p in second.messages[0].parts if hasattr(p, "text"))
    summary = json.loads((tmp_path / ep["artifacts"]["ee_summary"]).read_text())
    assert summary["lookups"] == 1 and summary["steps"] <= 28
    # a model must not be told where the pieces stand: only the goal's squares appear
    squares = set(ep["variation"]["squares"].values()) - {ep["variation"]["from"]}
    prefix = (tmp_path / ep["artifacts"]["prompt_prefix"]).read_text()
    goal_free = prefix.replace(ep["variation"]["goal"], "")
    assert not any(f" {sq} " in goal_free or f" {sq}." in goal_free for sq in squares)


def test_ee_agent_stops_on_unusable_replies_and_idle_fails(tmp_path):
    ep = _run(tmp_path, [EEAgent(_Model("I cannot help."), "fake/model")], "manip/chess_move")["episodes"][0]
    assert not ep["success"] and ep["exit_reason"] == "parse_failures"
    ep = _run(tmp_path / "idle", [EEAgent(_Model("a: L hold | R hold"), "fake/model")], "manip/stack")["episodes"][0]
    assert not ep["success"] and ep["exit_reason"] == "horizon"


def test_off_pad_contact_fails_the_grasp_rule(chess):
    """Closing on a piece far below its top touches it above the pad band."""
    session, c, _ = chess
    v = session.episode.values
    snap = session.oracle()
    piece = max((k for k in snap.objects if snap.objects[k].kind == "object"), key=lambda k: snap.objects[k].size[1])
    o = snap.objects[piece]
    arm = "left" if o.pos[1] > 0 else "right"
    x, y = int(o.pos[0] * 1000), int(o.pos[1] * 1000)
    low = int((o.pos[2] - o.size[1] / 2 + 0.012) * 1000)
    for z, g, n in ((low + 150, 0, 8), (low, 0, 6), (low, 1, 3)):
        for _ in range(n):
            c.step({a: (ArmTarget((x, y, z), 0, g) if a == arm else None) for a in ARMS})
    pairs = session.oracle().contacts_ever
    assert (piece, f"{arm}_hand") in pairs and (piece, f"{arm}_offpad") in pairs
    assert v["piece"]  # the goal names a piece


def test_suites_offer_the_interfaces_their_tasks_are_written_for():
    assert find_task("core/cube_to_tray").interfaces == frozenset({"tools", "code", "chunks", "scripted"})
    manip = load_suite("manip@1")
    assert {t.id for t in manip.tasks} == set(TASKS) and all(t.interfaces == {"ee", "scripted"} for t in manip.tasks)
    with pytest.raises(ValueError):
        build_subject("mock/press", "nonsense")


def test_a_task_names_its_ik_solver_and_the_default_is_the_robots_own(chess):
    import dataclasses

    from omakase_bench.harness.kinematics import ik
    from omakase_bench.harness.kinematics.ik import DlsSolver
    session, c, cfg = chess
    assert c.ik is session.api.ik and isinstance(c.ik, DlsSolver)    # ee.ik absent: robot.yaml ik.solver
    calls = []

    class Counting:
        def __init__(self, kin):
            self.kin = kin

        def solve_ik(self, arm, position, rotation, q_current, **kw):
            calls.append(arm)
            return self.kin.solve_ik(arm, position, rotation, q_current, **kw)

    ik.register_ik_solver("counting", Counting)
    try:
        named = dataclasses.replace(session.task, raw={**session.task.raw, "ee": {**session.task.raw["ee"], "ik": "counting"}})
        session.task = named
        other = EEController(session, cfg.limits)
        assert isinstance(other.ik, Counting)
        assert other.step({"left": ArmTarget((400, 170, 900), 0, 0), "right": None}) == [] and calls == ["left"]
        session.task = dataclasses.replace(named, raw={**named.raw, "ee": {**named.raw["ee"], "ik": "missing"}})
        with pytest.raises(ValueError, match="unknown IK solver 'missing'; registered: counting, dls"):
            EEController(session, cfg.limits)
    finally:
        del ik.IK_SOLVERS["counting"]


def test_without_ee_transcript_a_model_call_is_logged_as_before(chess):
    from omakase_bench.subjects.ee_agent import EEConfig
    session, c, cfg = chess
    assert EEConfig.from_task(session.task.raw, session.robot).transcript is False
    assert EEConfig.from_task(find_task("games/chess").raw, session.robot).transcript is True
    session.after_model_call(input_tokens=1, output_tokens=1, seconds=0.0, text="a: L hold")
    assert set(session.trace[-1]) == {"kind", "sim_time", "wall", "stop", "input_tokens", "output_tokens", "usd",
                                      "seconds", "text"}
    session.after_model_call(input_tokens=1, output_tokens=1, seconds=2.0, text="a: L hold", thinking="t", prompt="p")
    assert session.trace[-1]["thinking"] == "t" and session.trace[-1]["prompt"] == "p"

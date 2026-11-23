"""Multi-robot tasks: scene composition, per-robot views and sessions, scheduling, referee, faults."""
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from omakase_bench.backends.mujoco.backend import MujocoBackend
from omakase_bench.core.backend import BackendInfo
from omakase_bench.core.runner import RunConfig, run
from omakase_bench.harness.session import Recorder, RunBudget, StopEpisode
from omakase_bench.core.suite import find_task, load_suite
from omakase_bench.core.task import load_task, robot_for, scene_from_dict, scene_to_dict
from omakase_bench.core.team import TurnReport, inter_robot_contacts
from omakase_bench.harness.team_session import RobotSession, RobotView, World, robot_task, run_team_episode
from omakase_bench.core.task import Episode
from omakase_bench.subjects import build_subject
from omakase_bench.subjects.team import TeamSubject, build_team, parse_binding
from omakase_bench.tasks.multi_robot.block_relay.relay import RelayReferee

TASK = "team/block_relay"


def _variant(tmp_path: Path, **changes):
    raw = dict(find_task(TASK).raw)
    for key, value in changes.items():
        raw[key] = value
    path = tmp_path / f"variant_{len(list(tmp_path.glob('variant_*')))}.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_task(path)


def _world(task, seed: int = 0):
    ep = task.resolve(seed)
    robots = {r.id: robot_for(task, ep.scene, r.id) for r in ep.scene.robots}
    backend = MujocoBackend()
    backend.load_team(robots, ep.scene)
    backend.reset(ep.scene, seed=seed)
    return ep, robots, backend, World(backend, task.team, Recorder(enabled=False), task.budget.sim_seconds)


def _session(world, task, ep, robots, rid, on_action=lambda rid: None):
    slot = task.team.slot(rid)
    view = RobotView(world, rid, robots[rid], slot.cameras)
    episode = Episode(robot_task(task, slot), ep.seed, ep.scene, {**ep.values, "robot": rid}, ep.instruction)
    return RobotSession(world, rid, episode, view, robots[rid], run_budget=RunBudget(None, None),
                        time_mode="lockstep", on_action=on_action)


# --- schema ---------------------------------------------------------------------------------------

def test_task_declares_robots_schedule_and_referee():
    task = find_task(TASK)
    assert task.team.ids == ("a", "b") and task.team.schedule.mode == "turns"
    assert [(s.start_pose, s.idle_pose) for s in task.team.robots] == [("", "side_idle")] * 2
    assert task.team.referee.endswith("RelayReferee") and task.success == {}
    b = task.scene.placement("b")
    assert b.base == pytest.approx((0.95, 0.0, math.pi)) and b.lift == 0.15
    assert robot_for(task, task.scene, "b").name == "d1"
    scene = task.resolve(3).scene
    assert scene_from_dict(scene_to_dict(scene)) == scene
    assert find_task("core/touch_cube").team is None and load_suite("team@1").tasks[0].id == TASK


@pytest.mark.parametrize("change, message", [
    ({"robots": [{"id": "a"}, {"id": "a"}]}, "unique"),
    ({"robots": [{"id": "a/b"}]}, "letters, digits"),
    ({"schedule": {"mode": "round_robin"}}, "schedule.mode"),
    ({"faults": {"inter_robot_collision": "ignore"}}, "faults"),
    ({"schedule": {"mode": "turns", "turn": {"seconds": 3}}}, "schedule.turn"),
    ({"schedule": {"mode": "turns", "park": True}}, "unknown schedule field"),
    ({"robots": [{"id": "a", "idle_pose": "crouch"}]}, "idle_pose 'crouch' is not a pose"),
])
def test_bad_team_blocks_are_rejected_at_load(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        _variant(tmp_path, **change)


def test_bindings_precedence():
    task = find_task(TASK)
    made = []

    def make(spec, interface, robot=""):
        made.append((spec, interface, robot))
        return build_subject(spec, interface) if spec.startswith("scripted/") else type(
            "S", (), {"name": spec, "kind": "agent", "interface": interface})()

    team = build_team(task, make, {"b": ("mock/finish", "tools")})
    assert team.members["a"].name == "scripted/oracle" and team.members["b"].name == "mock/finish"
    team = build_team(task, make, {}, default=("mock/press", "tools"))
    assert {s.name for s in team.members.values()} == {"mock/press"} and team.members["a"] is not team.members["b"]
    assert parse_binding("a=openai/gpt-6.2-astra:ee") == ("a", "openai/gpt-6.2-astra", "ee")
    with pytest.raises(ValueError, match="no robot"):
        build_team(task, make, {"c": ("mock/finish", None)})


# --- composition and per-robot views --------------------------------------------------------------

def test_two_robots_in_one_scene_with_namespaced_names_and_own_views():
    task = find_task(TASK)
    ep, robots, backend, world = _world(task)
    try:
        q = backend.joint_positions()
        assert "a/left_arm_joint1" in q and "b/left_arm_joint1" in q and "left_arm_joint1" not in q
        assert {"a/head", "b/head", "a/left_wrist", "showcase"} <= set(backend.cameras())
        snap = backend.oracle()
        assert set(snap.tcp) == {"a/left", "a/right", "b/left", "b/right"}
        assert snap.bases["b"] == pytest.approx((0.95, 0.0, math.pi), abs=1e-3)
        home = np.array(robots["a"].arms["left"].home)
        assert np.allclose([q[f"b/{j}"] for j in robots["b"].arms["left"].joints], home, atol=0.02)
        view = RobotView(world, "b", robots["b"])
        own = view.joint_positions()
        assert set(own) == {k[2:] for k in q if k.startswith("b/")} and "left_arm_joint1" in own
        assert view.oracle().base == pytest.approx((0.95, 0.0, math.pi), abs=1e-3)
        assert set(view.oracle().tcp) == {"left", "right"}
        assert view.render("head", 64, 48).shape == (48, 64, 3)
        with pytest.raises(KeyError, match="no camera"):
            view.render("showcase", 64, 48)
        limited = RobotView(world, "a", robots["a"], cameras=["head"])
        with pytest.raises(KeyError):
            limited.render("left_wrist", 64, 48)
        # the robot API of b works in b's own base frame: its left arm is on the world's -y side
        session = _session(world, task, ep, robots, "b")
        assert session.api.get_state()["arms"]["left"]["tcp"][1] > 0
        assert snap.tcp["b/left"][1] < 0
    finally:
        backend.close()


def test_gripper_limits_apply_per_robot(tmp_path):
    raw = find_task(TASK).raw
    robots = [dict(r) for r in raw["robots"]]
    robots[1]["gripper"] = {"max_opening_mm": 40}
    task = _variant(tmp_path, robots=robots)
    assert task.team_gripper["a"] == {} and set(task.team_gripper["b"]) == {"left", "right"}
    ep, rs, backend, world = _world(task)
    try:
        a, b = _session(world, task, ep, rs, "a"), _session(world, task, ep, rs, "b")
        fingers = {f.joint: f.open for f in rs["a"].hand_of("left").fingers}
        assert a.command_joints(fingers) == []
        assert b.command_joints(fingers) == ["left: opening clamped to 40 mm"]
        b.advance(0.6)
        assert b.api._opening("left", b.backend.joint_positions()) == pytest.approx(0.040, abs=0.0015)
        assert a.api._opening("left", a.backend.joint_positions()) == pytest.approx(0.100, abs=0.002)
    finally:
        backend.close()


def test_robots_collide_and_contacts_become_a_fault(tmp_path):
    """0.70 m apart the open jaws of the two robots at home overlap at the table centre."""
    raw = find_task(TASK).raw
    robots = [dict(r) for r in raw["robots"]]
    robots[1]["base"] = [0.70, 0, 180]
    task = _variant(tmp_path, robots=robots)
    ep, rs, backend, world = _world(task)
    try:
        pairs = inter_robot_contacts(backend.oracle(), ("a", "b"))
        assert pairs and all(p[0].startswith("a/") and p[1].startswith("b/") for p in pairs)
        world.check_faults()
        assert world.faults[0]["kind"] == "inter_robot_collision" and world.faults[0]["robots"] == ["a", "b"]
        assert world.stop is not None and world.stop.reason == "fault:inter_robot_collision"
        assert inter_robot_contacts(_world(find_task(TASK))[2].oracle(), ("a", "b")) == []
    finally:
        backend.close()


def test_referee_restores_what_a_turn_displaced():
    task = find_task(TASK)
    ep, rs, backend, world = _world(task)
    try:
        ref = RelayReferee(ep.values, task.raw["params"])
        from omakase_bench.core.team import GameView
        game = GameView(("a", "b"), ep.values, task.raw["params"])
        before = backend.oracle()
        assert ref.start(game, before).next_robot == "a"
        name = ep.values["blocks"]["b"][0]
        pos = np.array(before.objects[name].pos) + [0.0, 0.05, 0.0]
        backend.set_object_pose(name, pos, before.objects[name].quat)
        after = backend.oracle()
        assert after.objects[name].pos == pytest.approx(tuple(pos), abs=1e-6)
        ruling = ref.after_turn(game, TurnReport(1, "a", "finished", before, after, 0.0, 1.0))
        assert not ruling.valid and set(ruling.restore) == {name} and ruling.next_robot == "b"
        assert "displaced" in ruling.messages["b"] and "a" not in ruling.messages
    finally:
        backend.close()


def test_a_backend_without_multi_robot_is_refused(tmp_path):
    class OneRobot:
        def info(self):
            return BackendInfo("isaac", "0", "isaac sim", "d1", "", 0.005, 0.02, frozenset())

    task = find_task(TASK)
    team = TeamSubject({rid: build_subject("scripted/oracle") for rid in task.team.ids})
    with pytest.raises(ValueError, match="holds one robot only"):
        run_team_episode(task, 0, team, OneRobot(), RunConfig(out_dir=tmp_path), tmp_path / "e")


# --- episodes ------------------------------------------------------------------------------------------

def test_turns_episode_alternates_and_accounts_per_robot(tmp_path):
    cfg = RunConfig(out_dir=tmp_path / "run", video=False, run_budget=RunBudget(None, None))
    res = run([build_subject("scripted/oracle")], [find_task(TASK)], [0], cfg, progress=lambda s: None)
    e = res["episodes"][0]
    assert e["success"], e["evidence"]
    assert [t["robot"] for t in e["turns"]] == ["a", "b", "a", "b"] and all(t["valid"] for t in e["turns"])
    assert e["faults"] == [] and e["subject"] == "a=scripted/oracle | b=scripted/oracle"
    assert {r: m["metrics"]["placed"] for r, m in e["robots"].items()} == {"a": 2, "b": 2}
    calls = {r: m["usage"]["api_calls"] for r, m in e["robots"].items()}
    assert calls["a"] > 0 and calls["b"] > 0 and e["usage"]["api_calls"] == calls["a"] + calls["b"]
    assert res["per_model"] == [{"subject": "scripted/oracle", "robots": 2, "episodes": 1, "usd": 0.0, "tokens": 0,
                                 "model_calls": 0, "api_calls": e["usage"]["api_calls"]}]
    trace = (tmp_path / "run" / e["artifacts"]["trace"]).read_text().splitlines()
    acting = [l for l in trace if '"kind": "api"' in l]
    # while one robot acts the other sends nothing: api entries come in per-turn blocks
    blocks = [acting[0].split('"robot": "')[1][0]]
    for line in acting[1:]:
        r = line.split('"robot": "')[1][0]
        if r != blocks[-1]:
            blocks.append(r)
    assert blocks == ["a", "b", "a", "b"]


def test_simultaneous_episode_runs_both_robots_in_lockstep(tmp_path):
    """Both robots move one block at once. Their ee start poses sit further back and their zone slots
    further apart than in the turns task: there, both arms out at once touch (a fault)."""
    raw = find_task(TASK).raw
    task = _variant(tmp_path, schedule={"mode": "simultaneous"},
                    ee={**raw["ee"], "start": {"left": [0.30, 0.20, 1.0, 180], "right": [0.30, -0.20, 1.0, 180]}},
                    params={**raw["params"], "slot_y": [0.16, 0.22]})
    cfg = RunConfig(out_dir=tmp_path / "run", video=False, run_budget=RunBudget(None, None))
    res = run([build_subject("scripted/oracle")], [task], [0], cfg, progress=lambda s: None)
    e = res["episodes"][0]
    assert {r: m["metrics"]["placed"] for r, m in e["robots"].items()} == {"a": 1, "b": 1}
    assert not e["success"] and e["faults"] == []
    trace = (tmp_path / "run" / e["artifacts"]["trace"]).read_text().splitlines()
    times = {"a": [], "b": []}
    for line in trace:
        if '"kind": "api"' in line:
            r = line.split('"robot": "')[1][0]
            times[r].append(float(line.split('"sim_time": ')[1].split(",")[0]))
    # both robots act over the same stretch of simulated time
    assert min(times["b"]) < max(times["a"]) and min(times["a"]) < max(times["b"])


def test_turn_caps_end_a_turn_not_the_episode(tmp_path):
    task = _variant(tmp_path, schedule={"mode": "turns", "max_turns": 2, "turn": {"api_calls": 3}})
    team = TeamSubject({rid: build_subject("scripted/oracle") for rid in task.team.ids})
    cfg = RunConfig(out_dir=tmp_path / "run", video=False, run_budget=RunBudget(None, None))
    e = run_team_episode(task, 0, team, MujocoBackend(), cfg, tmp_path / "run" / "e")
    assert [t["ended"] for t in e["turns"]] == ["turn_end:api_calls", "turn_end:api_calls"]
    assert [r["usage"]["api_calls"] for r in e["robots"].values()] == [3, 3]


# --- poses and transfers ---------------------------------------------------------------------------

def test_named_poses_may_set_the_lift():
    from omakase_bench.robots.spec import load_robot
    d1 = load_robot("d1")
    assert d1.pose_lift("side_idle") == 0.20 and d1.pose_lift("home") is None and d1.pose_lift("") is None
    assert set(d1.pose_arm_targets("side_idle")) == {j for a in d1.arms.values() for j in a.joints}


def test_idle_robots_hold_their_idle_pose_and_transfers_are_logged(tmp_path):
    cfg = RunConfig(out_dir=tmp_path / "run", video=False, run_budget=RunBudget(None, None))
    res = run([build_subject("scripted/oracle")], [find_task(TASK)], [1], cfg, progress=lambda s: None)
    e = res["episodes"][0]
    assert e["success"] and e["faults"] == []
    import json
    trace = [json.loads(l) for l in (tmp_path / "run" / e["artifacts"]["trace"]).read_text().splitlines()]
    moves = [(t["robot"], t["pose"]) for t in trace if t["kind"] == "transfer"]
    assert moves[:2] == [("a", "side_idle"), ("b", "side_idle")]
    # every turn starts from the idle pose (only the torso moves to the acting lift) and ends back in
    # it, and nothing is touched during a transfer
    assert moves[2:] == [(r, p) for r in "abab" for p in ("(arms stay)", "side_idle")]
    assert all(t["contacts"] == [] for t in trace if t["kind"] == "transfer")
    # while one robot acts the other sends nothing: its arms hold the idle pose
    start = next(i for i, t in enumerate(trace) if t["kind"] == "turn_start" and t["robot"] == "a")
    end = next(i for i, t in enumerate(trace) if t["kind"] == "turn_end" and t["robot"] == "a")
    assert not [t for t in trace[start:end] if t.get("robot") == "b"]


def test_a_transfer_that_touches_something_is_a_fault(tmp_path):
    """A box where robot a's left hand passes on its way from home to side_idle."""
    raw = find_task(TASK).raw
    scene = dict(raw["scene"])
    scene["fixtures"] = [*scene["fixtures"], {"name": "post", "type": "box", "size": [0.08, 0.08, 0.10],
                                              "pos": [0.05, 0.40, 1.05], "color": "dark"}]
    task = _variant(tmp_path, scene=scene)
    team = TeamSubject({rid: build_subject("scripted/oracle") for rid in task.team.ids})
    cfg = RunConfig(out_dir=tmp_path / "run", video=False, run_budget=RunBudget(None, None))
    e = run_team_episode(task, 0, team, MujocoBackend(), cfg, tmp_path / "run" / "e")
    assert e["exit_reason"] == "fault:transfer_collision" and not e["success"]
    assert e["faults"][0]["kind"] == "transfer_collision" and e["faults"][0]["robots"] == ["a"]
    assert any(p[1] == "post" for p in e["faults"][0]["parts"])


# --- joint travel ------------------------------------------------------------------------------------

def test_joint_travel_counts_swings_between_reversals():
    from omakase_bench.harness.session import JointTravel
    jt = JointTravel(["j"], period=0.0)
    for deg in [0, 30, 60, 100, 60, 0, 10, 0]:
        jt.update({"j": math.radians(deg)})
    out = jt.to_json()["j"]
    assert out["travel_deg"] == pytest.approx(220, abs=0.5)
    assert out["swings_over_90"] == 2 and out["largest_swing_deg"] == pytest.approx(100, abs=0.5)


def test_home_idle_transfers_and_relay_turns_keep_the_wrist_from_spinning(tmp_path):
    """No half turns: side_idle is on the branch of home, and the relay works at the jaw yaw of
    side_idle. The wrist (joint 5) never swings more than 90 deg; the shoulder pitch (joint 2)
    sweeps from home to the table once per move, never by half a turn."""
    from omakase_bench.harness.team_session import transfer
    task = find_task(TASK)
    ep, rs, backend, world = _world(task)
    try:
        s = _session(world, task, ep, rs, "a")
        transfer(world, s, "side_idle")
        transfer(world, s, "")
        travel = s.travel.to_json()
        for arm in ("left", "right"):
            for k in (2, 5):
                assert travel[f"{arm}_arm_joint{k}"]["swings_over_90"] == 0, (arm, k, travel)
        assert world.faults == []
    finally:
        backend.close()
    cfg = RunConfig(out_dir=tmp_path / "run", video=False, run_budget=RunBudget(None, None))
    e = run([build_subject("scripted/oracle")], [find_task(TASK)], [3], cfg, progress=lambda s: None)["episodes"][0]
    assert e["success"]
    for rid, m in e["robots"].items():
        for arm in ("left", "right"):
            assert m["joint_travel"][f"{arm}_arm_joint5"]["swings_over_90"] == 0, (rid, arm)
            assert m["joint_travel"][f"{arm}_arm_joint2"]["largest_swing_deg"] < 175, (rid, arm)


def test_a_step_that_would_jump_a_joint_is_refused_with_feedback():
    from omakase_bench.harness.ee_control import ArmTarget, EEController, EELimits
    from omakase_bench.subjects.ee_agent import EEAgent
    task = find_task(TASK)
    ep, rs, backend, world = _world(task)
    try:
        s = _session(world, task, ep, rs, "a")
        from omakase_bench.harness.team_session import transfer
        transfer(world, s, "side_idle")          # jaws down: the step has an IK solution
        c = EEController(s, EELimits(max_joint_step_rad=0.01))
        tip, yaw = c.tip_pose("left")
        target = ArmTarget(tuple(int(v) for v in (tip * 1000 + [0, 0, 40])), int(round(yaw)), 0)
        assert c.step({"left": target, "right": None}) == ["left"]
        row = EEAgent.obs_row(c, ["left"])
        assert "ikL: would turn joint" in row and "deg in one step" in row
    finally:
        backend.close()

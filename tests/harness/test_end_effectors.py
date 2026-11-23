"""Task-level end-effector limits: schema, validation at task load, enforcement in every interface."""
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import yaml

from omakase_bench.backends.mujoco.backend import MujocoBackend
from omakase_bench.core.runner import RunConfig, run
from omakase_bench.harness.session import RunBudget, Session
from omakase_bench.core.suite import find_task, load_suite
from omakase_bench.core.task import load_task
from omakase_bench.models.base import Capabilities, Reply, Usage
from omakase_bench.harness.docgen import python_stubs, tool_specs
from omakase_bench.harness.ee_control import ArmTarget, EEController, ParseError, format_row, parse_row
from omakase_bench.harness.end_effectors import LIMIT_TYPES, ParallelJawLimits, parse_task_limits
from omakase_bench.robots.spec import load_robot
from omakase_bench.subjects import build_subject
from omakase_bench.subjects.chunks import ChunkSubject
from omakase_bench.subjects.ee_agent import EEAgent, EEConfig
from omakase_bench.subjects.scripted_ee import make_controller, make_follower

ARMS = ("left", "right")
TASK = "grip/packed_grid_pick"


def _variant(tmp_path: Path, gripper, task: str = TASK):
    """``task`` with its gripper block replaced (None removes it), loaded from a temporary file."""
    raw = dict(find_task(task).raw)
    raw.pop("gripper", None)
    if gripper is not None:
        raw["gripper"] = gripper
    path = tmp_path / f"variant_{len(list(tmp_path.glob('variant_*')))}.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_task(path)


def _session(task, seed: int = 0, backend=None):
    robot = load_robot(task.robot)
    ep = task.resolve(seed)
    backend = backend or MujocoBackend()
    backend.load(robot, ep.scene)
    backend.reset(ep.scene, seed=seed)
    return Session(ep, backend, robot)


def _opening_mm(session, arm: str) -> float:
    return session.api._opening(arm, session.backend.joint_positions()) * 1000


# --- schema and validation ----------------------------------------------------------------------

def test_block_parses_with_per_arm_overrides():
    robot = load_robot("d1")
    lim = parse_task_limits({"max_opening_mm": 44, "min_closing_mm": 5, "max_force_n": 40,
                             "arms": {"right": {"max_opening_mm": 30, "preshape_mm": 25}}}, robot)
    assert set(lim) == set(ARMS) and all(isinstance(v, ParallelJawLimits) for v in lim.values())
    assert (lim["left"].max_opening, lim["left"].min_closing, lim["left"].preshape) == pytest.approx((0.044, 0.005, 0.044))
    assert (lim["right"].max_opening, lim["right"].preshape, lim["right"].max_force) == pytest.approx((0.030, 0.025, 40))
    assert parse_task_limits(None, robot) == {} and "parallel_gripper" in LIMIT_TYPES


@pytest.mark.parametrize("block, message", [
    ({"max_opening_mm": 120}, "hardware range 0..100 mm"),
    ({"max_opening_mm": 40, "min_closing_mm": 40}, "below max_opening_mm"),
    ({"max_opening_mm": 40, "preshape_mm": 50}, "preshape_mm 50 outside"),
    ({"max_force_n": 500}, "max_force_n 500 outside the hardware range"),
    ({"max_opening": 40}, "unknown parallel_gripper limit"),
    ({"arms": {"middle": {"max_opening_mm": 40}}}, "unknown arm"),
    ({"arms": {"right": {"max_opening_mm": 101}}}, "right arm"),
])
def test_values_outside_the_hardware_range_are_rejected_at_task_load(tmp_path, block, message):
    with pytest.raises(ValueError, match=message):
        _variant(tmp_path, block)


def test_absent_block_means_no_limits(tmp_path):
    assert find_task("manip/clutter_pick").gripper == {} and _variant(tmp_path, None).gripper == {}
    for spec in tool_specs(load_robot("d1")):
        assert "End-effector limits" not in spec.description


# --- enforcement ----------------------------------------------------------------------------------

def test_episode_starts_at_the_preshape_and_the_session_clamps_every_joint_command(tmp_path):
    session = _session(_variant(tmp_path, {"max_opening_mm": 44, "arms": {"right": {"preshape_mm": 30}}}))
    try:
        assert _opening_mm(session, "left") == pytest.approx(44, abs=1.5)
        assert _opening_mm(session, "right") == pytest.approx(30, abs=1.5)
        ee = session.robot.hand_of("left")
        notes = session.command_joints({f.joint: f.open for f in ee.fingers})      # fully open, joint space
        assert notes == ["left: opening clamped to 44 mm"]
        session.advance(0.6)
        assert _opening_mm(session, "left") == pytest.approx(44, abs=1.5)
        assert session.command_joints({f.joint: f.closed for f in ee.fingers}) == []
    finally:
        session.backend.close()


def test_tools_and_code_docs_state_the_limits_and_the_api_reports_clamps(tmp_path):
    task = _variant(tmp_path, {"max_opening_mm": 44, "min_closing_mm": 8})
    session = _session(task)
    try:
        docs = {t.name: t.description for t in tool_specs(session.robot, session.gripper)}
        assert "left gripper opens at most 44 mm and closes to no less than 8 mm" in docs["open_gripper"]
        assert "clamped" in docs["close_gripper"]
        assert "opens at most 44 mm" in python_stubs(session.robot, session.gripper)
        out = session.api.open_gripper("left")
        assert out["opening_m"] == pytest.approx(0.044, abs=0.0015) and out["gripper"] == "opening clamped to 44 mm (task limit)"
        out = session.api.close_gripper("left")     # nothing between the jaws
        assert out["opening_m"] == pytest.approx(0.008, abs=0.0015)
        assert out["closed_on_something"] is False and "closing stops at 8 mm" in out["gripper"]
    finally:
        session.backend.close()


def test_ee_rows_accept_openings_only_under_limits_and_report_clamps(tmp_path):
    assert parse_row("L 400 170 900 0 w30 | R hold", ARMS, widths=True)["left"] == ArmTarget((400, 170, 900), 0, 0, 30)
    assert format_row(parse_row("L 400 170 900 0 w30", ARMS, widths=True), ARMS) == "L 400 170 900 0 w30 | R hold"
    with pytest.raises(ParseError):
        parse_row("L 400 170 900 0 w30", ARMS)                  # no limits: g must be 0 or 1
    with pytest.raises(ParseError):
        parse_row("L 400 170 900 w30", ARMS, widths=True)       # yaw missing
    session = _session(find_task(TASK))
    try:
        cfg = EEConfig.from_task(session.task.raw, session.robot)
        c = EEController(session, cfg.limits)
        assert c.widths
        c.step(parse_row("L 400 170 1000 0 w80 | R hold", ARMS, widths=True))
        row = EEAgent.obs_row(c, [])
        assert "gL: opening clamped to 44 mm" in row and "gL w80" in row
        for _ in range(4):
            c.step(parse_row("L 400 170 1000 0 w20 | R hold", ARMS, widths=True))
        assert _opening_mm(session, "left") == pytest.approx(20, abs=1.5) and "clamped" not in EEAgent.obs_row(c, [])
        c.step(parse_row("L 400 170 1000 0 0 | R hold", ARMS, widths=True))
        c.step(parse_row("L 400 170 1000 0 0 | R hold", ARMS, widths=True))
        assert _opening_mm(session, "left") == pytest.approx(44, abs=1.5)
        prefix = EEAgent(None, "x").prefix(session, cfg)
        assert "this task limits them: the left jaws open at most 44 mm" in prefix and "w<mm>" in prefix
    finally:
        session.backend.close()


class _OpenPolicy:
    """A chunk policy that commands its grippers fully open and records what it is told."""
    name = "test/open"

    def __init__(self):
        self.seen = []

    def spec(self, robot):
        from omakase_bench.subjects.chunks import ChunkSpec
        joints = tuple(f.joint for arm in robot.arms for f in robot.hand_of(arm).fingers)
        self.fingers = [(f.joint, f.open) for arm in robot.arms for f in robot.hand_of(arm).fingers]
        return ChunkSpec(joints=joints, cameras={}, horizon=5, execute=5, hz=10.0)

    def act(self, obs):
        self.seen.append((obs.limits, obs.notes))
        return np.tile([v for _, v in self.fingers], (5, 1))


def test_chunks_are_clamped_and_the_policy_is_told(tmp_path):
    session = _session(_variant(tmp_path, {"max_opening_mm": 44}))
    policy = _OpenPolicy()
    try:
        ChunkSubject(policy, max_chunks=2).run(session)
        assert policy.seen[0][1] == () and "opens at most 44 mm" in policy.seen[0][0]
        assert policy.seen[1][1] == ("left: opening clamped to 44 mm", "right: opening clamped to 44 mm")
        assert _opening_mm(session, "left") == pytest.approx(44, abs=1.5)
    finally:
        session.backend.close()


def test_force_cap_needs_the_backend_capability(tmp_path):
    task = _variant(tmp_path, {"max_force_n": 30})
    session = _session(task)
    try:
        b = session.backend
        act = b._act[session.robot.hand_of("left").fingers[0].joint]
        assert tuple(b.model.actuator_forcerange[act]) == (-30, 30)
    finally:
        session.backend.close()

    class NoForce(MujocoBackend):
        def info(self):
            from dataclasses import replace
            return replace(super().info(), capabilities=frozenset())

    with pytest.raises(ValueError, match="gripper_force_limit"):
        _session(task, backend=NoForce())


# --- the example task -----------------------------------------------------------------------------

def _run(tmp_path, task, seeds):
    cfg = RunConfig(out_dir=tmp_path, video=False, run_budget=RunBudget(None, None))
    return run([build_subject("scripted/oracle")], [task], list(seeds), cfg, progress=lambda s: None)


def test_packed_grid_needs_the_limit(tmp_path):
    """The scripted grasp succeeds with the task's limit; with full opening the pads land on a neighbour."""
    limited = _run(tmp_path / "limited", find_task(TASK), (0, 1))["episodes"]
    assert [e["success"] for e in limited] == [True, True], [e["evidence"] for e in limited]
    for seed in (0, 1):
        session = _session(_variant(tmp_path, None), seed)
        try:
            c = make_controller(session)
            f = make_follower(session, c)
            for _ in range(12):
                c.step(parse_row(f.row(), c.arms))
            snap = session.oracle()
            target = session.episode.values["target"]
            hit = {a for a, b in snap.contacts_ever if b.endswith("_hand") and a != target}
            assert hit, "a full opening should land on a neighbouring cylinder"
            assert not session.success().ok
        finally:
            session.backend.close()


def test_grip_suite_shape():
    s = load_suite("grip@1")
    assert [t.id for t in s.tasks] == [TASK] and s.tasks[0].gripper["left"].max_opening == pytest.approx(0.044)


@pytest.mark.parametrize("limited", [True, False])
def test_tools_api_grasp_in_the_packed_grid(tmp_path, limited):
    """The robot API (tools and code interfaces) obeys the same limit: open_gripper opens to 44 mm
    and the pick succeeds; without the block it opens fully and a pad lands on a neighbour."""
    from omakase_bench.harness.session import StopEpisode
    task = find_task(TASK) if limited else _variant(tmp_path, None)
    session = _session(task, 3)
    api = session.api
    try:
        target = session.episode.values["target"]
        o = session.oracle().objects[target]
        p = api.world_to_base(o.pos)
        top = p[2] + o.size[1] / 2
        arm = "left" if p[1] > 0 else "right"
        out = api.open_gripper(arm)
        assert ("gripper" in out) == limited
        api.move_ee(arm, p[0], p[1], top + 0.15, "down", 0.0)     # clear of the grid for a joint-space move
        api.move_ee(arm, p[0], p[1], top, "down", 0.0)
        api.close_gripper(arm)
        with pytest.raises(StopEpisode) if limited else nullcontext():
            api.move_ee_by(arm, dz=0.13)      # success ends the episode
        hit = {a for a, b in session.oracle().contacts_ever if b == f"{arm}_hand" and a != target}
        assert (session.success_at is not None) == limited and bool(hit) != limited
    finally:
        session.backend.close()


class _RowModel:
    """Stands in for a language model on the ee interface: replies with the scripted solver's rows,
    with every left-arm open command written as an opening beyond the limit."""
    capabilities = Capabilities()
    name = "fake/rows"

    def __init__(self):
        self.agent, self.follower, self.requests = None, None, []

    def respond(self, request):
        self.requests.append(request)
        if self.follower is None:
            self.follower = make_follower(self.agent.session, self.agent.controller)
        row = self.follower.row()
        left, right = row.split(" | ")
        left = left[:-2] + " w90" if left.endswith(" 0") else left
        return Reply(text=f"a: {left} | {right}", usage=Usage(input_tokens=10, output_tokens=5))


def test_ee_agent_end_to_end_under_limits(tmp_path):
    model = _RowModel()
    agent = EEAgent(model, "fake/rows")
    model.agent = agent
    cfg = RunConfig(out_dir=tmp_path, video=False, run_budget=RunBudget(None, None))
    task = find_task(TASK)
    seed = next(s for s in range(10) if task.resolve(s).scene.element(task.resolve(s).values["target"]).pos[1] > 0)
    ep = run([agent], [task], [seed], cfg, progress=lambda s: None)["episodes"][0]
    assert ep["success"], ep["evidence"]
    text = "".join(p.text for m in model.requests[1].messages for p in m.parts if hasattr(p, "text"))
    assert "this task limits them" in text and "gL: opening clamped to 44 mm" in text

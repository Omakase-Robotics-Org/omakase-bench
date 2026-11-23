"""IK behind its interface: the robot names its solver, and the robot API calls that one."""
import numpy as np
import pytest

from omakase_bench.backends.mujoco.backend import MujocoBackend
from omakase_bench.core.suite import find_task
from omakase_bench.core.task import with_robot
from omakase_bench.harness.kinematics import ik
from omakase_bench.harness.kinematics.model import Kinematics
from omakase_bench.harness.session import RunBudget, Session
from omakase_bench.robots.spec import load_robot


@pytest.mark.parametrize("name", ["d1", "arm6"])
def test_the_default_solver_reaches_a_pose_below_home_on_any_robot(name):
    robot = load_robot(name)
    kin = Kinematics(robot)
    q = robot.default_joint_positions()
    kin.set_q(q)
    for arm in robot.arms:
        home, _ = kin.tcp_pose(arm)
        target = home + (0.05, 0.0, -0.03)
        res = ik.ik_solver(robot.ik_solver, kin).solve_ik(arm, target, kin.named_rotation(arm, "down", 0.0), q)
        assert res.ok, res.reason
        kin.set_q({**q, **dict(zip(robot.arms[arm].joints, res.q))})
        assert np.linalg.norm(kin.tcp_pose(arm)[0] - target) < 0.004
        # the model's own methods are the default solver
        assert kin.solve_ik(arm, target, kin.named_rotation(arm, "down", 0.0), q).q == res.q


def test_the_robot_api_uses_the_solver_the_robot_names(tmp_path):
    calls = []

    class Counting(ik.DlsSolver):
        def solve_ik(self, arm, *args, **kw):
            calls.append(arm)
            return super().solve_ik(arm, *args, **kw)

    ik.register_ik_solver("counting", Counting)
    try:
        d = tmp_path / "arm6"
        import shutil
        shutil.copytree(load_robot("arm6").preset_dir, d)
        (d / "robot.yaml").write_text((d / "robot.yaml").read_text() + "ik:\n  solver: counting\n")
        task = with_robot(find_task("core/cube_to_tray"), str(d))
        episode = task.resolve(0)
        robot = load_robot(d)
        backend = MujocoBackend()
        backend.load(robot, episode.scene)
        backend.reset(episode.scene, seed=0)
        session = Session(episode, backend, robot, run_budget=RunBudget(None, None))
        assert isinstance(session.api.ik, Counting)
        session.api.move_ee_by("arm", dz=0.02)
        assert calls and set(calls) == {"arm"}
        with pytest.raises(ValueError, match="unknown IK solver 'nope'; registered: counting, dls"):
            ik.ik_solver("nope", session.api.kin)
    finally:
        del ik.IK_SOLVERS["counting"]

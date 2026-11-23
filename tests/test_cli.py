"""The command line: the generic commands, the robot override and the commands task packages add."""
import json
import os
import subprocess
import sys

import pytest

from omakase_bench.cli import main


def test_list_shows_suites_with_their_state_and_task_packages(capsys):
    assert main(["list", "suites"]) == 0
    out = capsys.readouterr().out
    assert "suite core@1" in out and "frozen" in out and "suite games@1" in out and "CHANGED" not in out
    assert main(["list", "packages"]) == 0
    out = capsys.readouterr().out
    assert "package manipulation/cube_to_tray" in out and "package games/chess" in out
    assert main(["list", "tasks"]) == 0
    assert "task  games/chess" in capsys.readouterr().out


def test_run_with_a_robot_override(tmp_path, capsys):
    code = main(["run", "--task", "core/cube_to_tray", "--model", "scripted/oracle", "--robot", "arm6", "--no-video",
                 "--out", str(tmp_path)])
    assert code == 0
    results = json.loads((tmp_path / "results.json").read_text())
    assert results["episodes"][0]["success"] and results["provenance"]["backend"]["robot"] == "arm6"


def test_robot_check(tmp_path, capsys):
    assert main(["robot", "check", "arm6", "--out", str(tmp_path)]) == 0
    assert "IK reaches 54/54" in capsys.readouterr().out and (tmp_path / "arm6_outside.png").exists()


def test_chess_tools_are_commands_of_the_chess_package(tmp_path, capsys):
    with pytest.raises(SystemExit) as stop:
        main(["chess", "cpl", "--help"])
    assert stop.value.code == 0 and "omakase-bench chess cpl" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="expected one games/chess episode, found 0"):
        main(["chess", "resume", str(tmp_path)])
    with pytest.raises(SystemExit):
        main(["chess", "no_such_tool"])


def test_suite_freeze_needs_its_arguments(capsys):
    with pytest.raises(SystemExit):
        main(["suite", "freeze", "starter@1"])


def test_the_default_renderer_is_chosen_before_the_simulator_is_imported(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "MUJOCO_GL"}
    done = subprocess.run([sys.executable, "-m", "omakase_bench.cli", "robot", "check", "arm6", "--out", str(tmp_path)],
                          env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    assert (tmp_path / "arm6_head.png").exists()


def test_a_request_that_cannot_be_carried_out_is_a_message_and_exit_code_1(tmp_path, capsys):
    for argv, message in ((["run", "--task", "core/no_such_task", "--model", "scripted/oracle"], "no_such_task"),
                          (["run", "--task", "core/touch_cube", "--model", "scripted/oracle", "--seeds", "2-0"], "names no seed"),
                          (["run", "--task", "core/touch_cube", "--model", "scripted/oracle", "--episodes", "0"], "at least 1"),
                          (["run", "--task", "core/touch_cube", "--model", "nobody/model"], "unknown model provider")):
        assert main([*argv, "--out", str(tmp_path / "run"), "--no-video"]) == 1, argv
        captured = capsys.readouterr()
        assert message in captured.err and "Traceback" not in captured.err, argv


def test_a_task_named_with_a_suite_is_the_suites_own_version_and_only_that_is_official(tmp_path, capsys):
    import json

    from omakase_bench.core.suite import find_task, load_suite
    assert main(["run", "--suite", "manip@1", "--task", "manip/stack", "--model", "scripted/oracle", "--seeds", "0",
                 "--out", str(tmp_path / "own"), "--no-video"]) == 0
    provenance = json.loads((tmp_path / "own" / "results.json").read_text())["provenance"]
    own = next(t for t in load_suite("manip@1").tasks if t.id == "manip/stack")
    assert provenance["tasks"] == {"manip/stack": own.version} and provenance["suite"]["version"] == "1"
    from omakase_bench.core.results import build_results
    from omakase_bench.core.runner import RunConfig
    from omakase_bench.core.backend import make_backend
    suite, other = load_suite("manip@1"), find_task("core/touch_cube")

    def official(tasks):
        cfg = RunConfig(out_dir=tmp_path, suite=suite)
        return build_results(episodes=[], subjects=[], tasks=tasks, seeds=suite.seeds, cfg=cfg,
                             backend_info=make_backend("mujoco").info(), started=0.0, stopped="")["provenance"]["suite"]["official"]

    assert official(list(suite.tasks)) is True and official([other]) is False


def test_an_interrupted_run_keeps_the_episodes_that_finished(tmp_path):
    import json

    from omakase_bench.core.runner import RunConfig, run
    from omakase_bench.core.suite import find_task
    from omakase_bench.harness.session import RunBudget
    from omakase_bench.subjects import build_subject

    oracle = build_subject("scripted/oracle")

    class Interrupted:
        name, kind, interface, card = oracle.name, oracle.kind, oracle.interface, oracle.card
        episodes = 0

        def run(self, session):
            self.episodes += 1
            if self.episodes == 2:
                raise KeyboardInterrupt
            return oracle.run(session)

    cfg = RunConfig(out_dir=tmp_path, video=False, run_budget=RunBudget(None, None))
    with pytest.raises(KeyboardInterrupt):
        run([Interrupted()], [find_task("core/touch_cube")], [0, 1, 2], cfg, progress=lambda s: None)
    results = json.loads((tmp_path / "results.json").read_text())
    assert results["stopped"] == "interrupted:KeyboardInterrupt" and [e["seed"] for e in results["episodes"]] == [0]
    assert (tmp_path / "report.html").exists()

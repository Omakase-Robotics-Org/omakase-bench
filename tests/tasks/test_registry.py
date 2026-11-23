"""The task registry: every task package registers itself, and nothing else lists tasks."""
from pathlib import Path

import yaml

import omakase_bench.tasks as tasks
from omakase_bench.core.suite import find_task, task_files
from omakase_bench.subjects import build_subject

ROOT = Path(tasks.__file__).parent


def test_every_package_directory_is_registered_under_its_path():
    names = {p.name for p in tasks.packages()}
    assert names == {f"{d.parent.name}/{d.name}" for d in ROOT.glob("*/*") if (d / "__init__.py").exists()}
    assert {p.family for p in tasks.packages()} == {"manipulation", "mobile_manipulation", "multi_robot", "games"}
    assert all(p.directory == ROOT / p.name and p.summary for p in tasks.packages())


def test_every_task_definition_has_a_reference_solution():
    ids = {yaml.safe_load(f.read_text())["id"] for f in task_files()}
    assert ids == set(tasks.programs())
    assert set(build_subject("scripted/oracle").card().detail["programs"]) == ids
    for task_id in ids:
        assert tasks.package_for(task_id).scripted[task_id] is tasks.scripted_program(task_id)
    assert tasks.package_for("no/such_task") is None and tasks.scripted_program("no/such_task") is None


def test_a_published_task_is_defined_by_its_suites_copy_and_its_package_holds_the_code():
    assert tasks.definitions() == []                    # no task is under development in a package
    assert tasks.package_for("games/chess").definition is None
    assert find_task("games/chess").source.parent.name == "v1"
    assert tasks.package_for("manip/stack").definition is None
    assert find_task("manip/stack").source.parent.name == "v1"
    assert tasks.package_for("manip/stack").moves["manip/stack"]


def test_only_chess_adds_cli_commands():
    assert [p.name for p in tasks.packages() if p.commands] == ["games/chess"]

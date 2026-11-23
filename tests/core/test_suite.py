"""Suites: published hashes, open suites, publishing a new one."""
import pytest
import yaml

from omakase_bench.core import suite as suites
from omakase_bench.core.suite import freeze, list_suites, load_suite, published_hashes

PUBLISHED = {"core@1", "manip@1", "grip@1", "team@1", "games@1", "drafts@0"}


def test_published_suites_have_the_hashes_they_were_published_with():
    recorded = published_hashes()
    assert set(recorded) == PUBLISHED
    for ref in recorded:
        s = load_suite(ref)
        assert s.published and s.intact and s.content_hash == recorded[ref], ref


def test_every_suite_directory_is_published_with_one_version():
    assert {load_suite(str(d)).ref for d in list_suites()} == PUBLISHED
    games = load_suite("games@1")
    assert games.intact and [t.id for t in games.tasks] == ["games/chess"]
    assert (games.path / "chess.yaml").exists()


def test_freeze_publishes_a_suite_and_a_change_to_it_is_noticed(tmp_path):
    s = freeze("starter@1", ["core/cube_to_tray", "core/press_button"], [0, 1, 2], "two tasks", root=tmp_path)
    d = tmp_path / "starter/v1"
    assert s.path == d and sorted(f.name for f in d.iterdir()) == ["cube_to_tray.yaml", "press_button.yaml", "suite.yaml"]
    assert yaml.safe_load((d / "suite.yaml").read_text()) == {
        "id": "starter", "version": 1, "description": "two tasks", "tasks": ["cube_to_tray", "press_button"], "seeds": [0, 1, 2]}
    assert published_hashes(tmp_path) == {"starter@1": s.content_hash} and s.published and s.intact
    assert [t.id for t in s.tasks] == ["core/cube_to_tray", "core/press_button"] and s.seeds == (0, 1, 2)
    with pytest.raises(FileExistsError, match="publish starter@2"):
        freeze("starter@1", ["core/cube_to_tray"], [0], root=tmp_path)
    freeze("starter@2", ["core/cube_to_tray"], [0], root=tmp_path)
    assert list(published_hashes(tmp_path)) == ["starter@1", "starter@2"]
    task = d / "cube_to_tray.yaml"
    task.write_text(task.read_text().replace("sim_seconds: 90", "sim_seconds: 91"))
    changed = load_suite(str(d))
    assert changed.published and not changed.intact
    with pytest.raises(ValueError, match="expected <id>@<version number>"):
        freeze("starter", ["core/cube_to_tray"], [0], root=tmp_path)


def test_a_suite_hash_covers_the_package_definition_of_a_task_it_lists_by_id(tmp_path, monkeypatch):
    d = tmp_path / "open/v0"
    d.mkdir(parents=True)
    (d / "suite.yaml").write_text("id: open\nversion: 0\ntasks: [games/chess]\nseeds: [0]\n")
    definition = tmp_path / "task.yaml"                     # a package's working definition
    definition.write_text(suites.find_task("games/chess").source.read_text())
    monkeypatch.setattr(suites, "_definition", lambda task_id: definition)
    before = load_suite(str(d)).content_hash
    definition.write_text(definition.read_text().replace("max_illegal: 3", "max_illegal: 4"))
    assert load_suite(str(d)).content_hash != before

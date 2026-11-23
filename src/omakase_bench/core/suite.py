"""Suites: published lists of tasks and seeds, and finding a task by its id.

A suite is a directory ``suites/<id>/v<N>/`` holding ``suite.yaml`` (task list and seeds) and the YAML of
its tasks as they were when the suite was published. Its identity is the SHA-256 over those files, the
predicate vocabulary and the code its tasks name (``generator``, ``goal``). ``suites/frozen.yaml`` records
the hash of every published suite; a published suite whose files or code changed is reported as not
intact. A suite under development may list a task by its id instead of holding a copy; it then follows
the task package's ``task.yaml`` and has no recorded hash.
"""
from __future__ import annotations

import hashlib
import importlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .task import Task, load_task

SUITES_DIR = Path(__file__).resolve().parent.parent / "suites"
FROZEN_FILE = SUITES_DIR / "frozen.yaml"


def _frozen(root: Path | None = None) -> Mapping[str, Any]:
    path = (root or SUITES_DIR) / "frozen.yaml"
    return (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}


def published_hashes(root: Path | None = None) -> Mapping[str, str]:
    """Suite reference -> the hash it was published with, from the ``frozen.yaml`` of a suites directory."""
    return dict(_frozen(root).get("hashes") or {})


@dataclass(frozen=True)
class Suite:
    id: str
    version: str
    tasks: tuple[Task, ...]
    seeds: tuple[int, ...]
    content_hash: str
    path: Path

    @property
    def published(self) -> bool:
        """Its hash is recorded in ``suites/frozen.yaml``."""
        return self.ref in published_hashes(self.path.parent.parent)

    @property
    def intact(self) -> bool:
        """A published suite still has the hash it was published with; an unpublished one has none to keep."""
        return published_hashes(self.path.parent.parent).get(self.ref, self.content_hash) == self.content_hash

    @property
    def ref(self) -> str:
        return f"{self.id}@{self.version}"


def suite_hash(directory: Path) -> str:
    """SHA-256 over every YAML file of the suite directory (sorted by name) and the
    predicate vocabulary, so a changed task or a changed predicate is a different suite."""
    from . import predicates
    h = hashlib.sha256()
    modules: set[str] = set()
    for f in sorted(directory.glob("*.yaml")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
        modules.update(_code_refs(yaml.safe_load(f.read_text())))
    # a task the suite lists by id instead of holding a copy: its package's task.yaml
    for task in _listed(directory):
        if not (directory / f"{task}.yaml").exists():
            f = _definition(task)
            h.update(task.encode())
            h.update(f.read_bytes())
            modules.update(_code_refs(yaml.safe_load(f.read_text())))
    h.update(Path(predicates.__file__).read_bytes())
    # a suite whose tasks name generator or goal functions also depends on that code
    for mod in sorted(modules):
        h.update(mod.encode())
        h.update(Path(importlib.import_module(mod).__file__).read_bytes())
    return h.hexdigest()


def _listed(directory: Path) -> list[str]:
    return list(yaml.safe_load((directory / "suite.yaml").read_text())["tasks"])


def _definition(task_id: str) -> Path:
    """The ``task.yaml`` of the task package that defines ``task_id``."""
    from ..tasks import definitions
    for f in definitions():
        if yaml.safe_load(f.read_text()).get("id") == task_id:
            return f
    raise KeyError(f"no task {task_id!r}")


def _code_refs(raw: Any) -> set[str]:
    """Modules named by ``generator`` and by ``goal`` predicates anywhere in a task file."""
    out: set[str] = set()
    if isinstance(raw, Mapping):
        for k, v in raw.items():
            if k in ("generator", "goal") and isinstance(v, str) and ":" in v:
                out.add(v.partition(":")[0])
            else:
                out |= _code_refs(v)
    elif isinstance(raw, list):
        for v in raw:
            out |= _code_refs(v)
    return out


def list_suites() -> list[Path]:
    return sorted(p.parent for p in SUITES_DIR.glob("*/*/suite.yaml"))


def load_suite(ref: str) -> Suite:
    """``core@1`` or a directory containing ``suite.yaml``."""
    d = Path(ref)
    if not (d / "suite.yaml").exists():
        name, _, version = ref.partition("@")
        d = SUITES_DIR / name / f"v{version or '1'}"
    meta = yaml.safe_load((d / "suite.yaml").read_text())
    tasks = tuple(load_task(d / f"{t}.yaml" if (d / f"{t}.yaml").exists() else _definition(t)) for t in meta["tasks"])
    return Suite(meta["id"], str(meta["version"]), tasks, tuple(meta["seeds"]), suite_hash(d), d)


def task_files() -> list[Path]:
    """Every task definition: the copies in the suite directories, then the task packages' own."""
    from ..tasks import definitions
    return [f for f in sorted(SUITES_DIR.glob("*/*/*.yaml")) if f.name != "suite.yaml"] + definitions()


def find_task(ref: str) -> Task:
    """A YAML path, or a task id such as ``core/press_button``: the copy in the newest suite version that
    holds one, else the ``task.yaml`` of the task's package."""
    p = Path(ref)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return load_task(p)
    found = [f for f in task_files() if yaml.safe_load(f.read_text()).get("id") == ref or (f.stem == ref and f.name != "task.yaml")]
    if not found:
        raise KeyError(f"no task {ref!r}")
    return load_task(max(found, key=lambda f: (_suite_version(f.parent), str(f))))


def _suite_version(directory: Path) -> int:
    name = directory.name
    return int(name[1:]) if name[:1] == "v" and name[1:].isdigit() else -1


def freeze(ref: str, task_ids: list[str], seeds: list[int], description: str = "", root: Path | None = None) -> Suite:
    """Publish a suite: write ``<root>/<id>/v<N>/`` with ``suite.yaml`` and a copy of each task's YAML as it
    is now, and record the suite's hash in ``frozen.yaml``. An existing version is never overwritten."""
    root = root or SUITES_DIR
    name, _, version = ref.partition("@")
    if not name or not version.isdigit():
        raise ValueError(f"suite reference {ref!r}: expected <id>@<version number>")
    d = root / name / f"v{version}"
    if d.exists():
        raise FileExistsError(f"{d} exists: a published suite does not change; publish {name}@{int(version) + 1}")
    sources = {t: find_task(t).source for t in task_ids}
    stems = {t: t.split("/")[-1] for t in task_ids}
    if len(set(stems.values())) != len(stems):
        raise ValueError(f"task names {sorted(stems.values())} are not distinct")
    d.mkdir(parents=True)
    for t, source in sources.items():
        shutil.copyfile(source, d / f"{stems[t]}.yaml")
    (d / "suite.yaml").write_text(yaml.safe_dump(
        {"id": name, "version": int(version), "description": description, "tasks": list(stems.values()),
         "seeds": list(seeds)}, sort_keys=False, default_flow_style=None))
    suite = load_suite(str(d))
    frozen = root / "frozen.yaml"
    text = frozen.read_text() if frozen.exists() else ""
    if not re.search(r"^hashes:", text, re.M):
        text = text + ("" if not text or text.endswith("\n") else "\n") + "hashes:\n"
    # hashes is the last block of the file, so a new entry is one appended line
    frozen.write_text(text.rstrip("\n") + f"\n  {suite.ref}: {suite.content_hash}\n")
    return suite

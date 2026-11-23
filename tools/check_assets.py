"""Fail if a mesh file in the repository has no entry (or a stale sha256) in assets.yaml."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import yaml

MESH_SUFFIXES = {".stl", ".obj", ".dae", ".ply", ".glb", ".gltf", ".fbx", ".usd", ".usda", ".usdc", ".msh"}
SKIP_DIRS = {".git", ".venv", "runs", "__pycache__", "node_modules"}
REQUIRED = ("path", "sha256", "source", "commit", "license")


def check(root: Path) -> list[str]:
    manifest = yaml.safe_load((root / "assets.yaml").read_text()) or {}
    entries = {e["path"]: e for e in manifest.get("meshes") or []}
    problems = []
    for e in entries.values():
        missing = [k for k in REQUIRED if not e.get(k)]
        if missing:
            problems.append(f"{e.get('path')}: manifest entry lacks {', '.join(missing)}")
    for f in sorted(root.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in MESH_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in f.relative_to(root).parts):
            continue
        rel = f.relative_to(root).as_posix()
        entry = entries.get(rel)
        if entry is None:
            problems.append(f"{rel}: mesh file with no entry in assets.yaml")
        elif hashlib.sha256(f.read_bytes()).hexdigest() != entry["sha256"]:
            problems.append(f"{rel}: sha256 differs from assets.yaml")
    for rel in entries:
        if not (root / rel).is_file():
            problems.append(f"{rel}: listed in assets.yaml but missing")
    return problems


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    problems = check(root)
    for p in problems:
        print(p)
    print("assets: ok" if not problems else f"assets: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

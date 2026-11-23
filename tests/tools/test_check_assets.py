import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _checker():
    spec = importlib.util.spec_from_file_location("check_assets", ROOT / "tools" / "check_assets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_mesh_is_in_the_manifest():
    assert _checker().check(ROOT) == []


def test_unlisted_mesh_fails(tmp_path):
    (tmp_path / "assets.yaml").write_text("schema: omakase-bench/assets/1\nmeshes: []\n")
    (tmp_path / "part.stl").write_bytes(b"solid x\nendsolid x\n")
    problems = _checker().check(tmp_path)
    assert problems and "no entry" in problems[0]


def test_every_mesh_has_a_confirmed_licence(tmp_path):
    import yaml
    assert all(e["license"].startswith("Apache-2.0") for e in yaml.safe_load((ROOT / "assets.yaml").read_text())["meshes"])
    notice = (ROOT / "NOTICE").read_text()
    assert "Licensed under the Apache License" in notice and "AWAITING" not in notice.upper()
    assert "Apache License 2.0" in (ROOT / "src/omakase_bench/robots/d1/api/PROVENANCE.md").read_text()
    for path in ("src", "tools", "tests", "README.md", "docs", "assets.yaml"):
        for f in ([ROOT / path] if (ROOT / path).is_file() else sorted((ROOT / path).rglob("*"))):
            if f.is_file() and f.suffix in (".py", ".md", ".yaml") and f != Path(__file__).resolve():
                assert "rights confirmed" not in f.read_text(), f

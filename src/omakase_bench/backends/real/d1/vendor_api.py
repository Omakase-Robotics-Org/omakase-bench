"""Copy the subset of the D1 robot daemon's OpenAPI document that the bench speaks into the D1 preset.

    python -m omakase_bench.backends.real.d1.vendor_api SOURCE.json SOURCE_COMMIT

writes ``robots/d1/api/openapi.json``: the operations in ``OPERATIONS`` and the schemas in ``SCHEMAS`` with
every component they reference (schemas, responses, parameters), unchanged, and an ``x-vendored`` record of the
source and commit. ``tests/backends/real/d1/test_api.py`` checks the bench's requests against it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OPERATIONS = [
    ("post", "/v1/arm/{side}/mode"),
    ("post", "/v1/arm/{side}/move_joint"),
    ("post", "/v1/arm/{side}/move_joints"),
    ("post", "/v1/arm/move_joints_both"),
    ("post", "/v1/arm/{side}/estop"),
    ("post", "/v1/end_effectors/{side}/set"),
    ("post", "/v1/end_effectors/{side}/stop"),
    ("post", "/v1/slider/set_height"),
    ("post", "/v1/neck/cmd"),
    ("post", "/v1/chassis/cmd_vel"),
    ("get", "/v1/limits"),
    # measured state, read by the real backend
    ("get", "/v1/arm/feedback"),
    ("get", "/v1/end_effectors"),
    ("get", "/v1/slider/state"),
    ("get", "/v1/neck/state"),
    ("get", "/v1/chassis/amr/localization"),
    # the arm lease the real backend holds for a run
    ("get", "/v1/arm/lease"),
    ("post", "/v1/arm/lease"),
    ("delete", "/v1/arm/lease"),
]
#: schemas no vendored operation references: the frames of the WebSocket frontend, which carries the same
#: operations by their ``x-ws-method`` and is the real backend's transport for streamed motion
SCHEMAS = ["WsRequest", "WsResponse"]
OUT = Path(__file__).resolve().parents[3] / "robots/d1/api/openapi.json"


def refs(node, out: set[str]) -> None:
    if isinstance(node, dict):
        r = node.get("$ref")
        if isinstance(r, str) and r.startswith("#/components/"):
            out.add(r)
        for v in node.values():
            refs(v, out)
    elif isinstance(node, list):
        for v in node:
            refs(v, out)


def main(src: str, commit: str) -> int:
    spec = json.loads(Path(src).read_text())
    paths: dict = {}
    for method, path in OPERATIONS:
        paths.setdefault(path, {})[method] = spec["paths"][path][method]
    seen: set[str] = set()
    todo: set[str] = {f"#/components/schemas/{name}" for name in SCHEMAS}
    refs(paths, todo)
    components: dict = {}
    while todo:
        r = todo.pop()
        if r in seen:
            continue
        seen.add(r)
        _, _, group, name = r.split("/", 3)
        node = spec["components"][group][name]
        components.setdefault(group, {})[name] = node
        new: set[str] = set()
        refs(node, new)
        todo |= new - seen
    out = {"openapi": spec["openapi"], "info": spec["info"], "paths": paths,
           "components": {g: dict(sorted(v.items())) for g, v in sorted(components.items())},
           "x-vendored": {"source": "D1 robot daemon OpenAPI document (openapi/d1-firmwared.v1.json)",
                          "commit": commit, "operations": [f"{m.upper()} {p}" for m, p in OPERATIONS],
                          "schemas": SCHEMAS}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=False) + "\n")
    print(f"{OUT}: {len(OPERATIONS)} operations, {sum(len(v) for v in components.values())} components, "
          f"{OUT.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))

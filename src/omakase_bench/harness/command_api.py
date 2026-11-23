"""A robot's own command API, as far as the harness needs to know it.

Some robots accept joint-level commands over an API of their own (the D1's daemon speaks HTTP). A robot
description may ship that API's OpenAPI document and an adapter (``robot.yaml`` ``api``); the ``ee``
interface then lets a model write command lines that are requests of that API, the same method, path and
body a real robot accepts. This module holds what is common to any such API:

- the document: loading it, finding an operation, checking a request body against it (:func:`validate`);
- :class:`Request`, the response envelope and :class:`RobotCommandAPI`, which a simulated or a real
  endpoint implements;
- :class:`CommandAdapter`, which a robot description implements: how its command lines read, how they are
  described to a model, and what serves them in simulation;

The transport to a real robot is the real backend's (:mod:`omakase_bench.backends.real.http`).
"""
from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Protocol

Envelope = dict[str, Any]


# --- the document ----------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_spec(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _resolve(spec: Mapping, node: Mapping) -> Mapping:
    while "$ref" in node:
        _, _, group, name = node["$ref"].split("/", 3)
        node = spec["components"][group][name]
    return node


def operation(spec: Mapping, method: str, path: str) -> tuple[Mapping, dict[str, str]]:
    """The operation of ``path`` (with its path parameters filled in) and the parameter values."""
    for template, ops in spec["paths"].items():
        pattern = "^" + re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", template) + "$"
        m = re.match(pattern, path)
        if m and method.lower() in ops:
            return ops[method.lower()], m.groupdict()
    raise KeyError(f"{method.upper()} {path} is not an operation of the robot API document")


def _types(node: Mapping) -> list[str]:
    t = node.get("type")
    return [t] if isinstance(t, str) else list(t or [])


def check(spec: Mapping, schema: Mapping, value: Any, where: str = "body") -> list[str]:
    """Every way ``value`` breaks ``schema`` (the JSON Schema subset the document uses)."""
    s = _resolve(spec, schema)
    errors: list[str] = []
    for part in s.get("allOf", []):
        errors += check(spec, part, value, where)
    for key in ("anyOf", "oneOf"):
        if key in s and all(check(spec, alt, value, where) for alt in s[key]):
            errors.append(f"{where}: matches none of the allowed forms")
    if "const" in s and value != s["const"]:
        errors.append(f"{where}: must be {s['const']!r}")
    if "enum" in s and value not in s["enum"]:
        errors.append(f"{where}: {value!r} is not one of {s['enum']}")
    types = _types(s)
    if types:
        ok = {"null": value is None, "boolean": isinstance(value, bool),
              "integer": isinstance(value, int) and not isinstance(value, bool),
              "number": isinstance(value, (int, float)) and not isinstance(value, bool),
              "string": isinstance(value, str), "array": isinstance(value, list), "object": isinstance(value, dict)}
        if not any(ok.get(t, False) for t in types):
            errors.append(f"{where}: {value!r} is not of type {'/'.join(types)}")
            return errors
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in s and value < s["minimum"]:
            errors.append(f"{where}: {value} is below {s['minimum']}")
        if "maximum" in s and value > s["maximum"]:
            errors.append(f"{where}: {value} is above {s['maximum']}")
    if isinstance(value, list):
        if "minItems" in s and len(value) < s["minItems"]:
            errors.append(f"{where}: fewer than {s['minItems']} items")
        if "maxItems" in s and len(value) > s["maxItems"]:
            errors.append(f"{where}: more than {s['maxItems']} items")
        if "items" in s:
            for i, v in enumerate(value):
                errors += check(spec, s["items"], v, f"{where}[{i}]")
    if isinstance(value, dict):
        props = s.get("properties", {})
        for k in s.get("required", []):
            if k not in value:
                errors.append(f"{where}: missing {k!r}")
        for k, v in value.items():
            if k in props:
                errors += check(spec, props[k], v, f"{where}.{k}")
            elif s.get("additionalProperties") is False:
                errors.append(f"{where}: unknown field {k!r}")
    return errors


def skeleton(spec: Mapping, schema: Mapping) -> Any:
    """The smallest value that fits a schema of the document: required fields only, zeros and empty strings.
    What serves an operation in simulation fills in the fields it models."""
    s = _resolve(spec, schema)
    for key in ("oneOf", "anyOf"):
        if key in s:
            return skeleton(spec, next((o for o in s[key] if _resolve(spec, o).get("type") != "null"), s[key][0]))
    if "enum" in s:
        return s["enum"][0]
    kind = s.get("type")
    kind = next((k for k in kind if k != "null"), "null") if isinstance(kind, list) else kind
    if kind == "object":
        return {k: skeleton(spec, s["properties"][k]) for k in s.get("required", [])}
    if kind == "array":
        return [skeleton(spec, s["items"]) for _ in range(int(s.get("minItems", 0)))]
    return {"integer": 0, "number": 0.0, "boolean": False, "string": "", "null": None}.get(kind)


def response_data(spec: Mapping, method: str, path: str) -> Mapping:
    """The schema of the ``data`` field of an operation's success response."""
    op, _ = operation(spec, method, path)
    body = _resolve(spec, _resolve(spec, op["responses"]["200"])["content"]["application/json"]["schema"])
    return _resolve(spec, body["properties"]["data"])


def member(spec: Mapping, schema: Mapping, *keys: str) -> Mapping:
    """The sub-schema reached by ``keys`` (``"properties", "sides", "items"``), references followed."""
    for key in keys:
        schema = _resolve(spec, _resolve(spec, schema)[key])
    return schema


def validate(spec: Mapping, method: str, path: str, body: Any = None) -> list[str]:
    """Why a request does not fit the document ([] when it does): the operation, its path parameters and
    its JSON body."""
    try:
        op, params = operation(spec, method, path)
    except KeyError as exc:
        return [str(exc)]
    errors = []
    for p in op.get("parameters") or []:
        p = _resolve(spec, p)
        if p.get("in") == "path":
            errors += check(spec, p["schema"], params.get(p["name"]), f"path.{p['name']}")
    content = ((op.get("requestBody") or {}).get("content") or {}).get("application/json")
    if content is None:
        if body not in (None, {}):
            errors.append("this operation takes no body")
    elif body is None and (op.get("requestBody") or {}).get("required"):
        errors.append("this operation needs a body")
    elif body is not None:
        errors += check(spec, content["schema"], body)
    return errors


def _why(feedback: str) -> str:
    """The reason of a controller refusal line ('joints: refused at sample 3 of 9: ...' -> 'at sample 3 ...')."""
    text = feedback.split(": ", 1)[1] if ": " in feedback else feedback
    return re.sub(r"^refused[,:]?\s*", "", text)


def ok(data: Any = None) -> Envelope:
    return {"status": "ok", "data": data, "message": None}


def error(kind: str, message: str, http: int) -> Envelope:
    return {"status": "error", "data": None, "message": message, "kind": kind, "http_status": http}


# --- requests ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Request:
    method: str
    path: str
    body: Any = None
    #: simulated seconds to let pass after the request (the hold of a velocity command)
    then_wait_s: float = 0.0


class RobotCommandAPI(Protocol):
    """Anything that serves the daemon API: the simulated daemon or a client of a real one."""

    def request(self, method: str, path: str, body: Any = None) -> Envelope: ...

    def wait(self, seconds: float) -> None: ...


def summary(req: Request, env: Envelope) -> str:
    line = f"{req.method} {req.path} -> {env['status']}"
    if env["status"] != "ok":
        line += f" {env.get('http_status', '')} {env.get('kind', '')}: {env.get('message', '')}"
    elif env.get("note"):
        line += f" ({env['note']})"
    return line


# --- the robot's adapter ----------------------------------------------------------------------------------

class CommandAdapter(Protocol):
    """A robot's command lines (``robot.yaml`` ``api.adapter: module:factory``; ``factory(robot)`` builds one)."""

    def describe(self, robot, body, keys: Mapping[str, str]) -> tuple[str, list[str]]:
        """Model-facing text: how the lines name the arms (``keys``: arm -> row letter), and one description
        per command line that ``body`` (the task's ``ee.whole_body``) allows."""
        ...

    def run(self, controller, text: str) -> list[str]:
        """Carry out the command lines of ``text`` in order on the simulated robot behind ``controller``
        (an ``EEController``); one feedback line per request."""
        ...


def command_adapter(robot) -> CommandAdapter | None:
    """The adapter of a robot's own command API, or None when its robot.yaml has no ``api`` block."""
    if not robot.api:
        return None
    if "adapter" not in robot.api:
        raise ValueError(f"robot {robot.name}: its api block names no adapter (robot.yaml api.adapter: module:factory)")
    module, _, factory = str(robot.api["adapter"]).partition(":")
    return getattr(importlib.import_module(module), factory)(robot)

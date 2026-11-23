"""Robot API documentation for models, generated from the exposed functions'
signatures and docstrings: JSON-schema tool specs for the ``tools`` interface
and Python stubs for the ``code`` interface. Nothing is written by hand, so
the docs cannot drift from the code. Given a robot, the docs also state what
that robot's hardware rules out (``robot_notes``), such as sideways base motion, and given a
task's end-effector limits, what the gripper commands do under them."""
from __future__ import annotations

import inspect
import textwrap
import typing
from typing import TYPE_CHECKING, Any, Literal, Mapping, get_args, get_origin

from ..models.base import ToolSpec
from .api import Arm, ImageResult, exposed_functions

if TYPE_CHECKING:
    from ..robots.spec import RobotSpec

_JSON = {float: "number", int: "integer", str: "string", bool: "boolean"}


def _values(annotation: Any, robot: "RobotSpec | None") -> list:
    """The values of a ``Literal`` parameter; an arm parameter takes the robot's own arm names."""
    if annotation == Arm and robot is not None:
        return list(robot.arms)
    return list(get_args(annotation))


def _schema(annotation: Any, robot: "RobotSpec | None" = None) -> dict[str, Any]:
    if get_origin(annotation) is Literal:
        values = _values(annotation, robot)
        return {"type": _JSON[type(values[0])], "enum": values}
    if annotation in _JSON:
        return {"type": _JSON[annotation]}
    raise TypeError(f"unsupported API parameter type {annotation!r}")


def robot_notes(robot: "RobotSpec | None", gripper: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Per-function sentences describing ``robot``'s limits and the task's end-effector limits
    (``gripper``: arm -> limits), appended to the generic docs."""
    notes: dict[str, str] = {}
    if gripper:
        from .end_effectors import describe_limits
        text = describe_limits(gripper) + " A command beyond them is clamped and the result says so."
        notes["open_gripper"] = notes["close_gripper"] = text
    mb = robot.mobile_base if robot else None
    if mb is not None and mb.differential:
        notes["move_base"] = ("This robot's base is differential drive (two wheels): it cannot move sideways "
                              "or diagonally, so always pass left_m=0. Turn toward the target, drive "
                              "forward, then turn to the heading you need.")
    elif mb is not None:
        notes["move_base"] = "This robot's base is holonomic: it can translate in any direction."
    return notes


def _doc(fn, notes: dict[str, str] | None = None) -> str:
    text = " ".join(inspect.getdoc(fn).split())
    extra = (notes or {}).get(fn.__name__)
    return f"{text} {extra}" if extra else text


def tool_specs(robot: "RobotSpec | None" = None, gripper: Mapping[str, Any] | None = None) -> list[ToolSpec]:
    specs = []
    notes = robot_notes(robot, gripper)
    for name, fn in sorted(exposed_functions().items()):
        hints = typing.get_type_hints(fn)
        props, required = {}, []
        for p in list(inspect.signature(fn).parameters.values())[1:]:
            props[p.name] = _schema(hints[p.name], robot)
            if p.default is inspect.Parameter.empty:
                required.append(p.name)
            else:
                props[p.name]["default"] = p.default
        specs.append(ToolSpec(name, _doc(fn, notes), {"type": "object", "properties": props, "required": required,
                                               "additionalProperties": False}))
    return specs


def _annotation_text(a: Any, robot: "RobotSpec | None" = None) -> str:
    if get_origin(a) is Literal:
        return "Literal[" + ", ".join(repr(v) for v in _values(a, robot)) + "]"
    if a is ImageResult:
        return "dict"
    return getattr(a, "__name__", str(a))


def python_stubs(robot: "RobotSpec | None" = None, gripper: Mapping[str, Any] | None = None) -> str:
    """The robot API as Python signatures with docstrings, as shown to code-writing models."""
    out = []
    notes = robot_notes(robot, gripper)
    for name, fn in sorted(exposed_functions().items()):
        hints = typing.get_type_hints(fn)
        params = []
        for p in list(inspect.signature(fn).parameters.values())[1:]:
            text = f"{p.name}: {_annotation_text(hints[p.name], robot)}"
            if p.default is not inspect.Parameter.empty:
                text += f" = {p.default!r}"
            params.append(text)
        ret = _annotation_text(hints.get("return", Any))
        body = textwrap.indent(textwrap.fill(_doc(fn, notes), 92), "    ")
        out.append(f"def {name}({', '.join(params)}) -> {ret}:\n    \"\"\"\n{body}\n    \"\"\"")
    return "\n\n".join(out)


def coerce_arguments(name: str, arguments: dict[str, Any], robot: "RobotSpec | None" = None) -> dict[str, Any]:
    """Check a model's call against the signature (and ``robot``'s arm names); raise ValueError with a
    readable reason."""
    fns = exposed_functions()
    if name not in fns:
        raise ValueError(f"unknown function {name!r}; available: {', '.join(sorted(fns))}")
    fn = fns[name]
    hints = typing.get_type_hints(fn)
    params = {p.name: p for p in list(inspect.signature(fn).parameters.values())[1:]}
    unknown = set(arguments) - set(params)
    if unknown:
        raise ValueError(f"{name}: unknown argument(s) {', '.join(sorted(unknown))}")
    out = {}
    for pname, p in params.items():
        if pname not in arguments:
            if p.default is inspect.Parameter.empty:
                raise ValueError(f"{name}: missing argument {pname!r}")
            continue
        v, hint = arguments[pname], hints[pname]
        if get_origin(hint) is Literal:
            if v not in _values(hint, robot):
                raise ValueError(f"{name}: {pname} must be one of {_values(hint, robot)}, got {v!r}")
        elif hint is float:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"{name}: {pname} must be a number, got {v!r}")
            v = float(v)
        elif hint is int:
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"{name}: {pname} must be an integer, got {v!r}")
        elif hint is str and not isinstance(v, str):
            raise ValueError(f"{name}: {pname} must be a string, got {v!r}")
        out[pname] = v
    return out

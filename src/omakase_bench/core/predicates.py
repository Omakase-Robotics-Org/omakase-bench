"""The predicate vocabulary tasks are scored with.

A predicate is a YAML mapping with one key, evaluated on an
:class:`OracleSnapshot` (simulator ground truth). It never sees what a model
said. Vocabulary:

=====================  ==========================================================
``all: [p, ...]``       every sub-predicate holds
``any: [p, ...]``       at least one holds
``not: p``              negation
``touched: [a, b]``     a and b have been in contact since reset; ``b`` may be
                        ``any_hand``, ``<arm>_hand`` (``left_hand``) or ``robot``
``touching: [a, b]``    a and b are in contact now
``held: obj``           both jaws of one hand touch obj
``not_held: obj``       no hand holds obj
``inside: [obj, c]``    obj's centre is within container c's footprint (shrunk by
                        half the object) and below its rim, and obj is at rest
``height_above: [obj, ref, dz]``  obj's bottom is at least dz above ref's top
``displaced_lt: [obj, d]``  obj moved less than d metres from its reset position
``distance_lt: [a, b, d]``  centres (objects, ``left_tcp``/``right_tcp``) closer than d
``hands_clear_of: [obj, d]``  every TCP is at least d metres from obj's centre
``upright: [obj, deg]`` obj's z axis within deg of vertical
``pressed: button``     the button has been pressed since reset
``door_open: [door, f]``  the door or drawer is at least fraction f open
``base_in_region: r``   every corner of the robot footprint is inside region r (xy)
``in_region: [obj, r]`` obj's centre is inside region r
``python: mod:func``    ``func(snapshot, args)`` returns bool
=====================  ==========================================================
"""
from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from .backend import OracleSnapshot


@dataclass(frozen=True)
class Verdict:
    ok: bool
    evidence: str


class PredicateError(ValueError):
    pass


_REGISTRY: dict[str, Callable[[OracleSnapshot, Any, Mapping[str, Any]], Verdict]] = {}


def predicate(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def vocabulary() -> list[str]:
    return sorted(_REGISTRY)


def evaluate(pred: Mapping[str, Any], snap: OracleSnapshot, context: Mapping[str, Any] | None = None) -> Verdict:
    if not isinstance(pred, Mapping) or len(pred) != 1:
        raise PredicateError(f"a predicate is a one-key mapping, got {pred!r}")
    (name, args), = pred.items()
    fn = _REGISTRY.get(name)
    if fn is None:
        raise PredicateError(f"unknown predicate {name!r}; vocabulary: {', '.join(vocabulary())}")
    return fn(snap, args, context or {})


def validate(pred: Any) -> None:
    """Raise :class:`PredicateError` for an unknown name anywhere in a predicate tree."""
    if not isinstance(pred, Mapping) or len(pred) != 1:
        raise PredicateError(f"a predicate is a one-key mapping, got {pred!r}")
    (name, args), = pred.items()
    if name not in _REGISTRY:
        raise PredicateError(f"unknown predicate {name!r}")
    if name in ("all", "any"):
        for p in args:
            validate(p)
    elif name == "not":
        validate(args)


def _obj(snap: OracleSnapshot, name: str):
    if name not in snap.objects:
        raise PredicateError(f"no object {name!r} in the scene")
    return snap.objects[name]


def _point(snap: OracleSnapshot, name: str) -> np.ndarray:
    if name.endswith("_tcp") and name[:-4] in snap.tcp:
        return np.array(snap.tcp[name[:-4]])
    if name in snap.objects:
        return np.array(snap.objects[name].pos)
    if name in snap.articulated:
        return np.array(snap.articulated[name].pos)
    raise PredicateError(f"unknown point {name!r}")


def _half_height(o) -> float:
    if o.kind == "fixture" or len(o.size) == 3:
        return o.size[2] / 2 if len(o.size) == 3 else o.size[0]
    if len(o.size) == 2:
        return o.size[1] / 2
    return o.size[0]


def _yaw(quat) -> float:
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


@predicate("all")
def _all(snap, args, ctx):
    vs = [evaluate(p, snap, ctx) for p in args]
    return Verdict(all(v.ok for v in vs), "; ".join(v.evidence for v in vs))


@predicate("any")
def _any(snap, args, ctx):
    vs = [evaluate(p, snap, ctx) for p in args]
    return Verdict(any(v.ok for v in vs), " | ".join(v.evidence for v in vs))


@predicate("not")
def _not(snap, args, ctx):
    v = evaluate(args, snap, ctx)
    return Verdict(not v.ok, f"not({v.evidence})")


def _contact(pairs, a: str, b: str) -> bool:
    if b == "any_hand":
        return any(p == a and q.endswith("_hand") for p, q in pairs)        # whatever the robot's arms are called
    return (a, b) in pairs


@predicate("touched")
def _touched(snap, args, ctx):
    a, b = args
    ok = _contact(snap.contacts_ever, a, b)
    return Verdict(ok, f"{a} {'has' if ok else 'has not'} touched {b}")


@predicate("touching")
def _touching(snap, args, ctx):
    a, b = args
    ok = _contact(snap.contacts, a, b)
    return Verdict(ok, f"{a} {'is' if ok else 'is not'} touching {b}")


@predicate("held")
def _held(snap, args, ctx):
    hand = snap.held_by.get(args)
    return Verdict(hand is not None, f"{args} held by {hand}" if hand else f"{args} not held")


@predicate("not_held")
def _not_held(snap, args, ctx):
    hand = snap.held_by.get(args)
    return Verdict(hand is None, f"{args} held by {hand}" if hand else f"{args} not held")


@predicate("inside")
def _inside(snap, args, ctx):
    name, cname = args
    o, c = _obj(snap, name), _obj(snap, cname)
    rel = np.array(o.pos) - np.array(c.pos)
    yaw = _yaw(c.quat)
    lx = math.cos(yaw) * rel[0] + math.sin(yaw) * rel[1]
    ly = -math.sin(yaw) * rel[0] + math.cos(yaw) * rel[1]
    margin = 0.5 * min(o.size[:2]) if len(o.size) >= 2 else o.size[0]
    hx, hy, hz = c.size[0] / 2 - margin, c.size[1] / 2 - margin, c.size[2] / 2
    within = abs(lx) <= hx and abs(ly) <= hy and -hz - 0.01 <= rel[2] <= hz + 0.02
    still = float(np.linalg.norm(o.lin_vel)) < 0.05
    ok = within and still
    return Verdict(ok, f"{name} at ({lx:+.3f}, {ly:+.3f}, {rel[2]:+.3f}) m in {cname} frame"
                   f" (limits ±{hx:.3f}, ±{hy:.3f}), {'at rest' if still else 'moving'}")


@predicate("height_above")
def _height_above(snap, args, ctx):
    name, ref, dz = args
    o = _obj(snap, name)
    bottom = o.pos[2] - _half_height(o)
    if isinstance(ref, (int, float)):
        top = float(ref)
    else:
        r = _obj(snap, ref)
        top = r.pos[2] + _half_height(r)
    ok = bottom - top >= float(dz)
    return Verdict(ok, f"{name} bottom {bottom - top:+.3f} m above {ref}")


@predicate("displaced_lt")
def _displaced(snap, args, ctx):
    name, d = args
    moved = float(np.linalg.norm(np.array(_obj(snap, name).pos) - np.array(snap.initial[name])))
    return Verdict(moved < float(d), f"{name} moved {moved:.3f} m")


@predicate("distance_lt")
def _distance(snap, args, ctx):
    a, b, d = args
    dist = float(np.linalg.norm(_point(snap, a) - _point(snap, b)))
    return Verdict(dist < float(d), f"|{a} - {b}| = {dist:.3f} m")


@predicate("hands_clear_of")
def _clear(snap, args, ctx):
    name, d = args
    p = _point(snap, name)
    dists = {arm: float(np.linalg.norm(np.array(t) - p)) for arm, t in snap.tcp.items()}
    ok = all(v >= float(d) for v in dists.values())
    return Verdict(ok, "tcp distances " + ", ".join(f"{k} {v:.3f} m" for k, v in dists.items()))


@predicate("upright")
def _upright(snap, args, ctx):
    name, deg = (args if isinstance(args, list) else (args, 15))
    w, x, y, z = _obj(snap, name).quat
    zz = 1 - 2 * (x * x + y * y)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, zz))))
    return Verdict(tilt <= float(deg), f"{name} tilted {tilt:.1f} deg")


@predicate("pressed")
def _pressed(snap, args, ctx):
    a = snap.articulated.get(args)
    if a is None:
        raise PredicateError(f"no articulated object {args!r}")
    return Verdict(a.pressed_ever, f"{args} {'was' if a.pressed_ever else 'was not'} pressed "
                                   f"(travel now {a.q * 1000:.1f} mm)")


@predicate("door_open")
def _door_open(snap, args, ctx):
    name, frac = args
    a = snap.articulated[name]
    return Verdict(a.open_fraction >= float(frac), f"{name} {a.open_fraction * 100:.0f}% open")


def footprint_world(snap: OracleSnapshot, footprint) -> np.ndarray:
    x, y, yaw = snap.base
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[x + c * px - s * py, y + s * px + c * py] for px, py in footprint])


@predicate("base_in_region")
def _base_in_region(snap, args, ctx):
    region = snap.regions[args]
    corners = footprint_world(snap, ctx.get("footprint") or ((0.0, 0.0),))
    ok = all(region.lo[0] <= cx <= region.hi[0] and region.lo[1] <= cy <= region.hi[1] for cx, cy in corners)
    return Verdict(ok, f"base at ({snap.base[0]:.2f}, {snap.base[1]:.2f}) "
                       f"{'inside' if ok else 'not inside'} {args}")


@predicate("in_region")
def _in_region(snap, args, ctx):
    name, r = args
    ok = snap.regions[r].contains(_obj(snap, name).pos)
    return Verdict(ok, f"{name} {'in' if ok else 'not in'} {r}")


@predicate("python")
def _python(snap, args, ctx):
    ref = args if isinstance(args, str) else args["fn"]
    mod, _, fn = ref.partition(":")
    out = getattr(importlib.import_module(mod), fn)(snap, args if isinstance(args, Mapping) else {})
    return out if isinstance(out, Verdict) else Verdict(bool(out), f"{ref} -> {out}")

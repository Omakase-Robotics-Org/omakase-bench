"""The ``goal`` predicate: success checks that depend on what the episode asked for.

``goal: module:function`` calls ``function(snapshot, values, params)`` with the
episode's values (from the task's variation and generator) and the task's YAML
``params``; it returns a :class:`Verdict` or a bool. It lives beside the frozen
vocabulary in ``predicates.py`` so that suites which do not use it keep their
content hash; ``suite_hash`` adds the named module for suites that do.
"""
from __future__ import annotations

from .predicates import Verdict, predicate
from .task import resolve_ref


@predicate("goal")
def _goal(snap, args, ctx):
    out = resolve_ref(args)(snap, ctx.get("values", {}), ctx.get("params", {}))
    return out if isinstance(out, Verdict) else Verdict(bool(out), f"{args} -> {out}")

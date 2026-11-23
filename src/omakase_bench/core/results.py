"""``results.json``: one self-contained record of a run.

Schema ``omakase-bench/results/1``::

    schema, run_id, created, stopped
    provenance: bench version and git commit, suite (id, version, content hash,
                official), backend info, robot (name, sha256), time mode, host
    subjects:   [SubjectCard]
    episodes:   [per-episode record, see runner.run_episode]
    summary:    [{subject, task, n, successes, success_rate, ci95, mean_credit,
                  usd, tokens, median_success_sim_seconds}]
    overall:    [{subject, n, successes, success_rate, ci95, task_mean_rate, usd}]
    per_model:  multi-robot episodes only: [{subject, robots, episodes, usd, tokens, model_calls,
                api_calls}] per subject that drove a robot, summed over its robots and episodes
"""
from __future__ import annotations

import dataclasses
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from .. import __version__
from .stats import wilson

SCHEMA = "omakase-bench/results/1"


def _git_commit() -> str:
    try:
        root = Path(__file__).resolve().parents[3]
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=5)
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "") if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def summarize(episodes: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for e in episodes:
        groups.setdefault((e["subject"], e["task"]), []).append(e)
    summary = []
    for (subj, task), eps in groups.items():
        n = len(eps)
        k = sum(1 for e in eps if e["success"])
        times = [e["success_sim_seconds"] or e["sim_seconds"] for e in eps if e["success"]]
        lo, hi = wilson(k, n)
        summary.append({
            "subject": subj, "task": task, "n": n, "successes": k, "success_rate": round(k / n, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "mean_credit": round(sum(e["credit"] for e in eps) / n, 4),
            "usd": round(sum(e["usage"]["usd"] for e in eps), 4),
            "tokens": sum(e["usage"]["input_tokens"] + e["usage"]["output_tokens"] for e in eps),
            "model_calls": sum(e["usage"]["model_calls"] for e in eps),
            "median_success_sim_seconds": round(statistics.median(times), 2) if times else None,
            "exit_reasons": _count(e["exit_reason"] for e in eps),
        })
    overall = []
    for subj in dict.fromkeys(e["subject"] for e in episodes):
        rows = [s for s in summary if s["subject"] == subj]
        n = sum(r["n"] for r in rows)
        k = sum(r["successes"] for r in rows)
        lo, hi = wilson(k, n)
        overall.append({"subject": subj, "n": n, "successes": k, "success_rate": round(k / n, 4),
                        "ci95": [round(lo, 4), round(hi, 4)],
                        "task_mean_rate": round(sum(r["success_rate"] for r in rows) / len(rows), 4),
                        "usd": round(sum(r["usd"] for r in rows), 4), "tasks": len(rows)})
    return summary, overall


def per_model(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cost and calls of each subject that drove a robot of a multi-robot episode."""
    rows: dict[str, dict[str, Any]] = {}
    for e in episodes:
        for rid, r in (e.get("robots") or {}).items():
            row = rows.setdefault(r["subject"], {"subject": r["subject"], "robots": 0, "episodes": set(), "usd": 0.0,
                                                 "tokens": 0, "model_calls": 0, "api_calls": 0})
            u = r["usage"]
            row["robots"] += 1
            row["episodes"].add((e["subject"], e["task"], e["seed"]))
            row["usd"] += u["usd"]
            row["tokens"] += u["input_tokens"] + u["output_tokens"]
            row["model_calls"] += u["model_calls"]
            row["api_calls"] += u["api_calls"]
    return [{**r, "episodes": len(r["episodes"]), "usd": round(r["usd"], 5)} for r in rows.values()]


def _count(items) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return out


def build_results(*, episodes, subjects, tasks, seeds, cfg, backend_info, started, stopped) -> dict[str, Any]:
    summary, overall = summarize(episodes) if episodes else ([], [])
    models = per_model(episodes)
    suite = cfg.suite
    # official: the suite's own task versions on the suite's seeds; another version of a task under the
    # same id (a newer suite's) is not the published suite
    own = {(t.id, t.version) for t in suite.tasks} if suite is not None else set()
    official_seeds = (suite is not None and tuple(seeds) == tuple(suite.seeds)
                      and all((t.id, t.version) in own for t in tasks))
    robots = {t.robot for t in tasks}
    from ..robots.spec import load_robot
    return {
        "schema": SCHEMA,
        "run_id": cfg.out_dir.name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "stopped": stopped,
        "provenance": {
            "omakase_bench": __version__, "git_commit": _git_commit(),
            "suite": ({"id": suite.id, "version": suite.version, "content_hash": suite.content_hash,
                       "official": official_seeds and cfg.time_mode == "lockstep"} if suite else None),
            "tasks": {t.id: t.version for t in tasks}, "seeds": list(seeds),
            "backend": dataclasses.asdict(backend_info) | {"capabilities": sorted(backend_info.capabilities)},
            "robots": {r: load_robot(r).sha256() for r in robots},
            "time_mode": cfg.time_mode,
            "host": {"platform": platform.platform(), "python": platform.python_version()},
            "run_budget": {"max_model_calls": cfg.run_budget.max_model_calls, "max_usd": cfg.run_budget.max_usd,
                           "model_calls": cfg.run_budget.model_calls, "usd": round(cfg.run_budget.usd, 4)},
        },
        "subjects": [dataclasses.asdict(s.card()) for s in subjects],
        "episodes": episodes,
        "summary": summary,
        "overall": overall,
        **({"per_model": models} if models else {}),
    }


def write_results(out_dir: Path, results: dict[str, Any]) -> None:
    tmp = out_dir / "results.json.tmp"
    tmp.write_text(json.dumps(results, indent=1, default=str))
    tmp.replace(out_dir / "results.json")
    from ..report.html import write_report
    write_report(out_dir, results)


def load_results(path: Path) -> dict[str, Any]:
    p = Path(path)
    if p.is_dir():
        p = p / "results.json"
    data = json.loads(p.read_text())
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{p}: not an omakase-bench results file (schema {data.get('schema')!r})")
    data["_dir"] = str(p.parent)
    return data

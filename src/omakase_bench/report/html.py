"""Static HTML: the per-run report and the multi-run leaderboard. No external assets."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Iterable

STYLE = """
:root{--bg:#fbfbfa;--fg:#1d1d1f;--muted:#6b6b70;--line:#e3e3e0;--card:#fff;--ok:#1f8a4c;--bad:#c0392b;--bar:#3b6fd8}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececee;--muted:#9a9aa2;--line:#2c2c31;--card:#1c1c20;--ok:#3cc173;--bad:#ef6b5b;--bar:#6b9bff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1200px;margin:0 auto;padding:24px 16px 64px}h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:32px 0 10px}
.muted{color:var(--muted)}table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;font-variant-numeric:tabular-nums}th{font-weight:600;font-size:13px;color:var(--muted)}
.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}
.ci{position:relative;height:8px;background:var(--line);border-radius:4px;min-width:120px;margin-top:6px}
.ci span{position:absolute;top:0;height:8px;background:var(--bar);opacity:.35;border-radius:4px}.ci b{position:absolute;top:-3px;width:3px;height:14px;background:var(--bar)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px}
.ep{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px}.ep video{width:100%;border-radius:6px;background:#000}
code,pre{font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}pre{white-space:pre-wrap;background:var(--card);border:1px solid var(--line);padding:10px;border-radius:6px;max-height:340px;overflow:auto}
details{margin-top:6px}.kv{display:grid;grid-template-columns:max-content 1fr;gap:2px 14px;font-size:13px}
.wrap{overflow-x:auto}
"""


def _e(x: Any) -> str:
    return html.escape(str(x))


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _ci(rate: float, ci: list[float]) -> str:
    lo, hi = ci
    return (f'<div class="ci" title="Wilson 95% CI {_pct(lo)}–{_pct(hi)}"><span style="left:{100 * lo:.1f}%;'
            f'width:{100 * (hi - lo):.1f}%"></span><b style="left:calc({100 * rate:.1f}% - 1px)"></b></div>')


def _page(title: str, body: str) -> str:
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport "
            f"content='width=device-width,initial-scale=1'><title>{_e(title)}</title><style>{STYLE}</style></head>"
            f"<body><main>{body}</main></body></html>")


def render_report(results: dict[str, Any]) -> str:
    p = results["provenance"]
    suite = p.get("suite") or {}
    b = p["backend"]
    head = (f"<h1>omakase-bench run {_e(results['run_id'])}</h1><p class=muted>{_e(results['created'])} · "
            f"backend {_e(b['name'])} ({_e(b['engine'])}) · time mode {_e(p['time_mode'])}"
            + (f" · suite {_e(suite.get('id'))}@{_e(suite.get('version'))} "
               f"{'official' if suite.get('official') else 'unofficial seeds or timing'}" if suite else "")
            + (f" · <span class=bad>stopped: {_e(results['stopped'])}</span>" if results.get("stopped") else "")
            + "</p>")
    rows = "".join(
        f"<tr><td>{_e(o['subject'])}</td><td>{o['successes']}/{o['n']}</td><td>{_pct(o['success_rate'])}"
        f"{_ci(o['success_rate'], o['ci95'])}</td><td>{_pct(o['task_mean_rate'])}</td><td>${o['usd']:.2f}</td></tr>"
        for o in results["overall"])
    overall = (f"<h2>Subjects</h2><div class=wrap><table><tr><th>subject</th><th>episodes</th><th>success (Wilson 95% CI)"
               f"</th><th>mean over tasks</th><th>cost</th></tr>{rows}</table></div>")
    rows = "".join(
        f"<tr><td>{_e(s['subject'])}</td><td>{_e(s['task'])}</td><td>{s['successes']}/{s['n']}</td>"
        f"<td>{_pct(s['success_rate'])}{_ci(s['success_rate'], s['ci95'])}</td><td>{s['mean_credit']:.2f}</td>"
        f"<td>{s['median_success_sim_seconds'] if s['median_success_sim_seconds'] is not None else '–'}</td>"
        f"<td>{s['model_calls']}</td><td>{s['tokens']:,}</td><td>${s['usd']:.3f}</td>"
        f"<td class=muted>{_e(', '.join(f'{k} ×{v}' for k, v in s['exit_reasons'].items()))}</td></tr>"
        for s in results["summary"])
    per_task = (f"<h2>Per task</h2><div class=wrap><table><tr><th>subject</th><th>task</th><th>n</th><th>success</th>"
                f"<th>credit</th><th>median sim s to success</th><th>model calls</th><th>tokens</th><th>cost</th>"
                f"<th>exits</th></tr>{rows}</table></div>")
    cards = []
    for e in results["episodes"]:
        art = e.get("artifacts", {})
        video = f"<video src='{_e(art['video'])}' controls muted loop playsinline preload=metadata></video>" if "video" in art else ""
        program = ""
        if "program" in art:
            try:
                code = (Path(results.get("_dir", ".")) / art["program"]).read_text()
            except OSError:
                code = ""
            program = f"<details><summary>program</summary><pre>{_e(code)}</pre></details>"
        subgoals = " · ".join(f"{'✓' if g['ok'] else '✗'} {_e(g['name'])}" for g in e["subgoals"])
        cards.append(
            f"<div class=ep>{video}<div><b>{_e(e['subject'])}</b> · {_e(e['task'])} · seed {e['seed']} · "
            f"<span class={'ok' if e['success'] else 'bad'}>{'success' if e['success'] else 'fail'}</span></div>"
            f"<div class=muted>{_e(e['instruction'])}</div><div class=kv><span>exit</span><span>{_e(e['exit_reason'])}"
            f"{(' – ' + _e(e['exit_detail'])) if e['exit_detail'] else ''}</span><span>evidence</span><span>{_e(e['evidence'])}"
            f"</span><span>sub-goals</span><span>{subgoals}</span><span>model claim</span><span>{_e(e['model_claim'] or '–')}"
            f"</span><span>usage</span><span>{e['usage']['api_calls']} API calls, {e['usage']['model_calls']} model calls, "
            f"${e['usage']['usd']:.3f}, sim {e['sim_seconds']} s</span></div>{program}"
            f"<div class=muted><a href='{_e(art.get('trace', ''))}'>trace.jsonl</a></div></div>")
    episodes = f"<h2>Episodes</h2><div class=grid>{''.join(cards)}</div>"
    prov = f"<h2>Provenance</h2><pre>{_e(json.dumps(p, indent=1))}</pre>"
    return _page(f"omakase-bench {results['run_id']}", head + overall + per_task + episodes + prov)


def write_report(out_dir: Path, results: dict[str, Any]) -> Path:
    path = Path(out_dir) / "report.html"
    path.write_text(render_report({**results, "_dir": str(out_dir)}))
    return path


def render_leaderboard(runs: Iterable[dict[str, Any]], *, title: str = "omakase-bench leaderboard") -> str:
    """One row per (subject, backend, suite hash, time mode); each run's episodes pooled per row."""
    from ..core.stats import wilson
    rows: dict[tuple, dict[str, Any]] = {}
    for r in runs:
        p = r["provenance"]
        suite = p.get("suite") or {}
        key_suite = f"{suite.get('id')}@{suite.get('version')}" if suite else "+".join(sorted(p["tasks"]))
        for e in r["episodes"]:
            key = (e["subject"], p["backend"]["name"], key_suite, (suite.get("content_hash") or "")[:12], p["time_mode"])
            row = rows.setdefault(key, {"n": 0, "k": 0, "usd": 0.0, "tasks": {}, "official": bool(suite.get("official"))})
            row["n"] += 1
            row["k"] += int(e["success"])
            row["usd"] += e["usage"]["usd"]
            t = row["tasks"].setdefault(e["task"], [0, 0])
            t[0] += int(e["success"])
            t[1] += 1
    tasks = sorted({t for row in rows.values() for t in row["tasks"]})
    ordered = sorted(rows.items(), key=lambda kv: -kv[1]["k"] / max(kv[1]["n"], 1))
    head = "".join(f"<th>{_e(t.split('/')[-1])}</th>" for t in tasks)
    body = []
    for rank, ((subj, backend, suite, h, mode), row) in enumerate(ordered, 1):
        rate = row["k"] / row["n"]
        lo, hi = wilson(row["k"], row["n"])
        cells = "".join(f"<td>{row['tasks'][t][0]}/{row['tasks'][t][1]}</td>" if t in row["tasks"] else "<td>–</td>"
                        for t in tasks)
        body.append(f"<tr><td>{rank}</td><td><b>{_e(subj)}</b></td><td>{_pct(rate)}{_ci(rate, [lo, hi])}</td>{cells}"
                    f"<td>{_e(backend)}</td><td>{_e(suite)} <span class=muted>{_e(h)}</span>"
                    f"{'' if row['official'] else ' <span class=muted>(unofficial)</span>'}</td><td>{_e(mode)}</td>"
                    f"<td>${row['usd']:.2f}</td></tr>")
    table = (f"<div class=wrap><table><tr><th>#</th><th>subject</th><th>success (Wilson 95% CI)</th>{head}<th>backend</th>"
             f"<th>suite</th><th>timing</th><th>cost</th></tr>{''.join(body)}</table></div>")
    note = ("<p class=muted>Success is measured by the simulator's ground truth, never by a model's claim. "
            "Official scores use the suite's frozen seeds in lockstep timing; scores on different backends or "
            "suite hashes are not comparable.</p>")
    return _page(title, f"<h1>{_e(title)}</h1>{note}{table}")

"""omakase-bench command line: list | doctor | run | report | compare | leaderboard | showcase | robot | suite,
and the commands task packages add (``chess``)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__


def _load_env_files(paths: list[str]) -> list[str]:
    """Read KEY=VALUE lines into the environment; return the names only (values are never shown)."""
    names = []
    for p in paths:
        for line in Path(os.path.expanduser(p)).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.removeprefix("export ").partition("=")
            os.environ[key.strip()] = value.strip().strip("'\"")
            names.append(key.strip())
    return names


def _seeds(text: str | None, default: tuple[int, ...]) -> list[int]:
    if not text:
        return list(default)
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    if not out:
        raise ValueError(f"--seeds {text!r} names no seed; give a list or an ascending range such as 0-4")
    return out


def _options(pairs: list[str]) -> dict[str, str]:
    out = {}
    for p in pairs or []:
        k, sep, v = p.partition("=")
        if not sep:
            raise SystemExit(f"-M expects key=value, got {p!r}")
        out[k] = v
    return out


def cmd_list(args) -> int:
    from .core import goal_predicates  # noqa: F401  (registers the ``goal`` predicate)
    from .core.predicates import validate
    from .core.suite import find_task, list_suites, load_suite, task_files
    from .models import providers
    from .robots.spec import PRESETS_DIR
    from .tasks import packages
    what = args.what
    ok = True
    if what in ("suites", "all"):
        for d in list_suites():
            s = load_suite(str(d))
            state = "open" if not s.published else "frozen" if s.intact else "CHANGED since it was published"
            ok = ok and s.intact
            print(f"suite {s.ref:12s} {len(s.tasks)} tasks, {len(s.seeds)} seeds, hash {s.content_hash[:12]}, {state}")
    if what in ("tasks", "all"):
        for f in task_files():
            t = find_task(str(f))
            if t.success or t.team is None:
                validate(t.success)
            for g in t.subgoals:
                validate(g.predicate)
            t.resolve(0)
            print(f"task  {t.id:24s} v{t.version:7s} {t.status:7s} {','.join(sorted(t.interfaces)):28s} {t.title}")
    if what in ("packages", "all"):
        for p in packages():
            print(f"package {p.name:36s} {', '.join(sorted(p.scripted)):60s} {p.summary}")
    if what in ("models", "all"):
        from .models.pricing import PRICES
        print("models: provider/model, providers: " + ", ".join(providers()))
        print("  scripted/oracle (interface scripted), stub/hold (interface chunks)")
        for k in PRICES:
            print(f"  {k}")
    if what in ("backends", "all"):
        print("backends: mujoco (CPU, default), isaac (remote Isaac Sim server over TCP), real (a robot over its own API; "
              "disabled unless -B enable=1 and OMAKASE_BENCH_ROBOT_ENABLE=1, no default host)")
    if what in ("robots", "all"):
        for d in sorted(PRESETS_DIR.iterdir()):
            if (d / "robot.yaml").exists():
                print(f"robot {d.name}")
    return 0 if ok else 1


def _probe_gl(gl: str) -> tuple[bool, str]:
    code = ("import mujoco;m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos=\"0 0 1\"/>"
            "<geom type=\"box\" size=\".1 .1 .1\"/></worldbody></mujoco>');d=mujoco.MjData(m);"
            "r=mujoco.Renderer(m,32,32);mujoco.mj_forward(m,d);r.update_scene(d);print(r.render().mean())")
    try:
        out = subprocess.run([sys.executable, "-c", code], env={**os.environ, "MUJOCO_GL": gl},
                             capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return False, "timed out"
    if out.returncode == 0:
        return True, "renders"
    return False, (out.stderr.strip().splitlines() or ["failed"])[-1][:120]


def cmd_doctor(args) -> int:
    import mujoco
    from .sandbox import sandbox_report
    ok = True
    print(f"omakase-bench {__version__}, python {sys.version.split()[0]}, mujoco {mujoco.__version__}")
    for gl in ("egl", "osmesa"):
        good, msg = _probe_gl(gl)
        print(f"  MUJOCO_GL={gl:7s} {'ok  ' if good else 'FAIL'} {msg}")
    try:
        import imageio_ffmpeg
        print(f"  ffmpeg        ok   {imageio_ffmpeg.get_ffmpeg_exe()}")
    except Exception as exc:   # video is optional; report and continue
        ok = False
        print(f"  ffmpeg        FAIL {exc}")
    print(f"  sandbox       {json.dumps(sandbox_report())}")
    import importlib.util
    for key, library in (("ANTHROPIC_API_KEY", "anthropic"), ("OPENAI_API_KEY", "openai")):
        client = "installed" if importlib.util.find_spec(library) else f"not installed (uv sync --extra {library})"
        print(f"  {key:18s} {'set' if os.environ.get(key) else 'not set'}; {library} client {client}")
    if args.backend == "isaac":
        from .backends.isaac.client import IsaacBackend
        try:
            b = IsaacBackend(**_options(args.backend_option))
            print(f"  isaac         {b.ping()}")
            b.close()
        except Exception as exc:   # report any connection problem as a failed check
            ok = False
            print(f"  isaac         FAIL {exc}")
    return 0 if ok else 1


def _run_cap_shortfalls(tasks, teams, subjects, max_calls: int | None, max_usd: float | None) -> list[str]:
    """Why the run-wide caps are below what a task says one episode may need (task YAML ``run_budget:
    {model_calls, usd}``), per task driven by a model; scripted subjects make no model calls."""
    out = []
    for task in tasks:
        need = task.raw.get("run_budget") or {}
        team = teams.get(task.id)
        drivers = list(team.members.values()) if team is not None else subjects
        if not need or all(getattr(s, "kind", "") == "scripted" for s in drivers):
            continue
        calls, usd = need.get("model_calls"), need.get("usd")
        if calls is not None and max_calls is not None and max_calls < int(calls):
            out.append(f"{task.id}: one episode may take {int(calls)} model calls; --max-calls is {max_calls}")
        if usd is not None and max_usd is not None and max_usd < float(usd):
            out.append(f"{task.id}: one episode may spend ${float(usd):g}; --max-usd is {max_usd:g}")
    return out


def cmd_run(args) -> int:
    from .core.runner import RunConfig, run
    from .harness.session import RunBudget
    from .core.suite import find_task, load_suite
    from .subjects import build_subject
    from .subjects.team import build_team, parse_binding
    if args.env_file:
        names = _load_env_files(args.env_file)
        print(f"loaded {', '.join(names)} from env file(s)")
    suite = load_suite(args.suite) if args.suite else None
    if args.task:
        # with a suite, an id names the suite's own version of the task, not the newest one
        own = {t.id: t for t in suite.tasks} if suite else {}
        tasks = [own.get(t) or find_task(t) for t in args.task.split(",")]
    elif suite:
        tasks = list(suite.tasks)
    else:
        raise SystemExit("give --task or --suite")
    if getattr(args, "robot", None):
        from .core.task import with_robot
        tasks = [with_robot(t, args.robot) for t in tasks]
    default_seeds = suite.seeds if suite else (0,)
    seeds = _seeds(args.seeds, default_seeds)
    if args.episodes is not None:
        if args.episodes < 1:
            raise ValueError("--episodes must be at least 1")
        seeds = seeds[: args.episodes]
    opts = _options(args.model_option)

    def make(spec: str, interface: str | None, robot: str = ""):
        """A subject; ``-M <robot>.<key>=v`` options apply to that robot's model only."""
        interface = interface or ("scripted" if spec.startswith("scripted/") else
                                  "chunks" if spec.startswith("stub/") else "tools")
        own = {k: v for k, v in opts.items() if "." not in k}
        own.update({k.split(".", 1)[1]: v for k, v in opts.items() if robot and k.startswith(robot + ".")})
        return build_subject(spec, interface, code_mode=args.code_mode, program_timeout=args.program_timeout,
                             model_options=own)

    subjects = []
    for spec in (args.model.split(",") if args.model else []):
        for itf in (args.interface or "").split(",") if args.interface else [None]:
            subjects.append(make(spec, itf))
    teams = {}
    bindings = dict((rid, (spec, itf)) for rid, spec, itf in map(parse_binding, getattr(args, "bind", None) or []))
    for task in tasks:
        if task.team is not None and (bindings or not subjects):
            default = None
            if args.model and len(args.model.split(",")) == 1:
                default = (args.model, args.interface)
            teams[task.id] = build_team(task, make, bindings, default)
    if not subjects and any(t.team is None for t in tasks):
        raise SystemExit("give --model for the single-robot tasks")
    for subj in [*subjects, *(m for t in teams.values() for m in t.members.values())]:
        price = getattr(getattr(subj, "model", None), "price", None)
        if price is not None and price.assumed:
            print(f"{subj.name}: no published price is recorded for this model; its cost and the spend caps use an "
                  f"assumed ${price.input:g} / ${price.output:g} per million tokens (-M price_in=... -M price_out=...)")
    short = _run_cap_shortfalls(tasks, teams, subjects, args.max_calls, args.max_usd)
    if short and not getattr(args, "allow_partial", False):
        raise SystemExit("the run-wide caps would stop a task before it can finish:\n  " + "\n  ".join(short)
                         + "\nraise them, or pass --allow-partial to run until a cap is reached "
                           "(reported as a harness stop)")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    label = (suite.ref if suite else tasks[0].id.split("/")[-1]).replace("/", "_").replace("@", "-")
    out = Path(args.out) if args.out else Path("runs") / f"{stamp}_{label}_{args.backend}"
    cfg = RunConfig(out_dir=out, backend=args.backend, backend_options=_options(args.backend_option),
                    time_mode=args.time_mode, video=not args.no_video,
                    run_budget=RunBudget(args.max_calls, args.max_usd), suite=suite)
    results = run(subjects, tasks, seeds, cfg, teams=teams)
    print(f"\nresults: {out / 'results.json'}\nreport:  {out / 'report.html'}")
    for o in results["overall"]:
        print(f"  {o['subject']:40s} {o['successes']}/{o['n']} = {100 * o['success_rate']:.0f}% "
              f"(95% CI {100 * o['ci95'][0]:.0f}-{100 * o['ci95'][1]:.0f}%)  ${o['usd']:.3f}")
    for m in results.get("per_model", []):
        print(f"  per model: {m['subject']:30s} {m['robots']} robot seats in {m['episodes']} episodes, "
              f"{m['model_calls']} model calls, ${m['usd']:.3f}")
    if results.get("stopped"):
        print(f"  run stopped early: {results['stopped']}")
    return 0


def cmd_report(args) -> int:
    from .core.results import load_results
    from .report.html import write_report
    r = load_results(Path(args.run))
    path = write_report(Path(r["_dir"]), r)
    print(path)
    return 0


def cmd_leaderboard(args) -> int:
    from .core.results import load_results
    from .report.html import render_leaderboard
    runs = [load_results(Path(p)) for p in args.runs]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_leaderboard(runs, title=args.title))
    print(out)
    return 0


def cmd_compare(args) -> int:
    from .core.results import load_results
    from .core.stats import mcnemar_exact
    from .report.html import render_leaderboard
    from .report.video import compose_comparison, read_video
    runs = [load_results(Path(p)) for p in args.runs]
    episodes = [(r, e) for r in runs for e in r["episodes"]]
    tasks = sorted({e["task"] for _, e in episodes})
    task = args.task or tasks[0]
    subjects = list(dict.fromkeys(e["subject"] for _, e in episodes if e["task"] == task))
    if args.subjects:
        subjects = [s for s in args.subjects.split(",") if s in subjects]
    # paired comparison on shared seeds
    by = {(e["subject"], e["seed"]): e["success"] for _, e in episodes if e["task"] == task}
    for i, a in enumerate(subjects):
        for b in subjects[i + 1:]:
            shared = sorted({s for (x, s) in by if x == a} & {s for (x, s) in by if x == b})
            if not shared:
                continue
            ab = sum(1 for s in shared if by[(a, s)] and not by[(b, s)])
            ba = sum(1 for s in shared if by[(b, s)] and not by[(a, s)])
            print(f"{a} vs {b} on {task}: {len(shared)} shared seeds, discordant {ab}/{ba}, "
                  f"exact McNemar p = {mcnemar_exact(ab, ba):.3f}")
    if args.video:
        seed = args.seed
        if seed is None:
            common = set.intersection(*[{s for (x, s) in by if x == subj} for subj in subjects])
            seed = min(common)
        tiles, instruction = [], ""
        for subj in subjects:
            r, e = next((r, e) for r, e in episodes if e["subject"] == subj and e["task"] == task and e["seed"] == seed)
            raw = Path(r["_dir"]) / Path(e["artifacts"]["video"]).with_name("raw.mp4")
            code = None
            if "program" in e["artifacts"]:
                code = (Path(r["_dir"]) / e["artifacts"]["program"]).read_text()
            tiles.append({"frames": read_video(raw), "label": subj, "success": e["success"], "code": code})
            instruction = e["instruction"]
        header = f"{task}  seed {seed}  ·  {runs[0]['provenance']['backend']['engine']}  ·  success judged by simulator ground truth"
        compose_comparison(Path(args.video), tiles, 10.0, instruction=instruction, header=header)
        print(args.video)
    if args.html:
        Path(args.html).write_text(render_leaderboard(runs, title="omakase-bench comparison"))
        print(args.html)
    return 0


def cmd_showcase(args) -> int:
    """Run each model on one task and seed, then write the side-by-side video."""
    out = Path(args.out)
    run_args = argparse.Namespace(
        model=args.model, interface=args.interface, code_mode=args.code_mode, program_timeout=120.0, task=args.task,
        suite=None, seeds=str(args.seed), episodes=None, backend=args.backend, backend_option=args.backend_option,
        time_mode="lockstep", model_option=args.model_option, max_calls=args.max_calls, max_usd=args.max_usd,
        env_file=args.env_file, out=str(out / "run"), no_video=False)
    cmd_run(run_args)
    video = out / f"{args.task.split('/')[-1]}_seed{args.seed}.mp4"
    cmd_compare(argparse.Namespace(runs=[str(out / "run")], task=args.task, seed=args.seed, subjects=None,
                                   video=str(video), html=None))
    return 0


def cmd_suite_freeze(args) -> int:
    from .core.suite import freeze
    suite = freeze(args.suite, args.tasks.split(","), _seeds(args.seeds, ()), args.description)
    print(f"published {suite.ref}: {len(suite.tasks)} tasks, {len(suite.seeds)} seeds, hash {suite.content_hash}\n{suite.path}")
    return 0


def cmd_robot(args) -> int:
    from .robots.check import main as check
    return check(args.robot, args.out)


def main(argv: list[str] | None = None) -> int:
    # before anything imports mujoco, which reads it once: the task packages below do
    os.environ.setdefault("MUJOCO_GL", "egl")
    ap = argparse.ArgumentParser(prog="omakase-bench", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="list suites, tasks, task packages, models, backends, robots")
    p.add_argument("what", nargs="?", default="all", choices=["all", "suites", "tasks", "packages", "models", "backends", "robots"])
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("doctor", help="check rendering, video, sandbox, keys and a backend")
    p.add_argument("--backend", default="mujoco", choices=["mujoco", "isaac", "real"])
    p.add_argument("-B", "--backend-option", action="append", default=[], help="backend option key=value")
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("run", help="run subjects on tasks and write results.json + report.html")
    p.add_argument("--bind", action="append", default=[],
                   help="multi-robot tasks: <robot>=<provider/model>[:<interface>], e.g. a=openai/gpt-6.2-astra:ee")
    p.add_argument("--model", help="comma list: anthropic/<model>, openai/<model>, mailbox/<dir>, "
                                                  "mock/<behaviour>, scripted/oracle, stub/hold")
    p.add_argument("--interface", help="tools | code | chunks | ee | scripted (comma list allowed)")
    p.add_argument("--code-mode", default="one-shot", choices=["one-shot", "iterative"])
    p.add_argument("--program-timeout", type=float, default=120.0)
    p.add_argument("--task", help="comma list of task ids or YAML paths")
    p.add_argument("--suite", help="suite reference, e.g. core@1")
    p.add_argument("--robot", help="run the single-robot tasks on this robot (a name under robots/ or a directory "
                                   "with robot.yaml) instead of the one each task names")
    p.add_argument("--seeds", help="e.g. 0-19 or 0,3,5 (default: the suite's frozen seeds)")
    p.add_argument("--episodes", type=int, help="use only the first N seeds")
    p.add_argument("--backend", default="mujoco", choices=["mujoco", "isaac", "real"])
    p.add_argument("-B", "--backend-option", action="append", default=[], help="backend option key=value")
    p.add_argument("--time-mode", default="lockstep", choices=["lockstep", "realtime"])
    p.add_argument("-M", "--model-option", action="append", default=[],
                   help="model option key=value; <robot>.key=value for one robot of a multi-robot task")
    p.add_argument("--max-calls", type=int, default=400, help="run-wide cap on model requests")
    p.add_argument("--max-usd", type=float, default=10.0, help="run-wide cap on model spend")
    p.add_argument("--allow-partial", action="store_true",
                   help="start although the run-wide caps are below a task's run_budget (the episode then "
                        "ends as a harness stop when a cap is reached)")
    p.add_argument("--env-file", action="append", default=[], help="KEY=VALUE file to load (values never printed)")
    p.add_argument("--out", help="output directory (default runs/<time>_<suite>_<backend>)")
    p.add_argument("--no-video", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("report", help="re-render report.html for a run directory")
    p.add_argument("run")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("compare", help="paired statistics, side-by-side video and a table for several runs")
    p.add_argument("runs", nargs="+")
    p.add_argument("--task")
    p.add_argument("--seed", type=int)
    p.add_argument("--subjects", help="comma list, in tile order")
    p.add_argument("--video", help="write a side-by-side mp4 here")
    p.add_argument("--html", help="write a comparison table page here")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("leaderboard", help="static leaderboard page from results.json files")
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default="site/index.html")
    p.add_argument("--title", default="omakase-bench leaderboard")
    p.set_defaults(fn=cmd_leaderboard)

    p = sub.add_parser("showcase", help="run models on one task and seed and write a side-by-side video")
    p.add_argument("--model", default="scripted/oracle")
    p.add_argument("--interface")
    p.add_argument("--code-mode", default="one-shot", choices=["one-shot", "iterative"])
    p.add_argument("--task", default="drafts/elevator_call")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", default="mujoco", choices=["mujoco", "isaac", "real"])
    p.add_argument("-B", "--backend-option", action="append", default=[])
    p.add_argument("-M", "--model-option", action="append", default=[])
    p.add_argument("--max-calls", type=int, default=200)
    p.add_argument("--max-usd", type=float, default=5.0)
    p.add_argument("--env-file", action="append", default=[])
    p.add_argument("--out", default="runs/showcase")
    p.set_defaults(fn=cmd_showcase)

    p = sub.add_parser("robot", help="check a robot description: load it, solve IK around home, render stills")
    rsub = p.add_subparsers(dest="robot_cmd", required=True)
    p = rsub.add_parser("check", help="exit status 1 when an arm reaches none of its IK targets")
    p.add_argument("robot", help="a name under robots/ or a directory with robot.yaml")
    p.add_argument("--out", help="directory for PNG stills (an outside view and every robot camera)")
    p.set_defaults(fn=cmd_robot)

    p = sub.add_parser("suite", help="publish a suite")
    ssub = p.add_subparsers(dest="suite_cmd", required=True)
    p = ssub.add_parser("freeze", help="write suites/<id>/v<N>/ with a copy of each task's YAML and record its hash")
    p.add_argument("suite", help="<id>@<version>, e.g. starter@1; an existing version is never overwritten")
    p.add_argument("--tasks", required=True, help="comma list of task ids")
    p.add_argument("--seeds", required=True, help="e.g. 0-19 or 0,3,5")
    p.add_argument("--description", default="")
    p.set_defaults(fn=cmd_suite_freeze)

    from .tasks import packages
    for package in packages():
        if package.commands is not None:
            package.commands(sub)

    args = ap.parse_args(argv)
    from .core.backend import BackendUnsupported
    try:
        return args.fn(args)
    except (ValueError, KeyError, FileNotFoundError, BackendUnsupported) as exc:
        # what the command line asked for cannot be done: the message, not a traceback
        if os.environ.get("OMAKASE_BENCH_TRACEBACK"):
            raise
        print(f"omakase-bench: {exc.args[0] if isinstance(exc, KeyError) and exc.args else exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

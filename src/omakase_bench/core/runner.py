"""The single runner: for each subject, task and seed, reset, let the subject drive, score.

    episode = task.resolve(seed)
    backend.reset(episode.scene, seed=seed)
    session = Session(episode, backend, robot, ...)
    subject.run(session)            # until it returns or the session raises StopEpisode
    score = session.score()         # oracle only

Each episode writes ``episodes/<subject>/<task>/<seed>/`` (trace.jsonl,
episode.mp4, program files) and the run writes ``results.json`` and
``report.html``.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .backend import Backend, check_supported, make_backend
from .results import build_results, write_results
from ..harness.session import Recorder, RunBudget, Session, StopEpisode
from .task import Task, robot_for


@dataclass
class RunConfig:
    out_dir: Path
    backend: str = "mujoco"
    backend_options: dict[str, Any] = field(default_factory=dict)
    time_mode: str = "lockstep"
    video: bool = True
    video_fps: float = 10.0
    video_camera: str = "showcase"
    run_budget: RunBudget = field(default_factory=RunBudget)
    official: bool = True
    suite: Any = None


def safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def run_episode(task: Task, seed: int, subject, backend: Backend, cfg: RunConfig,
                episode_dir: Path) -> dict[str, Any]:
    capabilities = backend.info().capabilities
    check_supported(task, backend.info())
    if task.team is not None:
        from ..harness.team_session import run_team_episode
        return run_team_episode(task, seed, as_team(subject, task), backend, cfg, episode_dir)
    episode = task.resolve(seed)
    robot = robot_for(task, episode.scene)
    backend.load(robot, episode.scene)
    check_supported(task, backend.info())        # what a backend can deliver may depend on the robot it loaded
    if "reset" in capabilities:
        backend.reset(episode.scene, seed=seed)
    initial = backend.oracle() if "ground_truth" in capabilities else None
    recorder = Recorder(camera=cfg.video_camera, fps=cfg.video_fps, enabled=cfg.video)
    session = Session(episode, backend, robot, recorder=recorder, run_budget=cfg.run_budget,
                      time_mode=cfg.time_mode, footprint=robot.mobile_base.footprint if robot.mobile_base else None)
    session.episode_dir = episode_dir
    episode_dir.mkdir(parents=True, exist_ok=True)
    exit_reason, detail = "finished", ""
    t0 = time.monotonic()
    try:
        exit_reason = subject.run(session) or "finished"
    except StopEpisode as stop:
        exit_reason, detail = stop.reason, stop.detail
    except Exception as exc:   # a subject crash is a failed episode, not a failed run
        exit_reason, detail = "error:subject", f"{type(exc).__name__}: {exc}"
        session.log("error", traceback=traceback.format_exc()[-4000:])
    wall = time.monotonic() - t0
    score = session.score()
    artifacts = {}
    with (episode_dir / "trace.jsonl").open("w") as fh:
        for entry in session.trace:
            fh.write(json.dumps(entry, default=str) + "\n")
    artifacts["trace"] = "trace.jsonl"
    if cfg.video and recorder.frames:
        from ..report.video import write_episode_video, write_raw_video
        recorder.maybe_capture(backend, backend.sim_time(), force=True)
        write_raw_video(episode_dir / "raw.mp4", recorder.frames, cfg.video_fps)
        write_episode_video(episode_dir / "episode.mp4", recorder.frames, recorder.times, cfg.video_fps,
                            title=subject.name, instruction=episode.instruction,
                            success=score["success"])
        artifacts["video"] = "episode.mp4"
    for name, content in getattr(session, "artifacts", {}).items():
        (episode_dir / name).write_text(content)
        artifacts[name.split(".")[0]] = name
    result = {
        "subject": subject.name, "task": task.id, "task_version": task.version, "seed": seed,
        "instruction": episode.instruction, "variation": {k: v for k, v in episode.values.items()},
        "success": score["success"], "credit": score["credit"], "evidence": score["evidence"],
        "subgoals": score["subgoals"], "exit_reason": exit_reason, "exit_detail": detail,
        "model_claim": session.model_claim, "model_ids": sorted(session.model_ids),
        "interface": subject.interface, "sim_seconds": round(backend.sim_time(), 2),
        "success_sim_seconds": score["success_at"], "wall_seconds": round(wall, 2),
        "usage": session.usage_dict(), "initial_state": initial.to_json() if initial else None,
        "final_state": score["final_state"], "joint_travel": session.travel.to_json(),
        "artifacts": {k: str((episode_dir / v).relative_to(cfg.out_dir)) for k, v in artifacts.items()},
    }
    (episode_dir / "result.json").write_text(json.dumps(result, indent=1, default=str))
    return result


def as_team(subject, task: Task):
    """``subject`` for every robot of a multi-robot task (a shallow copy each), unless it is a team."""
    import copy
    from ..subjects.team import TeamSubject
    if isinstance(subject, TeamSubject):
        return subject
    return TeamSubject({rid: copy.copy(subject) for rid in task.team.ids})


def _offered(subject, task: Task) -> str:
    """'' when ``task`` offers the interface(s) of ``subject``, else the one it does not."""
    members = getattr(subject, "members", None)
    if task.team is None:
        return "" if members is None and subject.interface in task.interfaces else subject.interface
    for s in (members.values() if members else [subject]):
        if s.interface not in task.interfaces:
            return s.interface
    return ""


def run(subjects: Sequence[Any], tasks: Sequence[Task], seeds: Sequence[int], cfg: RunConfig,
        progress: Callable[[str], None] = print, teams: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Every subject on every task and seed. ``teams`` (task id -> team subject) drives a
    multi-robot task instead of ``subjects``, which otherwise drive every robot of it."""
    teams = dict(teams or {})
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    backend = make_backend(cfg.backend, **cfg.backend_options)
    episodes: list[dict[str, Any]] = []
    stopped = ""
    started = time.time()
    pairs = [(subject, task) for subject in subjects for task in tasks if task.id not in teams]
    pairs += [(teams[task.id], task) for task in tasks if task.id in teams]
    interrupted: BaseException | None = None
    try:
        try:
            for subject, task in pairs:
                missing = _offered(subject, task)
                if missing:
                    progress(f"skip {subject.name} on {task.id}: interface {missing} not offered")
                    continue
                for seed in seeds:
                    over = cfg.run_budget.exceeded()
                    if over:
                        stopped = f"budget_exceeded:{over}"
                        progress(f"run stopped: {stopped}")
                        break
                    name = as_team(subject, task).name if task.team is not None else subject.name
                    d = cfg.out_dir / "episodes" / safe(name) / safe(task.id) / str(seed)
                    res = run_episode(task, seed, subject, backend, cfg, d)
                    episodes.append(res)
                    progress(f"{res['subject']:32s} {task.id:22s} seed {seed:3d}  "
                             f"{'SUCCESS' if res['success'] else 'fail   '}  {res['exit_reason']:28s} "
                             f"sim {res['sim_seconds']:6.1f}s  ${res['usage']['usd']:.3f}")
                    if res["exit_reason"].startswith("budget_exceeded:run_"):
                        stopped = res["exit_reason"]
                        progress(f"run stopped: {stopped}")
                        break
                if stopped:
                    break
        except BaseException as exc:
            # Ctrl-C, or a failure outside an episode: the episodes that finished are still written
            if not episodes:
                raise
            interrupted, stopped = exc, f"interrupted:{type(exc).__name__}"
            progress(f"run stopped: {stopped}")
        info = backend.info()
    finally:
        backend.close()
    results = build_results(episodes=episodes, subjects=[*subjects, *teams.values()], tasks=tasks, seeds=seeds,
                            cfg=cfg,
                            backend_info=info, started=started, stopped=stopped)
    write_results(cfg.out_dir, results)
    if interrupted is not None:
        raise interrupted
    return results

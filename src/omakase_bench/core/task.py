"""Tasks: declared in YAML, resolved per seed into a concrete scene.

A task names a robot preset, a backend-neutral scene (fixtures, free objects,
articulated objects, regions, cameras), the variation axes sampled per seed,
an instruction template, a success predicate, weighted sub-goals and a
budget. A task whose episodes cannot be described by independent distributions
names a ``generator`` (``module:function``) that receives the scene, the
episode's random generator and the YAML ``params`` and returns the episode's
scene and values. An optional ``gripper`` block narrows the end effectors' range for the task
(:mod:`omakase_bench.harness.end_effectors`). Suites, and finding a task by its id, are :mod:`.suite`.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

COLORS: dict[str, tuple[float, float, float, float]] = {
    "red": (0.85, 0.15, 0.12, 1), "green": (0.15, 0.65, 0.25, 1), "blue": (0.15, 0.35, 0.85, 1),
    "yellow": (0.95, 0.8, 0.1, 1), "gray": (0.55, 0.55, 0.57, 1), "white": (0.95, 0.95, 0.95, 1),
    "black": (0.08, 0.08, 0.09, 1), "orange": (0.95, 0.5, 0.1, 1), "wood": (0.62, 0.47, 0.32, 1),
    "steel": (0.7, 0.72, 0.75, 1), "wall": (0.88, 0.86, 0.82, 1), "dark": (0.25, 0.25, 0.28, 1),
    "purple": (0.55, 0.2, 0.7, 1),
}


def rgba(value: Any) -> tuple[float, float, float, float]:
    if isinstance(value, str):
        return COLORS[value]
    v = tuple(float(x) for x in value)
    return v if len(v) == 4 else (*v, 1.0)


@dataclass(frozen=True)
class Shape:
    """A fixture (static) or a free object. ``type``: box | cylinder | sphere | tray | staunton.
    ``size``: box/tray full extents (x, y, z); cylinder (radius, height); sphere (radius,);
    staunton (height, base diameter), a chess piece of the kind named by ``variant``
    (:mod:`omakase_bench.objects.staunton`)."""
    name: str
    type: str
    size: tuple[float, ...]
    pos: tuple[float, float, float]
    yaw: float = 0.0                       # rad
    color: str | tuple = "gray"
    mass: float = 0.1
    friction: float = 1.0
    wall: float = 0.008                    # tray wall thickness
    label: str = ""                        # what perception reports as its kind
    variant: str = ""                      # staunton: pawn | knight | bishop | rook | queen | king


@dataclass(frozen=True)
class Articulated:
    """``type``: button (pushes along -normal into its mount) | sliding_door (pair of
    panels sliding apart along y) | hinged_door | drawer. ``params`` carries per-type
    numbers (travel, stiffness, press_threshold, opens_with, delay, hold_open...)."""
    name: str
    type: str
    pos: tuple[float, float, float]
    yaw: float = 0.0
    size: tuple[float, ...] = ()
    color: str | tuple = "steel"
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Region:
    """An axis-aligned box in world coordinates."""
    name: str
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]

    def contains(self, p) -> bool:
        return all(self.lo[i] <= p[i] <= self.hi[i] for i in range(3))


@dataclass(frozen=True)
class SceneCamera:
    name: str
    pos: tuple[float, float, float]
    lookat: tuple[float, float, float]
    fovy: float = 50.0


@dataclass(frozen=True)
class RobotPlacement:
    """One robot of a multi-robot scene: its id (the prefix of its names in the backend), preset,
    base pose in the world (x, y, yaw rad) and start configuration."""
    id: str
    preset: str
    base: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lift: float | None = None
    neck: Mapping[str, float] = field(default_factory=dict)
    home_pose: str = ""


@dataclass(frozen=True)
class SceneSpec:
    robot: str
    base: tuple[float, float, float] = (0.0, 0.0, 0.0)      # x, y, yaw (rad)
    lift: float | None = None
    neck: Mapping[str, float] = field(default_factory=dict)
    fixtures: tuple[Shape, ...] = ()
    objects: tuple[Shape, ...] = ()
    articulated: tuple[Articulated, ...] = ()
    regions: Mapping[str, Region] = field(default_factory=dict)
    cameras: tuple[SceneCamera, ...] = ()
    floor_color: str | tuple = (0.72, 0.72, 0.70, 1)
    #: named pose of the preset the arms start in and return to with go_home; '' = the preset's home
    home_pose: str = ""
    #: the robots of a multi-robot task (empty: the single robot above)
    robots: tuple[RobotPlacement, ...] = ()

    def placement(self, robot_id: str) -> RobotPlacement:
        return next(r for r in self.robots if r.id == robot_id)

    def element(self, name: str):
        for group in (self.fixtures, self.objects, self.articulated):
            for e in group:
                if e.name == name:
                    return e
        raise KeyError(name)


@dataclass(frozen=True)
class Budget:
    sim_seconds: float = 120.0
    wall_seconds: float = 900.0
    api_calls: int = 60          # robot API calls (tools turns, code RPCs)
    model_calls: int = 40        # language-model requests per episode
    usd: float | None = 2.0
    code_iterations: int = 4     # programs per episode for the iterative code interface


@dataclass(frozen=True)
class Subgoal:
    name: str
    predicate: Mapping[str, Any]
    weight: float = 1.0


@dataclass(frozen=True)
class Task:
    id: str
    version: str
    robot: str
    title: str
    instruction: str
    scene: SceneSpec
    variation: Mapping[str, Mapping[str, Any]]
    success: Mapping[str, Any]
    subgoals: tuple[Subgoal, ...]
    budget: Budget
    interfaces: frozenset[str]
    stop_on_success: bool = True
    status: str = "active"                # active | draft
    source: Path | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)
    #: arm -> end-effector limits of this task (``gripper:`` block), empty when the task sets none
    gripper: Mapping[str, Any] = field(default_factory=dict)
    #: robots, schedule and referee of a multi-robot task (``robots:`` block); None for one robot
    team: Any = None
    #: robot id -> its arm -> end-effector limits (multi-robot tasks)
    team_gripper: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def resolve(self, seed: int) -> "Episode":
        rng = np.random.default_rng(_seed_for(self.id, seed))
        values: dict[str, Any] = {}
        scene = self.scene
        for key, dist in self.variation.items():
            target, _, attr = key.rpartition(".")
            value = _sample(dist, rng)
            values[key] = value
            scene = _apply(scene, target, attr, value)
        if self.raw.get("generator"):
            scene, extra = resolve_ref(self.raw["generator"])(scene, rng, self.raw.get("params", {}))
            values.update(extra)
        _check_separation(scene, self.raw.get("min_separation", {}))
        names = {k.replace(".", "_"): (v if not isinstance(v, float) else round(v, 3))
                 for k, v in values.items()}
        # a multi-robot task formats its instruction per robot, with that robot's values
        instruction = (self.instruction.format(**names) if "{" in self.instruction and self.team is None
                       else self.instruction)
        return Episode(self, seed, scene, values, instruction)


@dataclass(frozen=True)
class Episode:
    task: Task
    seed: int
    scene: SceneSpec
    values: Mapping[str, Any]
    instruction: str


def resolve_ref(ref: str):
    """The object named by ``module:attribute``."""
    import importlib
    mod, _, attr = ref.partition(":")
    return getattr(importlib.import_module(mod), attr)


def _seed_for(task_id: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{task_id}#{seed}".encode()).digest()[:8], "little")


def _sample(dist: Mapping[str, Any], rng: np.random.Generator) -> Any:
    if "uniform" in dist:
        lo, hi = dist["uniform"]
        return float(rng.uniform(lo, hi))
    if "uniform_box" in dist:
        lo, hi = (np.array(v, float) for v in dist["uniform_box"])
        return tuple(float(v) for v in rng.uniform(lo, hi))
    if "choice" in dist:
        options = list(dist["choice"])
        return options[int(rng.integers(len(options)))]
    raise ValueError(f"unknown distribution {dist}")


def _apply(scene: SceneSpec, target: str, attr: str, value: Any) -> SceneSpec:
    if target.startswith("robots."):
        rid = target.partition(".")[2]
        robots = list(scene.robots)
        for i, r in enumerate(robots):
            if r.id == rid:
                if attr == "base":
                    b = tuple(float(v) for v in value)
                    robots[i] = replace(r, base=(b[0], b[1], math.radians(b[2])))
                elif attr == "lift":
                    robots[i] = replace(r, lift=float(value))
                else:
                    raise ValueError(f"unknown robot variation {attr}")
                return replace(scene, robots=tuple(robots))
        raise KeyError(f"variation names unknown robot {rid!r}")
    if target == "robot":
        if attr == "base":
            return replace(scene, base=tuple(value))
        if attr == "lift":
            return replace(scene, lift=float(value))
        raise ValueError(f"unknown robot variation {attr}")
    for group in ("fixtures", "objects", "articulated"):
        items = list(getattr(scene, group))
        for i, e in enumerate(items):
            if e.name != target:
                continue
            if attr in ("xy", "x", "y", "z"):
                pos = list(e.pos)
                if attr == "xy":
                    pos[0], pos[1] = value
                else:
                    pos["xyz".index(attr)] = float(value)
                items[i] = replace(e, pos=tuple(pos))
            elif attr == "yaw_deg":
                items[i] = replace(e, yaw=math.radians(float(value)))
            elif attr == "color":
                items[i] = replace(e, color=value)
            else:
                raise ValueError(f"unknown variation attribute {attr!r}")
            return replace(scene, **{group: tuple(items)})
    raise KeyError(f"variation names unknown scene element {target!r}")


def _check_separation(scene: SceneSpec, rules: Mapping[str, float]) -> None:
    """Reject a sample whose named pairs are closer than declared (a task bug, not bad luck:
    variation ranges must be chosen so this never fires)."""
    for pair, dist in rules.items():
        a, b = pair.split("|")
        pa, pb = scene.element(a).pos, scene.element(b).pos
        if math.dist(pa[:2], pb[:2]) < float(dist):
            raise ValueError(f"variation placed {a} and {b} closer than {dist} m")


# --- loading -----------------------------------------------------------------

def _shape(d: Mapping[str, Any]) -> Shape:
    return Shape(name=d["name"], type=d.get("type", "box"), size=tuple(float(v) for v in d["size"]),
                 pos=tuple(float(v) for v in d["pos"]), yaw=math.radians(float(d.get("yaw_deg", 0.0))),
                 color=d.get("color", "gray"), mass=float(d.get("mass", 0.1)),
                 friction=float(d.get("friction", 1.0)), wall=float(d.get("wall", 0.008)),
                 label=d.get("label", d.get("type", "box")))


def load_task(path: str | Path) -> Task:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    sc = raw["scene"]
    robot = sc.get("robot", {})
    home = robot.get("home") or ""
    team, placements = None, ()
    if raw.get("robots"):
        from .team import parse_team, placement_base
        team = parse_team(raw, path.name)
        placements = tuple(RobotPlacement(s.id, s.preset, placement_base(d), d.get("lift"),
                                          dict(d.get("neck") or {}), s.start_pose)
                           for s, d in zip(team.robots, raw["robots"]))
        _check_poses(team, path)
        robot = {"preset": placements[0].preset}
    scene = SceneSpec(
        robot=robot.get("preset", raw.get("robot", "d1")),
        base=(lambda b: (float(b[0]), float(b[1]), math.radians(float(b[2]))))(robot.get("base", [0, 0, 0])),
        lift=robot.get("lift"),
        neck=robot.get("neck", {}),
        fixtures=tuple(_shape(d) for d in sc.get("fixtures", [])),
        objects=tuple(_shape(d) for d in sc.get("objects", [])),
        articulated=tuple(Articulated(d["name"], d["type"], tuple(float(v) for v in d["pos"]),
                                      math.radians(float(d.get("yaw_deg", 0.0))),
                                      tuple(float(v) for v in d.get("size", ())), d.get("color", "steel"),
                                      d.get("params", {}))
                          for d in sc.get("articulated", [])),
        regions={n: Region(n, tuple(r["lo"]), tuple(r["hi"])) for n, r in sc.get("regions", {}).items()},
        cameras=tuple(SceneCamera(n, tuple(c["pos"]), tuple(c["lookat"]), float(c.get("fovy", 50)))
                      for n, c in sc.get("cameras", {}).items()),
        home_pose=home,
        robots=placements,
    )
    b = raw.get("budget", {})
    return Task(
        id=raw["id"], version=str(raw.get("version", "1.0.0")), robot=scene.robot,
        title=raw.get("title", raw["id"]), instruction=raw["instruction"].strip(), scene=scene,
        variation=raw.get("variation", {}), success=raw["success"] if team is None else raw.get("success") or {},
        subgoals=tuple(Subgoal(s["name"], s["pred"], float(s.get("weight", 1.0)))
                       for s in raw.get("subgoals", [])),
        budget=Budget(**{k: v for k, v in b.items()}),
        interfaces=frozenset(raw.get("interfaces", ["tools", "code", "chunks", "scripted"])),
        stop_on_success=bool(raw.get("stop_on_success", True)), status=raw.get("status", "active"),
        source=path, raw=raw, gripper=_gripper_limits(raw.get("gripper"), scene.robot, path), team=team,
        team_gripper={r.id: _gripper_limits(_merged(raw.get("gripper"), r.gripper), r.preset, path, r.id)
                      for r in (team.robots if team else ())})


def robot_for(task: "Task", scene: SceneSpec | None = None, robot_id: str = ""):
    """The robot preset of an episode (of robot ``robot_id`` in a multi-robot task), with the arm
    home its scene names."""
    from ..robots.spec import load_robot
    scene = scene or task.scene
    if robot_id:
        r = scene.placement(robot_id)
        return load_robot(r.preset).with_home(r.home_pose)
    return load_robot(task.robot).with_home(scene.home_pose)


def with_robot(task: "Task", robot: str) -> "Task":
    """A single-robot task on another robot preset: the same scene and success check, the robot at its own
    home pose, and without the lift and neck settings that robot has no joints for."""
    if task.team is not None:
        raise ValueError(f"{task.id} places its robots itself; a multi-robot task takes no robot override")
    from ..robots.spec import load_robot
    spec = load_robot(robot)
    scene = replace(task.scene, robot=robot, home_pose="", lift=task.scene.lift if spec.lift_joint else None,
                    neck={k: v for k, v in task.scene.neck.items() if k in spec.neck})
    return replace(task, robot=robot, scene=scene,
                   gripper=_gripper_limits(task.raw.get("gripper"), robot, task.source or Path(task.id)))


def _gripper_limits(block: Any, robot: str, path: Path, robot_id: str = "") -> Mapping[str, Any]:
    """The task's end-effector limits, checked against the robot preset's hardware range."""
    if block is None:
        return {}
    from ..harness.end_effectors import parse_task_limits
    from ..robots.spec import load_robot
    where = f"{path.name}: robot {robot_id}: gripper" if robot_id else f"{path.name}: gripper"
    return parse_task_limits(block, load_robot(robot), where)


def _check_poses(team, path: Path) -> None:
    from ..robots.spec import load_robot
    for s in team.robots:
        spec = load_robot(s.preset)
        for kind, pose in (("start_pose", s.start_pose), ("idle_pose", s.idle_pose)):
            if pose and pose not in spec.poses:
                raise ValueError(f"{path.name}: robot {s.id}: {kind} {pose!r} is not a pose of preset {s.preset}; "
                                 f"poses: {sorted(spec.poses)}")


def _merged(task_block: Any, robot_block: Mapping[str, Any]) -> Any:
    """A robot's gripper block: the task-wide block with the robot's own fields on top."""
    if not task_block and not robot_block:
        return None
    return {**(task_block or {}), **(robot_block or {})}


# --- wire form (for remote backends) ------------------------------------------------

def scene_to_dict(scene: SceneSpec) -> dict[str, Any]:
    import dataclasses
    d = dataclasses.asdict(scene)
    d["regions"] = {k: dataclasses.asdict(v) for k, v in scene.regions.items()}
    return d


def scene_from_dict(d: Mapping[str, Any]) -> SceneSpec:
    def tup(v):
        return tuple(tup(x) for x in v) if isinstance(v, list) else v

    def shape(x):
        return Shape(**{k: tup(v) for k, v in x.items()})

    return SceneSpec(
        robot=d["robot"], base=tuple(d["base"]), lift=d.get("lift"), neck=dict(d.get("neck") or {}),
        fixtures=tuple(shape(x) for x in d.get("fixtures", [])),
        objects=tuple(shape(x) for x in d.get("objects", [])),
        articulated=tuple(Articulated(x["name"], x["type"], tuple(x["pos"]), x.get("yaw", 0.0), tuple(x.get("size") or ()),
                                      tup(x.get("color", "steel")), dict(x.get("params") or {}))
                          for x in d.get("articulated", [])),
        regions={k: Region(k, tuple(r["lo"]), tuple(r["hi"])) for k, r in (d.get("regions") or {}).items()},
        cameras=tuple(SceneCamera(c["name"], tuple(c["pos"]), tuple(c["lookat"]), c.get("fovy", 50.0))
                      for c in d.get("cameras", [])),
        floor_color=tup(d.get("floor_color", (0.72, 0.72, 0.70, 1))), home_pose=d.get("home_pose", ""),
        robots=tuple(RobotPlacement(r["id"], r["preset"], tuple(r["base"]), r.get("lift"), dict(r.get("neck") or {}),
                                    r.get("home_pose", "")) for r in d.get("robots", [])))

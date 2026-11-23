"""The Backend Protocol: what a simulator or a real robot must offer.

Every backend: describe itself, take a robot description, report joint state, follow joint targets (the
harness sends every command through one place), let time pass, and deliver camera images. Kinematics, IK
and the model-facing robot API live above it, so a new engine or a new robot transport implements only
this.

What only some backends can do is a capability: a separate Protocol the backend also satisfies and a name
it announces in ``BackendInfo.capabilities``. Four of them are what makes a simulator a simulator:
``reset`` (build a scene and put it in the start state of a seed), ``ground_truth`` (privileged object
poses), ``contacts`` (exact contact detection in that ground truth) and ``deterministic`` (the same seed
gives the same episode). A real robot has none of the four. A task states what it needs
(:func:`required_capabilities`); the runner refuses a task on a backend that lacks it
(:class:`BackendUnsupported`) instead of running it unjudged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from ..robots.spec import RobotSpec
from .task import Region, SceneSpec


@dataclass(frozen=True)
class BackendInfo:
    name: str                      # "mujoco", "isaac"
    version: str                   # backend adapter version
    engine: str                    # e.g. "mujoco 4.0.1"
    robot: str
    robot_sha256: str
    physics_dt: float
    control_dt: float
    capabilities: frozenset[str] = frozenset()
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectState:
    name: str
    kind: str                      # fixture | object | button | sliding_door | ...
    label: str
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]   # w, x, y, z
    size: tuple[float, ...]
    lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ArticulatedState:
    name: str
    kind: str
    q: float                       # joint displacement (m or rad)
    pressed: bool = False
    pressed_ever: bool = False
    open_fraction: float = 0.0     # doors and drawers: 0 closed, 1 fully open
    lamp: bool = False             # an indicator the scene lights (e.g. a call button)
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: tuple[float, float, float] = (1.0, 0.0, 0.0)


@dataclass(frozen=True)
class OracleSnapshot:
    """Privileged world state, in world coordinates. Only scorers read this."""
    time: float
    objects: Mapping[str, ObjectState]
    articulated: Mapping[str, ArticulatedState]
    base: tuple[float, float, float]                  # x, y, yaw
    base_footprint: tuple[float, float]
    tcp: Mapping[str, tuple[float, float, float]]     # per arm, world frame
    gripper_opening: Mapping[str, float]              # per arm, metres
    contacts: frozenset[tuple[str, str]]              # (object, part) touching now
    contacts_ever: frozenset[tuple[str, str]]         # since reset
    held_by: Mapping[str, str]                        # object -> arm (both jaws touching it)
    initial: Mapping[str, tuple[float, float, float]] # object positions at reset
    regions: Mapping[str, Region]
    #: multi-robot scenes: robot id -> base pose (x, y, yaw); per-arm keys, contact parts and
    #: held_by values then carry the robot's prefix (``a/left``, ``a/left_pad0``, ``a/robot``)
    bases: Mapping[str, tuple[float, float, float]] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "time": round(self.time, 4),
            "objects": {k: {"pos": [round(v, 5) for v in o.pos], "quat": [round(v, 5) for v in o.quat]}
                        for k, o in sorted(self.objects.items())},
            "articulated": {k: {"q": round(a.q, 5), "pressed_ever": a.pressed_ever,
                                "open_fraction": round(a.open_fraction, 4)}
                            for k, a in sorted(self.articulated.items())},
            "base": [round(v, 5) for v in self.base],
            "tcp": {k: [round(v, 5) for v in p] for k, p in sorted(self.tcp.items())},
            "contacts_ever": sorted(list(c) for c in self.contacts_ever),
            "held_by": dict(self.held_by),
            **({"bases": {k: [round(v, 5) for v in b] for k, b in sorted(self.bases.items())}} if self.bases else {}),
        }


class BackendUnsupported(RuntimeError):
    """The backend cannot do what was asked of it; the message names what is missing."""


@runtime_checkable
class Backend(Protocol):
    """What every backend provides. In a simulator time advances only inside :meth:`step` (lockstep); on a
    real robot :meth:`step` lets that much wall-clock time pass."""

    def info(self) -> BackendInfo: ...
    def load(self, robot: RobotSpec, scene: SceneSpec) -> None: ...
    def joint_positions(self) -> dict[str, float]: ...
    def command_joints(self, targets: Mapping[str, float]) -> None: ...
    def step(self, seconds: float) -> None: ...
    def sim_time(self) -> float: ...
    def render(self, camera: str, width: int, height: int) -> np.ndarray: ...
    def cameras(self) -> Sequence[str]: ...
    def close(self) -> None: ...


# --- what a simulator adds --------------------------------------------------------

@runtime_checkable
class SceneReset(Protocol):
    """``reset``: build the scene and put robot and objects in the start state of ``seed``."""
    def reset(self, scene: SceneSpec, *, seed: int) -> None: ...


@runtime_checkable
class GroundTruth(Protocol):
    """``ground_truth``: privileged world state. With ``contacts`` its contact sets are exact."""
    def oracle(self) -> OracleSnapshot | None: ...


#: capabilities only a simulator has; ``contacts`` and ``deterministic`` are properties, not methods
SIMULATOR = frozenset({"reset", "ground_truth", "contacts", "deterministic"})


# --- other optional capabilities ------------------------------------------------

@runtime_checkable
class DepthCamera(Protocol):
    """A backend whose cameras also deliver depth: metres along the optical axis per pixel."""
    def render_depth(self, camera: str, width: int, height: int) -> np.ndarray: ...


@runtime_checkable
class GripperForceLimit(Protocol):
    """A backend that can cap the force of an arm's finger actuators (N per actuator) for an episode."""
    def set_gripper_force(self, arm: str, newtons: float) -> None: ...


@runtime_checkable
class MultiRobot(Protocol):
    """A backend that holds several robots in one scene. ``load_team`` replaces ``load``; the
    scene's ``robots`` give each robot's id, base pose and start configuration. Every name of a
    robot (joints, actuators, cameras, per-arm oracle keys, contact parts) is prefixed ``<id>/``.
    Robots collide with each other, so contacts between two robots' parts are in the oracle."""
    def load_team(self, robots: Mapping[str, RobotSpec], scene: SceneSpec) -> None: ...


@runtime_checkable
class ObjectPose(Protocol):
    """A backend that can put a free object back to a pose (a referee restoring a displaced piece)."""
    def set_object_pose(self, name: str, pos: Sequence[float], quat: Sequence[float]) -> None: ...


@runtime_checkable
class PoseClearance(Protocol):
    """A backend that can tell, without moving anything, how close a robot would come to fixtures, other
    robots (and optionally free objects) and how close its arms would come to its own body and to each
    other, with some of its joints set to other values: the smallest distance (m, capped) and the pair.
    ``prefix`` names the robot (``a/``)."""
    def pose_clearance(self, prefix: str, joints: Mapping[str, float], cap: float = 0.05,
                       objects: bool = False) -> tuple[float, str]: ...


CAPABILITIES = {
    "reset": SceneReset,
    "ground_truth": GroundTruth,
    "depth_camera": DepthCamera,
    "gripper_force_limit": GripperForceLimit,
    "multi_robot": MultiRobot,
    "object_pose": ObjectPose,
    "pose_clearance": PoseClearance,
}


def make_backend(name: str, **options) -> Backend:
    if name == "mujoco":
        from ..backends.mujoco.backend import MujocoBackend
        return MujocoBackend(**options)
    if name == "isaac":
        from ..backends.isaac.client import IsaacBackend
        return IsaacBackend(**options)
    if name == "real":
        from ..backends.real.backend import RealBackend
        return RealBackend(**options)
    raise ValueError(f"unknown backend {name!r}; known: {', '.join(BACKENDS)}")


BACKENDS = ("mujoco", "isaac", "real")


def required_capabilities(task) -> frozenset[str]:
    """What ``task`` needs from a backend: its YAML ``requires:`` when it has one, else what its definition
    implies. A scene to build needs ``reset``; success predicates and a referee read ``ground_truth``; a
    referee may name more (class attribute ``requires``); several robots need ``multi_robot``."""
    declared = task.raw.get("requires")
    if declared is not None:
        return frozenset(declared)
    needs = {"reset", "ground_truth"}
    if task.team is not None:
        from .task import resolve_ref
        needs.add("multi_robot")
        if task.team.referee:
            needs.update(getattr(resolve_ref(task.team.referee), "requires", ()))
    return frozenset(needs)


def check_supported(task, info: BackendInfo) -> None:
    """Raise :class:`BackendUnsupported` when ``task`` needs a capability the backend does not announce."""
    missing = sorted(required_capabilities(task) - info.capabilities)
    if missing:
        raise BackendUnsupported(
            f"task {task.id} needs {', '.join(missing)}, which backend {info.name} does not provide "
            f"(it provides: {', '.join(sorted(info.capabilities)) or 'none'})")

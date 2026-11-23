"""A robot preset: a URDF plus a small ``robot.yaml`` naming its joint groups.

Every backend and every subject sees a robot through this description, so a
new robot (or a new revision of one) is a new preset directory, not new code.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

PRESETS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Finger:
    joint: str
    open: float
    closed: float


@dataclass(frozen=True)
class EndEffector:
    name: str
    type: str
    link: str
    tcp_link: str
    approach_axis: tuple[float, float, float]
    fingers: tuple[Finger, ...]
    pad_links: tuple[str, ...]
    max_opening: float
    force: float
    #: fingertip pads: length along the jaw from its tip, and the part of it a grasp may touch (m);
    #: 0 when the preset does not declare pads
    pad_length: float = 0.0
    pad_band: float = 0.0
    #: distance of the fingertip centre past the TCP along the approach axis, jaws (open, closed)
    fingertip_offset: tuple[float, float] = (0.0, 0.0)
    #: wrist angle of the "down" orientation that end-effector rows call yaw 0 (deg)
    yaw_zero_deg: float = 0.0
    #: fingertip distance past the TCP along the approach axis that scripted programs plan with (m)
    tip_reach: float = 0.0


@dataclass(frozen=True)
class Arm:
    name: str
    joints: tuple[str, ...]
    end_effector: str
    home: tuple[float, ...]
    max_joint_speed: float
    #: joints that reach their targets first when the arm leaves its home in stages
    leave_home_first: tuple[str, ...] = ()
    #: "left" or "right" of the base frame (+y is left); '' for an arm that serves both sides
    side: str = ""
    #: first joint of the chain whose links are checked against the robot's own body; the links
    #: before it sit at the arm's mount and touch the body by construction
    self_collision_from: str = ""


@dataclass(frozen=True)
class Camera:
    name: str
    link: str
    fovy: float
    width: int
    height: int
    #: working range of the camera's depth stream along the optical axis (m); None = no depth
    depth_range: tuple[float, float] | None = None
    description: str = ""                 # the sensor, as stated to models


DRIVES = ("holonomic", "differential")
SIDES = ("left", "right")


@dataclass(frozen=True)
class Actuation:
    """Position servos of the simulated joints, per joint group: (kp, force limit)."""
    base_linear: tuple[float, float] = (60000.0, 6000.0)
    base_yaw: tuple[float, float] = (30000.0, 3000.0)
    lift: tuple[float, float] = (40000.0, 4000.0)
    #: the first ``arm_proximal_joints`` joints of an arm carry the rest of it
    arm_proximal: tuple[float, float] = (800.0, 150.0)
    arm_distal: tuple[float, float] = (200.0, 40.0)
    arm_proximal_joints: int = 4
    #: kp of the fingers; their force limit is the end effector's ``force``
    finger_kp: float = 3000.0
    other: tuple[float, float] = (50.0, 10.0)
    #: servo lag of each planar joint of a differential base (s)
    base_lag_s: float = 0.095


@dataclass(frozen=True)
class MobileBase:
    x: str
    y: str
    yaw: str
    max_speed: float
    max_yaw_rate: float
    footprint: tuple[tuple[float, float], ...]   # polygon in the base frame
    #: "holonomic" moves in any planar direction; "differential" (two driven wheels)
    #: only moves along its heading while turning, so it has no lateral velocity
    drive: str = "holonomic"

    @property
    def differential(self) -> bool:
        return self.drive == "differential"


@dataclass(frozen=True)
class RobotSpec:
    name: str
    version: int
    urdf_path: Path
    base_link: str
    arms: Mapping[str, Arm]
    end_effectors: Mapping[str, EndEffector]
    cameras: Mapping[str, Camera]
    mobile_base: MobileBase | None
    lift_joint: str | None
    lift_range: tuple[float, float]
    lift_speed: float
    neck: Mapping[str, str]
    defaults: Mapping[str, Any] = field(default_factory=dict)
    preset_dir: Path = Path(".")
    visuals: Mapping[str, Path] = field(default_factory=dict)
    default_visual: str = ""
    #: named joint poses of the preset (``spec.json`` ``poses`` and ``robot.yaml`` ``poses``):
    #: pose -> joint -> value; a pose may also set the torso lift (its joint is then in the map)
    poses: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    actuation: Actuation = Actuation()
    #: name pattern of collision geoms left out of the robot's self-clearance check
    exempt_geoms: str = "*_exempt"
    #: registered name of the robot's IK solver (``harness/kinematics/ik.py``)
    ik_solver: str = "dls"
    #: the robot's own command API (``api`` block), read by its adapter (``harness/command_api.py``)
    api: Mapping[str, Any] = field(default_factory=dict)
    #: on-robot defaults of the real backend (``real`` block): camera devices and topics, the calibration
    #: file, the lease
    real: Mapping[str, Any] = field(default_factory=dict)

    def arm_for(self, y: float) -> str:
        """The arm on the side of a point at ``y`` in the base frame; the first arm when none is on that side."""
        side = "left" if y >= 0 else "right"
        return next((n for n, a in self.arms.items() if a.side == side), next(iter(self.arms)))

    def pose_lift(self, pose: str) -> float | None:
        """The torso lift a named pose sets, or None when it sets arm joints only."""
        if not pose or not self.lift_joint:
            return None
        v = self.pose(pose).get(self.lift_joint)
        return None if v is None else float(v)

    def pose(self, name: str) -> Mapping[str, float]:
        if name not in self.poses:
            raise ValueError(f"robot {self.name} has no pose {name!r}; poses: {sorted(self.poses)}")
        return self.poses[name]

    def pose_arm_targets(self, name: str) -> dict[str, float]:
        """Arm joint values of a named pose (every arm joint of the preset)."""
        named = self.pose(name)
        return {j: float(named[j]) for a in self.arms.values() for j in a.joints}

    def with_home(self, pose: str) -> "RobotSpec":
        """This preset with every arm's home replaced by the named pose ('' keeps the preset's home)."""
        if not pose:
            return self
        from dataclasses import replace
        named = self.pose(pose)
        arms = {n: replace(a, home=tuple(float(named[j]) for j in a.joints)) for n, a in self.arms.items()}
        return replace(self, arms=arms)

    def visual_urdf(self, variant: str | None = None) -> Path:
        """The URDF to build a simulation model from: same kinematics and collisions,
        visuals per ``variant`` (``OMAKASE_BENCH_VISUAL`` or the preset default)."""
        import os
        variant = variant or os.environ.get("OMAKASE_BENCH_VISUAL") or self.default_visual
        if not variant or variant not in self.visuals:
            return self.urdf_path
        path = self.visuals[variant]
        return path if path.exists() else self.urdf_path

    def sha256(self) -> str:
        h = hashlib.sha256()
        h.update(self.urdf_path.read_bytes())
        for extra in ("robot.yaml", "spec.json"):
            if (self.preset_dir / extra).exists():
                h.update((self.preset_dir / extra).read_bytes())
        return h.hexdigest()

    def hand_of(self, arm: str) -> EndEffector:
        return self.end_effectors[self.arms[arm].end_effector]

    def default_joint_positions(self) -> dict[str, float]:
        q: dict[str, float] = {}
        for arm in self.arms.values():
            q.update(zip(arm.joints, arm.home))
        for ee in self.end_effectors.values():
            for f in ee.fingers:
                q[f.joint] = f.open
        if self.lift_joint:
            q[self.lift_joint] = float(self.defaults.get("lift", 0.0))
        neck = self.defaults.get("neck", {})
        for key, joint in self.neck.items():
            q[joint] = float(neck.get(key, 0.0))
        return q


def load_robot(name_or_dir: str | Path) -> RobotSpec:
    """``d1`` (a shipped preset) or a path to a directory holding ``robot.yaml``."""
    d = Path(name_or_dir)
    if not (d / "robot.yaml").exists():
        d = PRESETS_DIR / str(name_or_dir)
    if not (d / "robot.yaml").exists():
        known = sorted(p.name for p in PRESETS_DIR.iterdir() if (p / "robot.yaml").exists())
        raise FileNotFoundError(f"no robot preset {name_or_dir!r}; known: {known}")
    raw = yaml.safe_load((d / "robot.yaml").read_text())
    ees = {
        n: EndEffector(n, e["type"], e["link"], e["tcp_link"], tuple(e["approach_axis"]),
                       tuple(Finger(f["joint"], f["open"], f["closed"]) for f in e.get("fingers", [])),
                       tuple(e.get("pad_links", ())), float(e.get("max_opening", 0.0)),
                       float(e.get("force", 20.0)), float(e.get("pad_length", 0.0)),
                       float(e.get("pad_band", 0.0)),
                       (float((e.get("fingertip_offset") or {}).get("open", 0.0)),
                        float((e.get("fingertip_offset") or {}).get("closed", 0.0))),
                       float(e.get("yaw_zero_deg", 0.0)),
                       float(e.get("tip_reach", (e.get("fingertip_offset") or {}).get("open", 0.0))))
        for n, e in raw.get("end_effectors", {}).items()}
    poses = {}
    if (d / "spec.json").exists():
        poses = json.loads((d / "spec.json").read_text()).get("poses", {})
    lift_joint = (raw.get("torso_lift") or {}).get("joint")
    for name, p in (raw.get("poses") or {}).items():
        joints = {k: float(v) for k, v in (p.get("joints") or {}).items()}
        if "lift" in p:
            if not lift_joint:
                raise ValueError(f"{d / 'robot.yaml'}: pose {name} sets a lift, but the robot has no torso lift")
            joints[lift_joint] = float(p["lift"])
        poses[name] = joints

    def home(a) -> tuple[float, ...]:
        if isinstance(a["home"], str):
            pose = poses[a["home"]]
            return tuple(float(pose[j]) for j in a["joints"])
        return tuple(float(v) for v in a["home"])

    def arm(n: str, a) -> Arm:
        joints = tuple(a["joints"])
        side = a.get("side", n if n in SIDES else "")
        first = a.get("self_collision_from", joints[min(2, len(joints) - 1)])
        if side not in ("", *SIDES):
            raise ValueError(f"{d / 'robot.yaml'}: arms.{n}.side must be one of {SIDES}, got {side!r}")
        if first not in joints:
            raise ValueError(f"{d / 'robot.yaml'}: arms.{n}.self_collision_from {first!r} is not a joint of the arm")
        return Arm(n, joints, a["end_effector"], home(a), float(a.get("max_joint_speed", 1.0)),
                   tuple(a.get("leave_home_first", ())), side, first)

    arms = {n: arm(n, a) for n, a in raw.get("arms", {}).items()}
    act = raw.get("actuation") or {}
    unknown = set(act) - {f.name for f in dataclasses.fields(Actuation)}
    if unknown:
        raise ValueError(f"{d / 'robot.yaml'}: actuation has unknown field(s) {sorted(unknown)}")
    actuation = Actuation(**{k: (tuple(float(x) for x in v) if isinstance(v, (list, tuple)) else v) for k, v in act.items()})
    cams = {n: Camera(n, c["link"], float(c.get("fovy", 60.0)), int(c.get("width", 640)),
                      int(c.get("height", 480)),
                      tuple(float(v) for v in c["depth_range"]) if c.get("depth_range") else None,
                      str(c.get("description", "")))
            for n, c in raw.get("cameras", {}).items()}
    mb = raw.get("mobile_base")
    base = None
    if mb:
        drive = mb.get("drive", "holonomic")
        if drive not in DRIVES:
            raise ValueError(f"{d / 'robot.yaml'}: mobile_base.drive must be one of {DRIVES}, got {drive!r}")
        base = MobileBase(mb["joints"]["x"], mb["joints"]["y"], mb["joints"]["yaw"],
                          float(mb.get("max_speed", 0.5)), float(mb.get("max_yaw_rate", 1.0)),
                          tuple(tuple(float(v) for v in p) for p in mb["footprint"]), drive)
    lift = raw.get("torso_lift") or {}
    return RobotSpec(
        name=raw["name"], version=int(raw.get("version", 1)), urdf_path=d / raw["urdf"],
        base_link=raw.get("base_link", ""), arms=arms, end_effectors=ees, cameras=cams,
        mobile_base=base, lift_joint=lift.get("joint"),
        lift_range=tuple(lift.get("range", (0.0, 0.0))), lift_speed=float(lift.get("speed", 0.05)),
        neck=dict(raw.get("neck") or {}), defaults=raw.get("defaults") or {}, preset_dir=d,
        visuals={k: d / v for k, v in (raw.get("visuals") or {}).items()},
        default_visual=raw.get("default_visual", ""), poses=poses, actuation=actuation,
        exempt_geoms=str((raw.get("collision") or {}).get("exempt_geoms", "*_exempt")),
        ik_solver=str((raw.get("ik") or {}).get("solver", "dls")), api=raw.get("api") or {}, real=raw.get("real") or {})

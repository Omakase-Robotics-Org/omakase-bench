"""MuJoCo backend: CPU physics, EGL/OSMesa offscreen rendering, privileged oracle.

One robot (``load``) or several (``load_team``): in a multi-robot scene every name of
robot ``a`` (joints, actuators, cameras, sites, per-arm oracle keys, contact parts)
carries the prefix ``a/``; with one robot the prefix is empty and nothing changes.
"""
from __future__ import annotations

import math
import os
import warnings
from fnmatch import fnmatch
from typing import Mapping, Sequence

import mujoco
import numpy as np

from ...core.backend import SIMULATOR, ArticulatedState, BackendInfo, ObjectState, OracleSnapshot
from ...core.scene_logic import SceneLogic
from ...core.task import RobotPlacement, SceneSpec
from ...robots.spec import RobotSpec
from .scene_mjcf import build_scene, build_team_scene

VERSION = "0.1.0"
CONTROL_DT = 0.02


class MujocoBackend:
    name = "mujoco"

    def __init__(self, *, visual: str = ""):
        os.environ.setdefault("MUJOCO_GL", "egl")
        self.robot: RobotSpec | None = None
        self.scene: SceneSpec | None = None
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._depth_renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self.logic: SceneLogic | None = None
        self._contacts_ever: set[tuple[str, str]] = set()
        self._gripper_geoms: dict[int, str] = {}
        self._initial: dict[str, tuple[float, float, float]] = {}
        self.visual = visual or None
        #: (name prefix, preset, placement) per robot; one entry with prefix "" for a single robot
        self._team: list[tuple[str, RobotSpec, RobotPlacement | None]] = []

    # --- Backend Protocol ---------------------------------------------------
    def info(self) -> BackendInfo:
        return BackendInfo(
            name="mujoco", version=VERSION, engine=f"mujoco {mujoco.__version__}",
            robot=self.robot.name if self.robot else "", robot_sha256=self.robot.sha256() if self.robot else "",
            physics_dt=0.002, control_dt=CONTROL_DT, capabilities=SIMULATOR | {"depth_camera", "gripper_force_limit", "multi_robot", "object_pose", "pose_clearance"},
            detail={"gl": os.environ.get("MUJOCO_GL", ""),
                    "visual": (self.robot.visual_urdf(self.visual).name if self.robot else "")})

    def load(self, robot: RobotSpec, scene: SceneSpec) -> None:
        self.robot, self.scene = robot, scene
        self._team = [("", robot, None)]

    def load_team(self, robots: Mapping[str, RobotSpec], scene: SceneSpec) -> None:
        """Several robots, placed and configured by ``scene.robots``; ``robots`` maps id -> preset."""
        if not scene.robots:
            raise ValueError("load_team needs a scene with robots")
        self.scene = scene
        self._team = [(f"{r.id}/", robots[r.id], r) for r in scene.robots]
        self.robot = self._team[0][1]

    def reset(self, scene: SceneSpec, *, seed: int) -> None:
        assert self.robot is not None, "load() first"
        self.scene = scene
        self._close_renderers()
        if self._team[0][2] is None:
            xml = build_scene(self.robot, scene, self.visual)
        else:
            xml = build_team_scene([(r.id, spec) for _, spec, r in self._team], scene, self.visual)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*obj_decoder.*")
            self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._index()
        q: dict[str, float] = {}
        for prefix, spec, placement in self._team:
            q.update({prefix + j: v for j, v in self._start_q(spec, placement or scene).items()})
        for j, v in q.items():
            self.data.qpos[self.model.jnt_qposadr[self.model.joint(j).id]] = v
        self._apply_mimics()
        self.command_joints(q)
        self.logic = SceneLogic(scene)
        self.logic.reset()
        self._contacts_ever = set()
        mujoco.mj_forward(self.model, self.data)
        self._index_pads()
        self.step(0.3)
        self.data.time = 0.0
        self._contacts_ever = set()
        self.logic.reset()
        self._initial = {o.name: tuple(float(v) for v in self.data.body(o.name).xpos)
                         for o in scene.objects}

    @staticmethod
    def _start_q(spec: RobotSpec, at) -> dict[str, float]:
        """Start joints of one robot from a scene (single robot) or a placement: home, lift, neck, base."""
        q = spec.with_home(at.home_pose).default_joint_positions()
        lift = at.lift if at.lift is not None else spec.pose_lift(at.home_pose)
        if lift is not None and spec.lift_joint:
            q[spec.lift_joint] = float(lift)
        for k, v in at.neck.items():
            q[spec.neck[k]] = float(v)
        if spec.mobile_base:
            mb = spec.mobile_base
            q[mb.x], q[mb.y], q[mb.yaw] = at.base
        return q

    def _member(self, name: str) -> tuple[str, RobotSpec, str]:
        """(prefix, preset, unprefixed name) of the robot a prefixed name belongs to."""
        for prefix, spec, _ in self._team:
            if prefix and name.startswith(prefix):
                return prefix, spec, name[len(prefix):]
        return self._team[0][0], self._team[0][1], name

    def joint_positions(self) -> dict[str, float]:
        return {n: float(self.data.qpos[a]) for n, a in self._qadr.items()}

    def command_joints(self, targets: Mapping[str, float]) -> None:
        for name, v in targets.items():
            act = self._act.get(name)
            if act is None:
                raise KeyError(f"no actuated joint {name!r}")
            lo, hi = self.model.actuator_ctrlrange[act]
            if self.model.actuator_ctrllimited[act]:
                v = min(max(v, lo), hi)
            self.data.ctrl[act] = v

    def set_gripper_force(self, arm: str, newtons: float) -> None:
        """Cap the finger actuators of ``arm`` at ``newtons`` until the next reset (which rebuilds the model)."""
        prefix, spec, local = self._member(arm)
        for f in spec.hand_of(local).fingers:
            self.model.actuator_forcerange[self._act[prefix + f.joint]] = (-newtons, newtons)

    def pose_clearance(self, prefix: str, joints: Mapping[str, float], cap: float = 0.05,
                       objects: bool = False) -> tuple[float, str]:
        """Smallest distance between the robot named by ``prefix`` and fixtures, other robots and (with
        ``objects``) free objects, and between its arms and the rest of its own body or its other arm, with
        ``joints`` (unprefixed names) set to the given values. The simulation state is left as it was.
        Returns the distance (m, capped at ``cap``) and the pair it was measured between. Self clearance
        leaves out what the robot description exempts: the links of each arm before
        ``arms.<name>.self_collision_from`` and the geoms matching ``collision.exempt_geoms``."""
        m, d = self.model, self.data
        qpos = d.qpos.copy()
        for name, v in joints.items():
            d.qpos[m.jnt_qposadr[m.joint(prefix + name).id]] = v
        mujoco.mj_kinematics(m, d)
        pairs = self._clearance_pairs(prefix, objects)
        best, part, ft = cap, "", np.zeros(6)
        for a, b in pairs:
            if np.linalg.norm(d.geom_xpos[a] - d.geom_xpos[b]) - m.geom_rbound[a] - m.geom_rbound[b] > best:
                continue
            dist = mujoco.mj_geomDistance(m, d, a, b, best, ft)
            if dist < best:
                best, part = dist, f"{m.body(int(m.geom_bodyid[a])).name} and {m.body(int(m.geom_bodyid[b])).name}"
        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)
        return float(best), part

    def _clearance_pairs(self, prefix: str, objects: bool) -> list[tuple[int, int]]:
        key = (prefix, objects)
        cache = self.__dict__.setdefault("_pair_cache", {})
        if key in cache:
            return cache[key]
        m = self.model
        spec = next(sp for p, sp, _ in self._team if p == prefix)
        scene_names = {s.name for s in (*self.scene.fixtures, *self.scene.objects)}
        object_names = {s.name for s in self.scene.objects}

        def body(g: int) -> str:
            return m.body(int(m.geom_bodyid[g])).name

        def collides(g: int) -> bool:
            return bool(m.geom_contype[g] or m.geom_conaffinity[g]) and m.geom(g).name != "floor"

        def subtree(root: int) -> set[int]:
            out = set()
            for i in range(m.nbody):
                c = i
                while c > 0 and c != root:
                    c = int(m.body_parentid[c])
                if c == root:
                    out.add(i)
            return out

        own = [g for g in range(m.ngeom) if collides(g) and body(g).startswith(prefix) and body(g) not in scene_names] \
            if prefix else [g for g in range(m.ngeom) if collides(g) and body(g) not in scene_names | {"world"}]
        others = [g for g in range(m.ngeom) if collides(g) and g not in set(own)
                  and (objects or body(g) not in object_names)]
        pairs = [(a, b) for a in own for b in others
                 if m.geom_contype[a] & m.geom_conaffinity[b] or m.geom_contype[b] & m.geom_conaffinity[a]
                 or body(b) in scene_names]
        arms = {}
        for name, arm in spec.arms.items():
            root = int(m.jnt_bodyid[m.joint(prefix + arm.self_collision_from).id])
            arms[name] = subtree(root)
        all_arm = set()
        for name, arm in spec.arms.items():
            all_arm |= subtree(int(m.jnt_bodyid[m.joint(prefix + arm.joints[0]).id]))
        body_geoms = [g for g in own if int(m.geom_bodyid[g]) not in all_arm and not fnmatch(m.geom(g).name, spec.exempt_geoms)]
        arm_geoms = {n: [g for g in own if int(m.geom_bodyid[g]) in b] for n, b in arms.items()}
        names = list(arm_geoms)
        for n in names:
            pairs += [(a, b) for a in arm_geoms[n] for b in body_geoms]
        for i, n in enumerate(names):
            for n2 in names[i + 1:]:
                pairs += [(a, b) for a in arm_geoms[n] for b in arm_geoms[n2]]
        cache[key] = pairs
        return pairs

    def set_object_pose(self, name: str, pos, quat) -> None:
        """Put free object ``name`` at ``pos`` with orientation ``quat`` (w, x, y, z), at rest."""
        j = self.model.joint(f"{name}_free")
        a, v = int(self.model.jnt_qposadr[j.id]), int(self.model.jnt_dofadr[j.id])
        self.data.qpos[a:a + 7] = [*pos, *quat]
        self.data.qvel[v:v + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def step(self, seconds: float) -> None:
        n_ticks = max(1, int(round(seconds / CONTROL_DT)))
        substeps = int(round(CONTROL_DT / self.model.opt.timestep))
        for _ in range(n_ticks):
            self._scene_tick()
            for _ in range(substeps):
                mujoco.mj_step(self.model, self.data)
                self._collect_contacts()

    def sim_time(self) -> float:
        return float(self.data.time)

    def cameras(self) -> Sequence[str]:
        return [self.model.camera(i).name for i in range(self.model.ncam)]

    def render(self, camera: str, width: int, height: int) -> np.ndarray:
        key = (height, width)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height, width)
        r = self._renderers[key]
        r.update_scene(self.data, camera=camera)
        return r.render().copy()

    def oracle(self) -> OracleSnapshot:
        d = self.data
        objs = {}
        for group, kind in ((self.scene.objects, "object"), (self.scene.fixtures, "fixture")):
            for s in group:
                b = d.body(s.name)
                vel = np.zeros(6)
                if kind == "object":
                    mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_BODY, b.id, vel, 0)
                objs[s.name] = ObjectState(s.name, kind, s.label, tuple(float(v) for v in b.xpos),
                                           tuple(float(v) for v in b.xquat), s.size,
                                           tuple(float(v) for v in vel[3:]))
        arts = {}
        for a in self.scene.articulated:
            arts[a.name] = self._articulated_state(a)
        return OracleSnapshot(
            time=float(d.time), objects=objs, articulated=arts, base=self.base_pose(),
            base_footprint=(0.0, 0.0),
            tcp={p + arm: tuple(float(v) for v in d.site(f"{p}{a.end_effector}_tcp").xpos)
                 for p, spec, _ in self._team for arm, a in spec.arms.items()},
            gripper_opening={p + arm: self._opening(p + arm) for p, spec, _ in self._team for arm in spec.arms},
            contacts=frozenset(self._current_contacts()), contacts_ever=frozenset(self._contacts_ever),
            held_by=self._held(), initial=dict(self._initial), regions=dict(self.scene.regions),
            bases={r.id: self.base_pose(p) for p, _, r in self._team if r is not None})

    def render_depth(self, camera: str, width: int, height: int) -> np.ndarray:
        """Depth image of ``camera`` in metres along its optical axis (ideal, noise-free)."""
        key = (height, width)
        if key not in self._depth_renderers:
            self._depth_renderers[key] = mujoco.Renderer(self.model, height, width)
            self._depth_renderers[key].enable_depth_rendering()
        r = self._depth_renderers[key]
        r.update_scene(self.data, camera=camera)
        return r.render().copy()

    def close(self) -> None:
        self._close_renderers()

    def _close_renderers(self) -> None:
        for r in (*self._renderers.values(), *self._depth_renderers.values()):
            r.close()
        self._renderers.clear()
        self._depth_renderers.clear()

    # --- helpers --------------------------------------------------------------
    def base_pose(self, prefix: str | None = None) -> tuple[float, float, float]:
        """Base pose of the robot with ``prefix`` (default: the first robot)."""
        prefix = self._team[0][0] if prefix is None else prefix
        spec = next(s for p, s, _ in self._team if p == prefix)
        mb = spec.mobile_base
        if not mb:
            r = next(r for p, _, r in self._team if p == prefix)
            return tuple(r.base) if r is not None else (0.0, 0.0, 0.0)
        q = self.joint_positions()
        return (q[prefix + mb.x], q[prefix + mb.y], q[prefix + mb.yaw])

    def _index(self) -> None:
        m = self.model
        self._qadr = {m.joint(i).name: int(m.jnt_qposadr[i]) for i in range(m.njnt)
                      if int(m.jnt_type[i]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))}
        self._act = {}
        for i in range(m.nu):
            self._act[m.joint(m.actuator_trnid[i, 0]).name] = i
        self._mimic = []
        for i in range(m.neq):
            if int(m.eq_type[i]) == int(mujoco.mjtEq.mjEQ_JOINT):
                self._mimic.append((int(m.jnt_qposadr[m.eq_obj1id[i]]), int(m.jnt_qposadr[m.eq_obj2id[i]]),
                                    float(m.eq_data[i, 0]), float(m.eq_data[i, 1])))
        # geom owner names for contact bookkeeping
        owner_of_body: dict[int, str] = {}
        hand_root = {self.model.body(pre + ee.link).id: f"{pre}{arm}_hand" for pre, spec, _ in self._team
                     for arm, a in spec.arms.items() for ee in [spec.end_effectors[a.end_effector]]}
        pads = {self.model.body(pre + p).id: f"{pre}{arm}_pad{k}" for pre, spec, _ in self._team
                for arm, a in spec.arms.items() for k, p in enumerate(spec.end_effectors[a.end_effector].pad_links)}
        prefixes = [pre for pre, _, _ in self._team if pre]
        scene_names = {s.name for s in (*self.scene.fixtures, *self.scene.objects)}
        art_names = {a.name for a in self.scene.articulated}
        robot_bodies = set()
        for b in range(m.nbody):
            name = m.body(b).name
            chain, cur = [], b
            while cur > 0:
                chain.append(cur)
                cur = m.body_parentid[cur]
            owner = None
            for c in chain:
                if c in pads:
                    owner = pads[c]
                    break
                if c in hand_root:
                    owner = hand_root[c]
                    break
            if owner is None:
                top = m.body(chain[-1]).name if chain else ""
                if name in scene_names:
                    owner = name
                elif any(name == a or name.startswith(a + "_") for a in art_names):
                    owner = next(a for a in art_names if name == a or name.startswith(a + "_"))
                    if name.endswith("_mount"):
                        owner = f"{owner}_mount"
                elif top in scene_names:
                    owner = top
                else:
                    owner = next((f"{pre}robot" for pre in prefixes if name.startswith(pre)), "robot")
                    robot_bodies.add(b)
            owner_of_body[b] = owner
        self._geom_owner = [("floor" if m.geom(g).name == "floor" else owner_of_body[int(m.geom_bodyid[g])])
                            for g in range(m.ngeom)]
        self._pad_names = {v for v in pads.values()}

    def _apply_mimics(self) -> None:
        for fa, la, off, mult in self._mimic:
            self.data.qpos[fa] = off + mult * self.data.qpos[la]

    def _pairs(self):
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = self._geom_owner[c.geom1], self._geom_owner[c.geom2]
            if a != b:
                yield a, b

    @staticmethod
    def _norm(a: str, b: str) -> list[tuple[str, str]]:
        out = []
        for x, y in ((a, b), (b, a)):
            out.append((x, y))
            if "_pad" in y:
                out.append((x, y.split("_pad")[0] + "_hand"))
        return out

    def _current_contacts(self) -> set[tuple[str, str]]:
        s = set()
        for a, b in self._pairs():
            s.update(self._norm(a, b))
        return s

    def _collect_contacts(self) -> None:
        for a, b in self._pairs():
            self._contacts_ever.update(self._norm(a, b))
        if self._gripper_geoms:
            self._collect_off_pad()

    def _index_pads(self) -> None:
        """For end effectors that declare a pad band: which collision geoms belong to each gripper,
        and per jaw link the approach axis in the link frame and the fingertip coordinate along it."""
        m, d = self.model, self.data
        self._gripper_geoms: dict[int, str] = {}
        self._jaws: dict[int, tuple[np.ndarray, float, float]] = {}
        self._object_names = {o.name for o in self.scene.objects}
        for pre, spec, arm, a in [(pre, spec, arm, a) for pre, spec, _ in self._team for arm, a in spec.arms.items()]:
            ee = spec.end_effectors[a.end_effector]
            if ee.pad_band <= 0:
                continue
            root = m.body(pre + ee.link).id
            approach = d.site(f"{pre}{ee.name}_tcp").xmat.reshape(3, 3) @ np.array(ee.approach_axis, float)
            pad_bodies = {m.body(pre + p).id for p in ee.pad_links}
            tips: dict[int, float] = {}
            for g in range(m.ngeom):
                if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                    continue
                b = cur = int(m.geom_bodyid[g])
                while cur > 0 and cur != root:
                    cur = int(m.body_parentid[cur])
                if cur != root:
                    continue
                self._gripper_geoms[g] = pre + arm
                if b in pad_bodies:
                    axis = d.xmat[b].reshape(3, 3).T @ approach
                    half = _support(m, g, axis)
                    tips[b] = max(tips.get(b, -np.inf), float(m.geom_pos[g] @ axis + half))
            for b, tip in tips.items():
                self._jaws[b] = (d.xmat[b].reshape(3, 3).T @ approach, tip, ee.pad_band)

    def _collect_off_pad(self) -> None:
        """Record ``(object, "<arm>_offpad")`` when an object touches a gripper anywhere but the
        fingertip pads within the declared band."""
        d, m = self.data, self.model
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in self._gripper_geoms and self._geom_owner[g2] in self._object_names:
                g, obj = g1, self._geom_owner[g2]
            elif g2 in self._gripper_geoms and self._geom_owner[g1] in self._object_names:
                g, obj = g2, self._geom_owner[g1]
            else:
                continue
            b = int(m.geom_bodyid[g])
            jaw = self._jaws.get(b)
            if jaw is not None:
                axis, tip, band = jaw
                depth = tip - float((d.xmat[b].reshape(3, 3).T @ (c.pos - d.xpos[b])) @ axis)
                if -1e-3 <= depth <= band:
                    continue
            part = f"{self._gripper_geoms[g]}_offpad"
            self._contacts_ever.update(((obj, part), (part, obj)))

    def _held(self) -> dict[str, str]:
        cur = self._current_contacts()
        held = {}
        for o in self.scene.objects:
            for pre, spec, _ in self._team:
                for arm, a in spec.arms.items():
                    n = len(spec.end_effectors[a.end_effector].pad_links)
                    if n and all((o.name, f"{pre}{arm}_pad{k}") in cur for k in range(n)):
                        held[o.name] = pre + arm
        return held

    def _opening(self, arm: str) -> float:
        prefix, spec, local = self._member(arm)
        ee = spec.hand_of(local)
        if not ee.fingers:
            return 0.0
        q = self.joint_positions()
        frac = np.mean([(q[prefix + f.joint] - f.open) / (f.closed - f.open) for f in ee.fingers])
        return float(max(0.0, ee.max_opening * (1.0 - frac)))

    def _articulated_q(self, a) -> float:
        q = self.joint_positions()
        if a.type == "sliding_door":
            return q[f"{a.name}_l_joint"] + q[f"{a.name}_r_joint"]
        return q[f"{a.name}_joint"]

    def _articulated_state(self, a) -> ArticulatedState:
        q = self._articulated_q(a)
        t = float(self.data.time)
        normal = (math.cos(a.yaw), math.sin(a.yaw), 0.0)
        if a.type == "button":
            return ArticulatedState(a.name, a.type, q, pressed=self.logic.button_pressed(a, q),
                                    pressed_ever=self.logic.pressed_ever(a.name), lamp=self.logic.lamp(a, t),
                                    pos=a.pos, normal=normal)
        if a.type == "sliding_door":
            width = (a.size or (0.03, 0.9, 2.0))[1]
            return ArticulatedState(a.name, a.type, q, open_fraction=float(np.clip(q / width, 0, 1)),
                                    pos=a.pos, normal=normal)
        rng = self.model.jnt_range[self.model.joint(f"{a.name}_joint").id]
        return ArticulatedState(a.name, a.type, q, open_fraction=float(np.clip(q / rng[1], 0, 1)),
                                pos=a.pos, normal=normal)

    def _scene_tick(self) -> None:
        t = float(self.data.time)
        qs = {a.name: self._articulated_q(a) for a in self.scene.articulated}
        self.logic.update(t, qs)
        for a in self.scene.articulated:
            if a.type == "sliding_door":
                width = (a.size or (0.03, 0.9, 2.0))[1]
                target = self.logic.door_target(a, t) * width / 2
                self.data.ctrl[self._act[f"{a.name}_l_joint"]] = target
                self.data.ctrl[self._act[f"{a.name}_r_joint"]] = target
            elif a.type == "button":
                lamp = self.model.geom(f"{a.name}_lamp").id
                self.model.geom_rgba[lamp] = (1.0, 0.75, 0.1, 1.0) if self.logic.lamp(a, t) else (0.35, 0.35, 0.38, 1.0)


def _support(m: mujoco.MjModel, g: int, axis: np.ndarray) -> float:
    """Half extent of collision geom ``g`` along ``axis`` (body frame): exact for boxes, the
    bounding radius otherwise."""
    if int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_BOX):
        return float(m.geom_rbound[g])
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, m.geom_quat[g])
    return float(np.abs(rot.reshape(3, 3).T @ axis) @ m.geom_size[g])

"""The robot API every subject uses: the same functions as LLM tools, as a Python
module for written programs, and for hand-written scripted subjects.

Only methods decorated with :func:`exposed` are offered to models; their
signatures and docstrings are the documentation models see (``docgen``).
Coordinates are metres in the robot base frame (x forward, y left, z up from
the floor under the robot); angles are degrees.

A call that cannot be carried out raises :class:`Refused` before anything
moves; the reason is returned to the model.
"""
from __future__ import annotations

import functools
import io
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .kinematics.ik import ik_solver
from .kinematics.model import APPROACHES, Kinematics

if TYPE_CHECKING:
    from .session import Session

Arm = Literal["left", "right"]
Approach = Literal["down", "forward", "left", "right", "keep"]

CARTESIAN_SPEED = 0.15      # m/s, straight-line hand speed
SETTLE = 0.25               # s after each motion
BASE_SPEED_FRACTION = 0.6   # of the robot's maximum base speed
BUMP_BACKOFF = 0.03         # m the base retreats after its body touches something
LATERAL_TOLERANCE = 1e-4    # m of left_m a differential base treats as zero


class Refused(Exception):
    """Nothing moved; the message says why."""


@dataclass(frozen=True)
class ImageResult:
    camera: str
    png: bytes
    width: int
    height: int

    def to_json(self) -> dict[str, Any]:
        return {"camera": self.camera, "width": self.width, "height": self.height,
                "note": "image attached"}


def exposed(fn):
    """Mark a method as part of the model-facing API and route it through the session."""
    @functools.wraps(fn)
    def wrapper(self: "RobotAPI", *args, **kwargs):
        name = fn.__name__
        call_args = dict(zip(list(_params(fn))[: len(args)], args), **kwargs)
        self.session.before_api_call(name, call_args)
        try:
            result = fn(self, *args, **kwargs)
        except Refused as exc:
            self.session.after_api_call(name, call_args, None, refused=str(exc))
            raise
        logged = result.to_json() if isinstance(result, ImageResult) else result
        self.session.after_api_call(name, call_args, logged)
        return result
    wrapper.__exposed__ = True
    return wrapper


def _params(fn):
    import inspect
    return [p for p in inspect.signature(fn).parameters if p != "self"]


def exposed_functions() -> dict[str, Any]:
    return {n: getattr(RobotAPI, n) for n in dir(RobotAPI)
            if getattr(getattr(RobotAPI, n), "__exposed__", False)}


def _r(v, nd=3):
    return [round(float(x), nd) for x in v]


class RobotAPI:
    def __init__(self, session: "Session"):
        self.session = session
        self.spec = session.robot
        self.kin = Kinematics(self.spec)
        self.ik = ik_solver(self.spec.ik_solver, self.kin)
        q = self.session.backend.joint_positions()
        self._origin = self._base_pose(q)

    # --- frames ------------------------------------------------------------------
    def _base_pose(self, q=None) -> tuple[float, float, float]:
        q = q or self.session.backend.joint_positions()
        mb = self.spec.mobile_base
        return (q[mb.x], q[mb.y], q[mb.yaw]) if mb else (0.0, 0.0, 0.0)

    def world_to_base(self, p) -> np.ndarray:
        x, y, yaw = self._base_pose()
        dx, dy = p[0] - x, p[1] - y
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([c * dx + s * dy, -s * dx + c * dy, p[2]])

    def vec_to_base(self, v) -> np.ndarray:
        yaw = self._base_pose()[2]
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]])

    def _q(self) -> dict[str, float]:
        q = self.session.backend.joint_positions()
        self.kin.set_q(q)
        return q

    def _check_arm(self, arm: str) -> None:
        if arm not in self.spec.arms:
            raise Refused(f"unknown arm {arm!r}; arms: {', '.join(self.spec.arms)}")

    # --- motion primitives (not exposed) -------------------------------------------
    def _run_joint_path(self, joints: list[str], path: list[np.ndarray], durations: list[float]) -> None:
        """Follow joint waypoints (``path[0]`` is the start) with linear interpolation."""
        segs = list(zip(path[:-1], path[1:], durations))
        total = sum(durations)
        bounds = np.cumsum([0.0] + durations)

        def on_tick(t: float) -> None:
            k = int(np.searchsorted(bounds, t, side="right") - 1)
            k = min(max(k, 0), len(segs) - 1)
            a, b, d = segs[k]
            u = 1.0 if d <= 0 else min(1.0, (t - bounds[k]) / d)
            self.session.command_joints(dict(zip(joints, a + (b - a) * u)))

        self.session.advance(total + self.session.control_dt, on_tick)
        self.session.command_joints(dict(zip(joints, path[-1])))
        self.session.advance(SETTLE)

    def _joint_move(self, joints: list[str], target, speed: float) -> None:
        q = self.session.backend.joint_positions()
        start = np.array([q[j] for j in joints])
        target = np.array(target, float)
        dur = max(0.2, float(np.max(np.abs(target - start))) / speed)
        self._run_joint_path(joints, [start, target], [dur])

    def _rotation_for(self, arm: str, approach: str, wrist_deg: float) -> np.ndarray:
        if approach == "keep":
            return self.kin.tcp_pose(arm)[1]
        if approach not in APPROACHES:
            raise Refused(f"approach must be one of {', '.join([*APPROACHES, 'keep'])}")
        return self.kin.named_rotation(arm, approach, math.radians(wrist_deg))

    def _cartesian(self, arm: str, target, rotations: list[np.ndarray]) -> dict[str, Any]:
        """Straight-line move to ``target`` with the first of ``rotations`` that is reachable."""
        q = self._q()
        joints = list(self.spec.arms[arm].joints)
        start_p, start_r = self.kin.tcp_pose(arm)
        target = np.array(target, float)
        for rotation in rotations:
            end = self.ik.solve_ik(arm, target, rotation, q)
            if end.ok:
                break
        if not end.ok:
            raise Refused(f"cannot reach ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}) with that "
                          f"orientation: {end.reason}")
        dist = float(np.linalg.norm(target - start_p))
        n = max(1, int(math.ceil(dist / 0.01)))
        path = [np.array([q[j] for j in joints])]
        seed = dict(q)
        cartesian_ok = True
        q_start = _quat(start_r)
        q_end = _quat(rotation)
        for i in range(1, n + 1):
            u = i / n
            p = start_p + (target - start_p) * u
            r = _quat_to_mat(_slerp(q_start, q_end, u))
            sol = self.ik.solve_ik(arm, p, r, seed, restarts=0, iters=100)
            if not sol.ok or np.max(np.abs(np.array(sol.q) - path[-1])) > 0.35:
                cartesian_ok = False
                break
            path.append(np.array(sol.q))
            seed.update(zip(joints, sol.q))
        speed = self.spec.arms[arm].max_joint_speed
        if cartesian_ok:
            seg_time = max(dist / n / CARTESIAN_SPEED, 1e-3)
            durs = [max(seg_time, float(np.max(np.abs(b - a))) / speed) for a, b in zip(path[:-1], path[1:])]
            self._run_joint_path(joints, path, durs)
        else:
            self._joint_move(joints, end.q, speed)
        self._q()
        reached, _ = self.kin.tcp_pose(arm)
        err = float(np.linalg.norm(reached - target))
        out = {"arm": arm, "tcp": _r(reached), "error_mm": round(err * 1000, 1),
               "path": "straight line" if cartesian_ok else "joint space (no straight-line path)"}
        if err > 0.01:
            out["note"] = "stopped short of the target: something may be in the way"
        return out

    # --- exposed API --------------------------------------------------------------
    @exposed
    def get_state(self) -> dict:
        """Read the robot's own state: base pose relative to where the episode started, torso lift,
        and for each arm the gripper (TCP) position in the base frame, the direction the jaws point
        and the jaw opening."""
        q = self._q()
        x, y, yaw = self._base_pose(q)
        ox, oy, oyaw = self._origin
        c, s = math.cos(oyaw), math.sin(oyaw)
        arms = {}
        for arm in self.spec.arms:
            p, _ = self.kin.tcp_pose(arm)
            arms[arm] = {"tcp": _r(p), "jaw_direction": _r(self.kin.approach_axis(arm), 2),
                         "gripper_opening_m": round(self._opening(arm, q), 4)}
        state = {"sim_time_s": round(self.session.backend.sim_time(), 2),
                 "base_odometry": {"x": round(c * (x - ox) + s * (y - oy), 3),
                                   "y": round(-s * (x - ox) + c * (y - oy), 3),
                                   "yaw_deg": round(math.degrees(yaw - oyaw), 1)},
                 "arms": arms}
        if self.spec.lift_joint:
            state["lift_m"] = round(q[self.spec.lift_joint], 3)
        return state

    def _opening(self, arm: str, q) -> float:
        ee = self.spec.hand_of(arm)
        if not ee.fingers:
            return 0.0
        frac = np.mean([(q[f.joint] - f.open) / (f.closed - f.open) for f in ee.fingers])
        return float(max(0.0, ee.max_opening * (1.0 - frac)))

    @exposed
    def list_objects(self) -> list:
        """List what the robot perceives: every object, fixture, button, door and marked area,
        with its centre position in the base frame, its size (x, y, z extents in metres) and its
        yaw in degrees relative to the robot. Buttons report the centre of their face, the
        direction the face points and whether their lamp is lit; doors report how far open they are."""
        snap = self.session.oracle()
        if snap is None:
            raise Refused("this backend has no perception")
        yaw0 = self._base_pose()[2]
        out = []
        scene = self.session.episode.scene
        colors = {s.name: s.color for s in (*scene.fixtures, *scene.objects)}
        for name, o in snap.objects.items():
            w, x, y, z = o.quat
            yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)) - yaw0)
            yaw = (yaw + 180) % 360 - 180
            item = {"name": name, "kind": o.label, "position": _r(self.world_to_base(o.pos)),
                    "size": _r(_extent(o)), "yaw_deg": round(yaw, 1)}
            if isinstance(colors.get(name), str):
                item["color"] = colors[name]
            out.append(item)
        for a in scene.articulated:
            st = snap.articulated[a.name]
            if a.type == "button":
                protrude = float(a.params.get("protrusion", 0.012)) + float(a.params.get("plate_thickness", 0.006))
                face = np.array(a.pos) + np.array(st.normal) * protrude
                out.append({"name": a.name, "kind": "push button", "position": _r(self.world_to_base(face)),
                            "facing": _r(self.vec_to_base(st.normal), 2),
                            "diameter_m": round(2 * float(a.params.get("radius", 0.022)), 3),
                            "color": a.color if isinstance(a.color, str) else "",
                            "lamp": "lit" if st.lamp else "off"})
            else:
                size = a.size or (0.03, 0.9, 2.0)
                out.append({"name": a.name, "kind": a.type.replace("_", " "),
                            "position": _r(self.world_to_base(a.pos)), "width_m": round(size[1], 3),
                            "open_fraction": round(st.open_fraction, 2)})
        for r in scene.regions.values():
            centre = (np.array(r.lo) + np.array(r.hi)) / 2
            out.append({"name": r.name, "kind": "area", "position": _r(self.world_to_base(centre)),
                        "size": _r(np.array(r.hi) - np.array(r.lo))})
        return out

    @exposed
    def move_ee(self, arm: Arm, x: float, y: float, z: float, approach: Approach = "down",
                wrist_deg: float = 0.0) -> dict:
        """Move a gripper's TCP (the point midway between the fingertip pads) in a straight line to
        (x, y, z) in the base frame. ``approach`` is the direction the jaws point: "down" (toward
        the floor), "forward", "left", "right", or "keep" (the current orientation). ``wrist_deg``
        turns the jaws about that direction. At 0 the jaws close along the robot's y axis (for
        "down" and "forward") or x axis (for "left" and "right"). For "down", a positive angle turns
        them counter-clockwise seen from above, so ``wrist_deg`` equal to an object's ``yaw_deg``
        closes the jaws across a pair of its faces. The jaws are alike, so ``wrist_deg`` and
        ``wrist_deg + 180`` grip the same way; whichever is reachable is used. Refused if the pose
        is out of reach."""
        self._check_arm(arm)
        self._q()
        rotations = [self._rotation_for(arm, approach, wrist_deg)]
        if approach != "keep" and self.spec.hand_of(arm).type == "parallel_gripper":
            # the two jaws are alike: a half turn about the approach axis grips the same way
            rotations.append(self._rotation_for(arm, approach, wrist_deg + 180.0))
        return self._cartesian(arm, (x, y, z), rotations)

    @exposed
    def move_ee_by(self, arm: Arm, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> dict:
        """Move a gripper's TCP in a straight line by (dx, dy, dz) metres in the base frame,
        keeping its orientation."""
        self._check_arm(arm)
        self._q()
        p, rot = self.kin.tcp_pose(arm)
        return self._cartesian(arm, p + np.array([dx, dy, dz]), [rot])

    @exposed
    def open_gripper(self, arm: Arm) -> dict:
        """Open a gripper fully. Returns the measured jaw opening."""
        return self._gripper(arm, close=False)

    @exposed
    def close_gripper(self, arm: Arm) -> dict:
        """Close a gripper until the jaws stop. Returns the measured opening: an opening wider
        than a few millimetres means the jaws closed on something of about that width."""
        return self._gripper(arm, close=True)

    def _gripper(self, arm: str, close: bool) -> dict:
        self._check_arm(arm)
        ee = self.spec.hand_of(arm)
        lim = self.session.gripper.get(arm)
        targets = ({f.joint: (f.closed if close else f.open) for f in ee.fingers} if lim is None
                   else lim.grip_targets(ee, close))
        self.session.command_joints(targets)
        self.session.advance(0.9)
        q = self._q()
        opening = self._opening(arm, q)
        out = {"arm": arm, "opening_m": round(opening, 4)}
        if close:
            floor = 0.0 if lim is None else self._opening(arm, targets)
            out["closed_on_something"] = opening > floor + 0.005
        note = lim.grip_note(ee, close) if lim is not None else ""
        if note:
            out["gripper"] = note
        return out

    @exposed
    def go_home(self, arm: Arm) -> dict:
        """Move an arm back to its home posture in joint space. Raise the gripper clear of
        anything below it first: this motion is not a straight line."""
        self._check_arm(arm)
        a = self.spec.arms[arm]
        self._joint_move(list(a.joints), a.home, a.max_joint_speed)
        self._q()
        return {"arm": arm, "tcp": _r(self.kin.tcp_pose(arm)[0])}

    @exposed
    def set_lift(self, height_m: float) -> dict:
        """Raise or lower the torso. ``height_m`` is the lift travel, 0 (lowest) to the maximum."""
        if not self.spec.lift_joint:
            raise Refused("this robot has no torso lift")
        lo, hi = self.spec.lift_range
        if not lo <= height_m <= hi:
            raise Refused(f"lift height must be within [{lo}, {hi}] m")
        self._joint_move([self.spec.lift_joint], [height_m], self.spec.lift_speed)
        return {"lift_m": round(self._q()[self.spec.lift_joint], 3)}

    @exposed
    def look_at(self, x: float, y: float, z: float) -> dict:
        """Turn the head so its camera centres the point (x, y, z) in the base frame."""
        if "head" not in self.spec.cameras or not self.spec.neck:
            raise Refused("this robot has no movable head camera")
        pan_j, tilt_j = self.spec.neck.get("pan"), self.spec.neck.get("tilt")
        q = self._q()
        target = np.array([x, y, z], float)
        best = (q[pan_j], q[tilt_j])
        cam = self.spec.cameras["head"].link

        def misalign(pan: float, tilt: float) -> float:
            self.kin.set_q({**q, pan_j: pan, tilt_j: tilt})
            body = self.kin.data.body(cam)
            fwd = body.xmat.reshape(3, 3)[:, 2]
            d = target - body.xpos
            return 1.0 - float(np.dot(fwd, d / np.linalg.norm(d)))

        lo_p, hi_p = self.kin._range[pan_j]
        lo_t, hi_t = self.kin._range[tilt_j]
        for step in (0.2, 0.05, 0.01, 0.002):
            p0, t0 = best
            cands = [(min(max(p0 + i * step, lo_p), hi_p), min(max(t0 + j * step, lo_t), hi_t))
                     for i in range(-3, 4) for j in range(-3, 4)]
            best = min(cands, key=lambda c: misalign(*c))
        self._joint_move([pan_j, tilt_j], best, 1.5)
        return {"pan_deg": round(math.degrees(best[0]), 1), "tilt_deg": round(math.degrees(best[1]), 1),
                "centred": misalign(*best) < 0.002}

    @exposed
    def move_base(self, forward_m: float = 0.0, left_m: float = 0.0, turn_deg: float = 0.0) -> dict:
        """Drive the mobile base by ``forward_m`` (negative = backward) and ``left_m`` in the current
        base frame while turning by ``turn_deg`` (positive = counter-clockwise). A differential-drive
        base (two driven wheels) cannot move sideways or diagonally: it refuses a nonzero ``left_m``,
        so turn toward a point to the side, drive forward, then turn back; ``forward_m`` together
        with ``turn_deg`` drives an arc of that length. A holonomic base translates in a straight line
        while turning. The base stops if its body bumps into something, backs off a few centimetres
        and reports what it touched. Returns the measured motion in the frame the base had before
        the call."""
        mb = self.spec.mobile_base
        if mb is None:
            raise Refused("this robot has a fixed base")
        if mb.differential and abs(left_m) > LATERAL_TOLERANCE:
            raise Refused("this robot's base is differential drive (two wheels): it cannot move sideways or "
                          "diagonally, so left_m must be 0. To reach a point to the side, turn toward it "
                          "(turn_deg), drive forward (forward_m), then turn back")
        if math.hypot(forward_m, left_m) > 3.0 or abs(turn_deg) > 180:
            raise Refused("one move_base call may translate at most 3 m and turn at most 180 deg")
        x0, y0, yaw0 = self._base_pose()
        c, s = math.cos(yaw0), math.sin(yaw0)
        turn = math.radians(turn_deg)
        dur = max(math.hypot(forward_m, left_m) / (mb.max_speed * BASE_SPEED_FRACTION),
                  abs(turn) / (mb.max_yaw_rate * BASE_SPEED_FRACTION), 0.2)
        start = np.array([x0, y0, yaw0])
        if mb.differential:
            # constant (v, w) over the same duration: forward plus turn is an arc
            arc = _unicycle(start, forward_m / dur, turn / dur)
            retreat = -BUMP_BACKOFF * math.copysign(1.0, forward_m) if forward_m else 0.0

            def path(t: float) -> np.ndarray:
                return arc(min(t, dur))

            def backoff(pose: np.ndarray) -> np.ndarray:
                return retreat * np.array([math.cos(pose[2]), math.sin(pose[2]), 0.0])
        else:
            goal = np.array([x0 + c * forward_m - s * left_m, y0 + s * forward_m + c * left_m, yaw0 + turn])
            step = (goal - start) / max(np.linalg.norm((goal - start)[:2]), 1e-6)

            def path(t: float) -> np.ndarray:
                return start + (goal - start) * min(1.0, t / dur)

            def backoff(pose: np.ndarray) -> np.ndarray:
                return -BUMP_BACKOFF * np.array([step[0], step[1], 0.0])
        bumped = self._drive_base(path, dur, backoff)
        x, y, yaw = self._base_pose()
        dx, dy = x - x0, y - y0
        out = {"moved_forward_m": round(c * dx + s * dy, 3), "moved_left_m": round(-s * dx + c * dy, 3),
               "turned_deg": round(math.degrees(yaw - yaw0), 1)}
        if bumped:
            out["bumped_into"] = bumped
        return out

    def _drive_base(self, path, dur: float, backoff) -> list[str]:
        """Command the planar base joints along ``path(t)`` (x, y, yaw in the world) for ``dur``
        seconds. On body contact, stop and move ``backoff(pose)`` from the measured pose."""
        mb = self.spec.mobile_base
        names = (mb.x, mb.y, mb.yaw)
        bumped: list[str] = []

        class _Bump(Exception):
            pass

        def on_tick(t: float) -> None:
            touching = self._body_contacts()
            if touching:
                bumped.extend(touching)
                raise _Bump()
            self.session.command_joints(dict(zip(names, path(t))))

        try:
            self.session.advance(dur + 0.3, on_tick)
        except _Bump:
            pose = np.array(self._base_pose())
            self.session.command_joints(dict(zip(names, pose + backoff(pose))))
            self.session.advance(0.4)
        return sorted(set(bumped))

    def _body_contacts(self) -> list[str]:
        snap = self.session.oracle()
        if snap is None:
            return []
        return [b for a, b in snap.contacts if a == "robot" and b not in ("floor",)]

    @exposed
    def wait(self, seconds: float) -> dict:
        """Do nothing for ``seconds`` (at most 30) of simulated time."""
        if not 0 <= seconds <= 30:
            raise Refused("wait between 0 and 30 seconds")
        self.session.advance(seconds)
        return {"sim_time_s": round(self.session.backend.sim_time(), 2)}

    @exposed
    def camera_image(self, camera: str = "head") -> ImageResult:
        """Take a photo with one of the robot's cameras ("head", "left_wrist", "right_wrist")."""
        if camera not in self.spec.cameras:
            raise Refused(f"unknown camera {camera!r}; cameras: {', '.join(self.spec.cameras)}")
        c = self.spec.cameras[camera]
        w, h = min(c.width, 640), min(c.height, 480)
        img = self.session.backend.render(camera, w, h)
        return ImageResult(camera, encode_png(img), w, h)


def encode_png(img: np.ndarray) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def _extent(o) -> tuple[float, ...]:
    if o.label == "tray" or len(o.size) == 3:
        return tuple(o.size)
    if len(o.size) == 2:
        return (2 * o.size[0], 2 * o.size[0], o.size[1])
    return (2 * o.size[0],) * 3


def _unicycle(start: np.ndarray, v: float, w: float):
    """Pose (x, y, yaw) at time ``t`` of a base starting at ``start`` with constant forward speed
    ``v`` and yaw rate ``w``: the exact integral of x' = v cos(yaw), y' = v sin(yaw), yaw' = w,
    so the velocity has no component across the heading at any instant."""
    x0, y0, yaw0 = (float(v_) for v_ in start)

    def pose(t: float) -> np.ndarray:
        yaw = yaw0 + w * t
        if abs(w) < 1e-9:
            return np.array([x0 + v * t * math.cos(yaw0), y0 + v * t * math.sin(yaw0), yaw0])
        r = v / w
        return np.array([x0 + r * (math.sin(yaw) - math.sin(yaw0)), y0 - r * (math.cos(yaw) - math.cos(yaw0)), yaw])
    return pose


def _quat(m: np.ndarray) -> np.ndarray:
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(m, float).flatten())
    return q


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    import mujoco
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def _slerp(a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + u * (b - a)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - u) * th) * a + math.sin(u * th) * b) / math.sin(th)

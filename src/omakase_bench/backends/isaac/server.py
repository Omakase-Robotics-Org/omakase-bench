"""Isaac Sim backend server: runs inside an Isaac Lab Python on a GPU host.

    python -m omakase_bench.backends.isaac.server --port 8990 [--device cuda:0]

Serves the Backend Protocol over TCP (one client at a time) with frames of a
4-byte big-endian length and a JSON body. Every ``reset`` builds a fresh stage
from the scene description, so a reset is exactly the scene the client
resolved for its seed. The robot is the preset's URDF converted to USD with
Isaac Lab's ``UrdfConverter`` (cached by content hash); articulated fixtures
(buttons, doors, drawers) are small generated URDFs. Scene behaviour (latched
presses, lamps, powered doors) comes from the same ``SceneLogic`` the MuJoCo
backend uses.

The process exits when its client disconnects, or after ``--idle-timeout``
seconds without a client.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import socket
import struct
import sys
import time
import traceback
from pathlib import Path

PROTOCOL = 1
CONTROL_DT = 0.02
PHYSICS_DT = 1.0 / 200.0


def _send(conn: socket.socket, obj) -> None:
    data = json.dumps(obj).encode()
    conn.sendall(struct.pack("!I", len(data)) + data)


def _recv(conn: socket.socket, parent: int):
    """Next frame, or None when the client has gone (closed socket, or the launching
    process died and this server was re-parented)."""
    head = b""
    while len(head) < 4:
        try:
            chunk = conn.recv(4 - len(head))
        except socket.timeout:
            if os.getppid() != parent:
                return None
            continue
        if not chunk:
            return None
        head += chunk
    n = struct.unpack("!I", head)[0]
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = conn.recv(min(1 << 20, n - len(buf)))
        except socket.timeout:
            continue
        if not chunk:
            return None
        buf += chunk
    return json.loads(buf)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8990)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache", default=str(Path.home() / ".cache" / "omakase-bench" / "usd"))
    ap.add_argument("--idle-timeout", type=float, default=900.0)
    args, rest = ap.parse_known_args()

    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    app_args = lp.parse_args(rest)
    app_args.headless = True
    app_args.enable_cameras = True
    app = AppLauncher(app_args).app   # noqa: F841  (keeps Kit alive)

    world = IsaacWorld(device=args.device, cache=Path(args.cache))
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.bind, args.port))
    srv.listen(1)
    srv.settimeout(args.idle_timeout)
    print(f"OMAKASE_BENCH_ISAAC_READY port={args.port}", flush=True)
    try:
        conn, _ = srv.accept()
    except socket.timeout:
        print("no client within the idle timeout; exiting", flush=True)
        os._exit(0)
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn.settimeout(10.0)
    parent = os.getppid()
    try:
        while True:
            msg = _recv(conn, parent)
            if msg is None or msg.get("cmd") == "close":
                break
            try:
                reply = world.handle(msg)
                reply["ok"] = True
            except Exception as exc:   # report every failure to the client instead of dying
                reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                         "traceback": traceback.format_exc()[-3000:]}
            _send(conn, reply)
    except OSError as exc:
        print(f"client connection lost: {exc}", flush=True)
    finally:
        sys.stdout.flush()
        # SimulationApp.close() can hang for minutes in headless mode, and a normal
        # interpreter exit closes it; nothing is left to save
        os._exit(0)


def _urdf_box(name: str, size, xyz=(0, 0, 0), rgba=(0.7, 0.7, 0.7, 1)) -> str:
    return (f'<visual><origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}"/><geometry><box size="{size[0]} {size[1]} {size[2]}"/>'
            f'</geometry><material name="{name}"><color rgba="{rgba[0]} {rgba[1]} {rgba[2]} 1"/></material></visual>'
            f'<collision><origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}"/><geometry><box size="{size[0]} {size[1]} {size[2]}"/>'
            f'</geometry></collision>')


def _inertial(mass: float) -> str:
    i = max(mass * 1e-3, 1e-6)
    return f'<inertial><mass value="{mass}"/><inertia ixx="{i}" iyy="{i}" izz="{i}" ixy="0" ixz="0" iyz="0"/></inertial>'


class IsaacWorld:
    def __init__(self, device: str, cache: Path):
        from importlib.metadata import PackageNotFoundError, version
        self.device = device
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self.versions = {}
        for pkg in ("isaacsim", "isaaclab"):
            try:
                self.versions[pkg] = version(pkg)
            except PackageNotFoundError:
                self.versions[pkg] = "?"
        self.sim = None
        self.robot_spec = None
        self.scene = None

    # --- protocol --------------------------------------------------------------
    def handle(self, msg):
        cmd = msg["cmd"]
        if cmd == "hello":
            return {"protocol": PROTOCOL, "versions": self.versions, "device": self.device,
                    "physics_dt": PHYSICS_DT, "control_dt": CONTROL_DT}
        if cmd == "reset":
            from omakase_bench.core.task import scene_from_dict
            from omakase_bench.robots.spec import load_robot
            self.scene = scene_from_dict(msg["scene"])
            self.robot_spec = load_robot(msg["robot"]).with_home(self.scene.home_pose)
            t0 = time.time()
            self._build()
            return {"build_seconds": round(time.time() - t0, 2), "time": self.time}
        if cmd == "step":
            if msg.get("targets"):
                self._set_targets(msg["targets"])
            self._advance(float(msg["seconds"]))
            return {"time": self.time}
        if cmd == "joints":
            return {"q": self._joint_positions()}
        if cmd == "oracle":
            return {"snapshot": self._oracle()}
        if cmd == "render":
            return {"png": self._render(msg["camera"], int(msg["width"]), int(msg["height"]))}
        if cmd == "cameras":
            return {"cameras": sorted(self.cameras)}
        raise ValueError(f"unknown command {cmd!r}")

    # --- building -------------------------------------------------------------------
    def _usd_for_urdf(self, urdf_text: str, name: str) -> str:
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
        digest = hashlib.sha256(urdf_text.encode()).hexdigest()[:16]
        out = self.cache / f"{name}-{digest}"
        usd = out / f"{name}.usd"
        if usd.exists():
            return str(usd)
        out.mkdir(parents=True, exist_ok=True)
        src = out / f"{name}.urdf"
        src.write_text(urdf_text)
        cfg = UrdfConverterCfg(
            asset_path=str(src), usd_dir=str(out), usd_file_name=f"{name}.usd", force_usd_conversion=True,
            fix_base=True, merge_fixed_joints=False, self_collision=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                target_type="position", gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=10.0)))
        return UrdfConverter(cfg).usd_path

    def _build(self) -> None:
        import isaaclab.sim as sim_utils
        import torch
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
        from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
        from isaaclab.sim import SimulationCfg, SimulationContext
        from omakase_bench.core.scene_logic import SceneLogic
        from omakase_bench.core.task import rgba

        if self.sim is not None:
            self.sim.clear_all_callbacks()
            self.sim.clear_instance()
        sim_utils.create_new_stage()
        self.sim = SimulationContext(SimulationCfg(dt=PHYSICS_DT, device=self.device, render_interval=1))
        spec, scene = self.robot_spec, self.scene
        self.mimic = _mimics(spec.urdf_path.read_text())

        ground = sim_utils.GroundPlaneCfg(visible=False)
        ground.func("/World/ground", ground, translation=(0.0, 0.0, -0.002))
        floor = sim_utils.CuboidCfg(size=(30.0, 30.0, 0.01), visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=tuple(rgba(scene.floor_color)[:3]), roughness=0.8))
        floor.func("/World/floor", floor, translation=(0.0, 0.0, -0.008))
        light = sim_utils.DomeLightCfg(intensity=700.0, color=(1.0, 1.0, 1.0))
        light.func("/World/dome", light)
        key = sim_utils.DistantLightCfg(intensity=1600.0, angle=1.0)
        key.func("/World/key", key, orientation=(0.92, 0.2, 0.2, 0.25))

        # robot
        joints = [n for n in _joint_names(spec.urdf_path.read_text())]
        groups = self._actuator_groups(joints)
        self.robot = Articulation(ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=self._usd_for_urdf(_absolute_meshes(spec.visual_urdf()), spec.name),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=1.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False, solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1)),
            init_state=ArticulationCfg.InitialStateCfg(joint_pos=self._initial_q()),
            actuators={name: ImplicitActuatorCfg(joint_names_expr=[f"^{j}$" for j in js], stiffness=k, damping=d,
                                                 effort_limit_sim=e)
                       for name, (js, k, d, e) in groups.items() if js}))

        # fixtures (kinematic) and free objects
        self.rigid: dict[str, object] = {}
        self.owner: dict[str, str] = {}
        self.parts: dict[str, list[str]] = {}

        def cuboid(path, size, pos, yaw, color, *, kinematic, mass=0.1, friction=1.0):
            q = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
            return RigidObject(RigidObjectCfg(
                prim_path=path,
                spawn=sim_utils.CuboidCfg(
                    size=tuple(size),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=kinematic, disable_gravity=kinematic),
                    mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=friction, dynamic_friction=friction),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(rgba(color)[:3]), roughness=0.6)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(pos), rot=q)))

        for s in scene.fixtures:
            if s.type == "tray":
                x, y, z = s.size
                w = s.wall
                c, sn = math.cos(s.yaw), math.sin(s.yaw)
                parts = [((x, y, w), (0, 0, -z / 2 + w / 2)),
                         ((w, y, z), (x / 2 - w / 2, 0, 0)), ((w, y, z), (-x / 2 + w / 2, 0, 0)),
                         ((x, w, z), (0, y / 2 - w / 2, 0)), ((x, w, z), (0, -y / 2 + w / 2, 0))]
                names = []
                for k, (size, (px, py, pz)) in enumerate(parts):
                    pos = (s.pos[0] + c * px - sn * py, s.pos[1] + sn * px + c * py, s.pos[2] + pz)
                    n = f"{s.name}_part{k}"
                    self.rigid[n] = cuboid(f"/World/Fixtures/{n}", size, pos, s.yaw, s.color, kinematic=True,
                                           friction=s.friction)
                    self.owner[n] = s.name
                    names.append(n)
                self.parts[s.name] = names
            else:
                self.rigid[s.name] = cuboid(f"/World/Fixtures/{s.name}", _box_size(s), s.pos, s.yaw, s.color,
                                            kinematic=True, friction=s.friction)
                self.owner[s.name] = s.name
                self.parts[s.name] = [s.name]
        for s in scene.objects:
            self.rigid[s.name] = cuboid(f"/World/Objects/{s.name}", _box_size(s), s.pos, s.yaw, s.color,
                                        kinematic=False, mass=s.mass, friction=s.friction)
            self.owner[s.name] = s.name
            self.parts[s.name] = [s.name]

        # articulated fixtures as generated URDFs
        self.articulated: dict[str, object] = {}
        self.art_moving: dict[str, list[str]] = {}
        for a in scene.articulated:
            urdf, actuators, moving = _articulated_urdf(a, rgba)
            q = (math.cos(a.yaw / 2), 0.0, 0.0, math.sin(a.yaw / 2))
            art = Articulation(ArticulationCfg(
                prim_path=f"/World/Articulated/{a.name}",
                spawn=sim_utils.UsdFileCfg(usd_path=self._usd_for_urdf(urdf, f"art_{a.type}"),
                                           activate_contact_sensors=True,
                                           rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)),
                init_state=ArticulationCfg.InitialStateCfg(pos=tuple(a.pos), rot=q),
                actuators={k: ImplicitActuatorCfg(joint_names_expr=[v[0]], stiffness=v[1], damping=v[2],
                                                  effort_limit_sim=v[3]) for k, v in actuators.items()}))
            self.articulated[a.name] = art
            self.art_moving[a.name] = moving

        # contact sensors: per-arm pads and hands, and the robot body as a bumper
        targets = [f"/World/Fixtures/{n}" for s in scene.fixtures for n in self.parts[s.name]]
        targets += [f"/World/Objects/{s.name}" for s in scene.objects]
        targets += [f"/World/Articulated/{a.name}/{link}" for a in scene.articulated for link in self.art_moving[a.name]]
        self.filter_owners = [self._owner_of(p) for p in targets]
        self.sensors = {}
        for arm, a in spec.arms.items():
            ee = spec.end_effectors[a.end_effector]
            pads = "|".join(ee.pad_links)
            self.sensors[f"{arm}_pads"] = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/Robot/({pads})", filter_prim_paths_expr=targets, force_threshold=0.0))
            prefix = ee.link.rsplit("_", 1)[0]
            self.sensors[f"{arm}_hand"] = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/Robot/{prefix}_.*", filter_prim_paths_expr=targets, force_threshold=0.0))
        self.sensors["robot_body"] = ContactSensor(ContactSensorCfg(
            prim_path="/World/Robot/(base_link|torso_column_link|head_link|neck_pan_link)",
            filter_prim_paths_expr=targets, force_threshold=0.0))

        # cameras: robot cameras on their optical frames (ROS convention), scene cameras by look-at
        self.cameras = {}
        for name, c in spec.cameras.items():
            focal = 24.0
            vaperture = 2 * focal * math.tan(math.radians(c.fovy) / 2)
            self.cameras[name] = Camera(CameraCfg(
                prim_path=f"/World/Robot/{c.link}/{name}_cam", update_period=0, height=480, width=640,
                data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(
                    focal_length=focal, horizontal_aperture=vaperture * 640 / 480, vertical_aperture=vaperture,
                    clipping_range=(0.02, 30.0)),
                offset=CameraCfg.OffsetCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0), convention="ros")))
        self.scene_cams = {}
        for c in scene.cameras:
            focal = 24.0
            vaperture = 2 * focal * math.tan(math.radians(c.fovy) / 2)
            self.cameras[c.name] = Camera(CameraCfg(
                prim_path=f"/World/SceneCameras/{c.name}", update_period=0, height=480, width=640, data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=focal, horizontal_aperture=vaperture * 640 / 480,
                                                 vertical_aperture=vaperture, clipping_range=(0.05, 40.0))))
            self.scene_cams[c.name] = (c.pos, c.lookat)

        self.sim.reset()
        # initial states are only defaults until written to the simulation
        q0 = self.robot.data.default_joint_pos.clone()
        self.robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
        for art in self.articulated.values():
            aq = art.data.default_joint_pos.clone()
            art.write_joint_state_to_sim(aq, torch.zeros_like(aq))
        self._bind_pad_friction()
        for name, (pos, look) in self.scene_cams.items():
            self.cameras[name].set_world_poses_from_view(torch.tensor([pos], device=self.device, dtype=torch.float32),
                                                         torch.tensor([look], device=self.device, dtype=torch.float32))
        self.joint_index = {n: i for i, n in enumerate(self.robot.joint_names)}
        self.body_index = {n: i for i, n in enumerate(self.robot.body_names)}
        self.targets = self.robot.data.default_joint_pos.clone()
        self.logic = SceneLogic(scene)
        self.logic.reset()
        self.contacts_ever: set[tuple[str, str]] = set()
        self.time = 0.0
        self._lamp_state: dict[str, bool] = {}
        self._advance(0.3)
        self.time = 0.0
        self.contacts_ever = set()
        self.logic.reset()
        self.initial = {s.name: tuple(float(v) for v in self.rigid[s.name].data.root_pos_w[0].tolist())
                        for s in scene.objects}

    def _owner_of(self, path: str) -> str:
        last = path.rstrip("/").split("/")
        if path.startswith("/World/Articulated/"):
            return last[3]
        return self.owner.get(last[-1], last[-1])

    def _bind_pad_friction(self) -> None:
        import isaaclab.sim as sim_utils
        from pxr import UsdPhysics
        stage = sim_utils.get_current_stage()
        mat = sim_utils.RigidBodyMaterialCfg(static_friction=2.0, dynamic_friction=2.0, friction_combine_mode="max")
        mat.func("/World/Materials/pad", mat)
        for ee in self.robot_spec.end_effectors.values():
            for link in ee.pad_links:
                root = stage.GetPrimAtPath(f"/World/Robot/{link}")
                for prim in [root, *root.GetAllChildren()] if root.IsValid() else []:
                    for p in [prim, *prim.GetAllChildren()]:
                        if p.HasAPI(UsdPhysics.CollisionAPI):
                            sim_utils.bind_physics_material(str(p.GetPath()), "/World/Materials/pad")

    def _initial_q(self) -> dict[str, float]:
        spec, scene = self.robot_spec, self.scene
        q = spec.default_joint_positions()
        if scene.lift is not None and spec.lift_joint:
            q[spec.lift_joint] = float(scene.lift)
        for k, v in scene.neck.items():
            q[spec.neck[k]] = float(v)
        if spec.mobile_base:
            mb = spec.mobile_base
            q[mb.x], q[mb.y], q[mb.yaw] = scene.base
        for follower, (leader, mult, off) in self.mimic.items():
            q[follower] = off + mult * q.get(leader, 0.0)
        return q

    def _actuator_groups(self, joints):
        spec = self.robot_spec
        mb = spec.mobile_base
        base = [mb.x, mb.y, mb.yaw] if mb else []
        fingers = [f.joint for ee in spec.end_effectors.values() for f in ee.fingers]
        arms_hi = [j for a in spec.arms.values() for j in a.joints[:4]]
        arms_lo = [j for a in spec.arms.values() for j in a.joints[4:]]
        neck = list(spec.neck.values())
        lift = [spec.lift_joint] if spec.lift_joint else []
        followers = list(self.mimic)
        return {"base": (base, 1.0e6, 1.0e5, 1.0e5), "lift": (lift, 2.0e5, 2.0e4, 4000.0),
                "neck": (neck, 200.0, 20.0, 50.0), "arm_hi": (arms_hi, 1500.0, 60.0, 300.0),
                "arm_lo": (arms_lo, 400.0, 16.0, 80.0), "fingers": (fingers, 4000.0, 150.0, 100.0),
                "linkage": (followers, 60.0, 2.0, 100.0)}

    # --- stepping ------------------------------------------------------------------
    def _set_targets(self, targets) -> None:
        for name, v in targets.items():
            self.targets[0, self.joint_index[name]] = float(v)
        for follower, (leader, mult, off) in self.mimic.items():
            if leader in targets:
                self.targets[0, self.joint_index[follower]] = off + mult * float(targets[leader])

    def _advance(self, seconds: float) -> None:
        import torch
        ticks = max(1, int(round(seconds / CONTROL_DT)))
        sub = int(round(CONTROL_DT / PHYSICS_DT))
        for _ in range(ticks):
            self._scene_tick(torch)
            self.robot.set_joint_position_target(self.targets)
            for _ in range(sub):
                self.robot.write_data_to_sim()
                for a in self.articulated.values():
                    a.write_data_to_sim()
                self.sim.step(render=False)
                self.time += PHYSICS_DT
                self.robot.update(PHYSICS_DT)
                for a in self.articulated.values():
                    a.update(PHYSICS_DT)
                for r in self.rigid.values():
                    r.update(PHYSICS_DT)
                for s in self.sensors.values():
                    s.update(PHYSICS_DT)
                self.contacts_ever |= self._contacts()

    def _art_q(self, a) -> float:
        art = self.articulated[a.name]
        q = art.data.joint_pos[0]
        if a.type == "sliding_door":
            return float(q.sum())
        return float(q[0])

    def _scene_tick(self, torch) -> None:
        qs = {a.name: self._art_q(a) for a in self.scene.articulated}
        self.logic.update(self.time, qs)
        for a in self.scene.articulated:
            art = self.articulated[a.name]
            if a.type == "sliding_door":
                width = (a.size or (0.03, 0.9, 2.0))[1]
                target = self.logic.door_target(a, self.time) * width / 2
                art.set_joint_position_target(torch.full_like(art.data.joint_pos, target))
            elif a.type == "button":
                lit = self.logic.lamp(a, self.time)
                if self._lamp_state.get(a.name) != lit:
                    self._lamp_state[a.name] = lit
                    _set_lamp(f"/World/Articulated/{a.name}", lit)
            else:
                art.set_joint_position_target(torch.zeros_like(art.data.joint_pos))

    def _contacts(self) -> set[tuple[str, str]]:
        out: set[tuple[str, str]] = set()
        for key, sensor in self.sensors.items():
            fm = sensor.data.force_matrix_w
            if fm is None:
                continue
            hit = (fm[0].norm(dim=-1) > 1e-4)          # (bodies, filters)
            if key.endswith("_pads"):
                arm = key[: -len("_pads")]
                for b, body in enumerate(sensor.body_names):
                    for f in hit[b].nonzero().flatten().tolist():
                        k = self.robot_spec.end_effectors[self.robot_spec.arms[arm].end_effector].pad_links.index(body)
                        o = self.filter_owners[f]
                        out |= {(o, f"{arm}_pad{k}"), (f"{arm}_pad{k}", o)}
            else:
                part = "robot" if key == "robot_body" else key
                for f in hit.any(dim=0).nonzero().flatten().tolist():
                    o = self.filter_owners[f]
                    out |= {(o, part), (part, o)}
        return out

    # --- reading ---------------------------------------------------------------------
    def _joint_positions(self) -> dict[str, float]:
        q = self.robot.data.joint_pos[0].tolist()
        return {n: q[i] for n, i in self.joint_index.items()}

    def _oracle(self) -> dict:
        spec, scene = self.robot_spec, self.scene
        objects = {}
        for group, kind in ((scene.fixtures, "fixture"), (scene.objects, "object")):
            for s in group:
                if s.type == "tray":
                    pos, quat, vel = s.pos, (math.cos(s.yaw / 2), 0.0, 0.0, math.sin(s.yaw / 2)), (0, 0, 0)
                else:
                    d = self.rigid[s.name].data
                    pos, quat, vel = d.root_pos_w[0].tolist(), d.root_quat_w[0].tolist(), d.root_lin_vel_w[0].tolist()
                objects[s.name] = {"name": s.name, "kind": kind, "label": s.label, "pos": list(pos), "quat": list(quat),
                                   "size": list(s.size), "lin_vel": list(vel)}
        arts = {}
        for a in scene.articulated:
            q = self._art_q(a)
            normal = [math.cos(a.yaw), math.sin(a.yaw), 0.0]
            st = {"name": a.name, "kind": a.type, "q": q, "pos": list(a.pos), "normal": normal,
                  "pressed": False, "pressed_ever": False, "open_fraction": 0.0, "lamp": False}
            if a.type == "button":
                st.update(pressed=self.logic.button_pressed(a, q), pressed_ever=self.logic.pressed_ever(a.name),
                          lamp=self.logic.lamp(a, self.time))
            elif a.type == "sliding_door":
                width = (a.size or (0.03, 0.9, 2.0))[1]
                st["open_fraction"] = max(0.0, min(1.0, q / width))
            else:
                hi = float(a.params.get("max_deg", 100)) * math.pi / 180 if a.type == "hinged_door" else \
                    (a.size or (0.35, 0.4, 0.12))[0] * 0.8
                st["open_fraction"] = max(0.0, min(1.0, q / hi))
            arts[a.name] = st
        qd = self._joint_positions()
        mb = spec.mobile_base
        base = [qd[mb.x], qd[mb.y], qd[mb.yaw]] if mb else [0.0, 0.0, 0.0]
        tcp = {}
        opening = {}
        for arm, a in spec.arms.items():
            ee = spec.end_effectors[a.end_effector]
            tcp[arm] = self.robot.data.body_pos_w[0, self.body_index[ee.tcp_link]].tolist()
            frac = sum((qd[f.joint] - f.open) / (f.closed - f.open) for f in ee.fingers) / max(len(ee.fingers), 1)
            opening[arm] = max(0.0, ee.max_opening * (1.0 - frac))
        current = self._contacts()
        held = {}
        for s in scene.objects:
            for arm, a in spec.arms.items():
                n = len(spec.end_effectors[a.end_effector].pad_links)
                if n and all((s.name, f"{arm}_pad{k}") in current for k in range(n)):
                    held[s.name] = arm
        hand_pairs = set()
        for o, p in self.contacts_ever | current:
            if "_pad" in p:
                hand_pairs.add((o, p.split("_pad")[0] + "_hand"))
        return {"time": self.time, "objects": objects, "articulated": arts, "base": base,
                "tcp": tcp, "gripper_opening": opening,
                "contacts": sorted(map(list, current | {(o, p.split("_pad")[0] + "_hand") for o, p in current if "_pad" in p})),
                "contacts_ever": sorted(map(list, self.contacts_ever | hand_pairs)),
                "held_by": held, "initial": self.initial}

    def _render(self, camera: str, width: int, height: int) -> str:
        from PIL import Image
        cam = self.cameras[camera]
        self.sim.render()
        cam.update(0.0, force_recompute=True)
        img = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype("uint8")
        im = Image.fromarray(img)
        if im.size != (width, height):
            im = im.resize((width, height))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()


def _absolute_meshes(urdf_path: Path) -> str:
    """URDF text with mesh filenames made absolute, so the converted copy can live elsewhere."""
    import xml.etree.ElementTree as ET
    root = ET.parse(str(urdf_path)).getroot()
    for mesh in root.iter("mesh"):
        f = mesh.get("filename", "")
        if f and not os.path.isabs(f) and "://" not in f:
            mesh.set("filename", str((urdf_path.parent / f).resolve()))
    return ET.tostring(root, encoding="unicode")


def _box_size(s) -> tuple[float, float, float]:
    if s.type == "box":
        return tuple(s.size)
    if s.type == "cylinder":
        return (2 * s.size[0], 2 * s.size[0], s.size[1])
    return (2 * s.size[0],) * 3


def _joint_names(urdf: str) -> list[str]:
    import xml.etree.ElementTree as ET
    return [j.get("name") for j in ET.fromstring(urdf).iter("joint") if j.get("type") != "fixed"]


def _mimics(urdf: str) -> dict[str, tuple[str, float, float]]:
    import xml.etree.ElementTree as ET
    out = {}
    for j in ET.fromstring(urdf).iter("joint"):
        m = j.find("mimic")
        if m is not None:
            out[j.get("name")] = (m.get("joint"), float(m.get("multiplier", "1")), float(m.get("offset", "0")))
    return out


def _articulated_urdf(a, rgba):
    """URDF text, actuator settings {name: (joint regex, stiffness, damping, effort)} and the
    names of the moving links, for one articulated fixture in its own frame."""
    color = rgba(a.color)
    p = a.params
    if a.type == "button":
        radius = float(p.get("radius", 0.022))
        travel = float(p.get("travel", 0.008))
        protrude = float(p.get("protrusion", 0.012))
        plate_t = float(p.get("plate_thickness", 0.006))
        plate = a.size or (plate_t, 0.11, 0.16)
        pc = rgba(p.get("plate_color", "steel"))
        urdf = (f'<robot name="button"><link name="base">{_inertial(1.0)}{_urdf_box("plate", plate, (plate[0] / 2, 0, 0), pc)}'
                f'</link><link name="cap">{_inertial(0.02)}'
                f'<visual><origin xyz="{protrude / 2} 0 0" rpy="0 1.5708 0"/><geometry><cylinder radius="{radius}" '
                f'length="{protrude}"/></geometry><material name="cap"><color rgba="{color[0]} {color[1]} {color[2]} 1"/>'
                f'</material></visual><collision><origin xyz="{protrude / 2} 0 0" rpy="0 1.5708 0"/><geometry>'
                f'<cylinder radius="{radius}" length="{protrude}"/></geometry></collision></link>'
                f'<link name="lamp">{_inertial(0.001)}<visual><origin xyz="0.0005 0 0" rpy="0 1.5708 0"/><geometry>'
                f'<cylinder radius="{radius + 0.008}" length="0.001"/></geometry><material name="lamp">'
                f'<color rgba="0.35 0.35 0.38 1"/></material></visual></link>'
                f'<joint name="lamp_joint" type="fixed"><parent link="base"/><child link="lamp"/>'
                f'<origin xyz="{plate[0]} 0 0"/></joint>'
                f'<joint name="cap_joint" type="prismatic"><parent link="base"/><child link="cap"/>'
                f'<origin xyz="{plate[0]} 0 0"/><axis xyz="-1 0 0"/><limit lower="0" upper="{travel}" effort="50" '
                f'velocity="1"/></joint></robot>')
        return urdf, {"spring": ("cap_joint", float(p.get("stiffness", 300.0)), 2.0, 50.0)}, ["cap"]
    if a.type == "sliding_door":
        thick, width, height = a.size or (0.03, 0.9, 2.0)
        half = width / 2
        # 5 mm clear of the floor and the lintel, so the panels slide without rubbing
        links = "".join(
            f'<link name="panel_{s}">{_inertial(20.0)}{_urdf_box(f"p{s}", (thick, half, height - 0.01), (0, 0, 0), color)}</link>'
            f'<joint name="panel_{s}_joint" type="prismatic"><parent link="base"/><child link="panel_{s}"/>'
            f'<origin xyz="0 {sign * half / 2} {height / 2}"/><axis xyz="0 {sign} 0"/>'
            f'<limit lower="0" upper="{half}" effort="400" velocity="1"/></joint>'
            for s, sign in (("l", 1), ("r", -1)))
        urdf = f'<robot name="door"><link name="base">{_inertial(1.0)}</link>{links}</robot>'
        return urdf, {"motor": ("panel_.*_joint", 3000.0, 300.0, 400.0)}, ["panel_l", "panel_r"]
    if a.type == "hinged_door":
        thick, width, height = a.size or (0.03, 0.8, 2.0)
        hi = math.radians(float(p.get("max_deg", 100)))
        urdf = (f'<robot name="hinged"><link name="base">{_inertial(1.0)}</link><link name="leaf">{_inertial(8.0)}'
                f'{_urdf_box("leaf", (thick, width, height), (0, -width / 2, height / 2), color)}</link>'
                f'<joint name="hinge" type="revolute"><parent link="base"/><child link="leaf"/><axis xyz="0 0 1"/>'
                f'<limit lower="0" upper="{hi}" effort="100" velocity="2"/></joint></robot>')
        return urdf, {"hinge": ("hinge", 0.0, 5.0, 100.0)}, ["leaf"]
    if a.type == "drawer":
        depth, width, height = a.size or (0.35, 0.4, 0.12)
        urdf = (f'<robot name="drawer"><link name="base">{_inertial(1.0)}</link><link name="box">{_inertial(1.0)}'
                f'{_urdf_box("front", (0.02, width, height), (0, 0, 0), color)}</link>'
                f'<joint name="slide" type="prismatic"><parent link="base"/><child link="box"/><axis xyz="1 0 0"/>'
                f'<limit lower="0" upper="{depth * 0.8}" effort="100" velocity="1"/></joint></robot>')
        return urdf, {"slide": ("slide", 0.0, 5.0, 100.0)}, ["box"]
    raise ValueError(f"unknown articulated type {a.type!r}")


def _set_lamp(root: str, lit: bool) -> None:
    import isaaclab.sim as sim_utils
    from pxr import Gf, UsdShade
    stage = sim_utils.get_current_stage()
    color = Gf.Vec3f(1.0, 0.75, 0.1) if lit else Gf.Vec3f(0.35, 0.35, 0.38)
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith(root) and "lamp" in path.lower() and prim.IsA(UsdShade.Shader):
            for name in ("inputs:diffuse_color_constant", "inputs:diffuseColor", "inputs:diffuse_tint"):
                attr = prim.GetAttribute(name)
                if attr and attr.IsValid():
                    attr.Set(color)


if __name__ == "__main__":
    sys.exit(main())

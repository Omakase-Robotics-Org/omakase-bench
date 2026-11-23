"""Smoke test of a robot description: load it, solve IK around each arm's home, render stills.

    omakase-bench robot check <name or directory> [--out DIR]

It answers the questions a new ``robot.yaml`` raises before any task runs: does the URDF build, do the
names in ``robot.yaml`` exist in it, where is each tool centre point at home, how much of the space around
it does the IK solver reach with the gripper pointing down, and what do the cameras see.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .spec import RobotSpec, load_robot

#: IK targets: offsets from the home TCP (m) and wrist angles (deg), gripper pointing down
OFFSETS = (-0.10, 0.0, 0.10)
WRIST_DEG = (0.0, 90.0)
#: an outside view of the robot, placed from the height of its highest TCP
VIEW = {"direction": (1.0, -0.8, 0.6), "distance": 2.2, "fovy": 45.0}


def check(robot: RobotSpec, out: Path | None = None, echo=print) -> bool:
    """Print the findings; write PNG stills to ``out``; False when an arm reaches none of its targets."""
    from ..harness.kinematics.ik import ik_solver
    from ..harness.kinematics.model import Kinematics
    kin = Kinematics(robot)
    solver = ik_solver(robot.ik_solver, kin)
    q = robot.default_joint_positions()
    kin.set_q(q)
    base = robot.mobile_base
    echo(f"robot {robot.name} v{robot.version}: {kin.model.njnt} joints, "
         f"base {base.drive if base else 'fixed'}, lift {'yes' if robot.lift_joint else 'no'}, "
         f"neck {', '.join(robot.neck) or 'no'}, cameras {', '.join(robot.cameras) or 'none'}, IK solver {robot.ik_solver}")
    good = True
    top = 0.0
    for name, arm in robot.arms.items():
        home, _ = kin.tcp_pose(name)
        top = max(top, float(home[2]))
        solved = total = 0
        for dx in OFFSETS:
            for dy in OFFSETS:
                for dz in OFFSETS:
                    for wrist in WRIST_DEG:
                        total += 1
                        rotation = kin.named_rotation(name, "down", np.radians(wrist))
                        solved += solver.solve_ik(name, home + (dx, dy, dz), rotation, q).ok
        good = good and solved > 0
        echo(f"  arm {name}: {len(arm.joints)} joints, side {arm.side or 'any'}, home TCP "
             f"({home[0]:.3f}, {home[1]:.3f}, {home[2]:.3f}), IK reaches {solved}/{total} targets within "
             f"{max(OFFSETS):g} m of it")
    if out is not None:
        for path in render(robot, out, top):
            echo(f"  wrote {path}")
    return good


def render(robot: RobotSpec, out: Path, look_z: float = 1.0) -> list[Path]:
    """Stills of the robot at home in an empty scene: one outside view and one per robot camera."""
    from PIL import Image

    from ..backends.mujoco.backend import MujocoBackend
    from ..core.task import SceneCamera, SceneSpec
    d = np.array(VIEW["direction"], float)
    eye = np.array([0.0, 0.0, look_z * 0.7]) + d / np.linalg.norm(d) * VIEW["distance"]
    scene = SceneSpec(robot=robot.name, cameras=(SceneCamera("outside", tuple(eye), (0.0, 0.0, look_z * 0.7), VIEW["fovy"]),))
    backend = MujocoBackend()
    backend.load(robot, scene)
    backend.reset(scene, seed=0)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for camera in ("outside", *robot.cameras):
        path = out / f"{robot.name}_{camera}.png"
        Image.fromarray(backend.render(camera, 640, 480)).save(path)
        paths.append(path)
    backend.close()
    return paths


def main(name_or_dir: str, out: str | None = None) -> int:
    return 0 if check(load_robot(name_or_dir), Path(out) if out else None) else 1

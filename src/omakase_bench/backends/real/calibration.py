"""A robot's camera calibration file (schema ``omakase.camera_calibration/2``): lenses and mounts.

The file holds, per camera slot, the lens of the image as the camera daemon publishes it (intrinsics and a
distortion model) and the mount (the optical frame's pose in a parent link). With them a pixel becomes a ray
and a depth reading a point in the robot's base frame.

Distortion models and coefficient orders are the camera daemon's: ``none``; ``brown_conrady`` and
``inverse_brown_conrady`` with k1, k2, p1, p2, k3; ``brown_conrady_rational`` with k1, k2, p1, p2, k3, k4,
k5, k6; ``kannala_brandt`` (equidistant fisheye) with k1, k2, k3, k4. ``inverse_brown_conrady`` applies the
Brown-Conrady formula from the distorted image to the undistorted one, the other way round from
``brown_conrady``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

SCHEMA = "omakase.camera_calibration/2"
MODELS = ("none", "brown_conrady", "inverse_brown_conrady", "brown_conrady_rational", "kannala_brandt")


class CalibrationError(ValueError):
    """The file cannot be used; the message says which camera and why."""


def _brown(x: np.ndarray, y: np.ndarray, k: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    """The Brown-Conrady map of normalised coordinates (rational when eight coefficients are given)."""
    k1, k2, p1, p2, k3, k4, k5, k6 = (*k, 0.0, 0.0, 0.0)[:8] if len(k) == 5 else k
    r2 = x * x + y * y
    radial = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
    return (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)


def _invert(x: np.ndarray, y: np.ndarray, k: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    """The point the Brown-Conrady map sends to (x, y), by fixed-point iteration."""
    u, v = x.copy(), y.copy()
    for _ in range(20):
        fx, fy = _brown(u, v, k)
        u, v = u - (fx - x), v - (fy - y)
    return u, v


@dataclass(frozen=True)
class Lens:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str = "none"
    coefficients: tuple[float, ...] = ()

    def rays(self, u, v) -> np.ndarray:
        """Unit directions in the optical frame (x right, y down, z forward) of pixel centres (u, v). A
        fisheye pixel more than 90 degrees off the axis has a direction with z <= 0."""
        x = (np.asarray(u, float) - self.cx) / self.fx
        y = (np.asarray(v, float) - self.cy) / self.fy
        k = self.coefficients
        if self.model == "kannala_brandt":
            rd = np.hypot(x, y)
            theta = self._fisheye_angle(rd)
            planar = np.where(rd > 1e-12, np.sin(theta) / np.maximum(rd, 1e-12), 0.0)
            return np.stack([x * planar, y * planar, np.cos(theta)], axis=-1)
        if self.model in ("brown_conrady", "brown_conrady_rational"):
            x, y = _invert(x, y, k)
        elif self.model == "inverse_brown_conrady":
            x, y = _brown(x, y, k)
        d = np.stack([x, y, np.ones_like(x)], axis=-1)
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def _fisheye_angle(self, rd: np.ndarray) -> np.ndarray:
        """The angle off the axis whose distorted radius is ``rd``; NaN outside the range in which the
        polynomial grows with the angle (there the model has no unique inverse)."""
        k = self.coefficients
        grid = np.linspace(0.0, np.pi, 721)
        t2 = grid * grid
        value = grid * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))))
        falling = np.nonzero(np.diff(value) <= 0)[0]
        last = int(falling[0]) if len(falling) else len(grid) - 1
        theta = np.interp(rd, value[:last + 1], grid[:last + 1])
        return np.where(rd <= value[last], theta, np.nan)

    def pixels(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixel coordinates (u, v) of points in the optical frame. Except through a fisheye lens only
        points in front of the camera (z > 0) have one."""
        k = self.coefficients
        if self.model == "kannala_brandt":
            r = np.hypot(points[..., 0], points[..., 1])
            theta = np.arctan2(r, points[..., 2])
            t2 = theta * theta
            rd = theta * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))))
            scale = np.where(r > 1e-12, rd / np.maximum(r, 1e-12), 0.0)
            return points[..., 0] * scale * self.fx + self.cx, points[..., 1] * scale * self.fy + self.cy
        x, y = points[..., 0] / points[..., 2], points[..., 1] / points[..., 2]
        if self.model in ("brown_conrady", "brown_conrady_rational"):
            x, y = _brown(x, y, k)
        elif self.model == "inverse_brown_conrady":
            x, y = _invert(x, y, k)
        return x * self.fx + self.cx, y * self.fy + self.cy


@dataclass(frozen=True)
class Mount:
    """The optical frame (x right, y down, z forward) in ``parent_link``."""
    parent_link: str
    position: np.ndarray
    rotation: np.ndarray


@dataclass(frozen=True)
class Calibration:
    robot: str
    lenses: Mapping[str, Lens]
    mounts: Mapping[str, Mount]


def _rotation(quat_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def frame_offset(value) -> tuple[np.ndarray, np.ndarray]:
    """(position in metres, rotation matrix) of a frame given as ``{xyz_m, quat_xyzw}`` (the quaternion is
    optional) or as the text ``x,y,z`` or ``x,y,z,qx,qy,qz,qw`` of a command-line option."""
    if isinstance(value, Mapping):
        xyz, quat = value.get("xyz_m", (0.0, 0.0, 0.0)), value.get("quat_xyzw", (0.0, 0.0, 0.0, 1.0))
    else:
        numbers = [float(v) for v in str(value).split(",")]
        xyz, quat = numbers[:3], numbers[3:] or (0.0, 0.0, 0.0, 1.0)
    if len(xyz) != 3 or len(quat) != 4:
        raise CalibrationError(f"a frame offset is x, y, z in metres and optionally a quaternion x, y, z, w; got {value!r}")
    return np.array(xyz, float), _rotation(quat)


def _usable(layer: Mapping, where: str) -> None:
    """A layer whose gate failed is used only when the failure was overridden, as the schema requires."""
    gate = layer.get("gate") or {}
    if str(gate.get("verdict", "")).upper() == "FAIL" and not gate.get("override"):
        raise CalibrationError(f"{where}: its gate failed ({'; '.join(gate.get('reasons') or []) or 'no reason given'}) "
                               "and was not overridden")


def load_calibration(path: str | Path) -> Calibration:
    path = Path(path).expanduser()
    if not path.exists():
        raise CalibrationError(f"no camera calibration file at {path}")
    raw = json.loads(path.read_text())
    if raw.get("schema") != SCHEMA:
        raise CalibrationError(f"{path}: schema {raw.get('schema')!r}, expected {SCHEMA!r}")
    lenses, mounts = {}, {}
    for slot, cam in (raw.get("cameras") or {}).items():
        k = cam.get("intrinsics")
        if k:
            _usable(k, f"{path}: camera {slot} intrinsics")
            model = k["distortion"]["model"]
            coefficients = tuple(float(v) for v in k["distortion"]["coefficients"])
            need = {"none": (0,), "brown_conrady": (5,), "inverse_brown_conrady": (5,), "brown_conrady_rational": (8,),
                    "kannala_brandt": (4,)}.get(model)
            if need is None or (model != "none" and len(coefficients) not in need):
                raise CalibrationError(f"{path}: camera {slot}: distortion {model!r} with {len(coefficients)} "
                                       f"coefficients; models: {', '.join(MODELS)}")
            lenses[slot] = Lens(int(cam["stream"]["width"]), int(cam["stream"]["height"]), float(k["fx"]), float(k["fy"]),
                                float(k["cx"]), float(k["cy"]), model, coefficients)
        m = cam.get("mount")
        if m:
            _usable(m, f"{path}: camera {slot} mount")
            pose = m["T_parent_camera"]
            mounts[slot] = Mount(m["parent_link"], np.array(pose["xyz_m"], float), _rotation(pose["quat_xyzw"]))
    return Calibration(str(raw.get("robot", "")), lenses, mounts)

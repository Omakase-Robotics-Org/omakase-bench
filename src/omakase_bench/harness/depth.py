"""Depth lookup: a pixel of a robot camera becomes a point in the robot base frame.

The backend supplies the camera's depth image (the optional ``depth_camera``
capability); the pixel is deprojected with the camera's pinhole intrinsics from
the preset and its pose from forward kinematics of the measured joints. The
kinematic model holds the robot only, so no object state can enter, and
everything used here exists on a real robot: an aligned depth image, factory
intrinsics and the kinematic chain.

A pixel outside the view, or a depth outside the range the preset declares for
that camera, yields no point.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..core.backend import BackendUnsupported
from ..robots.spec import RobotSpec
from .kinematics.model import Kinematics

_LOCATE = re.compile(r"^[ \t]*locate[ \t]*:[ \t]*([A-Za-z_]\w*)[ \t,]+\(?[ \t]*(-?\d+)[ \t]*[, \t][ \t]*(-?\d+)",
                     re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class Lookup:
    view: str
    u: int
    v: int
    point_mm: tuple[int, int, int] | None   # base frame; None = no valid depth at the pixel
    reason: str = ""

    def text(self) -> str:
        where = (f"no depth{f' ({self.reason})' if self.reason else ''}" if self.point_mm is None
                 else "x {} y {} z {} mm".format(*self.point_mm))
        return f"lookup {self.view}({self.u},{self.v}) -> {where}"


def parse_lookups(text: str, views: Mapping[str, str], max_lookups: int = 3) -> list[tuple[str, int, int]]:
    """The first ``max_lookups`` lines ``locate: <view> <u> <v>`` of a reply that name a known view."""
    found = [(m.group(1).lower(), int(m.group(2)), int(m.group(3))) for m in _LOCATE.finditer(text)]
    return [f for f in found if f[0] in views][:max_lookups]


def depth_views(backend, views: Mapping[str, str]) -> dict[str, str]:
    """The views whose camera has depth on this backend. A backend whose depth covers only some cameras names
    them (``depth_cameras()``); one that does not is taken to have depth for every camera."""
    if not hasattr(backend, "render_depth"):
        return {}
    if not hasattr(backend, "depth_cameras"):
        return dict(views)
    cameras = set(backend.depth_cameras())
    return {view: camera for view, camera in views.items() if camera in cameras}


def camera_pose(kin: Kinematics, spec: RobotSpec, camera: str) -> tuple[np.ndarray, np.ndarray]:
    """(position, rotation) of the camera's optical frame (x right, y down, z forward) in the base
    frame, for the joints last given to ``kin.set_q``."""
    body = kin.data.body(spec.cameras[camera].link)
    return body.xpos.copy(), body.xmat.reshape(3, 3).copy()


def deproject(backend, kin: Kinematics, spec: RobotSpec, camera: str, u: int, v: int,
              view_width: int, view_height: int) -> tuple[np.ndarray | None, str]:
    """Base-frame point (m) seen at pixel (u, v) of a ``view_width`` x ``view_height`` view showing
    the camera's full field of view; (None, reason) when there is no valid depth."""
    if not hasattr(backend, "render_depth"):
        return None, "this backend has no depth cameras"
    cam = spec.cameras[camera]
    if cam.depth_range is None:
        return None, "this camera has no depth stream"
    if not (0 <= u < view_width and 0 <= v < view_height):
        return None, "pixel outside the view"
    w, h = cam.width, cam.height
    uf, vf = (u + 0.5) * w / view_width, (v + 0.5) * h / view_height
    try:
        z = float(backend.render_depth(camera, w, h)[int(vf), int(uf)])
    except BackendUnsupported:
        # a sensor that stopped delivering frames: the lookup has no answer, the episode goes on
        return None, "no depth image from this camera"
    lo, hi = cam.depth_range
    if not (lo <= z <= hi):
        return None, "outside the sensor's range"
    if hasattr(backend, "pixel_ray"):
        # a backend with calibrated lenses says where a pixel looks; the ideal pinhole below is the simulator's
        ray = backend.pixel_ray(camera, uf, vf, w, h)
        if ray is None:
            return None, "outside the calibrated part of the lens"
    else:
        f = 0.5 * h / math.tan(math.radians(cam.fovy) / 2)
        ray = np.array([(uf - w / 2) / f, (vf - h / 2) / f, 1.0])
    kin.set_q(backend.joint_positions())
    mount = backend.camera_mount(camera) if hasattr(backend, "camera_mount") else None
    if mount is None:
        pos, rot = camera_pose(kin, spec, camera)
    else:
        parent, offset, turn = mount                      # the calibrated optical frame in a link of the robot
        body = kin.data.body(parent)
        pos, rot = body.xpos + body.xmat.reshape(3, 3) @ offset, body.xmat.reshape(3, 3) @ turn
    return pos + rot @ (ray * z), ""


def lookup(backend, kin: Kinematics, spec: RobotSpec, views: Mapping[str, str], view: str, u: int, v: int,
           view_width: int, view_height: int) -> Lookup:
    """``views`` maps the names used in replies to camera names of the preset."""
    p, reason = deproject(backend, kin, spec, views[view], u, v, view_width, view_height)
    return Lookup(view, u, v, None if p is None else tuple(int(x) for x in np.round(p * 1000)), reason)

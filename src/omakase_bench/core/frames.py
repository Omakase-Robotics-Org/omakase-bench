"""Planar frame changes between the world and a robot base pose (x, y, yaw)."""
from __future__ import annotations

import math

import numpy as np


def to_world(base, p) -> np.ndarray:
    """A point (x, y) of the base frame, in the world."""
    x, y, yaw = base
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([x + c * p[0] - s * p[1], y + s * p[0] + c * p[1]])


def to_base(base, p) -> np.ndarray:
    """A world point in the base frame; coordinates past y are passed through."""
    x, y, yaw = base
    dx, dy = p[0] - x, p[1] - y
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * dx + s * dy, -s * dx + c * dy, *p[2:]])

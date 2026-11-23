"""Staunton chess pieces generated from documented proportions: a lathe mesh to look at, primitives to collide.

Every piece is described by its height and base diameter (a set's catalogue
numbers) and a profile in units of those two: radius as a fraction of the base
radius at a height given as a fraction of the piece's height. The profiles
follow the Staunton pattern (weighted base with a bead, tapering stem, collar,
head): pawn ball, rook turret with four crenels, bishop mitre with its slit,
queen coronet, king crown with a cross, and a knight whose horse head is an
extruded side silhouette on a turned base.

Head widths, the part the jaws close on 10-18 mm below the top, are the
fractions of the base diameter measured on Staunton sets (pawn ball 0.53,
bishop mitre 0.51, knight head 0.60 across, queen and king crown 0.55, rook
turret 0.74; stem under the collar about 0.5).

Collision geometry is a stack of primitives that follows the outline:
cylinders whose radius is the largest radius of their band (so a jaw that
clears the collision shape clears the visible piece), a sphere for the pawn
ball, boxes for the king's cross and the knight's neck, head and muzzle. The
mesh is visual only. Nothing here depends on a simulator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

KINDS = ("pawn", "knight", "bishop", "rook", "queen", "king")

# (height fraction, radius fraction of the base radius), bottom to top; the shared foot first
_FOOT = ((0.0, 1.0), (0.03, 1.0), (0.055, 0.93), (0.08, 0.97), (0.105, 0.86), (0.135, 0.78))
PROFILES: dict[str, tuple[tuple[float, float], ...]] = {
    "pawn": (*_FOOT, (0.22, 0.62), (0.50, 0.45), (0.56, 0.50), (0.585, 0.72), (0.625, 0.74), (0.65, 0.55),
             (0.675, 0.42)),
    "rook": (*_FOOT, (0.20, 0.66), (0.58, 0.56), (0.63, 0.60), (0.66, 0.70), (0.70, 0.62), (0.74, 0.62),
             (0.80, 0.74), (1.0, 0.74), (1.0, 0.56), (0.93, 0.56), (0.93, 0.0)),
    "bishop": (*_FOOT, (0.20, 0.60), (0.55, 0.42), (0.60, 0.48), (0.62, 0.68), (0.655, 0.70), (0.675, 0.45),
               (0.70, 0.40), (0.73, 0.46), (0.79, 0.51), (0.85, 0.50), (0.90, 0.40), (0.935, 0.24), (0.95, 0.14),
               (0.965, 0.18), (0.985, 0.15), (1.0, 0.0)),
    "queen": (*_FOOT, (0.18, 0.62), (0.52, 0.40), (0.58, 0.45), (0.60, 0.66), (0.635, 0.68), (0.655, 0.44),
              (0.72, 0.38), (0.80, 0.48), (0.86, 0.55), (0.885, 0.55), (0.885, 0.42), (0.91, 0.36),
              (0.935, 0.24), (0.95, 0.16), (0.97, 0.18), (0.99, 0.13), (1.0, 0.0)),
    "king": (*_FOOT, (0.17, 0.62), (0.52, 0.40), (0.58, 0.44), (0.60, 0.65), (0.63, 0.67), (0.65, 0.44),
             (0.70, 0.38), (0.79, 0.50), (0.84, 0.55), (0.865, 0.55), (0.875, 0.30), (0.885, 0.0)),
    "knight": (*_FOOT, (0.17, 0.70), (0.20, 0.74), (0.23, 0.72), (0.25, 0.60), (0.25, 0.0)),
}
#: pieces whose top rim is notched: kind -> (height fraction from which the rim is cut, notches, notch share)
CRENELS = {"rook": (0.90, 4, 0.35), "queen": (0.86, 10, 0.45)}
#: the jaws close on this part: kind -> width as a fraction of the base diameter (see the module notes)
HEAD_WIDTH = {"pawn": 0.53, "knight": 0.60, "bishop": 0.51, "rook": 0.74, "queen": 0.55, "king": 0.55}

# knight: side silhouette (u forward along the muzzle, v height fraction), u in base radii
_KNIGHT_OUTLINE = ((-0.74, 0.22), (-0.80, 0.42), (-0.78, 0.60), (-0.62, 0.80), (-0.42, 0.92), (-0.30, 0.985),
                   (-0.20, 1.0), (-0.10, 0.94), (0.05, 0.92), (0.30, 0.86), (0.55, 0.74), (0.80, 0.63),
                   (0.92, 0.56), (0.90, 0.47), (0.70, 0.45), (0.45, 0.48), (0.28, 0.44), (0.35, 0.34),
                   (0.52, 0.27), (0.55, 0.22))
#: half-thickness across the head (base radii): 0.60 of the base diameter, thinner towards the muzzle
_KNIGHT_HALF = 0.60
_KNIGHT_MUZZLE_HALF = 0.42


@dataclass(frozen=True)
class Primitive:
    """A collision primitive in the piece frame (origin at the piece's centre, z up, x along a
    knight's muzzle). ``size``: cylinder (radius, half-height), sphere (radius,), box half-extents."""
    type: str
    size: tuple[float, ...]
    pos: tuple[float, float, float]


def _check(kind: str) -> None:
    if kind not in PROFILES:
        raise ValueError(f"unknown Staunton piece {kind!r}; kinds: {', '.join(KINDS)}")


def profile(kind: str, height: float, base_d: float) -> np.ndarray:
    """The outline as (z from the bottom, radius) in metres; the pawn's ball is sampled onto it."""
    _check(kind)
    r0 = base_d / 2
    pts = [(v * height, s * r0) for v, s in PROFILES[kind]]
    if kind == "pawn":
        rb = HEAD_WIDTH["pawn"] * base_d / 2
        zc = height - rb
        z_neck, r_neck = pts[-1]
        start = math.asin(max(-1.0, min(1.0, (z_neck - zc) / rb)))
        for a in np.linspace(start, math.pi / 2, 10)[1:]:
            pts.append((zc + rb * math.sin(a), max(rb * math.cos(a), 0.0)))
        pts[-1] = (height, 0.0)
    return np.array(pts, float)


def head_width(kind: str, base_d: float) -> float:
    _check(kind)
    return HEAD_WIDTH[kind] * base_d


def _lathe(prof: np.ndarray, height: float, segments: int, crenel=None) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (z measured from the piece centre) and triangles of a surface of revolution. Rings with
    zero radius collapse to one vertex. ``crenel`` lowers the rim above a height fraction in notches."""
    verts: list[tuple[float, float, float]] = []
    rings: list[list[int]] = []
    for z, r in prof:
        if r < 1e-9:
            rings.append([len(verts)])
            verts.append((0.0, 0.0, z - height / 2))
            continue
        ring = []
        for k in range(segments):
            a = 2 * math.pi * k / segments
            zz = z
            if crenel is not None:
                from_frac, n, share = crenel
                if z >= from_frac * height - 1e-9 and ((a * n / (2 * math.pi)) % 1.0) < share:
                    zz = from_frac * height
            ring.append(len(verts))
            verts.append((r * math.cos(a), r * math.sin(a), zz - height / 2))
        rings.append(ring)
    faces: list[tuple[int, int, int]] = []
    for lo, hi in zip(rings, rings[1:]):
        if len(lo) == 1 and len(hi) == 1:
            continue
        if len(lo) == 1:
            faces += [(lo[0], hi[k], hi[(k + 1) % segments]) for k in range(segments)]
        elif len(hi) == 1:
            faces += [(lo[k], hi[0], lo[(k + 1) % segments]) for k in range(segments)]
        else:
            for k in range(segments):
                k1 = (k + 1) % segments
                faces += [(lo[k], hi[k], hi[k1]), (lo[k], hi[k1], lo[k1])]
    return np.array(verts, float), np.array(faces, int)


def _triangulate(poly: Sequence[tuple[float, float]]) -> list[tuple[int, int, int]]:
    """Ear clipping of a simple polygon given counter-clockwise."""
    idx = list(range(len(poly)))
    p = np.array(poly, float)
    out = []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        for i in range(len(idx)):
            a, b, c = idx[i - 1], idx[i], idx[(i + 1) % len(idx)]
            if cross(p[a], p[b], p[c]) <= 1e-12:
                continue
            if any(cross(p[a], p[b], p[j]) >= 0 and cross(p[b], p[c], p[j]) >= 0 and cross(p[c], p[a], p[j]) >= 0
                   for j in idx if j not in (a, b, c)):
                continue
            out.append((a, b, c))
            idx.pop(i)
            break
    out.append(tuple(idx))
    return out


def _knight_head(height: float, base_d: float) -> tuple[np.ndarray, np.ndarray]:
    r0 = base_d / 2
    outline = [(u * r0, v * height - height / 2) for u, v in _KNIGHT_OUTLINE]
    area = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(outline, outline[1:] + outline[:1]))
    if area < 0:
        outline = outline[::-1]
    n = len(outline)

    def half(u: float) -> float:
        t = min(1.0, max(0.0, (u / r0 - 0.25) / 0.65))
        return r0 * (_KNIGHT_HALF + (_KNIGHT_MUZZLE_HALF - _KNIGHT_HALF) * t)

    verts = [(u, half(u), z) for u, z in outline] + [(u, -half(u), z) for u, z in outline]
    tris = _triangulate(outline)
    faces = [(a, b, c) for a, b, c in tris] + [(a + n, c + n, b + n) for a, b, c in tris]
    for k in range(n):
        k1 = (k + 1) % n
        faces += [(k, k + n, k1 + n), (k, k1 + n, k1)]
    # the +y side was built counter-clockwise seen from +y; flip every face if the normals point inwards
    v = np.array(verts)
    f = np.array(faces, int)
    centre = v.mean(axis=0)
    out = sum(np.dot(np.cross(v[b] - v[a], v[c] - v[a]), (v[a] + v[b] + v[c]) / 3 - centre) for a, b, c in f)
    if out < 0:
        f = f[:, ::-1]
    return v, f


def mesh(kind: str, height: float, base_d: float, segments: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Visual mesh: vertices (m, piece frame) and triangles (outward normals)."""
    prof = profile(kind, height, base_d)
    verts, faces = _lathe(prof, height, segments, CRENELS.get(kind))
    if kind == "knight":
        hv, hf = _knight_head(height, base_d)
        faces = np.vstack([faces, hf + len(verts)])
        verts = np.vstack([verts, hv])
    elif kind == "king":
        verts, faces = _add_boxes(verts, faces, [b for b in _cross(height, base_d)])
    return verts, faces


def _cross(height: float, base_d: float) -> list[Primitive]:
    w = 0.07 * base_d
    z0 = 0.885 * height - height / 2
    bar = Primitive("box", (w, w, (height / 2 - z0) / 2), (0.0, 0.0, (z0 + height / 2) / 2))
    arm = Primitive("box", (2.4 * w, w, w), (0.0, 0.0, height / 2 - 0.045 * height))
    return [bar, arm]


def _add_boxes(verts: np.ndarray, faces: np.ndarray, boxes: Sequence[Primitive]) -> tuple[np.ndarray, np.ndarray]:
    for b in boxes:
        hx, hy, hz = b.size
        corners = np.array([(sx * hx, sy * hy, sz * hz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) + b.pos
        q = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6),
             (0, 6, 4), (1, 5, 7), (1, 7, 3)]
        faces = np.vstack([faces, np.array(q, int) + len(verts)])
        verts = np.vstack([verts, corners])
    return verts, faces


def collision(kind: str, height: float, base_d: float) -> list[Primitive]:
    """Collision primitives in the piece frame. Lathe pieces: one cylinder per band of the outline
    (the band's largest radius); the pawn's ball is a sphere; the knight's head three boxes."""
    _check(kind)
    r0 = base_d / 2
    prof = profile(kind, height, base_d)
    bands = {
        "pawn": (0.0, 0.03, 0.135, 0.56, 0.65, None),
        "rook": (0.0, 0.03, 0.135, 0.63, 0.74, 1.0),
        "bishop": (0.0, 0.03, 0.135, 0.60, 0.675, 0.95, 1.0),
        "queen": (0.0, 0.03, 0.135, 0.58, 0.655, 0.80, 0.885, 1.0),
        "king": (0.0, 0.03, 0.135, 0.58, 0.65, 0.79, 0.875),
        "knight": (0.0, 0.03, 0.135, 0.25),
    }[kind]
    out = []
    edges = [b for b in bands if b is not None]
    for lo, hi in zip(edges, edges[1:]):
        z0, z1 = lo * height, hi * height
        zs = np.linspace(z0, z1, 25)
        r = max(float(np.interp(z, prof[:, 0], prof[:, 1], left=0, right=0)) for z in zs)
        # an outline that runs back down (the rook's cup, the knight's base top) closes at the band edge
        r = max(r, max((s for zz, s in prof if z0 - 1e-9 <= zz <= z1 + 1e-9), default=0.0))
        out.append(Primitive("cylinder", (r, (z1 - z0) / 2), (0.0, 0.0, (z0 + z1) / 2 - height / 2)))
    if kind == "pawn":
        rb = HEAD_WIDTH["pawn"] * base_d / 2
        neck = 0.65 * height
        out.append(Primitive("cylinder", (0.45 * r0, (height - rb - neck) / 2 + 0.001),
                             (0.0, 0.0, (neck + height - rb) / 2 - height / 2)))
        out.append(Primitive("sphere", (rb,), (0.0, 0.0, height / 2 - rb)))
    elif kind == "king":
        out += _cross(height, base_d)
    elif kind == "knight":
        h, r = height, r0
        half, muzzle = _KNIGHT_HALF * r, _KNIGHT_MUZZLE_HALF * r

        def box(u0, u1, v0, v1, w):
            return Primitive("box", ((u1 - u0) * r / 2, w, (v1 - v0) * h / 2),
                             ((u0 + u1) * r / 2, 0.0, (v0 + v1) * h / 2 - h / 2))
        out += [box(-0.80, 0.50, 0.25, 0.62, half), box(-0.62, 0.40, 0.62, 1.0, half),
                box(0.30, 0.92, 0.45, 0.80, muzzle)]
    return out


def com_height(kind: str, height: float) -> float:
    """Centre of mass above the piece centre (negative: below). Club pieces are weighted in the base;
    taken at a quarter of the height from the bottom."""
    _check(kind)
    return 0.25 * height - height / 2

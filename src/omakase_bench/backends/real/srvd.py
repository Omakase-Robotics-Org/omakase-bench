"""Video frames of the robot's camera daemon as it publishes them on shared-memory topics.

One frame is one buffer, little-endian: a 72-byte header, for a depth frame a 136-byte depth block, then
tightly packed pixel rows.

    header   0  u32 magic "SRVD"      4  u16 schema version (1)   6  u8 timestamp source   7  u8 flags
             8  i64 timestamp, ns    16  u64 correlation id      24  32 bytes frame id (NUL padded)
            56  u32 width            60  u32 height              64  u8 pixel format      65  7 bytes zero
    flags    bit 0: the correlation id is set; bit 1: a depth block follows the header
    formats  1 rgba8, 2 bgra8, 3 gray8, 4 yuyv, 5 gray16 (depth)
    depth   72  u8 encoding (1 linear)   73  u8 significant bits   74  u8 intrinsics present   75  u8 distortion model
            80  f64 metres per value     88  f64 min m    96  f64 max m    104  f64 source depth units
           112  4 x f64 fx fy cx cy     144  8 x f64 distortion coefficients
           208  u16 per pixel: metres = value x metres per value, 0 = no measurement

A depth frame is on the depth imager's own pixel grid; it is not aligned to the colour image.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

MAGIC = 0x53525644
VERSION = 1
HEADER = struct.Struct("<IHBBqQ32sIIB7x")
DEPTH_BLOCK = struct.Struct("<BBBB4xdddd4d8d")
FORMATS = {1: ("rgba8", 4), 2: ("bgra8", 4), 3: ("gray8", 1), 4: ("yuyv", 2)}
GRAY16 = 5
FLAG_DEPTH = 2


class FrameError(ValueError):
    """The buffer is not a frame of this schema; the message names the rule it breaks."""


@dataclass(frozen=True)
class ColourFrame:
    timestamp_ns: int
    frame_id: str
    rgb: np.ndarray                    # uint8 (height, width, 3)


@dataclass(frozen=True)
class DepthFrame:
    timestamp_ns: int
    frame_id: str
    metres: np.ndarray                 # float32 (height, width); 0 = no measurement
    #: fx, fy, cx, cy of the depth imager, or None when the frame carries none
    intrinsics: tuple[float, float, float, float] | None


def _header(buf) -> tuple:
    if len(buf) < HEADER.size:
        raise FrameError(f"{len(buf)} bytes, the header alone is {HEADER.size}")
    magic, version, _source, flags, stamp, _corr, frame_id, width, height, fmt = HEADER.unpack_from(buf, 0)
    if magic != MAGIC:
        raise FrameError(f"magic 0x{magic:08x}, expected 0x{MAGIC:08x} ('SRVD')")
    if version != VERSION:
        raise FrameError(f"schema version {version}, expected {VERSION}")
    return flags, stamp, frame_id.split(b"\0", 1)[0].decode("utf-8", "replace"), width, height, fmt


def _yuyv_to_rgb(px: np.ndarray, width: int, height: int) -> np.ndarray:
    """BT.601 limited-range YUYV (Y0 U Y1 V per two pixels) to RGB."""
    yuyv = px.reshape(height, width // 2, 4).astype(np.float32)
    y = yuyv[:, :, [0, 2]].reshape(height, width) - 16.0
    u = np.repeat(yuyv[:, :, 1], 2, axis=1) - 128.0
    v = np.repeat(yuyv[:, :, 3], 2, axis=1) - 128.0
    rgb = np.stack([1.164 * y + 1.596 * v, 1.164 * y - 0.392 * u - 0.813 * v, 1.164 * y + 2.017 * u], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def decode_colour(buf) -> ColourFrame:
    flags, stamp, frame_id, width, height, fmt = _header(buf)
    if fmt not in FORMATS:
        raise FrameError(f"pixel format {fmt} is not a colour format ({sorted(FORMATS)})")
    name, bpp = FORMATS[fmt]
    if len(buf) - HEADER.size != width * height * bpp:
        raise FrameError(f"{len(buf) - HEADER.size} bytes of pixels, {width}x{height} {name} needs {width * height * bpp}")
    px = np.frombuffer(buf, np.uint8, width * height * bpp, HEADER.size)
    if name == "rgba8":
        rgb = px.reshape(height, width, 4)[:, :, :3].copy()
    elif name == "bgra8":
        rgb = px.reshape(height, width, 4)[:, :, 2::-1].copy()
    elif name == "gray8":
        rgb = np.repeat(px.reshape(height, width, 1), 3, axis=2)
    else:
        rgb = _yuyv_to_rgb(px, width, height)
    return ColourFrame(stamp, frame_id, rgb)


def decode_depth(buf) -> DepthFrame:
    flags, stamp, frame_id, width, height, fmt = _header(buf)
    if fmt != GRAY16 or not flags & FLAG_DEPTH:
        raise FrameError(f"pixel format {fmt}, flags {flags}: a depth frame is gray16 ({GRAY16}) with a depth block")
    offset = HEADER.size + DEPTH_BLOCK.size
    if len(buf) - offset != width * height * 2:
        raise FrameError(f"{len(buf) - offset} bytes of pixels, {width}x{height} gray16 needs {width * height * 2}")
    encoding, _bits, has_k, _model, scale, _lo, _hi, _units, fx, fy, cx, cy, *_dist = DEPTH_BLOCK.unpack_from(buf, HEADER.size)
    if encoding != 1 or not scale > 0.0:
        raise FrameError(f"depth encoding {encoding} with {scale!r} metres per value: only linear depth carries metres")
    values = np.frombuffer(buf, "<u2", width * height, offset).reshape(height, width)
    return DepthFrame(stamp, frame_id, values.astype(np.float32) * np.float32(scale), (fx, fy, cx, cy) if has_k else None)

"""Episode and comparison videos with overlays, written with imageio's bundled ffmpeg."""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Sequence

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf")
_MONO_CANDIDATES = ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf")
OK_COLOR = (40, 170, 90)
FAIL_COLOR = (200, 60, 50)


def font(size: int, mono: bool = False) -> ImageFont.ImageFont:
    for path in (_MONO_CANDIDATES if mono else _FONT_CANDIDATES):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def _banner(img: Image.Image, text: str, *, top: bool, size: int = 18, color=(0, 0, 0)) -> None:
    d = ImageDraw.Draw(img, "RGBA")
    f = font(size)
    lines = textwrap.wrap(text, width=max(20, int(img.width / (size * 0.55))))[:3]
    h = (size + 6) * len(lines) + 8
    y0 = 0 if top else img.height - h
    d.rectangle([0, y0, img.width, y0 + h], fill=(*color, 150))
    for i, line in enumerate(lines):
        d.text((10, y0 + 4 + i * (size + 6)), line, fill=(255, 255, 255), font=f)


def _badge(img: Image.Image, success: bool, size: int = 30) -> None:
    d = ImageDraw.Draw(img, "RGBA")
    text = "SUCCESS" if success else "FAIL"
    f = font(size)
    w = d.textlength(text, font=f)
    x, y = img.width - w - 30, 48
    d.rounded_rectangle([x - 12, y - 6, x + w + 12, y + size + 10], radius=8,
                        fill=(*(OK_COLOR if success else FAIL_COLOR), 230))
    d.text((x, y), text, fill=(255, 255, 255), font=f)


def _clock(img: Image.Image, t: float) -> None:
    d = ImageDraw.Draw(img, "RGBA")
    f = font(15)
    d.text((img.width - 90, img.height - 24), f"t={t:5.1f}s", fill=(255, 255, 255), font=f,
           stroke_width=2, stroke_fill=(0, 0, 0))


def annotate(frame: np.ndarray, *, title: str, instruction: str, t: float | None,
             success: bool | None) -> np.ndarray:
    img = Image.fromarray(frame)
    _banner(img, title, top=True, size=18, color=(20, 20, 30))
    if instruction:
        _banner(img, instruction, top=False, size=16, color=(20, 20, 30))
    if t is not None:
        _clock(img, t)
    if success is not None:
        _badge(img, success)
    return np.asarray(img)


def _write(path: Path, frames: Sequence[np.ndarray], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7, macro_block_size=16,
                            pixelformat="yuv420p") as w:
        for f in frames:
            w.append_data(f)


def write_episode_video(path: Path, frames: Sequence[np.ndarray], times: Sequence[float], fps: float, *,
                        title: str, instruction: str, success: bool, hold_s: float = 1.5) -> None:
    out = [annotate(f, title=title, instruction=instruction, t=t, success=None) for f, t in zip(frames, times)]
    last = annotate(frames[-1], title=title, instruction=instruction, t=times[-1], success=success)
    out += [last] * int(hold_s * fps)
    _write(path, out, fps)


def write_raw_video(path: Path, frames: Sequence[np.ndarray], fps: float) -> None:
    _write(path, frames, fps)


def read_video(path: Path) -> list[np.ndarray]:
    with imageio.get_reader(str(path)) as r:
        return [np.asarray(f) for f in r]


def code_panel(code: str, height: int, width: int = 560, title: str = "program") -> np.ndarray:
    img = Image.new("RGB", (width, height), (24, 26, 32))
    d = ImageDraw.Draw(img)
    f = font(13, mono=True)
    d.text((12, 10), title, fill=(150, 200, 255), font=font(15))
    y = 36
    for raw in code.splitlines():
        for line in textwrap.wrap(raw, width=74, subsequent_indent="    ", drop_whitespace=False) or [""]:
            if y > height - 20:
                d.text((12, y), "...", fill=(200, 200, 200), font=f)
                return np.asarray(img)
            d.text((12, y), line, fill=(220, 220, 220), font=f)
            y += 16
    return np.asarray(img)


def compose_comparison(path: Path, tiles: Sequence[dict], fps: float, *, instruction: str,
                       header: str = "", hold_s: float = 2.0) -> None:
    """Side by side: each tile is {frames, label, success, code (optional)}; shorter
    episodes hold their last frame. Frames are the raw recordings (no overlay)."""
    n = max(len(t["frames"]) for t in tiles) + int(hold_s * fps)
    h, w = tiles[0]["frames"][0].shape[:2]
    out = []
    panels = {i: code_panel(t["code"], h, title=f'{t["label"]}: program') for i, t in enumerate(tiles) if t.get("code")}
    for k in range(n):
        row = []
        for i, t in enumerate(tiles):
            fr = t["frames"][min(k, len(t["frames"]) - 1)]
            if fr.shape[:2] != (h, w):
                fr = np.asarray(Image.fromarray(fr).resize((w, h)))
            done = k >= len(t["frames"]) - 1
            img = annotate(fr, title=t["label"], instruction="", t=None, success=t["success"] if done else None)
            row.append(img)
            if i in panels:
                row.append(panels[i])
        frame = np.concatenate(row, axis=1)
        canvas = Image.new("RGB", (frame.shape[1], frame.shape[0] + 70), (14, 14, 18))
        canvas.paste(Image.fromarray(frame), (0, 0))
        d = ImageDraw.Draw(canvas)
        d.text((14, frame.shape[0] + 8), instruction, fill=(255, 255, 255), font=font(20))
        if header:
            d.text((14, frame.shape[0] + 40), header, fill=(170, 170, 180), font=font(15))
        arr = np.asarray(canvas)
        hh, ww = arr.shape[0] // 16 * 16, arr.shape[1] // 16 * 16
        out.append(arr[:hh, :ww])
    _write(path, out, fps)

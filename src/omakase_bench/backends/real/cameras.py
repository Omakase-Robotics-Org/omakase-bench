"""Camera frames of a real robot.

Three colour sources, chosen with the backend option ``camera_source``:

    v4l2      on the robot: the camera daemon's V4L2 loopback devices, one device per camera
    iceoryx   on the robot: the camera daemon's shared-memory video topics, one topic per camera
    http      from another host: one URL per camera that answers ``GET`` with a JPEG of the newest frame

and one depth source, on the robot only: the camera daemon's shared-memory depth topics
(:class:`DepthTopics`). A depth frame is on the depth imager's own grid; :class:`DepthTopics` re-draws it
on the colour image's grid with the colour lens of the robot's calibration file, which is what the
harness's depth lookup reads.

Which device or topic is which camera is configuration: the robot description's ``real.cameras`` block
holds the on-robot defaults and the backend options ``camera.<name>`` and ``depth.<name>`` replace them.
Nothing is opened unless the backend is enabled. The shared-memory topics are readable only by a process
of the camera daemon's account, with the ``iceoryx2`` Python binding at the daemon's version.
"""
from __future__ import annotations

import io
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

from ...core.backend import BackendUnsupported
from .calibration import Lens
from .srvd import FrameError, decode_colour, decode_depth

SOURCES = ("v4l2", "iceoryx", "http")
#: s a frame may be old before it counts as missing
MAX_AGE_S = 2.0


def _sized(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    from PIL import Image
    if rgb.shape[1] == width and rgb.shape[0] == height:
        return np.ascontiguousarray(rgb)
    return np.asarray(Image.fromarray(rgb).resize((width, height), Image.BILINEAR), dtype=np.uint8)


class CameraSource(Protocol):
    def names(self) -> tuple[str, ...]: ...

    def frame(self, camera: str, width: int, height: int) -> np.ndarray:
        """The camera's newest frame as RGB uint8, ``width`` x ``height``."""
        ...

    def close(self) -> None: ...


class HttpSnapshotCameras:
    def __init__(self, urls: Mapping[str, str], timeout_s: float = 5.0):
        for name, url in urls.items():
            if not re.match(r"^https?://", str(url)):
                raise ValueError(f"camera {name!r}: expected an http(s) URL, got {url!r}")
        self.urls, self.timeout_s = dict(urls), timeout_s

    def names(self) -> tuple[str, ...]:
        return tuple(self.urls)

    def close(self) -> None:
        pass

    def frame(self, camera: str, width: int, height: int) -> np.ndarray:
        """The camera's newest frame as RGB, scaled to ``width`` x ``height``."""
        from PIL import Image
        if camera not in self.urls:
            raise BackendUnsupported(f"camera {camera!r} has no source; configured: {', '.join(self.urls) or 'none'}")
        try:
            with urllib.request.urlopen(self.urls[camera], timeout=self.timeout_s) as resp:
                kind, data = resp.headers.get_content_type(), resp.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise BackendUnsupported(f"camera {camera!r}: no frame within {self.timeout_s:g} s: {exc}") from None
        if kind != "image/jpeg" or not data:
            raise BackendUnsupported(f"camera {camera!r}: expected a JPEG image, got {kind} ({len(data)} bytes)")
        return _sized(np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8), width, height)


# --- V4L2 loopback devices ----------------------------------------------------------------------------

class Capture(Protocol):
    """An open video device: ``read()`` returns (ok, BGR uint8 image), as OpenCV's ``VideoCapture`` does."""

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def release(self) -> None: ...


def open_v4l2(device: str) -> Capture:
    """The device through OpenCV's V4L2 backend (``pip install 'omakase-bench[real]'``)."""
    import cv2
    capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not capture.isOpened():
        raise BackendUnsupported(f"video device {device} could not be opened")
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # the newest frame, not a queue of old ones
    return capture


class V4l2Cameras:
    def __init__(self, devices: Mapping[str, str], open_capture: Callable[[str], Capture] = open_v4l2):
        self.devices = dict(devices)
        self._captures = {name: open_capture(device) for name, device in self.devices.items()}

    def names(self) -> tuple[str, ...]:
        return tuple(self.devices)

    def frame(self, camera: str, width: int, height: int) -> np.ndarray:
        if camera not in self._captures:
            raise BackendUnsupported(f"camera {camera!r} has no device; configured: {', '.join(self.devices) or 'none'}")
        ok, bgr = self._captures[camera].read()
        if not ok or bgr is None:
            raise BackendUnsupported(f"camera {camera!r}: no frame from {self.devices[camera]}")
        return _sized(bgr[:, :, ::-1], width, height)

    def close(self) -> None:
        for capture in self._captures.values():
            capture.release()
        self._captures = {}


# --- shared-memory topics -------------------------------------------------------------------------------

class TopicBus(Protocol):
    """The shared-memory layer: ``subscribe(topic)`` returns ``receive()``, which gives the next buffer of
    the topic or None when none is waiting."""

    def subscribe(self, topic: str) -> Callable[[], bytes | None]: ...

    def close(self) -> None: ...


class Iceoryx2Bus:
    """The camera daemon's topics through the ``iceoryx2`` Python binding. The binding's version has to be
    the daemon's (shared-memory layouts are tied to it), so it is installed on the robot, not as a
    dependency of this package."""

    #: the daemon's sample queue depth: a subscriber that falls further behind loses frames
    QUEUE_DEPTH = 8

    def __init__(self):
        try:
            import iceoryx2
        except ImportError:
            raise BackendUnsupported("shared-memory topics need the iceoryx2 Python binding at the camera daemon's "
                                     "version (pip install iceoryx2==<version>)") from None
        self._iox = iceoryx2
        self._node = iceoryx2.NodeBuilder.new().create(iceoryx2.ServiceType.Ipc)
        self._subscribers = []

    def subscribe(self, topic: str) -> Callable[[], bytes | None]:
        import ctypes
        iox = self._iox
        service = (self._node.service_builder(iox.ServiceName.new(topic))
                   .publish_subscribe(iox.Slice[ctypes.c_uint8]).open())
        subscriber = service.subscriber_builder().buffer_size(self.QUEUE_DEPTH).create()
        self._subscribers.append(subscriber)

        def receive() -> bytes | None:
            sample = subscriber.receive()
            if sample is None:
                return None
            payload = sample.payload()
            data = ctypes.string_at(payload.as_ptr(), payload.len())
            sample.delete()               # hand the chunk back at once: a held sample stalls the publisher's queue
            return data
        return receive

    def close(self) -> None:
        for subscriber in self._subscribers:
            subscriber.delete()
        self._subscribers = []


class _Newest:
    """Drains one topic on a thread and keeps its newest buffer, so that the publisher's queue never fills
    while the caller is busy."""

    def __init__(self, topic: str, receive: Callable[[], bytes | None]):
        self.topic, self._receive = topic, receive
        self._lock = threading.Lock()
        self._buffer: bytes | None = None
        self._at = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._drain, name=f"topic {topic}", daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        while self._running:
            newest = None
            while True:
                data = self._receive()
                if data is None:
                    break
                newest = data
            if newest is not None:
                with self._lock:
                    self._buffer, self._at = newest, time.monotonic()
            time.sleep(0.002)

    def newest(self, wait_s: float = 1.0) -> bytes:
        deadline = time.monotonic() + wait_s
        while True:
            with self._lock:
                buffer, age = self._buffer, time.monotonic() - self._at
            if buffer is not None and age <= MAX_AGE_S:
                return buffer
            if time.monotonic() >= deadline:
                raise BackendUnsupported(f"no frame on topic {self.topic} within {wait_s:g} s: is the camera daemon "
                                         "running, and is this process of its account?")
            time.sleep(0.01)

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)


class TopicCameras:
    """Colour frames from the camera daemon's video topics."""

    def __init__(self, topics: Mapping[str, str], bus: TopicBus):
        self.topics = dict(topics)
        self._readers = {name: _Newest(topic, bus.subscribe(topic)) for name, topic in self.topics.items()}

    def names(self) -> tuple[str, ...]:
        return tuple(self.topics)

    def frame(self, camera: str, width: int, height: int) -> np.ndarray:
        if camera not in self._readers:
            raise BackendUnsupported(f"camera {camera!r} has no topic; configured: {', '.join(self.topics) or 'none'}")
        try:
            return _sized(decode_colour(self._readers[camera].newest()).rgb, width, height)
        except FrameError as exc:
            raise BackendUnsupported(f"camera {camera!r}: topic {self.topics[camera]}: {exc}") from None

    def close(self) -> None:
        for reader in self._readers.values():
            reader.stop()


class DepthTopics:
    """Depth along the optical axis on the colour image's pixel grid, from the camera daemon's depth topics.

    Each depth pixel becomes a point through the depth imager's own intrinsics (carried by the frame), and
    the point is drawn where the colour lens of the calibration file sees it; where several land on one
    pixel the nearest wins.

    ``offsets``: per camera the pose of the depth imager's optical frame in the colour camera's optical
    frame, as (position in metres, rotation matrix). A camera without one is taken to have its depth imager
    at the colour camera: the camera daemon publishes no transform between them, so without a configured
    value the two are off by the sensor's baseline (about 15 mm on the head camera), a parallax of a few
    pixels at arm's length.

    ``aligned``: cameras whose depth topic is already on the colour camera's grid (the camera daemon rendered
    the point cloud into the colour view with the sensor's factory transform). Such a frame carries the
    colour camera's pinhole intrinsics, so it is used as it is: no lens of the calibration file, no offset."""

    def __init__(self, topics: Mapping[str, str], lenses: Mapping[str, Lens], bus: TopicBus,
                 offsets: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None, aligned: Sequence[str] = ()):
        self.aligned = frozenset(aligned)
        missing = sorted(set(topics) - set(lenses) - self.aligned)
        if missing:
            raise BackendUnsupported(f"depth for camera(s) {missing} needs their lens in the calibration file")
        self.topics, self.lenses, self.offsets = dict(topics), dict(lenses), dict(offsets or {})
        #: per aligned camera the pinhole of its newest frame
        self._frame_lens: dict[str, Lens] = {}
        self._readers = {name: _Newest(topic, bus.subscribe(topic)) for name, topic in self.topics.items()}

    def names(self) -> tuple[str, ...]:
        return tuple(self.topics)

    def depth(self, camera: str, width: int, height: int) -> np.ndarray:
        """Metres along the optical axis per pixel of the ``width`` x ``height`` colour image; 0 = none."""
        if camera not in self._readers:
            raise BackendUnsupported(f"camera {camera!r} has no depth topic; configured: {', '.join(self.topics) or 'none'}")
        try:
            frame = decode_depth(self._readers[camera].newest())
        except FrameError as exc:
            raise BackendUnsupported(f"camera {camera!r}: topic {self.topics[camera]}: {exc}") from None
        if frame.intrinsics is None:
            raise BackendUnsupported(f"camera {camera!r}: the depth frames carry no intrinsics")
        fx, fy, cx, cy = frame.intrinsics
        if camera in self.aligned:
            rows, cols = frame.metres.shape
            self._frame_lens[camera] = Lens(cols, rows, fx, fy, cx, cy)
            v = np.minimum(((np.arange(height) + 0.5) * rows / height).astype(np.int64), rows - 1)
            u = np.minimum(((np.arange(width) + 0.5) * cols / width).astype(np.int64), cols - 1)
            return np.ascontiguousarray(frame.metres[v][:, u])
        lens = self.lenses[camera]
        v, u = np.nonzero(frame.metres > 0)
        z = frame.metres[v, u].astype(np.float64)
        points = np.stack([(u + 0.5 - cx) / fx * z, (v + 0.5 - cy) / fy * z, z], axis=-1)
        if camera in self.offsets:
            position, rotation = self.offsets[camera]
            points = points @ rotation.T + position
            z = points[:, 2]
        pu, pv = lens.pixels(points)
        col = np.floor(pu * width / lens.width).astype(np.int64)
        row = np.floor(pv * height / lens.height).astype(np.int64)
        inside = (z > 0) & (col >= 0) & (col < width) & (row >= 0) & (row < height)
        out = np.full(height * width, np.inf)
        np.minimum.at(out, row[inside] * width + col[inside], z[inside])
        out[np.isinf(out)] = 0.0
        return out.reshape(height, width).astype(np.float32)

    def lens(self, camera: str) -> Lens:
        """The lens a pixel of the depth image is seen through: the calibrated one, or for an aligned camera
        the pinhole its frames carry."""
        if camera not in self.aligned:
            return self.lenses[camera]
        if camera not in self._frame_lens:
            self.depth(camera, 1, 1)
        return self._frame_lens[camera]

    def close(self) -> None:
        for reader in self._readers.values():
            reader.stop()

"""A real robot behind the Backend protocol.

The harness sends the same joint targets it sends a simulator and reads the same joint state and camera
images back, so the observations a subject gets are built by the same code. What is specific to one robot
is a *driver* (``robot.yaml`` ``api.driver: module:factory``): it turns joint targets into requests of the
robot's own API and that API's state readings into joint positions.

    motion          the driver's requests, over the WebSocket transport when ``ws_url`` is given (one
                    connection, the robot API's streaming route) and over HTTP otherwise; the base through
                    the API's velocity command, stopped when the targets stop moving
    lease           the robot API's arm lease is taken for the run, renewed on a thread for as long as the
                    backend is open (a model's thinking time is longer than the lease's lifetime) and
                    handed back when the backend closes; a lease another client holds refuses the run, and
                    one that is lost stops it at the next step
    joint state     measured: the driver reads the robot's state operations, at most every
                    ``STATE_PERIOD_S`` (slow devices every ``SLOW_STATE_PERIOD_S``); a joint the API does
                    not report holds its last target
    cameras         ``camera_source``: ``v4l2`` or ``iceoryx`` on the robot, ``http`` from another host
    depth           on the robot: the camera daemon's depth topics and the calibration file; the backend
                    then announces ``depth_camera``, names the cameras that have depth and gives the
                    harness calibrated pixel rays and mounts. The depth imager is taken to sit at the
                    colour camera unless the robot description or an option states its offset
                    (``depth_to_colour``)
    time            wall clock: ``step`` sleeps until one control period after the step before it, so the
                    time the requests of a period take is not added to it

The backend announces none of a simulator's capabilities: it builds no scene, has no ground truth, no exact
contacts and repeats nothing for a seed.

It is disabled unless enabled twice, by the option ``enable`` and by ``OMAKASE_BENCH_ROBOT_ENABLE=1``, and it
has no default host (options ``url``, ``ws_url``, or ``OMAKASE_BENCH_ROBOT_URL`` and
``OMAKASE_BENCH_ROBOT_WS_URL``). With ``dry_run`` it prints the requests, opens no socket, device or topic,
and reports the commanded targets as the joint state.

Options (``-B key=value``): ``url``, ``ws_url``, ``enable``, ``dry_run``, ``camera_source``
(``v4l2`` | ``iceoryx`` | ``http``, default ``http``), ``camera.<name>`` (a device, topic or URL: replaces the
robot description's on-robot default for that camera), ``depth.<name>`` (a depth topic), ``depth=0`` (no
depth), ``depth_frame.<name>`` (``depth``: the topic is on the depth imager's grid; ``color``: it is already on
the colour camera's grid, see :class:`~.cameras.DepthTopics`), ``depth_to_colour.<name>`` (``x,y,z`` in metres, optionally followed by a quaternion ``x,y,z,w``: the
depth imager's optical frame in the colour camera's), ``calibration`` (the calibration file), ``lease=0``
(take no lease), ``lease_holder``, ``lease_class``, ``lease_ttl_s``.
"""
from __future__ import annotations

import importlib
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from ...core.backend import BackendInfo, BackendUnsupported
from ...core.task import SceneSpec
from ...harness.command_api import Request, load_spec, operation, summary
from ...robots.spec import RobotSpec
from .calibration import Calibration, CalibrationError, frame_offset, load_calibration
from .cameras import SOURCES, CameraSource, DepthTopics, HttpSnapshotCameras, Iceoryx2Bus, TopicCameras, V4l2Cameras
from .http import HttpDaemonClient

VERSION = "0.3"
CONTROL_DT = 0.1             # s between joint-target updates sent to the robot
STATE_PERIOD_S = 0.05        # s a state reading is reused for
SLOW_STATE_PERIOD_S = 1.0    # s between readings of devices the driver calls slow
PREFIXES = ("camera.", "depth.", "depth_to_colour.", "depth_frame.")
DEPTH_FRAMES = ("depth", "color")
LEASE = {"holder": "omakase-bench", "class": "policy", "ttl_s": 30}


class RobotDriver(Protocol):
    """One robot's API, as the real backend needs it."""

    #: the lease operations: (method, path) to read, take and hand back, or None for an API without a lease
    lease: Mapping[str, tuple[str, str]] | None
    #: "measured", or "commanded: <why>" when the API reports no base pose and the base joints hold their targets
    base_pose: str

    def requests(self, robot: RobotSpec, q: Mapping[str, float], changed: Sequence[str], streaming: bool,
                 previous: Mapping[str, float], dt: float) -> list[Request]:
        """The requests that command the joints named in ``changed`` from ``previous`` to their values in
        ``q`` (every joint's target) within ``dt`` seconds. ``streaming``: the transport is the API's
        streaming route. Raises :class:`BackendUnsupported` for a joint the API cannot command."""
        ...

    def stop(self, robot: RobotSpec) -> list[Request]:
        """The requests that stop whatever keeps moving without a new command (a base velocity)."""
        ...

    def read(self, robot: RobotSpec, get: Callable[[str], Any], slow: bool) -> dict[str, float]:
        """Measured joint positions, from ``get(path)`` (the ``data`` of a state operation). ``slow``
        says whether devices that take long to read are due. Joints it does not return keep their last
        known value."""
        ...


def robot_driver(robot: RobotSpec) -> RobotDriver:
    name = robot.api.get("driver") if robot.api else None
    if not name:
        raise BackendUnsupported(f"robot {robot.name} names no driver for the real backend (robot.yaml api.driver)")
    module, _, factory = str(name).partition(":")
    return getattr(importlib.import_module(module), factory)(robot)


def _flag(value: Any) -> bool:
    return value is True or str(value).lower() in ("1", "true", "yes")


class RealBackend:
    def __init__(self, url: str | None = None, ws_url: str | None = None, enable: Any = False, dry_run: Any = False,
                 echo: Callable[[str], None] = print, camera_source: str = "http", calibration: str | None = None,
                 depth: Any = True, lease: Any = True, lease_holder: str | None = None, lease_class: str | None = None,
                 lease_ttl_s: Any = None, topic_bus=None, open_capture=None, **named: str):
        unknown = sorted(k for k in named if not k.startswith(PREFIXES))
        if unknown:
            raise ValueError(f"unknown option(s) {unknown}; see the options of the real backend")
        if camera_source not in SOURCES:
            raise ValueError(f"camera_source {camera_source!r}; sources: {', '.join(SOURCES)}")
        self.url, self.ws_url, self.enable, self.dry_run, self.echo = url, ws_url, _flag(enable), _flag(dry_run), echo
        self.camera_source, self.calibration_path = camera_source, calibration
        self._want_depth, self._want_lease = depth is True or _flag(depth), lease is True or _flag(lease)
        self._lease_options = {"holder": lease_holder, "class": lease_class, "ttl_s": lease_ttl_s}
        self._cameras = {k[len("camera."):]: v for k, v in named.items() if k.startswith("camera.")}
        self._depths = {k[len("depth."):]: v for k, v in named.items() if k.startswith("depth.")}
        self._depth_frames = {k[len("depth_frame."):]: v for k, v in named.items() if k.startswith("depth_frame.")}
        self._depth_offsets = {k[len("depth_to_colour."):]: v for k, v in named.items() if k.startswith("depth_to_colour.")}
        #: the shared-memory layer and the device opener; tests put stand-ins here
        self._given_bus, self._open_capture = topic_bus, open_capture
        self._bus = None
        self.robot: RobotSpec | None = None
        self.client: HttpDaemonClient | None = None
        self.stream = None
        self.driver: RobotDriver | None = None
        self.eyes: CameraSource | None = None
        self.depths: DepthTopics | None = None
        self.calibration: Calibration | None = None
        self.notes: dict[str, Any] = {}
        self._lease: dict[str, Any] | None = None
        self._lease_at = 0.0
        self._lease_lock = threading.Lock()
        self._lease_stop = threading.Event()
        self._lease_lost: BackendUnsupported | None = None
        self._period_end = -1e9
        self._target: dict[str, float] = {}
        self._measured: dict[str, float] = {}
        self._read_at = self._slow_read_at = -1e9
        self._drifting = self._commanded_drift = False
        self._time = 0.0

    def info(self) -> BackendInfo:
        """Before a robot is loaded ``depth_camera`` is announced when the options allow depth; whether the
        robot delivers it is known once it is loaded, and the announcement is exact from then on."""
        may_have_depth = self.robot is None and self._want_depth and not self.dry_run and self.camera_source != "http"
        caps = frozenset({"depth_camera"}) if self.depths is not None or may_have_depth else frozenset()
        return BackendInfo(
            name="real", version=VERSION, engine="robot API over HTTP and WebSocket",
            robot=self.robot.name if self.robot else "", robot_sha256=self.robot.sha256() if self.robot else "",
            physics_dt=CONTROL_DT, control_dt=CONTROL_DT, capabilities=caps,
            detail={"joint_state": "commanded" if self.dry_run else "measured",
                    "motion": "websocket stream" if self.stream is not None else "http",
                    "camera_source": self.camera_source, "cameras": sorted(self.eyes.names()) if self.eyes else [],
                    "depth": sorted(self.depths.names()) if self.depths else [],
                    "lease": {k: self._lease[k] for k in ("holder", "class", "epoch")} if self._lease else None,
                    "base_pose": self._base_pose_source(),
                    "dry_run": self.dry_run, **self.notes})

    def _base_pose_source(self) -> str | None:
        """Where the base joints' values come from: "measured", "commanded: <why>", None without a mobile base."""
        if self.robot is None or self.robot.mobile_base is None:
            return None
        return "commanded: dry run" if self.dry_run else getattr(self.driver, "base_pose", "measured")

    # --- set-up -------------------------------------------------------------------
    def load(self, robot: RobotSpec, scene: SceneSpec) -> None:
        if self.client is not None:
            self.close()                 # the runner loads once per episode: one lease, one set of connections
        self.driver = robot_driver(robot)
        self._document = load_spec(str(Path(robot.preset_dir) / robot.api["spec"]))
        self.client = HttpDaemonClient(self._document, self.url, enable=self.enable, dry_run=self.dry_run, echo=self.echo)
        unknown = sorted((set(self._cameras) | set(self._depths) | set(self._depth_offsets) | set(self._depth_frames))
                         - set(robot.cameras))
        if unknown:
            raise ValueError(f"robot {robot.name} has no camera(s) {unknown}; cameras: {', '.join(robot.cameras)}")
        self.robot = robot
        self._target = robot.with_home(scene.home_pose).default_joint_positions()
        if robot.mobile_base:
            base = robot.mobile_base
            self._target.update(zip((base.x, base.y, base.yaw), scene.base))
        self._measured = {}
        self._read_at = self._slow_read_at = self._period_end = -1e9
        if self.dry_run:
            return
        if self.camera_source == "http":
            self.eyes = HttpSnapshotCameras(self._cameras)       # checks the URLs; fetches nothing yet
            self.notes["depth_note"] = "no depth over the http camera source"
        try:
            if self.ws_url:
                from .ws import WsDaemonClient
                self.stream = WsDaemonClient(self._document, self.ws_url, enable=self.enable)
            self._take_lease()
            self._open_cameras(robot)
        except BaseException:
            self.close()
            raise

    def _open_cameras(self, robot: RobotSpec) -> None:
        defaults = (robot.real.get("cameras") or {})
        source = self.camera_source
        if source == "http":
            return
        chosen = {**(defaults.get(source) or {}), **self._cameras}
        depth_topics = {**(defaults.get("depth") or {}), **self._depths} if self._want_depth else {}
        if source == "iceoryx" or depth_topics:
            self._bus = self._given_bus or Iceoryx2Bus()
        self.eyes = (V4l2Cameras(chosen, self._open_capture) if self._open_capture else V4l2Cameras(chosen)) \
            if source == "v4l2" else TopicCameras(chosen, self._bus)
        if not depth_topics:
            self.notes["depth_note"] = "no depth topic configured" if self._want_depth else "switched off (depth=0)"
            return
        frames = {name: str({**(robot.real.get("depth_frame") or {}), **self._depth_frames}.get(name, "depth"))
                  for name in depth_topics}
        wrong = sorted(f"{name}={mode}" for name, mode in frames.items() if mode not in DEPTH_FRAMES)
        if wrong:
            raise ValueError(f"depth_frame {', '.join(wrong)}; a depth topic is on the grid of: {', '.join(DEPTH_FRAMES)}")
        aligned = {name for name, mode in frames.items() if mode == "color"}
        path = self.calibration_path or robot.real.get("calibration")
        try:
            if not path:
                raise CalibrationError("no camera calibration file is configured")
            self.calibration = load_calibration(path)
        except CalibrationError as exc:
            if self.calibration_path or set(self._depths) - aligned:      # asked for by option: its absence is an error
                raise BackendUnsupported(f"depth needs the camera calibration: {exc}") from None
            if not aligned:
                self.notes["depth_note"] = f"off: {exc}"
                return
            # colour-aligned depth needs no lens; without the file the description's nominal camera pose is used
            self.calibration = Calibration("", {}, {})
            self.notes["depth_note"] = f"aligned cameras only, at their nominal pose: {exc}"
        usable = {name: topic for name, topic in depth_topics.items() if name in self.calibration.lenses or name in aligned}
        if not usable:
            self.notes["depth_note"] = "off: the calibration file has no lens for a camera with a depth topic"
            return
        try:
            offsets = {name: frame_offset(value) for name, value in
                       {**(robot.real.get("depth_to_colour") or {}), **self._depth_offsets}.items() if name in usable}
        except (CalibrationError, ValueError) as exc:
            raise BackendUnsupported(f"depth_to_colour: {exc}") from None
        for name in sorted(aligned & set(offsets)):
            position, rotation = offsets.pop(name)
            if np.any(position) or not np.allclose(rotation, np.eye(3)):
                self.echo(f"real backend: depth_to_colour.{name} is ignored: its depth topic is already on the colour "
                          "camera's grid (depth_frame: color)")
        self.depths = DepthTopics(usable, self.calibration.lenses, self._bus, offsets, aligned & set(usable))
        self.notes["depth_frame"] = {name: frames[name] for name in usable}
        # present only with depth: the harness asks for them by name
        self.render_depth, self.pixel_ray, self.camera_mount = self._render_depth, self._pixel_ray, self._camera_mount
        self.depth_cameras = self.depths.names

    # --- lease --------------------------------------------------------------------
    def _lease_call(self, verb: str, body: Mapping | None = None) -> Any:
        method, path = self.driver.lease[verb]
        reply = self.client.request(method, path, body)
        if reply.get("status") != "ok":
            raise BackendUnsupported(f"the arm lease: {summary(Request(method, path), reply)}")
        return reply.get("data")

    def _take_lease(self) -> None:
        if not self._want_lease or not getattr(self.driver, "lease", None):
            return
        defaults = {**LEASE, **(self.robot.real.get("lease") or {})}
        want = {k: (self._lease_options[k] if self._lease_options[k] is not None else defaults[k]) for k in defaults}
        want["ttl_s"] = int(want["ttl_s"])
        held = self._lease_call("read")
        if held and held.get("holder") != want["holder"]:
            raise BackendUnsupported(f"the arms are leased to {held['holder']!r} (class {held.get('class')}, "
                                     f"{held.get('expires_in_s')} s left): another client is driving this robot. "
                                     "Nothing was sent.")
        self._lease = dict(self._lease_call("take", want))
        self._lease_at = time.monotonic()
        self._lease_ttl = want["ttl_s"]
        self._lease_request = want
        self._lease_lost = None
        self._lease_stop = threading.Event()
        threading.Thread(target=self._keep_lease, args=(self._lease_stop,), name="arm lease", daemon=True).start()

    def _keep_lease(self, stop: threading.Event) -> None:
        """Renew the lease every third of its lifetime until the backend closes or the lease is lost. Steps
        alone would not do: no step is taken while a model thinks, which lasts longer than the lease."""
        while not stop.wait(self._lease_ttl / 3):
            try:
                self._renew_lease(stop)
            except BackendUnsupported as exc:
                self._lease_lost = exc
                return

    def _renew_lease(self, stop: threading.Event | None = None) -> None:
        """Renew the lease: from a step when a third of its lifetime has passed, from the lease thread (which
        gives its ``stop``) whenever it wakes."""
        with self._lease_lock:
            if self._lease is None or (stop or self._lease_stop).is_set():
                return
            if stop is None and time.monotonic() - self._lease_at < self._lease_ttl / 3:
                return
            renewed = self._lease_call("take", self._lease_request)
            if renewed.get("epoch") != self._lease["epoch"]:
                raise BackendUnsupported("the arm lease was lost and taken again between two renewals: another "
                                         "client drove the arms in between")
            self._lease_at = time.monotonic()

    # --- state ------------------------------------------------------------------
    def _get(self, path: str) -> Any:
        reply = self.client.request("GET", path)
        if reply.get("status") != "ok":
            raise BackendUnsupported(f"the robot's state could not be read: {summary(Request('GET', path), reply)}")
        return reply.get("data")

    def joint_positions(self) -> dict[str, float]:
        """Measured joint positions; joints the robot's API does not report hold their last target."""
        if self.dry_run:
            return dict(self._target)
        now = time.monotonic()
        if now - self._read_at >= STATE_PERIOD_S:
            slow = now - self._slow_read_at >= SLOW_STATE_PERIOD_S
            reading = self.driver.read(self.robot, self._get, slow)
            self._measured.update(reading)
            base = self.robot.mobile_base
            if base and base.x not in reading:
                for joint in (base.x, base.y, base.yaw):       # no longer measured: back to the targets
                    self._measured.pop(joint, None)
            self._read_at = now
            if slow:
                self._slow_read_at = now
        return {**self._target, **self._measured}

    # --- commands ---------------------------------------------------------------
    def _send(self, requests: Sequence[Request]) -> None:
        transport = self.stream or self.client
        for request in requests:
            body = request.body
            op, _ = operation(self._document, request.method, request.path)
            if self._lease is not None and op.get("x-arm-lease"):
                body = {**(body or {}), "holder": self._lease["holder"]}
            reply = transport.request(request.method, request.path, body)
            if reply.get("status") != "ok":
                raise BackendUnsupported(f"the robot refused a command: {summary(request, reply)}")

    def command_joints(self, targets: Mapping[str, float]) -> None:
        unknown = sorted(set(targets) - set(self._target))
        if unknown:
            raise KeyError(f"no commanded joint {unknown[0]!r}")
        changed = [j for j, v in targets.items() if abs(float(v) - self._target[j]) > 1e-6]
        if not changed:
            return
        q = {**self._target, **{j: float(v) for j, v in targets.items()}}
        previous = self._target
        base = self.robot.mobile_base
        base_joints = [j for j in ((base.x, base.y, base.yaw) if base else ()) if j in changed]
        if base_joints and not self._drifting:
            # A base motion starts: its first velocity covers the step from where the base is. The last target
            # is where the motion before it was told to end, which the base did not reach exactly.
            now = self.joint_positions()
            previous = {**self._target, **{j: now[j] for j in base_joints}}
        requests = self.driver.requests(self.robot, q, changed, self.stream is not None, previous, CONTROL_DT)
        if base_joints:
            # before sending: a velocity command that got no answer may still have been taken
            self._drifting = self._commanded_drift = True
        self._send(requests)
        self._target = q
        self._read_at = -1e9          # the next reading is taken after this command

    # --- time and cameras ---------------------------------------------------------
    def step(self, seconds: float) -> None:
        if self._drifting and not self._commanded_drift:
            # no new base target in this period: the velocity command would otherwise be held
            self._send(self.driver.stop(self.robot))
            self._drifting = False
        self._commanded_drift = False
        if self._lease_lost is not None:
            raise self._lease_lost
        if self.dry_run:
            self.client.wait(seconds)
        else:
            self._renew_lease()
            # The period ends ``seconds`` after the one before it: what its requests took is part of it. A
            # base velocity is the step per period, so a longer period would carry the base further. After a
            # pause (a model call) the cadence starts anew.
            spent = time.monotonic() - self._period_end
            self.client.wait(seconds if spent >= 2 * seconds else max(0.0, seconds - spent))
            self._period_end = time.monotonic()
        self._time += seconds

    def sim_time(self) -> float:
        return self._time

    def render(self, camera: str, width: int, height: int) -> np.ndarray:
        if self.eyes is None:
            raise BackendUnsupported(f"camera {camera!r}: a dry run has no cameras")
        return self.eyes.frame(camera, width, height)

    def _render_depth(self, camera: str, width: int, height: int) -> np.ndarray:
        return self.depths.depth(camera, width, height)

    def _pixel_ray(self, camera: str, u: float, v: float, width: int, height: int) -> np.ndarray | None:
        """The direction a pixel of the ``width`` x ``height`` image looks in, scaled to z = 1, from the
        calibrated lens (for colour-aligned depth the frame's own pinhole); None where the lens model does not
        reach."""
        lens = self.depths.lens(camera)
        ray = lens.rays(u * lens.width / width, v * lens.height / height)
        return None if not np.isfinite(ray).all() or ray[2] <= 1e-6 else ray / ray[2]

    def _camera_mount(self, camera: str):
        """(parent link, position, rotation) of the camera's calibrated optical frame, or None when the
        calibration file has no mount for it (the description's nominal pose is used then)."""
        mount = self.calibration.mounts.get(camera)
        return None if mount is None else (mount.parent_link, mount.position, mount.rotation)

    def cameras(self) -> Sequence[str]:
        return self.eyes.names() if self.eyes is not None else ()

    def close(self) -> None:
        """Stop the base, hand the lease back, close devices, topics and connections; each step is tried
        whatever the one before it did."""
        steps = []
        with self._lease_lock:               # a renewal under way finishes; none starts after this
            self._lease_stop.set()
        if self.client is not None and not self.dry_run:
            if self._drifting:
                steps.append(lambda: self._send(self.driver.stop(self.robot)))
            if self._lease is not None:
                steps.append(lambda: self._lease_call("release", {"holder": self._lease["holder"]}))
        steps += [s.close for s in (self.depths, self.eyes, self._bus, self.stream) if s is not None]
        for step in steps:
            try:
                step()
            except Exception as exc:        # closing goes on: the lease expires by itself, the rest is local
                self.echo(f"real backend: while closing: {exc}")
        self._drifting, self._lease = False, None
        self.depths = self.eyes = self.stream = self.client = self._bus = None
        for name in ("render_depth", "pixel_ray", "camera_mount", "depth_cameras"):
            self.__dict__.pop(name, None)

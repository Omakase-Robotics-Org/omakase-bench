"""Isaac Sim backend, client side: drives ``server.py`` on a GPU host.

With ``host=local`` (the recommended setup: run the harness on the GPU host
itself) the client starts the server as a child process. With an ssh
destination it copies this package to the host, starts the server there with
a port forward and connects; every control tick is one round trip, so this is
only practical on a low-latency link. The server exits when the client
disconnects. ``url=host:port`` connects to a server started by hand.

Options (``-B key=value``): ``host`` (``local`` or an ssh destination), ``remote_dir``,
``setup`` (shell run before Python on the host, e.g. sourcing library paths),
``python`` (the Isaac Lab interpreter), ``port`` (0: a free one), ``url``,
``startup_timeout``, ``log`` (server output file). The first four name a site's
machines and paths, so an option left out is read from the environment
(``OMAKASE_BENCH_ISAAC_HOST``, ``_REMOTE_DIR``, ``_SETUP``, ``_PYTHON``) before its
default: ``local``, ``~/omakase-bench``, no setup, ``python``.
"""
from __future__ import annotations

import base64
import io
import json
import os
import shlex
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ...core.backend import ArticulatedState, BackendInfo, ObjectState, OracleSnapshot
from ...core.task import Region, SceneSpec, scene_to_dict
from ...robots.spec import RobotSpec

VERSION = "0.1.0"
PACKAGE_ROOT = Path(__file__).resolve().parents[3]   # .../src


def _site(value: str | None, name: str, default: str) -> str:
    """An option, else ``OMAKASE_BENCH_ISAAC_<name>`` from the environment, else ``default``."""
    return value if value is not None else os.environ.get(f"OMAKASE_BENCH_ISAAC_{name}", default)


class IsaacBackend:
    name = "isaac"

    def __init__(self, *, host: str | None = None, remote_dir: str | None = None, setup: str | None = None,
                 python: str | None = None, port: int | str = 0, url: str = "",
                 startup_timeout: float | str = 240.0, log: str = ""):
        self.host = _site(host, "HOST", "local")
        self.remote_dir = _site(remote_dir, "REMOTE_DIR", "~/omakase-bench")
        self.setup = _site(setup, "SETUP", "") or "true"        # "true": the shell's no-op before "&&"
        self.python = _site(python, "PYTHON", "python")
        self.port, self.url, self.startup_timeout = int(port), url, float(startup_timeout)
        self.log_path = Path(log) if log else None
        self.robot: RobotSpec | None = None
        self.scene: SceneSpec | None = None
        self._sock: socket.socket | None = None
        self._proc: subprocess.Popen | None = None
        self._pending: dict[str, float] = {}
        self._hello: dict[str, Any] = {}
        self._time = 0.0
        self._q: dict[str, float] | None = None

    # --- connection --------------------------------------------------------------
    def _connect(self) -> None:
        if self._sock is not None:
            return
        if self.url:
            host, _, port = self.url.rpartition(":")
            self._sock = socket.create_connection((host or "127.0.0.1", int(port)), timeout=600)
        else:
            self._launch()
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._hello = self._call("hello")

    def _launch(self) -> None:
        if self.host == "local":
            src = PACKAGE_ROOT
            self.port = self.port or _free_port()
            cmd = ["bash", "-c", f"{self.setup} && export PYTHONPATH={src}:$PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES && "
                                 f"exec {self.python} -m omakase_bench.backends.isaac.server --port {self.port}"]
            self._start(cmd, self.port)
            return
        rd = self.remote_dir
        self.port = self.port or 8990
        subprocess.run(["ssh", self.host, f"mkdir -p {rd}/src"], check=True, timeout=60)
        subprocess.run(["rsync", "-a", "--delete", "--exclude", "__pycache__", str(PACKAGE_ROOT / "omakase_bench"),
                        f"{self.host}:{rd}/src/"], check=True, timeout=300)
        local_port = _free_port()
        remote = (f"cd {rd} && {self.setup} && export PYTHONPATH={rd}/src:$PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES && "
                  f"exec {self.python} -m omakase_bench.backends.isaac.server --port {self.port}")
        cmd = ["ssh", "-o", "ExitOnForwardFailure=yes", "-L", f"{local_port}:127.0.0.1:{self.port}", self.host, remote]
        self._start(cmd, local_port)

    def _start(self, cmd: list[str], local_port: int) -> None:
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                      start_new_session=True)
        ready = threading.Event()
        log = self.log_path.open("w") if self.log_path else None

        def pump():
            for line in self._proc.stdout:
                if log:
                    log.write(line)
                    log.flush()
                if "OMAKASE_BENCH_ISAAC_READY" in line:
                    ready.set()
        threading.Thread(target=pump, daemon=True).start()
        started = time.monotonic()
        while not ready.wait(1.0):
            if self._proc.poll() is not None:
                raise RuntimeError(f"Isaac server ({self.host}) exited with code {self._proc.returncode} before it was "
                                   "ready: is Isaac Sim installed there?")
            if time.monotonic() - started > self.startup_timeout:
                self._proc.kill()
                raise RuntimeError(f"Isaac server ({self.host}) did not report ready in {self.startup_timeout:.0f} s")
        deadline = time.time() + 30
        while True:
            try:
                self._sock = socket.create_connection(("127.0.0.1", local_port), timeout=600)
                return
            except OSError:
                if time.time() > deadline:
                    raise
                time.sleep(0.5)

    def _call(self, cmd: str, **kw) -> dict[str, Any]:
        data = json.dumps({"cmd": cmd, **kw}).encode()
        self._sock.sendall(struct.pack("!I", len(data)) + data)
        head = _recv_exact(self._sock, 4)
        reply = json.loads(_recv_exact(self._sock, struct.unpack("!I", head)[0]))
        if not reply.get("ok"):
            raise RuntimeError(f"isaac server: {reply.get('error')}\n{reply.get('traceback', '')}")
        return reply

    def ping(self) -> str:
        self._connect()
        return json.dumps(self._hello)

    # --- Backend Protocol ------------------------------------------------------------
    def info(self) -> BackendInfo:
        v = self._hello.get("versions", {})
        return BackendInfo(
            name="isaac", version=VERSION, engine=f"isaacsim {v.get('isaacsim', '?')} / isaaclab {v.get('isaaclab', '?')}",
            robot=self.robot.name if self.robot else "", robot_sha256=self.robot.sha256() if self.robot else "",
            physics_dt=float(self._hello.get("physics_dt", 0.005)), control_dt=float(self._hello.get("control_dt", 0.02)),
            capabilities=frozenset({"reset", "ground_truth", "contacts"}), detail={"host": self.host if not self.url else self.url})

    def load(self, robot: RobotSpec, scene: SceneSpec) -> None:
        self.robot, self.scene = robot, scene
        self._connect()

    def reset(self, scene: SceneSpec, *, seed: int) -> None:
        self.scene = scene
        self._pending = {}
        r = self._call("reset", robot=self.robot.name, scene=scene_to_dict(scene), seed=seed)
        self._time = r["time"]
        self._q = None

    def joint_positions(self) -> dict[str, float]:
        if self._q is None:
            self._q = self._call("joints")["q"]
        return dict(self._q)

    def command_joints(self, targets: Mapping[str, float]) -> None:
        """Targets are sent with the next :meth:`step`, saving one round trip per control tick."""
        self._pending.update({k: float(v) for k, v in targets.items()})

    def step(self, seconds: float) -> None:
        r = self._call("step", targets=self._pending, seconds=seconds)
        self._pending = {}
        self._time = r["time"]
        self._q = None

    def sim_time(self) -> float:
        return self._time

    def cameras(self) -> Sequence[str]:
        return self._call("cameras")["cameras"]

    def render(self, camera: str, width: int, height: int) -> np.ndarray:
        from PIL import Image
        png = base64.b64decode(self._call("render", camera=camera, width=width, height=height)["png"])
        return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))

    def oracle(self) -> OracleSnapshot:
        s = self._call("oracle")["snapshot"]
        objects = {k: ObjectState(v["name"], v["kind"], v["label"], tuple(v["pos"]), tuple(v["quat"]), tuple(v["size"]),
                                  tuple(v["lin_vel"])) for k, v in s["objects"].items()}
        arts = {k: ArticulatedState(v["name"], v["kind"], v["q"], v["pressed"], v["pressed_ever"], v["open_fraction"],
                                    v["lamp"], tuple(v["pos"]), tuple(v["normal"])) for k, v in s["articulated"].items()}
        return OracleSnapshot(
            time=s["time"], objects=objects, articulated=arts, base=tuple(s["base"]), base_footprint=(0.0, 0.0),
            tcp={k: tuple(v) for k, v in s["tcp"].items()}, gripper_opening=s["gripper_opening"],
            contacts=frozenset(tuple(c) for c in s["contacts"]), contacts_ever=frozenset(tuple(c) for c in s["contacts_ever"]),
            held_by=s["held_by"], initial={k: tuple(v) for k, v in s["initial"].items()},
            regions={k: Region(k, r.lo, r.hi) for k, r in self.scene.regions.items()})

    def close(self) -> None:
        if self._sock is not None:
            try:
                data = json.dumps({"cmd": "close"}).encode()
                self._sock.sendall(struct.pack("!I", len(data)) + data)
            except OSError:
                pass
            self._sock.close()
            self._sock = None
        if self._proc is not None:
            try:
                self._proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            raise ConnectionError("isaac server closed the connection")
        buf += chunk
    return bytes(buf)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port

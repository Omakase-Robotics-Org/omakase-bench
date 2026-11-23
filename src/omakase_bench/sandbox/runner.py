"""Parent side of the sandbox: start the child, serve robot calls, enforce the wall timeout."""
from __future__ import annotations

import json
import os
import resource
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

CHILD = Path(__file__).with_name("child.py")
OUTPUT_LIMIT = 32_000


@dataclass
class ProgramResult:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    calls: int
    wall_seconds: float
    stop: Exception | None = None            # a StopEpisode raised by the harness during the program
    images: list[Any] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def _limits(timeout: float, memory_bytes: int) -> Callable[[], None]:
    def apply() -> None:
        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (int(timeout) + 5, int(timeout) + 5))
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return apply


def _unshare_prefix() -> list[str]:
    """``unshare -rn`` (no network namespace access) where unprivileged namespaces are allowed."""
    exe = shutil.which("unshare")
    if not exe:
        return []
    try:
        ok = subprocess.run([exe, "-rn", "true"], capture_output=True, timeout=5).returncode == 0
    except (OSError, subprocess.SubprocessError):
        ok = False
    return [exe, "-rn"] if ok else []


_PREFIX: list[str] | None = None


def sandbox_report() -> dict[str, Any]:
    global _PREFIX
    if _PREFIX is None:
        _PREFIX = _unshare_prefix()
    return {"network_namespace": bool(_PREFIX), "audit_hook": True,
            "rlimits": ["CPU", "AS", "FSIZE=0", "NPROC=0"], "import_allow_list": True}


def run_program(code: str, dispatch: Callable[[str, list, dict], Any], *, timeout: float = 60.0,
                memory_bytes: int = 2 << 30) -> ProgramResult:
    """Run ``code`` in a child interpreter. ``dispatch(fn, args, kwargs)`` serves each robot call
    and returns a JSON-able result; it raises ``ValueError`` (or the API's ``Refused``) to report
    an error to the program, or a ``StopEpisode`` to end the program and the episode."""
    global _PREFIX
    if _PREFIX is None:
        _PREFIX = _unshare_prefix()
    req_r, req_w = os.pipe()
    resp_r, resp_w = os.pipe()
    workdir = tempfile.mkdtemp(prefix="omakase-bench-sbx-")
    t0 = time.monotonic()
    cmd = [*_PREFIX, sys.executable, "-I", str(CHILD), str(req_w), str(resp_r)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            pass_fds=(req_w, resp_r), cwd=workdir, env={"PATH": "/usr/bin:/bin", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                                 "MKL_NUM_THREADS": "1"},
                            preexec_fn=_limits(timeout, memory_bytes))
    os.close(req_w)
    os.close(resp_r)
    out: dict[str, bytes] = {"stdout": b"", "stderr": b""}

    def pump(name: str, stream) -> None:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break
            if len(out[name]) < OUTPUT_LIMIT:
                out[name] += chunk

    threads = [threading.Thread(target=pump, args=(n, s), daemon=True)
               for n, s in (("stdout", proc.stdout), ("stderr", proc.stderr))]
    for t in threads:
        t.start()
    try:
        proc.stdin.write(code.encode())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    calls, timed_out, stop = 0, False, None
    buf = b""
    images: list[Any] = []
    try:
        while True:
            remaining = timeout - (time.monotonic() - t0)
            if remaining <= 0:
                timed_out = True
                break
            ready, _, _ = select.select([req_r], [], [], min(remaining, 0.5))
            if not ready:
                if proc.poll() is not None:
                    break
                continue
            chunk = os.read(req_r, 65536)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, _, buf = buf.partition(b"\n")
                calls += 1
                reply = _serve(line, dispatch, images)
                if "stop" in reply:
                    stop = reply.pop("_exc")
                try:
                    os.write(resp_w, json.dumps(reply, default=str).encode() + b"\n")
                except BrokenPipeError:
                    pass
                if stop is not None:
                    break
            if stop is not None:
                break
    finally:
        if proc.poll() is None and (timed_out or stop is not None):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        for t in threads:
            t.join(timeout=2)
        for fd in (req_r, resp_w):
            try:
                os.close(fd)
            except OSError:
                pass
        shutil.rmtree(workdir, ignore_errors=True)
    return ProgramResult(exit_code=proc.returncode, stdout=out["stdout"].decode(errors="replace"),
                         stderr=out["stderr"].decode(errors="replace"), timed_out=timed_out, calls=calls,
                         wall_seconds=round(time.monotonic() - t0, 2), stop=stop, images=images)


def _serve(line: bytes, dispatch, images: list) -> dict[str, Any]:
    from ..harness.session import StopEpisode
    try:
        req = json.loads(line)
        fn, args, kwargs = req["fn"], list(req.get("args") or []), dict(req.get("kwargs") or {})
    except (ValueError, KeyError, TypeError) as exc:
        return {"ok": False, "error": f"malformed request: {exc}"}
    try:
        result = dispatch(fn, args, kwargs)
    except StopEpisode as exc:
        return {"ok": False, "stop": exc.reason, "_exc": exc}
    except Exception as exc:   # refusals and invalid calls go back to the program as RobotError
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if hasattr(result, "png"):
        images.append(result)
        result = result.to_json()
    return {"ok": True, "result": result}

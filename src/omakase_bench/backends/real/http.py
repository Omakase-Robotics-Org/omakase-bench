"""The HTTP transport of the real backend: requests of a robot's own API, sent to a robot.

:class:`HttpDaemonClient` never chooses a host: the base URL comes from its argument or
``OMAKASE_BENCH_ROBOT_URL``. It refuses to exist unless it is enabled by its ``enable`` argument and
``OMAKASE_BENCH_ROBOT_ENABLE=1`` together. With ``dry_run`` it prints each request and opens no socket.
Every request is checked against the robot's OpenAPI document before it is sent.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Mapping

from ...harness.command_api import Envelope, error, ok, operation, validate

ENABLE_ENV = "OMAKASE_BENCH_ROBOT_ENABLE"
URL_ENV = "OMAKASE_BENCH_ROBOT_URL"
TIMEOUT_MARGIN_S = 5.0       # added to an operation's own x-timeout-seconds


class HttpDaemonClient:
    """Sends API requests to a robot daemon over HTTP. It has no default host, and it only exists when the
    caller enables it (``enable=True``) and the environment does too (``OMAKASE_BENCH_ROBOT_ENABLE=1``).
    With ``dry_run`` every request is printed (or handed to ``echo``) and nothing is sent."""

    def __init__(self, spec: Mapping, base_url: str | None = None, *, enable: bool = False, dry_run: bool = False,
                 timeout_s: float = 35.0, echo: Callable[[str], None] = print):
        self.spec, self.dry_run, self.timeout_s, self.echo = spec, dry_run, timeout_s, echo
        self.base_url = (base_url or os.environ.get(URL_ENV, "")).rstrip("/")
        if not dry_run:
            if not enable or os.environ.get(ENABLE_ENV) != "1":
                raise PermissionError(f"the robot HTTP client needs enable=True and {ENABLE_ENV}=1")
            if not re.match(r"^https?://", self.base_url):
                raise ValueError(f"no robot URL: pass base_url or set {URL_ENV} (http://host:port)")

    def request(self, method: str, path: str, body: Any = None) -> Envelope:
        bad = validate(self.spec, method, path, body)
        if bad:
            return error("invalid", "; ".join(bad), 400)
        if self.dry_run:
            self.echo(f"{method.upper()} {self.base_url or '<robot>'}{path} {json.dumps(body) if body is not None else ''}".rstrip())
            return ok()
        import urllib.error
        import urllib.request
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base_url + path, data=data, method=method.upper(),
                                     headers={"Content-Type": "application/json", "Accept": "application/json"})
        # a blocking operation states how long it may take (x-timeout-seconds: a waited lift move, 300 s)
        op, _ = operation(self.spec, method, path)
        timeout = max(self.timeout_s, float(op.get("x-timeout-seconds", 0.0)) + TIMEOUT_MARGIN_S)
        import http.client
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status, raw = resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, exc.read()
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError) as exc:
            # no answer: the caller stops its command line here; whether the robot moved is unknown
            return error("unreachable", f"no answer from the robot within {timeout:g} s: {exc}", 0)
        try:
            env = json.loads(raw.decode() or "{}")
        except ValueError:
            env = None
        if not isinstance(env, dict):
            # not the API's envelope: something else answers at this address
            return error("device", f"HTTP {status}: the answer is not the API's JSON envelope", status)
        if status >= 400:
            env.setdefault("http_status", status)
        return env

    def wait(self, seconds: float) -> None:
        if self.dry_run:
            self.echo(f"# wait {seconds:g} s")
            return
        import time
        time.sleep(seconds)

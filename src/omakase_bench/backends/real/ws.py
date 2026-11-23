"""The WebSocket transport of the real backend: the same operations over one persistent connection.

A robot API that offers its operations over WebSocket marks each with ``x-ws-method`` in its OpenAPI
document. :class:`WsDaemonClient` sends such an operation as a request frame ``{id, method, params}`` and
returns the response frame with the same ``id`` in the shape of the REST envelope (``status``, ``data``,
``message``, and ``kind`` on an error). Responses may arrive out of order and the connection also carries
unsolicited event frames; both are passed over until the awaited ``id`` arrives.

It is the transport for streamed motion: one connection, no request set-up per control tick. Like the
HTTP transport it never chooses a host (argument or ``OMAKASE_BENCH_ROBOT_WS_URL``), is disabled unless
``enable`` and ``OMAKASE_BENCH_ROBOT_ENABLE=1`` are both given, and checks every request against the
document before sending it. It needs the ``websockets`` package (``pip install 'omakase-bench[real]'``).
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Mapping

from ...harness.command_api import Envelope, error, operation, validate
from .http import ENABLE_ENV

WS_URL_ENV = "OMAKASE_BENCH_ROBOT_WS_URL"


class WsDaemonClient:
    def __init__(self, spec: Mapping, url: str | None = None, *, enable: bool = False, timeout_s: float = 35.0):
        self.spec, self.timeout_s = spec, timeout_s
        self.url = url or os.environ.get(WS_URL_ENV, "")
        if not enable or os.environ.get(ENABLE_ENV) != "1":
            raise PermissionError(f"the robot WebSocket client needs enable=True and {ENABLE_ENV}=1")
        if not re.match(r"^wss?://", self.url):
            raise ValueError(f"no robot WebSocket URL: pass ws_url or set {WS_URL_ENV} (ws://host:port/path)")
        from websockets.exceptions import WebSocketException
        from websockets.sync.client import connect
        self._lost = (TimeoutError, OSError, WebSocketException)
        # entered by hand and left in close(): the connection lives as long as the backend
        self._connection = connect(self.url, open_timeout=timeout_s)
        self._socket = self._connection.__enter__()
        self._next_id = 0

    def method(self, http_method: str, path: str) -> str:
        """The ``x-ws-method`` of an operation, with the path's parameters filled in (``arm.a.move_joints``)."""
        op, params = operation(self.spec, http_method, path)
        name = op.get("x-ws-method")
        if not name:
            raise KeyError(f"{http_method.upper()} {path} is not offered over WebSocket")
        for key, value in params.items():
            name = name.replace("{" + key + "}", value)
        return name

    def request(self, method: str, path: str, body: Any = None) -> Envelope:
        bad = validate(self.spec, method, path, body)
        if bad:
            return error("invalid", "; ".join(bad), 400)
        self._next_id += 1
        frame = {"id": self._next_id, "method": self.method(method, path)}
        if body is not None:
            frame["params"] = body
        deadline = time.monotonic() + self.timeout_s
        try:
            self._socket.send(json.dumps(frame))
            while True:
                try:
                    reply = json.loads(self._socket.recv(timeout=max(0.0, deadline - time.monotonic())))
                except ValueError:
                    continue                      # not a frame of this protocol
                if isinstance(reply, dict) and "event" not in reply and reply.get("id") == self._next_id:
                    return reply
        except self._lost as exc:
            # no answer, or the connection is gone: whether the robot moved is unknown
            return error("unreachable", f"no answer from the robot within {self.timeout_s:g} s: {exc or type(exc).__name__}", 0)

    def wait(self, seconds: float) -> None:
        time.sleep(seconds)

    def close(self) -> None:
        self._connection.__exit__(None, None, None)

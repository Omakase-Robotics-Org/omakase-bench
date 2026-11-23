"""A model asked through files: a person or an interactive agent session answers.

Request N is written to ``<dir>/req-NNN.json`` (messages added since the
previous request; the system text and tools when they change) with images
beside it; the adapter then waits for ``<dir>/rep-NNN.json``::

    {"calls": [{"name": "move_ee", "arguments": {...}}], "text": ""}

An empty ``calls`` list ends the turn without a tool call (for the code
interface, put the program in ``text``).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .base import Capabilities, Image, Message, Reply, Request, ToolCall, Usage


class MailboxAdapter:
    capabilities = Capabilities(images=True)

    def __init__(self, directory: str, *, poll_s: float = 0.3, timeout_s: float = 6 * 3600.0):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name = f"mailbox/{self.dir.name}"
        self.poll_s, self.timeout_s = poll_s, timeout_s
        self._n = len(list(self.dir.glob("req-*.json")))
        self._sent = 0
        self._tools: Any = None
        self._system: str | None = None

    def _dump(self, n: int, i: int, message: Message) -> dict[str, Any]:
        def parts(ps, tag):
            out = []
            for j, p in enumerate(ps):
                if isinstance(p, Image):
                    ext = "png" if p.media_type.endswith("png") else "jpg"
                    path = self.dir / f"req-{n:03d}-{i}{tag}{j}.{ext}"
                    path.write_bytes(p.data)
                    out.append({"image": path.name, "label": p.label})
                else:
                    out.append({"text": p.text})
            return out
        return {"role": message.role, "parts": parts(message.parts, "p"),
                "calls": [{"id": c.id, "name": c.name, "arguments": dict(c.arguments)} for c in message.calls],
                "results": [{"call_id": r.call_id, "is_error": r.is_error, "parts": parts(r.parts, f"r{k}_")}
                            for k, r in enumerate(message.results)]}

    def respond(self, request: Request) -> Reply:
        n = self._n
        self._n += 1
        t0 = time.monotonic()
        if len(request.messages) < self._sent:
            self._sent = 0
        tools = [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in request.tools]
        body: dict[str, Any] = {"n": n, "max_calls": request.max_calls,
                                "new_messages": [self._dump(n, i, m) for i, m in
                                                 enumerate(request.messages[self._sent:], start=self._sent)]}
        if request.system != self._system:
            body["system"] = request.system
            self._system = request.system
        if tools != self._tools:
            body["tools"] = tools
            self._tools = tools
        self._sent = len(request.messages) + 1
        tmp = self.dir / f"req-{n:03d}.json.tmp"
        tmp.write_text(json.dumps(body, indent=1))
        tmp.rename(self.dir / f"req-{n:03d}.json")
        answer = self.dir / f"rep-{n:03d}.json"
        while not answer.exists():
            if time.monotonic() - t0 > self.timeout_s:
                return Reply(stop="error", error=f"no answer in {answer} after {self.timeout_s:.0f} s")
            time.sleep(self.poll_s)
        time.sleep(0.1)
        data = json.loads(answer.read_text())
        calls = tuple(ToolCall(c.get("id") or f"call_{n}_{k}", c["name"], c.get("arguments") or {})
                      for k, c in enumerate(data.get("calls") or []))
        return Reply(calls=calls, text=str(data.get("text") or ""), stop="tool_use" if calls else "end",
                     usage=Usage(seconds=round(time.monotonic() - t0, 2)), model="mailbox")

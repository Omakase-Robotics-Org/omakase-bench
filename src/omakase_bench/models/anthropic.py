"""Anthropic Messages API adapter (streaming, adaptive thinking, server-side refusal fallback).

The reply's content blocks are kept whole in ``provider_state`` and sent back
unchanged, which carries the model's reasoning across turns, so the history
is append-only.
"""
from __future__ import annotations

import base64
import time
from typing import Any

from .base import Capabilities, Image, Reply, Request, ToolCall, Usage, with_retries
from .pricing import Price, cost

_FALLBACK_BETA = "server-side-fallback-2026-07-01"
#: blocks of a model that declined mid-output: not sent back, and their tool calls are not carried out
_DROPPED_BEFORE_FALLBACK = ("thinking", "redacted_thinking", "tool_use")
#: error types of a 200 response whose stream then fails (the status says nothing there), and statuses worth another try
_RETRYABLE_TYPES = ("overloaded_error", "api_error")
_RETRYABLE_STATUS = (408, 409)


def _blocks(parts) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in parts:
        if isinstance(p, Image):
            if p.label:
                out.append({"type": "text", "text": p.label})
            out.append({"type": "image", "source": {"type": "base64", "media_type": p.media_type,
                                                    "data": base64.standard_b64encode(p.data).decode()}})
        elif p.text:
            out.append({"type": "text", "text": p.text,
                        **({"cache_control": {"type": "ephemeral"}} if p.cache_prefix else {})})
    return out


def to_wire(request: Request) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    for m in request.messages:
        if m.role == "assistant":
            content = list(m.provider_state) if m.provider_state else (
                _blocks(m.parts) + [{"type": "tool_use", "id": c.id, "name": c.name, "input": dict(c.arguments)}
                                    for c in m.calls])
            messages.append({"role": "assistant", "content": content})
            continue
        content = [{"type": "tool_result", "tool_use_id": r.call_id,
                    "content": _blocks(r.parts) or [{"type": "text", "text": "(no output)"}],
                    **({"is_error": True} if r.is_error else {})} for r in m.results]
        content += _blocks(m.parts)
        messages.append({"role": "user", "content": content})
    tools = [{"name": t.name, "description": t.description,
              "input_schema": dict(t.parameters) or {"type": "object", "properties": {}}} for t in request.tools]
    return {"messages": messages, "tools": tools}


def from_wire(message: Any, seconds: float, price: Price) -> Reply:
    u = message.usage
    cached = int(getattr(u, "cache_read_input_tokens", 0) or 0)
    written = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
    usage = Usage(input_tokens=int(u.input_tokens or 0), output_tokens=int(u.output_tokens or 0),
                  cached_tokens=cached, cache_write_tokens=written, seconds=seconds)
    usage = Usage(**{**usage.__dict__, "usd": cost(price, input_tokens=usage.input_tokens,
                                                    output_tokens=usage.output_tokens, cached_tokens=cached,
                                                    cache_write_tokens=written)})
    text = "".join(b.text for b in message.content if b.type == "text")
    model = getattr(message, "model", "") or ""
    if message.stop_reason == "refusal":
        return Reply(text=text, stop="refusal", usage=usage, model=model)
    if message.stop_reason == "max_tokens":
        return Reply(text=text, stop="max_tokens", usage=usage, model=model,
                     error="the reply hit max_tokens before it finished")
    # After a server-side fallback the content holds a ``fallback`` block per switch. What the declining
    # models produced before the last one is theirs alone: the API refuses their thinking and tool-use blocks
    # when they are sent back, and the block itself has to stay where it was.
    last = max((i for i, b in enumerate(message.content) if b.type == "fallback"), default=-1)
    content = [b for i, b in enumerate(message.content) if i >= last or b.type not in _DROPPED_BEFORE_FALLBACK]
    calls = tuple(ToolCall(b.id, b.name, dict(b.input or {})) for b in content if b.type == "tool_use")
    # by_alias: the fallback block's field ``from`` is ``from_`` in the SDK's model
    state = [b.model_dump(by_alias=True, exclude_none=True) for b in content]
    thinking = "\n".join(t for b in content if b.type == "thinking" and (t := (getattr(b, "thinking", "") or "").strip()))
    return Reply(calls=calls, text=text, stop="tool_use" if calls else "end", usage=usage,
                 provider_state=state, model=model, thinking=thinking)


class AnthropicAdapter:
    capabilities = Capabilities(images=True, images_in_tool_results=True, append_only_history=True)

    def __init__(self, model: str, *, price: Price, effort: str = "medium", max_tokens: int = 32000,
                 fallbacks: bool = True, attempts: int = 3, client: Any = None):
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.fallbacks, self.attempts, self.price = fallbacks, attempts, price
        self.name = f"anthropic/{model}"
        self._client = client

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(max_retries=0, timeout=600)
        return self._client

    def _kwargs(self, request: Request) -> dict[str, Any]:
        wire = to_wire(request)
        kw: dict[str, Any] = dict(model=self.model, max_tokens=self.max_tokens, system=request.system,
                                  messages=wire["messages"], output_config={"effort": self.effort})
        # a request that marks its own repeating prefix is cached there; otherwise the whole
        # conversation so far is, which suits append-only histories
        if not any(getattr(p, "cache_prefix", False) for m in request.messages for p in m.parts):
            kw["cache_control"] = {"type": "ephemeral"}
        if wire["tools"]:
            kw["tools"] = wire["tools"]
            kw["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": request.max_calls <= 1}
        if request.thinking_summary:
            # the display only: thinking runs and is billed the same whether its summary is returned or not
            kw["thinking"] = {"type": "adaptive", "display": "summarized"}
        if self.fallbacks:
            kw["betas"] = [_FALLBACK_BETA]
            kw["fallbacks"] = "default"
        return kw

    def _once(self, request: Request) -> Reply:
        import anthropic
        t0 = time.monotonic()
        try:
            with self._get_client().beta.messages.stream(**self._kwargs(request)) as stream:
                message = stream.get_final_message()
        except anthropic.RateLimitError as exc:
            return Reply(stop="error", error=f"retryable: rate limited: {exc.message}")
        except anthropic.InternalServerError as exc:
            return Reply(stop="error", error=f"retryable: HTTP {exc.status_code}: {exc.message}")
        except anthropic.APIStatusError as exc:
            again = getattr(exc, "type", None) in _RETRYABLE_TYPES or exc.status_code in _RETRYABLE_STATUS
            return Reply(stop="error", error=f"{'retryable: ' if again else ''}HTTP {exc.status_code}: {exc.message}")
        except anthropic.APIConnectionError as exc:
            return Reply(stop="error", error=f"retryable: connection: {exc}")
        return from_wire(message, round(time.monotonic() - t0, 2), self.price)

    def respond(self, request: Request) -> Reply:
        return with_retries(lambda: self._once(request), attempts=self.attempts)

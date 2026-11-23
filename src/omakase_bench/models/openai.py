"""OpenAI Responses API adapter (function calling, stateless input with encrypted reasoning).

Prompt caching follows the model generation (OpenAI prompt-caching guide):

- GPT-6.2 and later append top-level ``instructions`` after ``input``, where nothing is cached, so the
  system prompt goes out as the first ``developer`` message of ``input``. Earlier models keep
  ``instructions``.
- GPT-5.6 and later take explicit cache breakpoints. A request that marks its repeating prefix
  (``Text(cache_prefix=True)``) carries a breakpoint on that block and asks for explicit mode
  (``prompt_cache_options``), so that only the prefix is written to the cache: in implicit mode the
  whole request would be billed as a cache write on every call. Requests without a marked prefix are
  left to the implicit cache.
- Usage: cache reads and cache writes are reported apart (``input_tokens_details.cached_tokens`` and
  ``cache_write_tokens`` with its per-lifetime split) and each input token is costed at one rate.
"""
from __future__ import annotations

import base64
import json
import re
import time
from typing import Any

from .base import Capabilities, Image, Reply, Request, ToolCall, Usage, with_retries
from .pricing import Price, cost


def generation(model: str) -> tuple[int, int]:
    """(major, minor) of a ``gpt-<major>.<minor>...`` model name; (0, 0) for any other name."""
    m = re.match(r"gpt-(\d+)(?:\.(\d+))?", model)
    return (int(m.group(1)), int(m.group(2) or 0)) if m else (0, 0)


#: from this generation on, top-level instructions follow the input and cannot be cached
DEVELOPER_MESSAGE_FROM = (6, 2)
#: from this generation on, explicit cache breakpoints and prompt_cache_options exist
BREAKPOINTS_FROM = (5, 6)
CACHE_TTLS = ("30m", "12h")
SUMMARIES = ("auto", "concise", "detailed")


def _content(parts, breakpoints: bool = False) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in parts:
        if isinstance(p, Image):
            if p.label:
                out.append({"type": "input_text", "text": p.label})
            out.append({"type": "input_image", "detail": "high",
                        "image_url": f"data:{p.media_type};base64," + base64.b64encode(p.data).decode()})
        elif p.text:
            block: dict[str, Any] = {"type": "input_text", "text": p.text}
            if breakpoints and getattr(p, "cache_prefix", False):
                block["prompt_cache_breakpoint"] = {"mode": "explicit"}
            out.append(block)
    return out


def _marks_prefix(request: Request) -> bool:
    return any(getattr(p, "cache_prefix", False) for m in request.messages for p in m.parts)


def to_wire(request: Request, *, developer_system: bool = False, breakpoints: bool = False) -> list[dict[str, Any]]:
    """The ``input`` items. ``developer_system``: the system prompt leads as a developer message;
    ``breakpoints``: a marked cache prefix carries an explicit cache breakpoint."""
    items: list[dict[str, Any]] = []
    if developer_system and request.system:
        items.append({"role": "developer", "content": [{"type": "input_text", "text": request.system}]})
    for m in request.messages:
        if m.role == "assistant":
            if m.provider_state:
                items.extend(m.provider_state)
            else:
                text = "".join(p.text for p in m.parts if not isinstance(p, Image))
                if text:
                    items.append({"role": "assistant", "content": text})
                items.extend({"type": "function_call", "call_id": c.id, "name": c.name,
                              "arguments": json.dumps(dict(c.arguments))} for c in m.calls)
            continue
        extra: list[dict[str, Any]] = []
        for r in m.results:
            text = "\n".join(p.text for p in r.parts if not isinstance(p, Image))
            items.append({"type": "function_call_output", "call_id": r.call_id, "output": text or "(no output)"})
            extra += _content([p for p in r.parts if isinstance(p, Image)])
        content = extra + _content(m.parts, breakpoints)
        if content:
            items.append({"role": "user", "content": content})
    return items


def _cache_key(request: Request) -> dict[str, str]:
    """Routes requests that share a marked prefix to the same prompt cache."""
    import hashlib
    for m in request.messages:
        for p in m.parts:
            if getattr(p, "cache_prefix", False):
                return {"prompt_cache_key": hashlib.sha256(p.text.encode()).hexdigest()[:32]}
    return {}


def from_wire(response: Any, seconds: float, price: Price) -> Reply:
    u = getattr(response, "usage", None)
    d = u.model_dump() if hasattr(u, "model_dump") else {}
    details = d.get("input_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    written = int(details.get("cache_write_tokens") or 0)
    long_lived = int((details.get("cache_write_tokens_details") or {}).get("ttl_12h_tokens") or 0)
    total_in = int(d.get("input_tokens") or 0)
    out = int(d.get("output_tokens") or 0)
    plain = max(0, total_in - cached - written)          # every input token is billed at one rate
    # a 12-hour write costs twice a 30-minute one (2.5x and 1.25x the input rate)
    usd = cost(price, input_tokens=plain, output_tokens=out, cached_tokens=cached,
               cache_write_tokens=written + long_lived)
    usage = Usage(input_tokens=plain, output_tokens=out, cached_tokens=cached, cache_write_tokens=written,
                  reasoning_tokens=int((d.get("output_tokens_details") or {}).get("reasoning_tokens") or 0),
                  seconds=seconds, usd=usd)
    text = getattr(response, "output_text", "") or ""
    model = getattr(response, "model", "") or ""
    if getattr(response, "status", "") == "incomplete":
        if getattr(getattr(response, "incomplete_details", None), "reason", "") == "content_filter":
            return Reply(text=text, stop="refusal", usage=usage, model=model)
        return Reply(text=text, stop="max_tokens", usage=usage, error="the response is incomplete", model=model)
    if any(getattr(part, "type", "") == "refusal" for i in response.output if getattr(i, "type", "") == "message"
           for part in (getattr(i, "content", None) or [])):
        return Reply(text=text, stop="refusal", usage=usage, model=model)
    try:
        calls = tuple(ToolCall(getattr(i, "call_id", "") or i.id, i.name, json.loads(i.arguments or "{}"))
                      for i in response.output if getattr(i, "type", "") == "function_call")
    except ValueError as exc:
        return Reply(text=text, stop="error", usage=usage, model=model, error=f"a tool call's arguments are not JSON: {exc}")
    state = [i.model_dump(exclude_none=True) for i in response.output]
    thinking = "\n".join(t for i in response.output if getattr(i, "type", "") == "reasoning"
                         for part in (getattr(i, "summary", None) or [])
                         if (t := (getattr(part, "text", "") or "").strip()))
    return Reply(calls=calls, text=text, stop="tool_use" if calls else "end", usage=usage,
                 provider_state=state, model=model, thinking=thinking)


class OpenAIAdapter:
    capabilities = Capabilities(images=True)

    def __init__(self, model: str, *, price: Price, reasoning: str = "medium", max_output_tokens: int = 16000,
                 attempts: int = 3, cache_ttl: str = "30m", reasoning_summary: str = "auto", client: Any = None):
        if cache_ttl not in CACHE_TTLS:
            raise ValueError(f"cache_ttl must be one of {CACHE_TTLS}, got {cache_ttl!r}")
        if reasoning_summary not in SUMMARIES:
            raise ValueError(f"reasoning_summary must be one of {SUMMARIES}, got {reasoning_summary!r}")
        #: which summarizer a request that asks for a thinking summary names (reasoning.summary)
        self.reasoning_summary = reasoning_summary
        self.model, self.reasoning, self.max_output_tokens = model, reasoning, max_output_tokens
        self.price, self.attempts, self.cache_ttl = price, attempts, cache_ttl
        self.name = f"openai/{model}"
        self._client = client

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(max_retries=0, timeout=600)
        return self._client

    def _kwargs(self, request: Request) -> dict[str, Any]:
        gen = generation(self.model)
        developer = gen >= DEVELOPER_MESSAGE_FROM
        explicit = gen >= BREAKPOINTS_FROM and _marks_prefix(request)
        kw: dict[str, Any] = dict(
            model=self.model, input=to_wire(request, developer_system=developer, breakpoints=explicit),
            tools=[{"type": "function", "name": t.name, "description": t.description,
                    "parameters": dict(t.parameters)} for t in request.tools],
            parallel_tool_calls=request.max_calls > 1, store=False,
            include=["reasoning.encrypted_content"], max_output_tokens=self.max_output_tokens,
            **_cache_key(request))
        reasoning = {**({"effort": self.reasoning} if self.reasoning else {}),
                     **({"summary": self.reasoning_summary} if request.thinking_summary else {})}
        if reasoning:
            kw["reasoning"] = reasoning
        if not developer:
            kw["instructions"] = request.system
        if explicit:
            # sent as a body field so that it does not depend on the SDK version naming the parameter
            kw["extra_body"] = {"prompt_cache_options": {"mode": "explicit", "ttl": self.cache_ttl}}
        return kw

    def _once(self, request: Request) -> Reply:
        import openai
        t0 = time.monotonic()
        try:
            response = self._get_client().responses.create(**self._kwargs(request))
        except (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError) as exc:
            return Reply(stop="error", error=f"retryable: {type(exc).__name__}: {exc}")
        except openai.OpenAIError as exc:
            return Reply(stop="error", error=f"{type(exc).__name__}: {exc}")
        return from_wire(response, round(time.monotonic() - t0, 2), self.price)

    def respond(self, request: Request) -> Reply:
        return with_retries(lambda: self._once(request), attempts=self.attempts)

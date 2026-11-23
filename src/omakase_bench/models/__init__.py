"""Model registry: ``provider/model`` names resolve to adapters.

    anthropic/<model>      Anthropic Messages API (ANTHROPIC_API_KEY)
    openai/<model>         OpenAI Responses API (OPENAI_API_KEY)
    mailbox/<dir>          files in, files out: a person or another session answers
    mock/<behaviour>       deterministic stand-ins for tests

Options (``-M key=value``): ``effort`` / ``reasoning``, ``max_tokens``,
``price_in`` / ``price_out`` (USD per million tokens), ``attempts``, ``cache_ttl``
(OpenAI, ``30m`` or ``12h``: the lifetime of a marked prompt prefix in the cache),
``reasoning_summary`` (OpenAI, ``auto``, ``concise`` or ``detailed``: the summarizer of a
request that asks for a thinking summary).
A provider's SDK is imported only when its adapter is built.
"""
from __future__ import annotations

from typing import Any, Callable

from .base import (Capabilities, Image, Message, ModelAdapter, Reply, Request, Text, ToolCall, ToolResult,
                   ToolSpec, Usage)
from .pricing import price_for

Factory = Callable[..., ModelAdapter]
_REGISTERED: dict[str, Factory] = {}


def register(provider: str, factory: Factory) -> None:
    _REGISTERED[provider] = factory


def providers() -> list[str]:
    return sorted({"anthropic", "openai", "mailbox", "mock", *_REGISTERED})


def parse_spec(spec: str) -> tuple[str, str]:
    provider, sep, model = spec.partition("/")
    if not sep or not model:
        raise ValueError(f"model spec must be provider/model, got {spec!r}")
    return provider, model


def _need(library: str, extra: str) -> None:
    """Before the first episode, not as an error in each of them."""
    import importlib.util
    if importlib.util.find_spec(library) is None:
        raise ValueError(f"the {library} client library is not installed: uv sync --extra {extra} "
                         f"(or pip install 'omakase-bench[{extra}]')")


def build(spec: str, **options: Any) -> ModelAdapter:
    provider, model = parse_spec(spec)
    price = price_for(spec, _float(options.pop("price_in", None)), _float(options.pop("price_out", None)))
    if provider in _REGISTERED:
        return _REGISTERED[provider](model, **options)
    if provider == "anthropic":
        _need("anthropic", "anthropic")
        from .anthropic import AnthropicAdapter
        return AnthropicAdapter(model, price=price, **_typed(options, effort=str, max_tokens=int, attempts=int))
    if provider == "openai":
        _need("openai", "openai")
        from .openai import OpenAIAdapter
        return OpenAIAdapter(model, price=price, **_typed(options, reasoning=str, max_output_tokens=int,
                                                          attempts=int, cache_ttl=str,
                                                          reasoning_summary=str))
    if provider == "mailbox":
        from .mailbox import MailboxAdapter
        return MailboxAdapter(model, **_typed(options, poll_s=float, timeout_s=float))
    if provider == "mock":
        from .mock import MockAdapter
        return MockAdapter(model)
    raise ValueError(f"unknown model provider {provider!r}; known: {', '.join(providers())}")


def _float(v):
    return None if v is None else float(v)


def _typed(options: dict[str, Any], **types) -> dict[str, Any]:
    out = {}
    for k, v in options.items():
        if k not in types:
            raise ValueError(f"unknown model option {k!r}; known: {', '.join(types)}")
        out[k] = types[k](v)
    return out


__all__ = ["Capabilities", "Image", "Message", "ModelAdapter", "Reply", "Request", "Text", "ToolCall",
           "ToolResult", "ToolSpec", "Usage", "build", "parse_spec", "providers", "register"]

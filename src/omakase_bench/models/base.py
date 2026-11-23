"""The one seam between an agent and a language model.

An agent builds a :class:`Request` from provider-neutral parts and gets a
:class:`Reply`; an adapter translates both to one provider's wire format.
Adapters are stateless: the conversation is the ``messages`` of each request.
What a provider needs carried between turns (reasoning blocks, response ids)
travels in :attr:`Message.provider_state`, stored with the assistant message
and sent back unchanged.

An adapter never raises for a provider failure: it returns
``Reply(stop="error", error=...)`` after a bounded number of retries.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence, Union, runtime_checkable


@dataclass(frozen=True)
class Text:
    """``cache_prefix`` marks the end of a prompt prefix that repeats across requests; adapters of
    providers with prompt caching place a cache boundary after it."""
    text: str
    cache_prefix: bool = False


@dataclass(frozen=True)
class Image:
    """An encoded image; ``label`` is shown to the model before it."""
    data: bytes
    media_type: str = "image/png"
    label: str = ""


Part = Union[Text, Image]


@dataclass(frozen=True)
class ToolSpec:
    """A tool as JSON Schema (``parameters`` is an object schema)."""
    name: str
    description: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    parts: tuple[Part, ...]
    is_error: bool = False


@dataclass(frozen=True)
class Message:
    """``user``: parts plus the results of the previous assistant calls;
    ``assistant``: text, calls and the provider's opaque state."""
    role: str
    parts: tuple[Part, ...] = ()
    calls: tuple[ToolCall, ...] = ()
    results: tuple[ToolResult, ...] = ()
    provider_state: Any = None


@dataclass(frozen=True)
class Request:
    system: str
    messages: Sequence[Message]
    tools: Sequence[ToolSpec] = ()
    #: the most tool calls one reply may carry
    max_calls: int = 1
    #: ask the provider for a readable summary of the model's reasoning (``Reply.thinking``). It changes
    #: what is returned, not how the model reasons; the reasoning itself stays the provider's state.
    thinking_summary: bool = False


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0          # uncached input
    output_tokens: int = 0
    reasoning_tokens: int = 0      # part of output_tokens where the provider reports it
    cached_tokens: int = 0         # input read from cache
    cache_write_tokens: int = 0
    seconds: float = 0.0
    usd: float = 0.0


STOPS = ("tool_use", "end", "max_tokens", "refusal", "error")


@dataclass(frozen=True)
class Reply:
    calls: tuple[ToolCall, ...] = ()
    text: str = ""
    stop: str = "end"
    usage: Usage = field(default_factory=Usage)
    provider_state: Any = None
    error: str = ""
    model: str = ""                # the model id the provider reports
    thinking: str = ""             # the provider's readable summary of the reasoning, when asked for

    def as_message(self) -> Message:
        return Message("assistant", parts=(Text(self.text),) if self.text else (),
                       calls=self.calls, provider_state=self.provider_state)


@dataclass(frozen=True)
class Capabilities:
    images: bool = True
    images_in_tool_results: bool = False
    append_only_history: bool = False


@runtime_checkable
class ModelAdapter(Protocol):
    name: str
    capabilities: Capabilities

    def respond(self, request: Request) -> Reply: ...


def with_retries(call: Callable[[], Reply], *, attempts: int = 3, base_delay: float = 2.0,
                 retryable: Callable[[Reply], bool] = lambda r: r.stop == "error" and r.error.startswith("retryable"),
                 sleep: Callable[[float], None] = time.sleep) -> Reply:
    """Run ``call`` at most ``attempts`` times, backing off exponentially on retryable errors."""
    reply = call()
    for k in range(1, attempts):
        if not retryable(reply):
            return reply
        sleep(base_delay * (2 ** (k - 1)) * (1 + 0.25 * random.random()))
        reply = call()
    return reply

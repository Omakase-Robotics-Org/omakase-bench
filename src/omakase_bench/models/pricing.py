"""USD per million tokens, used to cost every model call.

Prices are first-party list prices; an entry marked ``assumed`` is not a
published price and should be overridden with ``-M price_in=... -M price_out=...``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cached_input: float
    cache_write: float
    assumed: bool = False


PRICES: dict[str, Price] = {
    "anthropic/claude-opus-5-5": Price(4.0, 20.0, 0.20, 5.0),
    "anthropic/claude-sonnet-5-5": Price(2.0, 10.0, 0.10, 2.5),
    "anthropic/claude-haiku-5-5": Price(0.10, 0.50, 0.01, 0.125),
    "anthropic/claude-fable-5-2": Price(8.0, 40.0, 0.20, 10.0),
    # cache_write is the 30-minute rate; a 12-hour write costs twice that (openai.py)
    "openai/gpt-6.2-astra": Price(8.0, 40.0, 0.40, 10.0),
}
DEFAULT = Price(10.0, 50.0, 1.0, 12.5, assumed=True)


def price_for(spec: str, override_in: float | None = None, override_out: float | None = None) -> Price:
    p = PRICES.get(spec, DEFAULT)
    if override_in is not None or override_out is not None:
        p = Price(override_in if override_in is not None else p.input,
                  override_out if override_out is not None else p.output,
                  p.cached_input, p.cache_write, assumed=False)
    return p


def cost(price: Price, *, input_tokens: int, output_tokens: int, cached_tokens: int = 0,
         cache_write_tokens: int = 0) -> float:
    return (input_tokens * price.input + output_tokens * price.output + cached_tokens * price.cached_input
            + cache_write_tokens * price.cache_write) / 1e6

import pytest

from omakase_bench.models import build, parse_spec
from omakase_bench.models.base import Reply, with_retries
from omakase_bench.models.pricing import cost, price_for


def test_spec_parsing():
    assert parse_spec("anthropic/claude-opus-5-5") == ("anthropic", "claude-opus-5-5")
    assert build("mock/finish").name == "mock/finish"


def test_retries_are_bounded():
    calls, sleeps = [], []

    def failing():
        calls.append(1)
        return Reply(stop="error", error="retryable: HTTP 529")
    reply = with_retries(failing, attempts=3, sleep=sleeps.append)
    assert len(calls) == 3 and len(sleeps) == 2 and reply.stop == "error"
    assert sleeps[1] > sleeps[0]


def test_non_retryable_errors_are_not_retried():
    calls = []

    def bad():
        calls.append(1)
        return Reply(stop="error", error="HTTP 400: bad request")
    with_retries(bad, attempts=5, sleep=lambda s: None)
    assert len(calls) == 1


def test_costing():
    p = price_for("anthropic/claude-opus-5-5")
    assert cost(p, input_tokens=1_000_000, output_tokens=0) == 4.0
    assert cost(p, input_tokens=0, output_tokens=100_000) == 2.0
    assert price_for("openai/unknown").assumed


def test_mailbox_round_trip(tmp_path):
    import json
    import threading

    from omakase_bench.models.base import Message, Request, Text, ToolSpec
    from omakase_bench.models.mailbox import MailboxAdapter

    box = MailboxAdapter(str(tmp_path), poll_s=0.01, timeout_s=10)

    def answer():
        req = tmp_path / "req-000.json"
        while not req.exists():
            pass
        body = json.loads(req.read_text())
        assert body["system"] == "sys" and body["tools"][0]["name"] == "get_state"
        (tmp_path / "rep-000.json").write_text(json.dumps({"calls": [{"name": "get_state", "arguments": {}}]}))
    t = threading.Thread(target=answer)
    t.start()
    reply = box.respond(Request("sys", [Message("user", (Text("go"),))],
                                [ToolSpec("get_state", "read", {"type": "object", "properties": {}})]))
    t.join()
    assert reply.stop == "tool_use" and reply.calls[0].name == "get_state"


def test_a_subject_card_states_the_price_and_whether_it_is_assumed():
    from omakase_bench.models.pricing import PRICES, price_for
    from omakase_bench.subjects.base import price_detail

    class Priced:
        price = price_for("openai/gpt-6.2-astra")

    d = price_detail(Priced())["price_usd_per_mtok"]
    assert d["assumed"] is PRICES["openai/gpt-6.2-astra"].assumed and d["input"] == Priced.price.input
    assert price_detail(object()) == {}
    Priced.price = price_for("openai/gpt-6.2-astra", 3.0, 12.0)          # -M price_in / price_out
    assert price_detail(Priced())["price_usd_per_mtok"] == {"input": 3.0, "output": 12.0, "cached_input": 0.4,
                                                             "cache_write": 10.0, "assumed": False}


# --- the OpenAI adapter on the wire (no client, no network: the request as built, a reply as parsed) ---------------------------------------------

def _openai(model, **options):
    from omakase_bench.models.openai import OpenAIAdapter
    from omakase_bench.models.pricing import price_for
    return OpenAIAdapter(model, price=price_for(f"openai/{model}"), client=object(), reasoning="low", **options)


def _reply(model, usage):
    from types import SimpleNamespace

    from omakase_bench.models.openai import from_wire
    from omakase_bench.models.pricing import price_for
    response = SimpleNamespace(usage=SimpleNamespace(model_dump=lambda: usage), output_text="a: L hold | R hold",
                               model=model, status="completed", output=[])
    return from_wire(response, 0.0, price_for(f"openai/{model}"))


def _ee_request():
    from omakase_bench.models.base import Image, Message, Request, Text
    parts = (Text("PREFIX", cache_prefix=True), Text("rows"), Image(b"png", "image/png", label="Current:"), Text("ask"))
    return Request("SYSTEM", [Message("user", parts)], (), max_calls=0)


def test_gpt_6_2_gets_the_system_prompt_as_the_first_developer_message_and_an_explicit_breakpoint():
    kw = _openai("gpt-6.2-astra")._kwargs(_ee_request())
    assert "instructions" not in kw                       # appended after the input by this generation: never cached
    assert kw["input"][0] == {"role": "developer", "content": [{"type": "input_text", "text": "SYSTEM"}]}
    user = kw["input"][1]
    assert user["role"] == "user" and user["content"][0] == {
        "type": "input_text", "text": "PREFIX", "prompt_cache_breakpoint": {"mode": "explicit"}}
    assert all("prompt_cache_breakpoint" not in block for block in user["content"][1:])
    assert [b["type"] for b in user["content"]] == ["input_text", "input_text", "input_text", "input_image", "input_text"]
    # only the marked prefix is written to the cache; the default lifetime is 30 minutes
    assert kw["extra_body"] == {"prompt_cache_options": {"mode": "explicit", "ttl": "30m"}}
    assert "prompt_cache_retention" not in kw and len(kw["prompt_cache_key"]) == 32
    assert kw["reasoning"] == {"effort": "low"} and kw["store"] is False
    assert _openai("gpt-6.2-astra", cache_ttl="12h")._kwargs(_ee_request())["extra_body"]["prompt_cache_options"]["ttl"] == "12h"
    with pytest.raises(ValueError):
        _openai("gpt-6.2-astra", cache_ttl="1h")


def test_a_request_without_a_marked_prefix_is_left_to_the_implicit_cache():
    from omakase_bench.models.base import Message, Request, Text
    kw = _openai("gpt-6.2-astra")._kwargs(Request("SYSTEM", [Message("user", (Text("TASK"),))], ()))
    assert kw["input"][0]["role"] == "developer" and "extra_body" not in kw and "prompt_cache_key" not in kw
    assert kw["input"][1] == {"role": "user", "content": [{"type": "input_text", "text": "TASK"}]}


def test_earlier_openai_models_keep_their_request():
    kw = _openai("gpt-5.5")._kwargs(_ee_request())
    assert kw["instructions"] == "SYSTEM" and "extra_body" not in kw
    assert kw["input"] == [{"role": "user", "content": [
        {"type": "input_text", "text": "PREFIX"}, {"type": "input_text", "text": "rows"},
        {"type": "input_text", "text": "Current:"},
        {"type": "input_image", "detail": "high", "image_url": "data:image/png;base64,cG5n"},
        {"type": "input_text", "text": "ask"}]}]
    # between the two changes: breakpoints exist, instructions still lead the prompt
    kw = _openai("gpt-5.6")._kwargs(_ee_request())
    assert kw["instructions"] == "SYSTEM" and kw["input"][0]["role"] == "user"
    assert kw["input"][0]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    from omakase_bench.models.openai import generation
    assert [generation(m) for m in ("gpt-6.2-astra", "gpt-10.1", "gpt-5", "o4-mini")] == [(6, 2), (10, 1), (5, 0), (0, 0)]


def test_openai_cache_reads_and_writes_are_costed_once_each_at_their_own_rate():
    usage = {"input_tokens": 10_000, "output_tokens": 1_000,
             "input_tokens_details": {"cached_tokens": 4_000, "cache_write_tokens": 3_000,
                                      "cache_write_tokens_details": {"ttl_30m_tokens": 2_000, "ttl_12h_tokens": 1_000}},
             "output_tokens_details": {"reasoning_tokens": 200}}
    u = _reply("gpt-6.2-astra", usage).usage
    assert (u.input_tokens, u.cached_tokens, u.cache_write_tokens, u.output_tokens, u.reasoning_tokens) == (
        3_000, 4_000, 3_000, 1_000, 200)
    # 3k plain at $8, 4k read at $0.40, 2k written for 30 min at $10, 1k for 12 h at $20, 1k out at $40 per million
    assert u.usd == pytest.approx((3_000 * 8 + 4_000 * 0.40 + 2_000 * 10 + 1_000 * 20 + 1_000 * 40) / 1e6)
    assert _reply("gpt-6.2-astra", {"input_tokens": 915_091, "output_tokens": 5_439}).usage.usd == pytest.approx(
        7.538, abs=0.001)
    from omakase_bench.models.pricing import PRICES
    p = PRICES["openai/gpt-6.2-astra"]
    assert (p.input, p.output, p.cached_input, p.cache_write, p.assumed) == (8.0, 40.0, 0.40, 10.0, False)


# --- readable thinking summaries (opt-in per request) --------------------------------------------------------

def test_a_thinking_summary_is_asked_for_only_when_the_request_says_so():
    import dataclasses

    from omakase_bench.models.anthropic import AnthropicAdapter
    from omakase_bench.models.pricing import price_for
    plain, asked = _ee_request(), dataclasses.replace(_ee_request(), thinking_summary=True)
    claude = AnthropicAdapter("claude-opus-5-5", price=price_for("anthropic/claude-opus-5-5"), effort="low", client=object())
    assert "thinking" not in claude._kwargs(plain)
    kw = claude._kwargs(asked)
    assert kw["thinking"] == {"type": "adaptive", "display": "summarized"} and kw["output_config"] == {"effort": "low"}
    gpt = _openai("gpt-6.2-astra")
    assert gpt._kwargs(plain)["reasoning"] == {"effort": "low"}
    assert gpt._kwargs(asked)["reasoning"] == {"effort": "low", "summary": "auto"}
    assert gpt._kwargs(asked)["include"] == ["reasoning.encrypted_content"]       # the reasoning itself stays state


def test_thinking_summaries_are_read_from_both_providers_replies():
    from types import SimpleNamespace as NS

    from omakase_bench.models import anthropic as claude
    from omakase_bench.models import openai as gpt
    from omakase_bench.models.pricing import price_for

    def block(**kw):
        return NS(model_dump=lambda **options: dict(kw), **kw)

    message = NS(usage=NS(input_tokens=10, output_tokens=5), stop_reason="end_turn", model="claude-opus-5-5",
                 content=[block(type="thinking", thinking="The e-pawn is free. ", signature="sig"),
                          block(type="thinking", thinking="", signature="sig"),
                          block(type="text", text="move: e4")])
    reply = claude.from_wire(message, 1.5, price_for("anthropic/claude-opus-5-5"))
    assert reply.thinking == "The e-pawn is free." and reply.text == "move: e4" and reply.usage.seconds == 1.5
    assert reply.provider_state[0]["signature"] == "sig"
    response = NS(usage=NS(model_dump=lambda: {"input_tokens": 10, "output_tokens": 5}), output_text="move: e5",
                  model="gpt-6.2-astra", status="completed",
                  output=[block(type="reasoning", encrypted_content="opaque",
                                summary=[NS(type="summary_text", text="Mirror the centre."), NS(type="summary_text", text="Then develop.")]),
                          block(type="message")])
    reply = gpt.from_wire(response, 2.0, price_for("openai/gpt-6.2-astra"))
    assert reply.thinking == "Mirror the centre.\nThen develop." and reply.text == "move: e5"
    assert _reply("gpt-6.2-astra", {"input_tokens": 1, "output_tokens": 1}).thinking == ""


# --- provider failures and refusals -------------------------------------------------------------------------------------

def test_after_a_server_side_fallback_only_the_serving_models_blocks_are_kept_and_carried_out():
    """What a model that declined mid-output left before the last fallback block is not sent back, the block
    keeps its place and its wire field name, and only the serving model's tool call runs."""
    from types import SimpleNamespace as NS

    pytest.importorskip("anthropic")
    from anthropic.types.beta import BetaFallbackBlock

    from omakase_bench.models import anthropic as claude
    from omakase_bench.models.pricing import price_for

    def block(**kw):
        return NS(model_dump=lambda **options: dict(kw), **kw)

    switch = BetaFallbackBlock.model_validate({"type": "fallback", "from": {"model": "claude-fable-5-2"},
                                               "to": {"model": "claude-opus-5-5"}, "trigger": {"type": "refusal", "category": "cyber"}})
    assert "from" in switch.model_dump(by_alias=True) and "from_" in switch.model_dump()
    message = NS(usage=NS(input_tokens=10, output_tokens=5), stop_reason="tool_use", model="claude-opus-5-5",
                 content=[block(type="thinking", thinking="", signature="first"),
                          block(type="text", text="Let me look."),
                          block(type="tool_use", id="declined", name="open_gripper", input={"arm": "left"}),
                          switch,
                          block(type="thinking", thinking="", signature="second"),
                          block(type="tool_use", id="served", name="get_state", input={})])
    reply = claude.from_wire(message, 1.0, price_for("anthropic/claude-opus-5-5"))
    assert [c.id for c in reply.calls] == ["served"]
    assert [b["type"] for b in reply.provider_state] == ["text", "fallback", "thinking", "tool_use"]
    assert "from" in reply.provider_state[1] and "from_" not in reply.provider_state[1]
    assert reply.provider_state[2]["signature"] == "second"


def test_an_overloaded_error_inside_a_stream_is_retried(monkeypatch):
    """The stream of a 200 response can end in an error event: the status says nothing, the error type does."""
    anthropic = pytest.importorskip("anthropic")
    import httpx2 as httpx

    from omakase_bench.models import base
    from omakase_bench.models.base import Message, Request, Text
    from omakase_bench.models.anthropic import AnthropicAdapter
    from omakase_bench.models.pricing import price_for

    monkeypatch.setattr(base.time, "sleep", lambda seconds: None)
    request = httpx.Request("POST", "https://example.invalid/v1/messages")

    def failure(status, kind):
        return anthropic.APIStatusError(kind, response=httpx.Response(status, request=request),
                                        body={"type": "error", "error": {"type": kind, "message": kind}})

    class Client:
        def __init__(self, error):
            self.error, self.attempts = error, 0
            self.beta = self.messages = self

        def stream(self, **kwargs):
            self.attempts += 1
            raise self.error

    for error, attempts in ((failure(200, "overloaded_error"), 3), (failure(400, "invalid_request_error"), 1)):
        client = Client(error)
        adapter = AnthropicAdapter("claude-opus-5-5", price=price_for("anthropic/claude-opus-5-5"), client=client)
        reply = adapter.respond(Request("system", [Message("user", (Text("hello"),))]))
        assert reply.stop == "error" and client.attempts == attempts, error


def test_openai_refusals_and_malformed_tool_arguments_are_replies_not_exceptions():
    from types import SimpleNamespace as NS

    from omakase_bench.models.openai import from_wire
    from omakase_bench.models.pricing import price_for

    def response(output, **kw):
        return NS(usage=NS(model_dump=lambda: {"input_tokens": 1, "output_tokens": 1}), output_text="", model="gpt-6.2-astra",
                  status=kw.pop("status", "completed"), output=output, **kw)

    def item(**kw):
        return NS(model_dump=lambda **options: dict(kw), **kw)

    price = price_for("openai/gpt-6.2-astra")
    broken = from_wire(response([item(type="function_call", id="c1", call_id="c1", name="move_ee", arguments='{"x": 0.1,')]), 1.0, price)
    assert broken.stop == "error" and "not JSON" in broken.error and broken.calls == ()
    refused = from_wire(response([item(type="message", content=[NS(type="refusal", refusal="I cannot help with that.")])]), 1.0, price)
    assert refused.stop == "refusal"
    filtered = from_wire(response([], status="incomplete", incomplete_details=NS(reason="content_filter")), 1.0, price)
    assert filtered.stop == "refusal"
    cut = from_wire(response([], status="incomplete", incomplete_details=NS(reason="max_output_tokens")), 1.0, price)
    assert cut.stop == "max_tokens"


def test_names_that_would_run_under_a_misleading_label_are_refused(monkeypatch):
    import importlib.util

    from omakase_bench.subjects import build_subject

    with pytest.raises(ValueError, match="unknown scripted subject 'scripted/gpt'; known: scripted/oracle"):
        build_subject("scripted/gpt")
    with pytest.raises(ValueError, match="unknown mock behaviour 'zzz'; known: finish, loop, press, place, script:<file>"):
        build("mock/zzz")
    with pytest.raises(ValueError, match="a model drives tools, code or ee"):
        build_subject("mock/place", "chunks")
    found = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "anthropic" else found(name, *a))
    with pytest.raises(ValueError, match="the anthropic client library is not installed: uv sync --extra anthropic"):
        build("anthropic/claude-opus-5-5")

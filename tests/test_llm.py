"""Provider translation. No database, no network, no API key.

The fallback client sits between the agent and a foreign wire format, which
makes it the easiest place in the project to lose information silently: a tool
call whose arguments never arrive, an error flag that gets dropped, a tool
result that lands next to the wrong call. Each of those produces a run that
looks fine and measures the wrong thing, so they are asserted directly.
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.agent import TOOL_SCHEMAS
from app.llm import (
    PROVIDERS,
    LLMError,
    OpenAICompatClient,
    ProviderSpec,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    _parse_arguments,
    from_openai_response,
    to_openai_messages,
    to_openai_tools,
)

FAKE = ProviderSpec(
    name="fake",
    base_url="https://fake.invalid/v1",
    default_model="fake-model",
    api_key_env=("FAKE_KEY",),
)


def client_returning(*payloads: dict, status: int = 200) -> tuple[OpenAICompatClient, list[dict]]:
    """A client wired to a canned transport, plus the list of bodies it sent."""
    sent: list[dict] = []
    queue = list(payloads)

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        body = queue.pop(0) if queue else payloads[-1]
        return httpx.Response(status, json=body)

    return (
        OpenAICompatClient(
            FAKE, "key", transport=httpx.MockTransport(handler), max_retries=1
        ),
        sent,
    )


class TestToolTranslation:
    def test_every_agent_tool_survives_translation(self) -> None:
        translated = to_openai_tools(TOOL_SCHEMAS)
        assert len(translated) == len(TOOL_SCHEMAS)
        assert {t["function"]["name"] for t in translated} == {
            t["name"] for t in TOOL_SCHEMAS
        }

    def test_input_schema_becomes_parameters_unchanged(self) -> None:
        """The schema is what constrains the model's arguments; it must not be rewritten."""
        source = next(t for t in TOOL_SCHEMAS if t["name"] == "test_hypothesis")
        translated = next(
            t for t in to_openai_tools(TOOL_SCHEMAS) if t["function"]["name"] == "test_hypothesis"
        )
        assert translated["function"]["parameters"] == source["input_schema"]
        assert "index_ddls" in translated["function"]["parameters"]["properties"]


class TestMessageTranslation:
    def test_system_prompt_leads(self) -> None:
        out = to_openai_messages("be terse", [{"role": "user", "content": "hi"}])
        assert out[0] == {"role": "system", "content": "be terse"}
        assert out[1] == {"role": "user", "content": "hi"}

    def test_assistant_tool_use_becomes_tool_calls(self) -> None:
        out = to_openai_messages(
            None,
            [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": [
                        TextBlock(text="checking the plan"),
                        ToolUseBlock(id="tu_1", name="explain_query", input={"sql": "SELECT 1"}),
                    ],
                },
            ],
        )
        assistant = out[-1]
        assert assistant["role"] == "assistant"
        assert assistant["content"] == "checking the plan"
        assert assistant["tool_calls"][0]["id"] == "tu_1"
        assert assistant["tool_calls"][0]["function"]["name"] == "explain_query"
        assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {
            "sql": "SELECT 1"
        }

    def test_provider_fields_on_a_tool_call_are_echoed_back(self) -> None:
        """Gemini 3.x 400s on the next turn without its thought_signature.

        Reconstructing a tool call from (id, name, input) alone is lossy in a
        way nothing catches until a real multi-turn run: the first request
        succeeds, and the second is rejected outright.
        """
        signature = {"google": {"thought_signature": "EroBCrcBARFNMg9t"}}
        out = to_openai_messages(
            None,
            [
                {
                    "role": "assistant",
                    "content": [
                        ToolUseBlock(
                            id="tu_1",
                            name="explain_query",
                            input={"sql": "SELECT 1"},
                            extra={"extra_content": signature},
                        )
                    ],
                }
            ],
        )
        assert out[0]["tool_calls"][0]["extra_content"] == signature
        # The reconstructed fields survive alongside it.
        assert out[0]["tool_calls"][0]["function"]["name"] == "explain_query"
        assert out[0]["tool_calls"][0]["id"] == "tu_1"

    def test_a_tool_call_round_trips_through_both_translators(self) -> None:
        """The path an actual loop takes: response -> history -> next request."""
        response = from_openai_response(
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "QM51W29P",
                                    "type": "function",
                                    "extra_content": {"google": {"thought_signature": "abc"}},
                                    "function": {
                                        "name": "benchmark",
                                        "arguments": '{"sql":"SELECT 1"}',
                                    },
                                }
                            ]
                        },
                    }
                ]
            }
        )
        replayed = to_openai_messages(
            None, [{"role": "assistant", "content": response.content}]
        )
        assert replayed[0]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "abc"

    def test_thinking_blocks_are_not_replayed_as_assistant_text(self) -> None:
        """Private reasoning must not come back looking like a stated claim."""
        out = to_openai_messages(
            None,
            [
                {
                    "role": "assistant",
                    "content": [
                        ThinkingBlock(thinking="maybe the sku index is useless"),
                        TextBlock(text="testing it"),
                    ],
                }
            ],
        )
        assert out[0]["content"] == "testing it"
        assert "useless" not in json.dumps(out)

    def test_each_tool_result_becomes_its_own_tool_message(self) -> None:
        """Anthropic batches results in one user turn; OpenAI wants one each."""
        out = to_openai_messages(
            None,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tu_1", "content": '{"a":1}'},
                        {"type": "tool_result", "tool_use_id": "tu_2", "content": '{"b":2}'},
                    ],
                }
            ],
        )
        assert [m["role"] for m in out] == ["tool", "tool"]
        assert [m["tool_call_id"] for m in out] == ["tu_1", "tu_2"]

    def test_tool_results_stay_adjacent_to_their_assistant_turn(self) -> None:
        """Order is the only thing binding a result to its call in this format."""
        out = to_openai_messages(
            None,
            [
                {
                    "role": "assistant",
                    "content": [ToolUseBlock(id="tu_1", name="benchmark", input={})],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tu_1", "content": "{}"}
                    ],
                },
            ],
        )
        assert [m["role"] for m in out] == ["assistant", "tool"]
        assert out[1]["tool_call_id"] == out[0]["tool_calls"][0]["id"]

    def test_error_flag_survives_as_visible_text(self) -> None:
        """A dropped is_error is a rejection the model never learns from."""
        out = to_openai_messages(
            None,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_1",
                            "content": '{"message": "column does not exist"}',
                            "is_error": True,
                        }
                    ],
                }
            ],
        )
        assert out[0]["content"].startswith("TOOL ERROR: ")
        assert "column does not exist" in out[0]["content"]

    def test_observation_bodies_are_passed_through_verbatim(self) -> None:
        """The measured numbers are the observation. Truncating them breaks the loop."""
        observation = json.dumps(
            {"verdict": "REJECTED", "improvement_pct": -3.2, "checksum_match": True}
        )
        out = to_openai_messages(
            None,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t", "content": observation}
                    ],
                }
            ],
        )
        assert out[0]["content"] == observation


class TestResponseTranslation:
    def test_tool_calls_become_tool_use_blocks(self) -> None:
        response = from_openai_response(
            {
                "model": "fake-model",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_0",
                                    "function": {
                                        "name": "test_hypothesis",
                                        "arguments": '{"hypothesis_id":"h1","kind":"index"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            }
        )
        assert response.stop_reason == "tool_use"
        block = response.content[0]
        assert block.type == "tool_use"
        assert block.name == "test_hypothesis"
        assert block.input == {"hypothesis_id": "h1", "kind": "index"}

    def test_finish_reason_maps_to_anthropic_stop_reason(self) -> None:
        def stop_reason_for(reason: str) -> str | None:
            return from_openai_response(
                {"choices": [{"finish_reason": reason, "message": {"content": "x"}}]}
            ).stop_reason

        assert stop_reason_for("stop") == "end_turn"
        assert stop_reason_for("length") == "max_tokens"

    def test_reasoning_is_captured_when_the_provider_offers_it(self) -> None:
        response = from_openai_response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"reasoning_content": "seq scan on order_items", "content": "ok"},
                    }
                ]
            }
        )
        assert [b.type for b in response.content] == ["thinking", "text"]

    def test_cached_tokens_are_not_double_counted(self) -> None:
        """Anthropic's input_tokens excludes cache reads; the ledger assumes that."""
        response = from_openai_response(
            {
                "choices": [{"finish_reason": "stop", "message": {"content": "x"}}],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 50,
                    "prompt_tokens_details": {"cached_tokens": 800},
                },
            }
        )
        assert response.usage.input_tokens == 200
        assert response.usage.cache_read_input_tokens == 800
        total = response.usage.input_tokens + response.usage.cache_read_input_tokens
        assert total == 1000

    def test_missing_usage_is_zero_not_a_crash(self) -> None:
        """Several free providers omit usage entirely; the budget still has to work."""
        response = from_openai_response(
            {"choices": [{"finish_reason": "stop", "message": {"content": "x"}}]}
        )
        assert response.usage.input_tokens == 0
        assert response.usage.output_tokens == 0

    def test_empty_choices_raises(self) -> None:
        with pytest.raises(LLMError, match="no choices"):
            from_openai_response({"choices": []})


class TestMalformedArguments:
    def test_broken_json_does_not_take_the_run_down(self) -> None:
        parsed = _parse_arguments('{"sql": "SELECT 1"')
        assert "_malformed_arguments" in parsed

    def test_a_json_scalar_is_not_accepted_as_arguments(self) -> None:
        assert "_malformed_arguments" in _parse_arguments('"just a string"')

    def test_empty_arguments_are_an_empty_dict(self) -> None:
        assert _parse_arguments("") == {}
        assert _parse_arguments(None) == {}


class TestClientContract:
    def test_create_returns_an_anthropic_shaped_response(self) -> None:
        client, sent = client_returning(
            {
                "model": "fake-model",
                "choices": [{"finish_reason": "stop", "message": {"content": "done"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            }
        )
        response = client.messages.create(
            model="fake-model",
            max_tokens=100,
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
            tools=TOOL_SCHEMAS,
            # Anthropic-only knobs the agent always passes.
            cache_control={"type": "ephemeral"},
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
        )
        assert response.content[0].text == "done"
        assert sent[0]["model"] == "fake-model"
        assert len(sent[0]["tools"]) == len(TOOL_SCHEMAS)
        # The Anthropic-only fields must not be forwarded to a foreign API.
        assert "cache_control" not in sent[0]
        assert "thinking" not in sent[0]
        assert "output_config" not in sent[0]
        client.close()

    def test_a_missing_key_fails_at_construction_not_mid_run(self) -> None:
        with pytest.raises(LLMError, match="FAKE_KEY"):
            OpenAICompatClient(FAKE, None)

    def test_a_non_retryable_error_names_the_provider(self) -> None:
        client, _ = client_returning({"error": "bad model"}, status=400)
        with pytest.raises(LLMError, match="fake request failed"):
            client.messages.create(
                model="nope", max_tokens=10, messages=[{"role": "user", "content": "x"}]
            )
        client.close()


class TestProviderRegistry:
    def test_every_provider_declares_a_usable_default(self) -> None:
        for name, spec in PROVIDERS.items():
            assert spec.default_model, name
            assert spec.base_url or name == "anthropic", name
            assert spec.api_key_env or not spec.requires_key, name

    def test_ollama_needs_no_key(self) -> None:
        assert PROVIDERS["ollama"].requires_key is False

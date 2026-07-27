"""Model clients. Anthropic is the reference; OpenAI-compatible providers are a fallback.

`OptimizerAgent` depends on the `AnthropicClient` Protocol, not on the Anthropic
SDK, so any object exposing `.messages.create(...)` and returning Anthropic-shaped
content blocks can drive the loop. This module supplies two such objects:

  * the real `anthropic.Anthropic` client, used unchanged, and
  * `OpenAICompatClient`, which translates to and from the OpenAI
    chat-completions shape that Gemini, Groq, Cerebras, OpenRouter, GitHub
    Models and Ollama all speak.

Nothing about the verifier changes with the provider. The whole point of the
project is that the model's output is measured rather than trusted, so a weaker
free model produces a worse *hit rate* -- more rejected hypotheses, fewer
accepted ones -- but never a wrong number in the report. That is the property
worth having when the model is swappable.

What is genuinely lost on the fallback path, and is reported as such:

  * **Prompt caching.** Anthropic's explicit `cache_control` has no portable
    equivalent. Some providers cache implicitly and report `cached_tokens`;
    that is passed through when present, and is zero otherwise.
  * **Thinking blocks.** Reasoning text is exposed inconsistently
    (`reasoning_content`, `reasoning`, or not at all). It is captured when
    offered, and the Phase 6 timeline simply has less to render when it is not.

Model IDs drift faster than anything else here. The per-provider defaults below
are starting points; override with OPTIQUERY_MODEL and check the provider's own
model list if a request 404s.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import httpx

DEFAULT_TIMEOUT_S = 180.0
DEFAULT_MAX_RETRIES = 4


class LLMError(RuntimeError):
    """A provider call failed in a way retrying will not fix."""


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderSpec:
    """Everything that differs between one OpenAI-compatible host and another."""

    name: str
    base_url: str
    default_model: str
    api_key_env: tuple[str, ...]
    requires_key: bool = True
    notes: str = ""
    extra_headers: dict[str, str] = field(default_factory=dict)


PROVIDERS: dict[str, ProviderSpec] = {
    "anthropic": ProviderSpec(
        name="anthropic",
        base_url="",  # native SDK, not HTTP-translated
        default_model="claude-sonnet-4-6",
        api_key_env=("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
        notes="Reference implementation: prompt caching and thinking blocks both work.",
    ),
    "gemini": ProviderSpec(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        default_model="gemini-2.5-flash",
        api_key_env=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        notes="Free tier at aistudio.google.com/apikey. Long context, strong tool calling.",
    ),
    "groq": ProviderSpec(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
        api_key_env=("GROQ_API_KEY",),
        notes="Free tier at console.groq.com. Fast; context is the binding constraint.",
    ),
    "cerebras": ProviderSpec(
        name="cerebras",
        base_url="https://api.cerebras.ai/v1",
        default_model="llama-3.3-70b",
        api_key_env=("CEREBRAS_API_KEY",),
        notes="Free tier at cloud.cerebras.ai.",
    ),
    "openrouter": ProviderSpec(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        default_model="meta-llama/llama-3.3-70b-instruct:free",
        api_key_env=("OPENROUTER_API_KEY",),
        notes="Free models carry a ':free' suffix; not all of them support tool calling.",
        extra_headers={"X-Title": "OptiQuery"},
    ),
    "github": ProviderSpec(
        name="github",
        base_url="https://models.github.ai/inference",
        default_model="openai/gpt-4o",
        api_key_env=("GITHUB_TOKEN", "GITHUB_MODELS_TOKEN"),
        notes="Free with a GitHub account; a fine-grained PAT with the models scope.",
    ),
    "ollama": ProviderSpec(
        name="ollama",
        base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/v1",
        default_model="qwen3:8b",
        api_key_env=(),
        requires_key=False,
        notes="Fully local, no key, no rate limit. Pick a model whose card lists tool support.",
    ),
}

#: Order used when OPTIQUERY_PROVIDER is unset: whichever key exists wins.
AUTODETECT_ORDER: tuple[str, ...] = ("anthropic", "gemini", "groq", "cerebras", "openrouter", "github")


def resolve_api_key(spec: ProviderSpec) -> str | None:
    for name in spec.api_key_env:
        value = os.environ.get(name)
        if value:
            return value
    return None


def available_providers() -> list[str]:
    """Providers whose credentials are actually present in this environment."""
    return [
        key
        for key, spec in PROVIDERS.items()
        if not spec.requires_key or resolve_api_key(spec) is not None
    ]


# ---------------------------------------------------------------------------
# Anthropic-shaped response objects
# ---------------------------------------------------------------------------


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ThinkingBlock:
    thinking: str
    type: str = "thinking"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class Response:
    """The subset of an Anthropic Message that `OptimizerAgent` reads."""

    content: list[Any]
    stop_reason: str | None
    usage: Usage
    model: str
    raw: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Block access that works on dicts and on SDK objects alike
# ---------------------------------------------------------------------------


def _btype(block: Any) -> str:
    if isinstance(block, dict):
        return str(block.get("type", ""))
    return str(getattr(block, "type", "") or "")


def _bget(block: Any, key: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


# ---------------------------------------------------------------------------
# Anthropic -> OpenAI
# ---------------------------------------------------------------------------


def to_openai_tools(tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for tool in tools
    ]


def to_openai_messages(
    system: str | None, messages: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Flatten Anthropic message blocks into the OpenAI message sequence.

    Two shape differences matter and both are load-bearing:

      * Anthropic puts tool calls inside the assistant's content list; OpenAI
        puts them in a sibling `tool_calls` field.
      * Anthropic returns every tool result in one user message; OpenAI wants
        one `role: "tool"` message per call, each immediately following the
        assistant turn that requested it. Emitting them in call order preserves
        that adjacency.
    """
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})

    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content")

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if not isinstance(content, Iterable):
            continue

        blocks = list(content)

        if role == "assistant":
            texts = [
                str(_bget(b, "text", "")) for b in blocks if _btype(b) == "text"
            ]
            tool_calls = [
                {
                    "id": str(_bget(b, "id", "")),
                    "type": "function",
                    "function": {
                        "name": str(_bget(b, "name", "")),
                        "arguments": json.dumps(_bget(b, "input", {}) or {}, default=str),
                    },
                }
                for b in blocks
                if _btype(b) == "tool_use"
            ]
            # Thinking blocks are dropped: no portable representation, and
            # replaying them as assistant text would let one turn's private
            # reasoning be mistaken for a claim in the next.
            assistant: dict[str, Any] = {"role": "assistant"}
            joined = "\n".join(t for t in texts if t).strip()
            assistant["content"] = joined or None
            if tool_calls:
                assistant["tool_calls"] = tool_calls
            out.append(assistant)
            continue

        pending_text: list[str] = []
        for block in blocks:
            kind = _btype(block)
            if kind == "tool_result":
                body = _bget(block, "content", "")
                if not isinstance(body, str):
                    body = json.dumps(body, default=str)
                if _bget(block, "is_error", False):
                    # OpenAI's tool role has no error flag. Marking it inline
                    # keeps the failure visible to the model, which is the
                    # entire reason errors are returned as observations.
                    body = "TOOL ERROR: " + body
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(_bget(block, "tool_use_id", "")),
                        "content": body,
                    }
                )
            elif kind == "text":
                pending_text.append(str(_bget(block, "text", "")))

        if pending_text:
            out.append({"role": "user", "content": "\n".join(pending_text)})

    return out


# ---------------------------------------------------------------------------
# OpenAI -> Anthropic
# ---------------------------------------------------------------------------

_FINISH_REASONS = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}


def _parse_arguments(raw: Any) -> dict[str, Any]:
    """Tool arguments, or a visibly broken payload.

    A model that emits malformed JSON must not take the run down. The raw text
    is passed through under a key no tool accepts, so the registry rejects it
    and the rejection re-enters context as an observation the model can act on.
    """
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"_malformed_arguments": str(raw)}
    return parsed if isinstance(parsed, dict) else {"_malformed_arguments": str(raw)}


def from_openai_response(payload: dict[str, Any]) -> Response:
    choices = payload.get("choices") or []
    if not choices:
        raise LLMError(f"provider returned no choices: {json.dumps(payload)[:500]}")

    choice = choices[0]
    message = choice.get("message") or {}
    blocks: list[Any] = []

    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        blocks.append(ThinkingBlock(thinking=reasoning))

    text = message.get("content")
    if isinstance(text, list):  # some providers return content parts
        text = "".join(
            part.get("text", "") for part in text if isinstance(part, dict)
        )
    if isinstance(text, str) and text.strip():
        blocks.append(TextBlock(text=text))

    tool_calls = message.get("tool_calls") or []
    for index, call in enumerate(tool_calls):
        function = call.get("function") or {}
        blocks.append(
            ToolUseBlock(
                id=str(call.get("id") or f"call_{index}"),
                name=str(function.get("name") or ""),
                input=_parse_arguments(function.get("arguments")),
            )
        )

    raw_usage = payload.get("usage") or {}
    details = raw_usage.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    prompt_tokens = int(raw_usage.get("prompt_tokens") or 0)
    usage = Usage(
        # Anthropic reports input_tokens as the uncached remainder; subtracting
        # keeps the agent's token ledger comparable across providers.
        input_tokens=max(prompt_tokens - cached, 0),
        output_tokens=int(raw_usage.get("completion_tokens") or 0),
        cache_read_input_tokens=cached,
        cache_creation_input_tokens=0,
    )

    stop_reason = "tool_use" if tool_calls else _FINISH_REASONS.get(
        str(choice.get("finish_reason") or ""), "end_turn"
    )
    return Response(
        content=blocks,
        stop_reason=stop_reason,
        usage=usage,
        model=str(payload.get("model") or ""),
        raw=payload,
    )


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------


class _Messages:
    def __init__(self, client: "OpenAICompatClient") -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: Sequence[dict[str, Any]],
        system: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        # Anthropic-only knobs. Accepted and ignored so the agent needs no
        # per-provider branching; what they cost is documented at module level.
        cache_control: Any = None,
        thinking: Any = None,
        output_config: Any = None,
        **_ignored: Any,
    ) -> Response:
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": to_openai_messages(system, messages),
        }
        if tools:
            body["tools"] = to_openai_tools(tools)
            body["tool_choice"] = "auto"
        if self._client.temperature is not None:
            body["temperature"] = self._client.temperature

        payload = self._client.post("/chat/completions", body)
        return from_openai_response(payload)


class OpenAICompatClient:
    """Satisfies `AnthropicClient` by translating to chat-completions.

    Retries are not a nicety here. Free tiers rate-limit aggressively, and a 429
    twelve iterations into a run would discard every hypothesis measured so far,
    so 429s and 5xx are retried with backoff that honours `Retry-After`.
    """

    def __init__(
        self,
        spec: ProviderSpec,
        api_key: str | None = None,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        temperature: float | None = 0.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if spec.requires_key and not api_key:
            raise LLMError(
                f"provider '{spec.name}' needs a key in one of: {', '.join(spec.api_key_env)}"
            )
        self.spec = spec
        self.api_key = api_key
        self.max_retries = max_retries
        self.temperature = temperature
        headers = {"Content-Type": "application/json", **spec.extra_headers}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx.Client(
            base_url=spec.base_url,
            headers=headers,
            timeout=timeout_s,
            transport=transport,
        )

    @property
    def messages(self) -> _Messages:
        return _Messages(self)

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        last_error = ""
        for attempt in range(self.max_retries + 1):
            response = self._http.post(path, json=body)
            if response.status_code < 400:
                try:
                    return response.json()
                except ValueError as exc:
                    raise LLMError(f"{self.spec.name} returned non-JSON: {exc}") from exc

            last_error = f"HTTP {response.status_code}: {response.text[:800]}"
            retryable = response.status_code == 429 or response.status_code >= 500
            if not retryable or attempt == self.max_retries:
                raise LLMError(f"{self.spec.name} request failed -- {last_error}")

            retry_after = response.headers.get("retry-after")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2.0**attempt
            time.sleep(min(delay, 60.0))

        raise LLMError(f"{self.spec.name} request failed -- {last_error}")

    def close(self) -> None:
        self._http.close()


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClientBundle:
    """A client plus the model name the agent should ask for."""

    client: Any
    provider: str
    model: str

    def describe(self) -> str:
        return f"{self.provider}:{self.model}"


def build_client(
    provider: str | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> ClientBundle:
    """Pick a provider from the environment and return a ready client.

    Precedence: explicit argument, then OPTIQUERY_PROVIDER, then the first
    provider in AUTODETECT_ORDER whose key is present. Anthropic leads that
    order because it is the only path with prompt caching and thinking blocks.
    """
    name = (provider or os.environ.get("OPTIQUERY_PROVIDER") or "").strip().lower()
    if not name:
        found = [p for p in AUTODETECT_ORDER if p in available_providers()]
        if not found:
            raise LLMError(
                "no model provider configured. Set one of ANTHROPIC_API_KEY, "
                "GEMINI_API_KEY, GROQ_API_KEY, CEREBRAS_API_KEY, OPENROUTER_API_KEY "
                "or GITHUB_TOKEN, or set OPTIQUERY_PROVIDER=ollama to run locally."
            )
        name = found[0]

    if name not in PROVIDERS:
        raise LLMError(f"unknown provider '{name}'. Known: {', '.join(sorted(PROVIDERS))}")

    spec = PROVIDERS[name]
    chosen_model = model or os.environ.get("OPTIQUERY_MODEL") or spec.default_model

    if name == "anthropic":
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - declared in requirements
            raise LLMError("the anthropic package is not installed") from exc
        key = resolve_api_key(spec)
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        return ClientBundle(client=Anthropic(api_key=key), provider=name, model=chosen_model)

    return ClientBundle(
        client=OpenAICompatClient(spec, resolve_api_key(spec), **kwargs),
        provider=name,
        model=chosen_model,
    )

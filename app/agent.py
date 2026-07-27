"""The ReAct loop: an LLM proposes, the verifier decides, the trace records both.

The single most important property of this file is that **every observation goes
back into the message history in full** -- measured runtimes, rejection reasons,
the server's own error text. A tool result that does not re-enter context is a
result the model cannot learn from, and a model that cannot see why its last
index was rejected proposes the same index again until the budget is gone.
`HypothesisLedger` makes that memory explicit rather than hoping attention does
the work: every observation carries a compact record of everything already tried
and how it scored.

Two structural guarantees sit outside the model's control:

  1. **Nothing is recommended that was not measured.** `finish` is cross-checked
     against the set of hypotheses the verifier actually accepted. A
     recommendation the model asserts but never tested is dropped from the
     report and recorded under `unverified_claims`.

  2. **Every limit degrades to a partial report.** Hitting the iteration cap or
     the token budget stops the loop and keeps everything verified up to that
     point. A run that ends early still produces a real report, marked partial,
     with the reason recorded.

Model: claude-sonnet-4-6 with adaptive thinking. Thinking blocks are echoed back
unmodified as part of `response.content`, which the API requires and which also
gives the Phase 6 timeline something honest to render.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, Sequence

from app.db import SqlGuardError
from app.shadow import ShadowIsolationError
from app.tools import ToolContext, ToolError, ToolRegistry
from app.verifier import (
    MIN_IMPROVEMENT_PCT,
    OVERSIZED_INDEX_PCT,
    BenchmarkError,
    Hypothesis,
    HypothesisResult,
    Verdict,
    Verifier,
)

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_ITERATIONS = 12
DEFAULT_MAX_TOKENS = 16_000
DEFAULT_TOKEN_BUDGET = 400_000
DEFAULT_EFFORT = "high"


class StopReason(str, Enum):
    """Why the loop ended."""

    FINISHED = "finished"
    ITERATION_LIMIT = "iteration_limit"
    TOKEN_BUDGET = "token_budget"
    MODEL_STOPPED = "model_stopped_without_finishing"
    ERROR = "error"


class AnthropicClient(Protocol):
    """The slice of the Anthropic SDK this module uses.

    Declared as a Protocol so tests can drive the loop with a scripted client
    and assert on its control flow -- iteration caps, budget exhaustion, the
    deduplication ledger -- without an API key or a network call.
    """

    @property
    def messages(self) -> Any: ...


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

# Ordering is fixed and the descriptions are static: tools render at the very
# front of the prompt, so any change here invalidates the whole prompt cache.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_schema",
        "description": (
            "Reconstructed DDL, column types, estimated row counts, table sizes, and "
            "per-column planner statistics (n_distinct, correlation, null_frac, "
            "avg_width) from pg_stats. n_distinct tells you whether an index on a "
            "column would be selective; correlation tells you whether scanning that "
            "index would be sequential or random I/O. Neither is derivable from the "
            "DDL. Omit `table` to get every table."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {"type": "string", "description": "Restrict to one table."},
                "database": {
                    "type": "string",
                    "enum": ["primary", "shadow"],
                    "description": "Defaults to primary (read-only production).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "list_indexes",
        "description": (
            "Existing indexes with their definitions, sizes, and usage counters from "
            "pg_stat_user_indexes. Check this before proposing an index: an index "
            "that already exists is not a fix, and idx_scan shows which existing "
            "indexes are already earning their keep."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {"type": "string", "description": "Restrict to one table."},
                "database": {
                    "type": "string",
                    "enum": ["primary", "shadow"],
                    "description": "Defaults to primary.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "explain_query",
        "description": (
            "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) on the shadow database, plus a "
            "distilled summary naming every sequential scan and how many rows it read "
            "and discarded. Use this to find WHERE the time goes and WHICH plan "
            "Postgres chose. Do NOT use its timings to judge whether a change helped: "
            "per-tuple instrumentation inflates them several-fold. That is what "
            "benchmark and test_hypothesis are for."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "A single SELECT/WITH/TABLE/VALUES statement."},
                "analyze": {
                    "type": "boolean",
                    "description": "True actually executes the query. False gives estimates only.",
                },
            },
            "required": ["sql"],
        },
    },
    {
        "name": "benchmark",
        "description": (
            "Time a query on shadow: one discarded warm-up run, then the median of "
            "`runs` timed runs, plus a row count and a sha256 checksum of the result "
            "set. Whatever indexes currently exist on shadow are in effect. This is "
            "an exploration tool -- it does not judge anything and does not reset "
            "shadow. Use test_hypothesis for anything that might become a "
            "recommendation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string"},
                "runs": {"type": "integer", "description": "Timed runs after the warm-up. Default 5."},
            },
            "required": ["sql"],
        },
    },
    {
        "name": "create_index_on_shadow",
        "description": (
            "Build an index on the shadow database and report its build time and "
            "size. The index name is required (shadow is reset by name). "
            "CONCURRENTLY and IF NOT EXISTS are rejected. This is for exploration -- "
            "e.g. building an index and then calling explain_query to see whether the "
            "planner would use it. Indexes built this way are NOT automatically "
            "cleaned up; test_hypothesis clears them before it measures anything."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ddl": {
                    "type": "string",
                    "description": "CREATE [UNIQUE] INDEX <name> ON <table> [USING <method>] (<cols>) [WHERE ...]",
                }
            },
            "required": ["ddl"],
        },
    },
    {
        "name": "drop_index_on_shadow",
        "description": (
            "Drop an index from the shadow database by name. Baseline indexes "
            "(primary keys) are refused."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "test_hypothesis",
        "description": (
            "Apply a proposed fix on shadow, benchmark it against the original, rule "
            "on it, and reset shadow to its baseline. THIS IS THE ONLY TOOL WHOSE "
            "RESULTS CAN BECOME RECOMMENDATIONS -- anything you assert without "
            "testing it here is dropped from the final report.\n\n"
            "The verdict is not negotiable. A hypothesis is ACCEPTED only if the "
            "result checksum is byte-identical to the original AND the median runtime "
            f"improves by at least {MIN_IMPROVEMENT_PCT:.0f}%. A rewrite that is 1000x "
            "faster and returns different rows is REJECTED, and the reason comes back "
            "to you with both checksums and both row counts.\n\n"
            "Every hypothesis starts from a clean shadow and ends with one, so results "
            "are attributable to exactly the change you proposed. Testing the same "
            "hypothesis twice returns the first result without re-running it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "hypothesis_id": {
                    "type": "string",
                    "description": "Short unique id for this hypothesis, e.g. 'h1'.",
                },
                "kind": {
                    "type": "string",
                    "enum": ["index", "rewrite", "index+rewrite"],
                },
                "summary": {
                    "type": "string",
                    "description": "One sentence: what this change is and why you expect it to help.",
                },
                "index_ddls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Indexes to build before measuring. Required for kind 'index' "
                        "and 'index+rewrite'. All of them are applied together and "
                        "judged as one unit."
                    ),
                },
                "rewritten_sql": {
                    "type": "string",
                    "description": (
                        "The rewritten query to measure instead of the original. "
                        "Required for kind 'rewrite' and 'index+rewrite'. It must "
                        "return exactly the same rows."
                    ),
                },
            },
            "required": ["hypothesis_id", "kind", "summary"],
        },
    },
    {
        "name": "finish",
        "description": (
            "End the investigation and return your final recommendations. Every "
            "recommendation must correspond to a hypothesis that test_hypothesis "
            "ACCEPTED; unbacked ones are dropped and recorded as unverified. An empty "
            "recommendation list is a legitimate answer -- 'this query is already "
            "close to optimal' is a real finding."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "diagnosis": {
                    "type": "string",
                    "description": "What was actually slow and why, in the planner's own numbers.",
                },
                "recommendations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string", "enum": ["index", "rewrite"]},
                            "summary": {"type": "string"},
                            "ddl": {"type": "string", "description": "Required when kind is 'index'."},
                            "rewritten_sql": {
                                "type": "string",
                                "description": "Required when kind is 'rewrite'.",
                            },
                            "hypothesis_id": {
                                "type": "string",
                                "description": "The accepted hypothesis this comes from.",
                            },
                        },
                        "required": ["kind", "summary"],
                    },
                },
            },
            "required": ["diagnosis", "recommendations"],
        },
    },
]


SYSTEM_PROMPT = f"""\
You are OptiQuery, a Postgres query optimisation agent. You are given one slow \
query. Your job is to find out why it is slow and to prove which fixes actually \
help.

# The setup

There are two databases with identical data and identical settings.

- `primary` is production. You can read its schema and its index list. You \
cannot write to it, and no tool you have is capable of writing to it.
- `shadow` is a disposable clone. Every index you build and every query you \
benchmark happens there.

Both run with max_parallel_workers_per_gather = 0, so all timings are \
single-threaded and comparable.

# How you are judged

A deterministic verifier decides what ships, not you. `test_hypothesis` accepts \
a change only when BOTH hold:

1. The result checksum matches the original exactly. Same rows, same values. A \
   rewrite that returns 0 rows instead of 15 is rejected no matter how fast it is.
2. The median runtime improves by at least {MIN_IMPROVEMENT_PCT:.0f}%.

An index larger than {OVERSIZED_INDEX_PCT:.0f}% of its table is flagged in the \
report but not rejected.

You cannot argue with a rejection. You can only propose something else.

# How to work

1. Read the plan first. `explain_query` with analyze=true tells you which node \
   owns the time and how many rows each sequential scan read to produce how few. \
   Do not guess from the SQL text.
2. Get the statistics. `get_schema` gives you n_distinct and correlation per \
   column. A column with 50,000 distinct values over 3,000,000 rows is worth \
   indexing; a column with 5 is usually not. High correlation means an index \
   scan on that column is cheap.
3. Check what already exists with `list_indexes` before proposing anything.
4. Propose one hypothesis at a time through `test_hypothesis`, and read the \
   result before proposing the next. Each result tells you the before and after \
   medians, whether the checksum matched, and exactly why it was rejected if it \
   was.
5. Call `finish` when you are done.

# Rules

- Never propose the same index DDL or the same rewrite twice. Every \
  test_hypothesis result includes a ledger of everything you have already tried \
  and how it scored. Read it. Repeating a tested hypothesis wastes an iteration \
  and returns the same answer.
- If a rewrite is rejected on checksum, the rewrite is wrong. Work out which \
  rows differ -- the observation gives you both row counts -- rather than \
  retrying variants blindly.
- Group indexes that only work together into a single hypothesis. Two indexes \
  that each do nothing alone but transform the plan together should be tested as \
  one unit.
- Only recommend what was accepted. Asserting an untested fix in `finish` gets \
  it dropped and recorded as an unverified claim, which looks worse than a short \
  list.
- You have a hard limit of {DEFAULT_MAX_ITERATIONS} iterations. Spend the early \
  ones on diagnosis and the rest on testing. If you are running out, call \
  `finish` with what you have proven rather than being cut off mid-hypothesis.
"""


# ---------------------------------------------------------------------------
# Deduplication ledger
# ---------------------------------------------------------------------------

_WHITESPACE = re.compile(r"\s+")


def normalize_sql(sql: str) -> str:
    """Fingerprint for duplicate detection.

    Case and whitespace only. Deliberately not a parser: two DDLs that differ
    in column order really are different indexes, and collapsing them would hide
    a legitimate second hypothesis behind the first one's verdict.
    """
    return _WHITESPACE.sub(" ", sql.strip().rstrip(";")).lower()


@dataclass
class LedgerEntry:
    hypothesis_id: str
    fingerprint: tuple[str, ...]
    summary: str
    verdict: str
    improvement_pct: float
    checksum_match: bool
    reasons: list[str]

    def one_line(self) -> str:
        detail = (
            f"{self.improvement_pct:+.1f}%, checksum "
            f"{'match' if self.checksum_match else 'DIFFERS'}"
        )
        if self.reasons:
            detail += f" -- {self.reasons[0]}"
        return f"[{self.verdict}] {self.hypothesis_id}: {self.summary} ({detail})"


@dataclass
class HypothesisLedger:
    """Everything tried so far, replayed into every observation.

    Attention over a long transcript is not a reliable memory of what has
    already failed. This is: the ledger is rendered into every test_hypothesis
    result, so the most recent thing in context always contains the full list of
    tested hypotheses and their verdicts.
    """

    entries: list[LedgerEntry] = field(default_factory=list)

    def fingerprint_of(self, hypothesis: Hypothesis) -> tuple[str, ...]:
        parts = [normalize_sql(ddl) for ddl in hypothesis.index_ddls]
        if hypothesis.rewritten_sql:
            parts.append("REWRITE:" + normalize_sql(hypothesis.rewritten_sql))
        return tuple(sorted(parts))

    def find(self, fingerprint: tuple[str, ...]) -> LedgerEntry | None:
        return next((e for e in self.entries if e.fingerprint == fingerprint), None)

    def record(self, hypothesis: Hypothesis, result: HypothesisResult) -> LedgerEntry:
        entry = LedgerEntry(
            hypothesis_id=hypothesis.hypothesis_id,
            fingerprint=self.fingerprint_of(hypothesis),
            summary=hypothesis.summary,
            verdict=result.verdict.value,
            improvement_pct=result.improvement_pct,
            checksum_match=result.checksum_match,
            reasons=list(result.reasons),
        )
        self.entries.append(entry)
        return entry

    def render(self) -> list[str]:
        return [entry.one_line() for entry in self.entries]

    @property
    def tested_ddls(self) -> set[str]:
        return {
            part for entry in self.entries for part in entry.fingerprint
            if not part.startswith("REWRITE:")
        }


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------

@dataclass
class ToolCallRecord:
    tool_use_id: str
    name: str
    arguments: dict[str, Any]
    observation: dict[str, Any]
    duration_ms: float
    is_error: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "tool_use_id": self.tool_use_id,
            "name": self.name,
            "arguments": self.arguments,
            "observation": self.observation,
            "duration_ms": round(self.duration_ms, 1),
            "is_error": self.is_error,
        }


@dataclass
class IterationRecord:
    """One (thought, tool_call, observation) triple, as the spec requires."""

    iteration: int
    thinking: str
    text: str
    tool_calls: list[ToolCallRecord]
    stop_reason: str | None
    usage: dict[str, int]

    def to_json(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "thinking": self.thinking,
            "text": self.text,
            "tool_calls": [call.to_json() for call in self.tool_calls],
            "stop_reason": self.stop_reason,
            "usage": self.usage,
        }


@dataclass
class TokenLedger:
    """Cumulative token accounting across the run.

    input_tokens is the *uncached* remainder, so a total that ignored the cache
    fields would understate a cached run by an order of magnitude. All four are
    tracked; `total` is what the budget is checked against.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def add(self, usage: Any) -> dict[str, int]:
        delta = {
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            "cache_read_input_tokens": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            "cache_creation_input_tokens": int(
                getattr(usage, "cache_creation_input_tokens", 0) or 0
            ),
        }
        self.input_tokens += delta["input_tokens"]
        self.output_tokens += delta["output_tokens"]
        self.cache_read_input_tokens += delta["cache_read_input_tokens"]
        self.cache_creation_input_tokens += delta["cache_creation_input_tokens"]
        return delta

    @property
    def total(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_input_tokens
        )

    def to_json(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "total_tokens": self.total,
        }


@dataclass
class AgentRun:
    """The artifact Phase 5 renders and Phase 6 ships as static JSON."""

    query_name: str
    original_sql: str
    model: str
    started_at: str
    finished_at: str
    stop_reason: str
    partial: bool
    iterations_used: int
    max_iterations: int
    tokens: dict[str, int]
    token_budget: int
    diagnosis: str
    baseline: dict[str, Any] | None
    baseline_drift: dict[str, Any] | None
    accepted: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    recommendations: list[dict[str, Any]]
    unverified_claims: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "query_name": self.query_name,
            "original_sql": self.original_sql,
            "model": self.model,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "stop_reason": self.stop_reason,
            "partial": self.partial,
            "iterations_used": self.iterations_used,
            "max_iterations": self.max_iterations,
            "tokens": self.tokens,
            "token_budget": self.token_budget,
            "diagnosis": self.diagnosis,
            "baseline": self.baseline,
            "baseline_drift": self.baseline_drift,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "recommendations": self.recommendations,
            "unverified_claims": self.unverified_claims,
            "trace": self.trace,
            "error": self.error,
        }

    def save(self, directory: Path | str) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.query_name}.json"
        path.write_text(json.dumps(self.to_json(), indent=2, default=str), encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Content block helpers (SDK objects and test doubles both go through these)
# ---------------------------------------------------------------------------

def _block_type(block: Any) -> str:
    return str(getattr(block, "type", "") or "")


def _block_to_json(block: Any) -> dict[str, Any]:
    """Plain-dict view of a content block, for the persisted trace."""
    kind = _block_type(block)
    if kind == "text":
        return {"type": "text", "text": getattr(block, "text", "")}
    if kind == "thinking":
        return {"type": "thinking", "thinking": getattr(block, "thinking", "")}
    if kind == "tool_use":
        return {
            "type": "tool_use",
            "id": getattr(block, "id", ""),
            "name": getattr(block, "name", ""),
            "input": dict(getattr(block, "input", {}) or {}),
        }
    return {"type": kind or "unknown"}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    model: str = DEFAULT_MODEL
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_tokens: int = DEFAULT_MAX_TOKENS
    token_budget: int = DEFAULT_TOKEN_BUDGET
    effort: str = DEFAULT_EFFORT
    benchmark_runs: int = 5


class OptimizerAgent:
    """Drives the loop. Owns the trace, the budget, and the verified/claimed split."""

    def __init__(
        self,
        ctx: ToolContext,
        client: AnthropicClient,
        config: AgentConfig | None = None,
    ) -> None:
        self.ctx = ctx
        self.client = client
        self.config = config or AgentConfig()
        self.registry = ToolRegistry(ctx)
        self.verifier = Verifier(ctx.shadow, runs=self.config.benchmark_runs)
        self.ledger = HypothesisLedger()
        self.results: list[HypothesisResult] = []
        self.original_sql = ""
        self._finish_payload: dict[str, Any] | None = None

    # -- tool dispatch -----------------------------------------------------

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Execute one tool call. Returns (observation, is_error).

        Errors are returned as observations rather than raised. A rejected DDL
        or a malformed rewrite is information the model needs on its next turn;
        tearing down the run instead would throw away everything already proven.
        """
        try:
            if name == "test_hypothesis":
                return self._test_hypothesis(arguments), False
            if name == "finish":
                payload = self.registry.call("finish", arguments and {
                    "recommendations": arguments.get("recommendations", [])
                })
                self._finish_payload = {
                    "diagnosis": arguments.get("diagnosis", ""),
                    "recommendations": payload["recommendations"],
                }
                return payload, False
            return self.registry.call(name, arguments), False
        except (ToolError, SqlGuardError, BenchmarkError, ValueError, TypeError) as exc:
            return {"error": type(exc).__name__, "message": str(exc)}, True

    def _test_hypothesis(self, arguments: dict[str, Any]) -> dict[str, Any]:
        hypothesis = Hypothesis(
            hypothesis_id=str(arguments.get("hypothesis_id", f"h{len(self.results) + 1}")),
            kind=arguments["kind"],
            summary=str(arguments.get("summary", "")),
            index_ddls=tuple(arguments.get("index_ddls") or ()),
            rewritten_sql=arguments.get("rewritten_sql"),
        )

        fingerprint = self.ledger.fingerprint_of(hypothesis)
        duplicate = self.ledger.find(fingerprint)
        if duplicate is not None:
            # Short-circuit rather than re-measure. Re-running would burn ~15s
            # of index builds and benchmarks to produce the same verdict, and
            # would add a second entry claiming to be a distinct hypothesis.
            return {
                "already_tested": True,
                "previous_hypothesis_id": duplicate.hypothesis_id,
                "verdict": duplicate.verdict,
                "improvement_pct": round(duplicate.improvement_pct, 2),
                "checksum_match": duplicate.checksum_match,
                "reasons": duplicate.reasons,
                "message": (
                    f"This is the same change as {duplicate.hypothesis_id}, which was "
                    f"already tested and {duplicate.verdict}. Nothing was re-run. "
                    "Propose something different."
                ),
                "hypotheses_tested_so_far": self.ledger.render(),
            }

        # Exploration via create_index_on_shadow may have left indexes behind.
        # Clear them first so this measurement is attributable to this
        # hypothesis and nothing else.
        cleanup = self.ctx.shadow.reset()

        result = self.verifier.evaluate(self.original_sql, hypothesis)
        self.results.append(result)
        self.ledger.record(hypothesis, result)

        payload = result.to_json()
        payload["cleared_before_test"] = cleanup.dropped
        payload["hypotheses_tested_so_far"] = self.ledger.render()
        return payload

    # -- the loop ----------------------------------------------------------

    def run(self, query_name: str, original_sql: str) -> AgentRun:
        self.original_sql = original_sql
        started = datetime.now(timezone.utc)
        tokens = TokenLedger()
        trace: list[IterationRecord] = []
        stop_reason = StopReason.MODEL_STOPPED
        error: str | None = None

        # Measured before the model sees anything, so the model cannot influence
        # the number every improvement is compared against.
        self.ctx.shadow.reset()
        baseline = self.verifier.baseline_for(original_sql)

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    f"Optimise this query. It currently runs in a median of "
                    f"{baseline.median_ms:.1f}ms and returns {baseline.row_count} rows.\n\n"
                    f"```sql\n{original_sql}\n```"
                ),
            }
        ]

        iteration = 0
        try:
            while True:
                if iteration >= self.config.max_iterations:
                    stop_reason = StopReason.ITERATION_LIMIT
                    break
                if tokens.total >= self.config.token_budget:
                    stop_reason = StopReason.TOKEN_BUDGET
                    break

                response = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    system=SYSTEM_PROMPT,
                    # Caches the whole rendered prefix (tools, system, and every
                    # turn so far) at the last cacheable block. In a 12-iteration
                    # loop the transcript is re-sent every turn, so without this
                    # the same observations are re-billed at full price a dozen
                    # times.
                    cache_control={"type": "ephemeral"},
                    thinking={"type": "adaptive"},
                    output_config={"effort": self.config.effort},
                    tools=TOOL_SCHEMAS,
                    messages=messages,
                )
                iteration += 1
                usage_delta = tokens.add(getattr(response, "usage", None))

                content = list(getattr(response, "content", []) or [])
                thinking = "\n".join(
                    getattr(b, "thinking", "") for b in content if _block_type(b) == "thinking"
                ).strip()
                text = "\n".join(
                    getattr(b, "text", "") for b in content if _block_type(b) == "text"
                ).strip()
                tool_uses = [b for b in content if _block_type(b) == "tool_use"]

                record = IterationRecord(
                    iteration=iteration,
                    thinking=thinking,
                    text=text,
                    tool_calls=[],
                    stop_reason=getattr(response, "stop_reason", None),
                    usage=usage_delta,
                )
                trace.append(record)

                if not tool_uses:
                    # end_turn with no tool call: the model considers itself done
                    # but never called finish, so there is nothing verified to
                    # ship beyond what it already tested.
                    stop_reason = StopReason.MODEL_STOPPED
                    break

                # Thinking blocks must be echoed back unmodified, so the whole
                # content list goes back rather than a reconstruction of it.
                messages.append({"role": "assistant", "content": content})

                tool_results: list[dict[str, Any]] = []
                for block in tool_uses:
                    name = str(getattr(block, "name", ""))
                    arguments = dict(getattr(block, "input", {}) or {})
                    started_call = time.perf_counter()
                    observation, is_error = self._call_tool(name, arguments)
                    elapsed_ms = (time.perf_counter() - started_call) * 1000.0

                    record.tool_calls.append(
                        ToolCallRecord(
                            tool_use_id=str(getattr(block, "id", "")),
                            name=name,
                            arguments=arguments,
                            observation=observation,
                            duration_ms=elapsed_ms,
                            is_error=is_error,
                        )
                    )
                    # THE critical line: the full observation, including measured
                    # runtimes and rejection reasons, goes back into the message
                    # history verbatim.
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": str(getattr(block, "id", "")),
                            "content": json.dumps(observation, default=str),
                            "is_error": is_error,
                        }
                    )

                # All results for a turn go back in one user message; splitting
                # them trains the model out of parallel tool calls.
                messages.append({"role": "user", "content": tool_results})

                if self._finish_payload is not None:
                    stop_reason = StopReason.FINISHED
                    break

        except ShadowIsolationError:
            # Not recoverable and not swallowed: shadow can no longer stand in
            # for primary, so every later measurement would be meaningless.
            raise
        except Exception as exc:  # noqa: BLE001 - recorded, then degraded gracefully
            stop_reason = StopReason.ERROR
            error = f"{type(exc).__name__}: {exc}"

        return self._build_run(
            query_name=query_name,
            original_sql=original_sql,
            started=started,
            stop_reason=stop_reason,
            iterations_used=iteration,
            tokens=tokens,
            trace=trace,
            baseline=baseline,
            error=error,
        )

    # -- result assembly ---------------------------------------------------

    def _build_run(
        self,
        query_name: str,
        original_sql: str,
        started: datetime,
        stop_reason: StopReason,
        iterations_used: int,
        tokens: TokenLedger,
        trace: Sequence[IterationRecord],
        baseline: Any,
        error: str | None,
    ) -> AgentRun:
        accepted = [r for r in self.results if r.verdict is Verdict.ACCEPTED]
        rejected = [r for r in self.results if r.verdict is not Verdict.ACCEPTED]

        claimed = (self._finish_payload or {}).get("recommendations", [])
        recommendations, unverified = self._split_verified(claimed, accepted)

        # An accepted hypothesis the model failed to carry into `finish` is still
        # a measured win. Losing it because the model ran out of iterations
        # would throw away work the verifier already paid for.
        if stop_reason is not StopReason.FINISHED:
            recommendations = [
                {
                    "kind": "index" if result.hypothesis.kind == "index" else "rewrite",
                    "summary": result.hypothesis.summary,
                    "ddl": result.index_reports[0].production_ddl if result.index_reports else None,
                    "rewritten_sql": result.hypothesis.rewritten_sql,
                    "hypothesis_id": result.hypothesis.hypothesis_id,
                    "source": "recovered_from_accepted_hypotheses",
                }
                for result in accepted
            ]

        drift: dict[str, Any] | None = None
        try:
            drift = self.verifier.check_baseline_drift(original_sql)
        except (BenchmarkError, ShadowIsolationError) as exc:
            drift = {"error": f"{type(exc).__name__}: {exc}"}

        finished = datetime.now(timezone.utc)
        return AgentRun(
            query_name=query_name,
            original_sql=original_sql,
            model=self.config.model,
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
            stop_reason=stop_reason.value,
            partial=stop_reason is not StopReason.FINISHED,
            iterations_used=iterations_used,
            max_iterations=self.config.max_iterations,
            tokens=tokens.to_json(),
            token_budget=self.config.token_budget,
            diagnosis=(self._finish_payload or {}).get("diagnosis", ""),
            baseline=baseline.to_json() if baseline else None,
            baseline_drift=drift,
            accepted=[r.to_json() for r in accepted],
            rejected=[r.to_json() for r in rejected],
            recommendations=recommendations,
            unverified_claims=unverified,
            trace=[record.to_json() for record in trace],
            error=error,
        )

    def _split_verified(
        self, claimed: Sequence[dict[str, Any]], accepted: Sequence[HypothesisResult]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Drop any recommendation the verifier never accepted.

        This is the structural half of "no recommendation reaches the report
        without measurement". The model can claim whatever it likes in `finish`;
        only claims that match an accepted hypothesis survive, and the rest are
        published as unverified claims rather than quietly discarded.
        """
        by_id = {r.hypothesis.hypothesis_id: r for r in accepted}
        by_ddl: dict[str, HypothesisResult] = {
            normalize_sql(ddl): r for r in accepted for ddl in r.hypothesis.index_ddls
        }
        by_rewrite = {
            normalize_sql(r.hypothesis.rewritten_sql): r
            for r in accepted
            if r.hypothesis.rewritten_sql
        }

        verified: list[dict[str, Any]] = []
        unverified: list[dict[str, Any]] = []
        for item in claimed:
            match = by_id.get(str(item.get("hypothesis_id", "")))
            if match is None and item.get("ddl"):
                match = by_ddl.get(normalize_sql(str(item["ddl"])))
            if match is None and item.get("rewritten_sql"):
                match = by_rewrite.get(normalize_sql(str(item["rewritten_sql"])))

            if match is None:
                unverified.append(
                    {
                        **item,
                        "dropped_because": (
                            "no accepted hypothesis backs this recommendation; it was "
                            "never measured, so it is not recommended"
                        ),
                    }
                )
                continue

            enriched = dict(item)
            enriched["hypothesis_id"] = match.hypothesis.hypothesis_id
            enriched["before_ms"] = round(match.baseline.median_ms, 2) if match.baseline else None
            enriched["after_ms"] = round(match.optimized.median_ms, 2) if match.optimized else None
            enriched["improvement_pct"] = round(match.improvement_pct, 2)
            enriched["speedup"] = round(match.speedup, 2)
            if match.index_reports:
                enriched["production_ddl"] = [
                    report.production_ddl for report in match.index_reports
                ]
            verified.append(enriched)
        return verified, unverified


def optimize_query(
    ctx: ToolContext,
    client: AnthropicClient,
    query_name: str,
    sql: str,
    config: AgentConfig | None = None,
) -> AgentRun:
    """One query in, one verified run artifact out."""
    return OptimizerAgent(ctx, client, config).run(query_name, sql)

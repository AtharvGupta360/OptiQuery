"""The ReAct loop, driven by a scripted client against a real database.

The model is faked; nothing else is. Every observation these tests assert on was
produced by the real verifier against real tables, which is the only way to test
the property the loop exists for: that a measured rejection re-enters the
model's context in full. A mocked verifier would let the loop pass these tests
while feeding the model numbers nobody measured.

What is deliberately NOT tested here is whether the model behaves well. That
belongs in TestLiveAgent at the bottom, which needs a real provider. Everything
above it is about control flow the loop guarantees regardless of the model:
limits degrade to partial reports, duplicates are not re-measured, and nothing
reaches `recommendations` that the verifier did not accept.
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass, field
from typing import Any

import pytest

from app.agent import (
    AgentConfig,
    HypothesisLedger,
    OptimizerAgent,
    StopReason,
    normalize_sql,
    optimize_query,
)
from app.llm import LLMError, TextBlock, ThinkingBlock, ToolUseBlock, Usage, build_client
from app.tools import ToolContext, list_indexes
from app.verifier import Hypothesis

pytestmark = pytest.mark.db

# A full seq scan of orders (1M rows), which is slow enough that the verifier's
# 20% threshold is a real margin rather than timing noise, and cheap enough that
# a whole agent run costs a couple of seconds. Deliberately not one of the seed
# queries: those take ~1.5s each, and at ~20 runs per suite that is minutes of
# waiting to test control flow that has nothing to do with which query it is.
#
# The literal is high-cardinality, so HELPFUL_INDEX turns the scan into a single
# index lookup -- a win no amount of variance can manufacture. IRRELEVANT_INDEX
# is on a column the predicate never mentions, so it cannot help, and any
# "improvement" it shows is noise the median is supposed to absorb.
SQL = (
    "SELECT o.id, o.total_amount FROM orders o "
    "WHERE o.email_snapshot = 'Paulo.Lindqvist138671@Example.com' ORDER BY o.id"
)
HELPFUL_INDEX = "CREATE INDEX ix_agent_orders_email ON orders (email_snapshot)"
IRRELEVANT_INDEX = "CREATE INDEX ix_agent_orders_payment ON orders (payment_method)"

# Faster than the original and returns 5 rows instead of 15. Used wherever a
# test needs a guaranteed rejection, because it is rejected on the checksum --
# a fact about the result set, not about the clock. Asserting REJECTED on a
# merely-useless index would make the test depend on timing noise staying under
# 20%, which is the one thing a benchmark cannot promise.
WRONG_REWRITE = SQL.replace("ORDER BY o.id", "ORDER BY o.id LIMIT 5")

_ids = itertools.count(1)


# ---------------------------------------------------------------------------
# Scripted client
# ---------------------------------------------------------------------------


def tool_use(_tool_name: str, **arguments: Any) -> ToolUseBlock:
    """Positional-only name so a tool taking a `name` argument still works."""
    return ToolUseBlock(id=f"tu_{next(_ids)}", name=_tool_name, input=arguments)


def finish_block(diagnosis: str = "seq scan on orders", **kwargs: Any) -> ToolUseBlock:
    return tool_use("finish", diagnosis=diagnosis, **kwargs)


@dataclass
class Turn:
    """One scripted model response."""

    blocks: list[Any]
    stop_reason: str = "tool_use"
    usage: Usage = field(default_factory=lambda: Usage(input_tokens=100, output_tokens=50))


@dataclass
class Reply:
    """Anthropic-shaped response object, same surface the SDK returns."""

    content: list[Any]
    stop_reason: str
    usage: Usage


class ScriptedClient:
    """Plays a fixed script and records every request it was sent.

    The recorded requests are the point: they are how a test proves an
    observation actually re-entered the message history, rather than proving
    only that the agent stored it somewhere internal.
    """

    def __init__(self, *turns: Turn, repeat_last: bool = False) -> None:
        self._turns = list(turns)
        self._repeat_last = repeat_last
        self.requests: list[dict[str, Any]] = []

    @property
    def messages(self) -> "ScriptedClient":
        return self

    def create(self, **kwargs: Any) -> Reply:
        # The agent appends to its message list in place, so snapshot it.
        self.requests.append({**kwargs, "messages": list(kwargs.get("messages", []))})
        if not self._turns:
            raise AssertionError("script exhausted: the loop asked for more turns than scripted")
        turn = self._turns[0] if (self._repeat_last and len(self._turns) == 1) else self._turns.pop(0)
        return Reply(content=list(turn.blocks), stop_reason=turn.stop_reason, usage=turn.usage)

    # -- assertions helpers ------------------------------------------------

    def tool_results_in(self, request_index: int) -> list[dict[str, Any]]:
        """Every tool_result block visible to the model on a given request."""
        results: list[dict[str, Any]] = []
        for message in self.requests[request_index]["messages"]:
            content = message.get("content")
            if isinstance(content, list):
                results.extend(
                    block
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "tool_result"
                )
        return results


def agent_for(ctx: ToolContext, client: ScriptedClient, **overrides: Any) -> OptimizerAgent:
    # Three runs, not one. A single timed run has no median to be robust with,
    # which is how an index on a column this query never touches measured 20%
    # faster and got ACCEPTED while these tests were being written.
    config = AgentConfig(model="scripted", benchmark_runs=3, **overrides)
    return OptimizerAgent(ctx, client, config)


# ---------------------------------------------------------------------------
# Observation feedback -- the property the loop exists for
# ---------------------------------------------------------------------------


class TestObservationsReenterContext:
    def test_a_tool_result_is_visible_on_the_next_request(self, ctx: ToolContext) -> None:
        client = ScriptedClient(
            Turn([tool_use("list_indexes", table="users")]),
            Turn([finish_block(recommendations=[])]),
        )
        agent_for(ctx, client).run("obs", SQL)

        results = client.tool_results_in(1)
        assert len(results) == 1
        assert "users_pkey" in results[0]["content"]

    def test_measured_numbers_reach_the_model_verbatim(self, ctx: ToolContext) -> None:
        """A rejection the model cannot read is a rejection it will repeat."""
        client = ScriptedClient(
            Turn(
                [
                    tool_use(
                        "test_hypothesis",
                        hypothesis_id="h1",
                        kind="rewrite",
                        summary="add a LIMIT, which drops rows the original returns",
                        rewritten_sql=WRONG_REWRITE,
                    )
                ]
            ),
            Turn([finish_block(recommendations=[])]),
        )
        agent_for(ctx, client).run("obs", SQL)

        observation = json.loads(client.tool_results_in(1)[0]["content"])
        assert observation["verdict"] == "REJECTED"
        assert observation["checksum_match"] is False
        # Rejected on the result set, and the observation says so -- a rejection
        # with no stated reason teaches nothing.
        assert any("checksum" in reason.lower() for reason in observation["reasons"])
        # Both row counts come back, so the model can work out which rows differ
        # instead of retrying variants blindly.
        assert observation["baseline"]["row_count"] == 15
        assert observation["optimized"]["row_count"] == 5
        assert observation["baseline"]["median_ms"] > 0
        assert observation["optimized"]["median_ms"] > 0

    def test_parallel_calls_return_in_a_single_user_message(self, ctx: ToolContext) -> None:
        """Splitting them trains the model out of issuing parallel calls."""
        client = ScriptedClient(
            Turn([tool_use("list_indexes"), tool_use("get_schema", table="users")]),
            Turn([finish_block(recommendations=[])]),
        )
        agent_for(ctx, client).run("obs", SQL)

        user_turns = [
            m
            for m in client.requests[1]["messages"]
            if m["role"] == "user" and isinstance(m["content"], list)
        ]
        assert len(user_turns) == 1
        assert len(user_turns[0]["content"]) == 2

    def test_assistant_content_is_echoed_unmodified(self, ctx: ToolContext) -> None:
        """Thinking blocks must survive the round trip; the API requires it."""
        thinking = ThinkingBlock(thinking="the filter on email_snapshot is unindexed")
        client = ScriptedClient(
            Turn([thinking, TextBlock(text="checking"), tool_use("list_indexes")]),
            Turn([finish_block(recommendations=[])]),
        )
        agent_for(ctx, client).run("obs", SQL)

        assistant = [m for m in client.requests[1]["messages"] if m["role"] == "assistant"]
        assert assistant[0]["content"][0] is thinking

    def test_a_tool_error_is_an_observation_not_a_crash(self, ctx: ToolContext) -> None:
        client = ScriptedClient(
            Turn([tool_use("explain_query", sql="SELECT * FROM table_that_does_not_exist")]),
            Turn([finish_block(recommendations=[])]),
        )
        run = agent_for(ctx, client).run("obs", SQL)

        result = client.tool_results_in(1)[0]
        assert result["is_error"] is True
        assert "does not exist" in result["content"]
        # The run continued and finished normally.
        assert run.stop_reason == StopReason.FINISHED.value

    def test_an_unknown_tool_is_reported_back(self, ctx: ToolContext) -> None:
        client = ScriptedClient(
            Turn([tool_use("drop_table", name="users")]),
            Turn([finish_block(recommendations=[])]),
        )
        agent_for(ctx, client).run("obs", SQL)
        assert client.tool_results_in(1)[0]["is_error"] is True


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_a_repeated_hypothesis_is_not_remeasured(self, ctx: ToolContext) -> None:
        proposal = dict(
            kind="index",
            summary="index email_snapshot",
            index_ddls=[HELPFUL_INDEX],
        )
        client = ScriptedClient(
            Turn([tool_use("test_hypothesis", hypothesis_id="h1", **proposal)]),
            # Same DDL, different id and whitespace -- still the same index.
            Turn(
                [
                    tool_use(
                        "test_hypothesis",
                        hypothesis_id="h2",
                        kind="index",
                        summary="try indexing email_snapshot",
                        index_ddls=["CREATE  INDEX ix_agent_orders_email ON orders (email_snapshot)"],
                    )
                ]
            ),
            Turn([finish_block(recommendations=[])]),
        )
        agent = agent_for(ctx, client)
        run = agent.run("dedup", SQL)

        second = json.loads(client.tool_results_in(2)[-1]["content"])
        assert second["already_tested"] is True
        assert second["previous_hypothesis_id"] == "h1"
        # Measured once, not twice.
        assert len(agent.results) == 1
        assert len(run.accepted) + len(run.rejected) == 1

    def test_every_observation_carries_the_full_ledger(self, ctx: ToolContext) -> None:
        """Attention over a long transcript is not a reliable memory of failure."""
        client = ScriptedClient(
            Turn(
                [
                    tool_use(
                        "test_hypothesis",
                        hypothesis_id="h1",
                        kind="rewrite",
                        summary="add a LIMIT",
                        rewritten_sql=WRONG_REWRITE,
                    )
                ]
            ),
            Turn(
                [
                    tool_use(
                        "test_hypothesis",
                        hypothesis_id="h2",
                        kind="index",
                        summary="index email_snapshot",
                        index_ddls=[HELPFUL_INDEX],
                    )
                ]
            ),
            Turn([finish_block(recommendations=[])]),
        )
        agent_for(ctx, client).run("ledger", SQL)

        second = json.loads(client.tool_results_in(2)[-1]["content"])
        ledger = second["hypotheses_tested_so_far"]
        assert len(ledger) == 2
        assert any(line.startswith("[REJECTED] h1") for line in ledger)

    def test_differing_column_order_is_not_a_duplicate(self) -> None:
        """Two indexes over the same columns in a different order really differ."""
        ledger = HypothesisLedger()
        first = ledger.fingerprint_of(
            Hypothesis("h1", "index", "a,b", index_ddls=("CREATE INDEX i ON t (a, b)",))
        )
        second = ledger.fingerprint_of(
            Hypothesis("h2", "index", "b,a", index_ddls=("CREATE INDEX i ON t (b, a)",))
        )
        assert first != second

    def test_fingerprints_ignore_case_and_whitespace(self) -> None:
        assert normalize_sql("CREATE  INDEX i ON t (a);") == normalize_sql("create index i on t (a)")


# ---------------------------------------------------------------------------
# Limits degrade to a partial report
# ---------------------------------------------------------------------------


class TestLimits:
    def test_the_iteration_cap_stops_the_loop(self, ctx: ToolContext) -> None:
        client = ScriptedClient(Turn([tool_use("list_indexes")]), repeat_last=True)
        run = agent_for(ctx, client, max_iterations=3).run("cap", SQL)

        assert run.stop_reason == StopReason.ITERATION_LIMIT.value
        assert run.iterations_used == 3
        assert run.partial is True
        assert len(client.requests) == 3

    def test_the_token_budget_stops_the_loop(self, ctx: ToolContext) -> None:
        client = ScriptedClient(
            Turn([tool_use("list_indexes")], usage=Usage(input_tokens=900, output_tokens=200)),
            repeat_last=True,
        )
        run = agent_for(ctx, client, max_iterations=50, token_budget=2_000).run("budget", SQL)

        assert run.stop_reason == StopReason.TOKEN_BUDGET.value
        assert run.tokens["total_tokens"] >= 2_000
        assert run.partial is True

    def test_truncated_output_is_not_reported_as_the_model_finishing(
        self, ctx: ToolContext
    ) -> None:
        """Thinking models return empty content when reasoning eats the budget."""
        client = ScriptedClient(Turn([], stop_reason="max_tokens"))
        run = agent_for(ctx, client).run("truncated", SQL)

        assert run.stop_reason == StopReason.OUTPUT_TRUNCATED.value
        assert run.partial is True

    def test_an_ordinary_end_turn_is_still_reported_as_such(self, ctx: ToolContext) -> None:
        client = ScriptedClient(Turn([TextBlock(text="I am done")], stop_reason="end_turn"))
        run = agent_for(ctx, client).run("stopped", SQL)
        assert run.stop_reason == StopReason.MODEL_STOPPED.value

    def test_a_verified_win_survives_being_cut_off(self, ctx: ToolContext) -> None:
        """The whole point of degrading gracefully: keep what was already proven."""
        client = ScriptedClient(
            Turn(
                [
                    tool_use(
                        "test_hypothesis",
                        hypothesis_id="h1",
                        kind="index",
                        summary="index email_snapshot",
                        index_ddls=[HELPFUL_INDEX],
                    )
                ]
            ),
            Turn([tool_use("list_indexes")]),
            repeat_last=True,
        )
        run = agent_for(ctx, client, max_iterations=2).run("partial", SQL)

        assert run.stop_reason == StopReason.ITERATION_LIMIT.value
        assert len(run.accepted) == 1
        # Recovered even though `finish` was never reached.
        assert len(run.recommendations) == 1
        assert run.recommendations[0]["source"] == "recovered_from_accepted_hypotheses"
        assert "CONCURRENTLY" in run.recommendations[0]["ddl"]


# ---------------------------------------------------------------------------
# Nothing unmeasured reaches the report
# ---------------------------------------------------------------------------


class TestVerifiedSplit:
    def test_an_unbacked_recommendation_is_dropped(self, ctx: ToolContext) -> None:
        """The structural half of 'no recommendation without measurement'."""
        client = ScriptedClient(
            Turn(
                [
                    finish_block(
                        recommendations=[
                            {
                                "kind": "index",
                                "summary": "obviously this will help, no need to test it",
                                "ddl": "CREATE INDEX ix_never_tested ON orders (status)",
                            }
                        ]
                    )
                ]
            )
        )
        run = agent_for(ctx, client).run("unbacked", SQL)

        assert run.recommendations == []
        assert len(run.unverified_claims) == 1
        assert "never measured" in run.unverified_claims[0]["dropped_because"]

    def test_a_backed_recommendation_carries_its_measurements(self, ctx: ToolContext) -> None:
        client = ScriptedClient(
            Turn(
                [
                    tool_use(
                        "test_hypothesis",
                        hypothesis_id="h1",
                        kind="index",
                        summary="index email_snapshot",
                        index_ddls=[HELPFUL_INDEX],
                    )
                ]
            ),
            Turn(
                [
                    finish_block(
                        recommendations=[
                            {
                                "kind": "index",
                                "summary": "index email_snapshot",
                                "ddl": HELPFUL_INDEX,
                                "hypothesis_id": "h1",
                            }
                        ]
                    )
                ]
            ),
        )
        run = agent_for(ctx, client).run("backed", SQL)

        assert run.stop_reason == StopReason.FINISHED.value
        assert run.unverified_claims == []
        recommendation = run.recommendations[0]
        assert recommendation["before_ms"] > recommendation["after_ms"]
        assert recommendation["improvement_pct"] >= 20.0
        assert recommendation["production_ddl"][0].startswith("CREATE INDEX CONCURRENTLY")

    def test_a_rejected_hypothesis_is_reported_not_hidden(self, ctx: ToolContext) -> None:
        """The rejected list is the evidence that the agent measures."""
        client = ScriptedClient(
            Turn(
                [
                    tool_use(
                        "test_hypothesis",
                        hypothesis_id="h1",
                        kind="rewrite",
                        summary="add a LIMIT, dropping rows the original returns",
                        rewritten_sql=WRONG_REWRITE,
                    )
                ]
            ),
            Turn([finish_block(recommendations=[])]),
        )
        run = agent_for(ctx, client).run("rejected", SQL)

        assert len(run.rejected) == 1
        assert run.rejected[0]["verdict"] == "REJECTED"
        assert run.rejected[0]["reasons"]
        assert run.recommendations == []


# ---------------------------------------------------------------------------
# The artifact Phase 5 and 6 consume
# ---------------------------------------------------------------------------


class TestTrace:
    def test_the_trace_records_thought_tool_and_observation(self, ctx: ToolContext) -> None:
        client = ScriptedClient(
            Turn([ThinkingBlock(thinking="check the indexes"), tool_use("list_indexes")]),
            Turn([finish_block(recommendations=[])]),
        )
        run = agent_for(ctx, client).run("trace", SQL)

        first = run.trace[0]
        assert first["thinking"] == "check the indexes"
        assert first["tool_calls"][0]["name"] == "list_indexes"
        assert first["tool_calls"][0]["observation"]["indexes"]
        assert first["usage"]["input_tokens"] == 100

    def test_the_run_serialises_to_json_on_disk(self, ctx: ToolContext, tmp_path) -> None:
        client = ScriptedClient(Turn([finish_block(recommendations=[])]))
        run = agent_for(ctx, client).run("q1_demo", SQL)

        path = run.save(tmp_path)
        assert path.name == "q1_demo.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        # Exactly the keys the frontend reads.
        for key in ("original_sql", "baseline", "accepted", "rejected", "recommendations", "trace"):
            assert key in payload
        assert payload["baseline"]["median_ms"] > 0

    def test_the_baseline_is_measured_before_the_model_sees_anything(
        self, ctx: ToolContext
    ) -> None:
        """The model must not be able to influence the number it is judged against."""
        client = ScriptedClient(Turn([finish_block(recommendations=[])]))
        run = agent_for(ctx, client).run("baseline", SQL)

        opening = client.requests[0]["messages"][0]["content"]
        quoted = re.search(r"median of ([\d.]+)ms", opening)
        assert quoted, opening
        # Compared numerically: the prompt and the artifact round independently.
        assert float(quoted.group(1)) == pytest.approx(run.baseline["median_ms"], abs=0.1)
        assert f"returns {run.baseline['row_count']} rows" in opening


class TestIsolation:
    def test_primary_is_untouched_by_a_full_run(self, ctx: ToolContext) -> None:
        client = ScriptedClient(
            Turn(
                [
                    tool_use(
                        "test_hypothesis",
                        hypothesis_id="h1",
                        kind="index",
                        summary="index email_snapshot",
                        index_ddls=[HELPFUL_INDEX],
                    )
                ]
            ),
            Turn([finish_block(recommendations=[])]),
        )
        optimize_query(ctx, client, "iso", SQL, AgentConfig(model="scripted", benchmark_runs=1))

        indexes = list_indexes(ctx, database="primary")["indexes"]
        assert len(indexes) == 5
        assert all(index["is_primary"] for index in indexes)

    def test_exploration_indexes_do_not_survive_a_run_that_never_tested_anything(
        self, ctx: ToolContext
    ) -> None:
        """The leak found by the first live run.

        A model that only explores never reaches _test_hypothesis's reset, so
        its indexes were still in place when the next query was optimised --
        silently giving that query a baseline it never earned.
        """
        before = {i["name"] for i in list_indexes(ctx, database="shadow")["indexes"]}
        client = ScriptedClient(
            Turn([tool_use("create_index_on_shadow", ddl=HELPFUL_INDEX)]),
            repeat_last=True,
        )
        run = agent_for(ctx, client, max_iterations=2).run("leak", SQL)

        assert run.stop_reason == StopReason.ITERATION_LIMIT.value
        after = {i["name"] for i in list_indexes(ctx, database="shadow")["indexes"]}
        assert after == before
        assert run.baseline_drift["indexes_cleared_before_measuring"] == [
            "ix_agent_orders_email"
        ]
        # And the drift number is measured on a clean shadow, not through the
        # index the model happened to leave behind.
        assert "error" not in run.baseline_drift

    def test_shadow_returns_to_baseline_after_the_run(self, ctx: ToolContext) -> None:
        before = {i["name"] for i in list_indexes(ctx, database="shadow")["indexes"]}
        client = ScriptedClient(
            Turn([tool_use("create_index_on_shadow", ddl=HELPFUL_INDEX)]),
            Turn(
                [
                    tool_use(
                        "test_hypothesis",
                        hypothesis_id="h1",
                        kind="index",
                        summary="index payment_method",
                        index_ddls=[IRRELEVANT_INDEX],
                    )
                ]
            ),
            Turn([finish_block(recommendations=[])]),
        )
        agent_for(ctx, client).run("iso", SQL)

        after = {i["name"] for i in list_indexes(ctx, database="shadow")["indexes"]}
        assert after == before

    def test_exploration_indexes_are_cleared_before_a_hypothesis_is_measured(
        self, ctx: ToolContext
    ) -> None:
        """Otherwise a hypothesis inherits a head start it never proposed."""
        client = ScriptedClient(
            Turn([tool_use("create_index_on_shadow", ddl=HELPFUL_INDEX)]),
            Turn(
                [
                    tool_use(
                        "test_hypothesis",
                        hypothesis_id="h1",
                        kind="index",
                        summary="index payment_method",
                        index_ddls=[IRRELEVANT_INDEX],
                    )
                ]
            ),
            Turn([finish_block(recommendations=[])]),
        )
        agent_for(ctx, client).run("clear", SQL)

        observation = json.loads(client.tool_results_in(2)[-1]["content"])
        assert observation["cleared_before_test"] == ["ix_agent_orders_email"]


# ---------------------------------------------------------------------------
# Live model. Skipped without a provider.
# ---------------------------------------------------------------------------


def _live_bundle():
    try:
        return build_client()
    except LLMError as exc:
        pytest.skip(f"no model provider configured: {exc}")


def _assert_the_run_actually_ran(run) -> None:
    """Guard against a vacuous pass.

    Every assertion below is over a collection the run produces. A run that
    died on its first request produces empty collections, and every "the model
    never did X" assertion then holds trivially -- which is exactly what
    happened here: two live tests passed green while every run was failing on
    iteration 2 with a provider error. A live test that cannot distinguish
    "the model behaved" from "the model never got to act" is not a test.
    """
    if run.error:
        # Quota and transport failures are environmental, not a code defect.
        if "429" in run.error or "quota" in run.error.lower():
            pytest.skip(f"provider rate-limited: {run.error[:200]}")
        pytest.fail(f"live run failed: {run.error[:500]}")
    assert run.iterations_used >= 2, "the model never got past its first turn"


@pytest.mark.live
class TestLiveAgent:
    def test_the_agent_never_proposes_the_same_index_twice(
        self, ctx: ToolContext, seed_queries
    ) -> None:
        """The spec's required property, against a real model.

        This is the failure the ledger exists to prevent: a model that cannot
        see its own rejections re-proposes the same index until the budget is
        gone. Asserted on the DDLs the model actually emitted, not on the
        deduplicated ledger -- the ledger would make a repeat invisible, which
        is exactly what must not be relied on.
        """
        bundle = _live_bundle()
        run = optimize_query(
            ctx,
            bundle.client,
            "live_q2",
            seed_queries["q2_unindexed_fk_join"].sql,
            AgentConfig(model=bundle.model, max_iterations=8, benchmark_runs=3),
        )

        _assert_the_run_actually_ran(run)

        proposed: list[str] = []
        for iteration in run.trace:
            for call in iteration["tool_calls"]:
                if call["name"] != "test_hypothesis":
                    continue
                for ddl in call["arguments"].get("index_ddls") or []:
                    proposed.append(normalize_sql(ddl))

        # Without this the assertion below is vacuously true for any run in
        # which the model proposed nothing at all.
        assert len(proposed) >= 2, (
            f"the model proposed {len(proposed)} indexes; at least two are needed "
            "for 'never twice' to mean anything"
        )
        duplicates = {ddl for ddl in proposed if proposed.count(ddl) > 1}
        assert not duplicates, f"the model re-proposed: {sorted(duplicates)}"

    def test_a_live_run_recommends_only_what_it_measured(
        self, ctx: ToolContext, seed_queries
    ) -> None:
        bundle = _live_bundle()
        run = optimize_query(
            ctx,
            bundle.client,
            "live_q1",
            seed_queries["q1_seq_scan_high_cardinality"].sql,
            AgentConfig(model=bundle.model, max_iterations=8, benchmark_runs=3),
        )

        _assert_the_run_actually_ran(run)

        # The property is about what survives the verifier, so the verifier has
        # to have ruled on something.
        assert run.accepted or run.rejected, "no hypothesis was tested at all"

        accepted_ids = {result["hypothesis"]["hypothesis_id"] for result in run.accepted}
        for recommendation in run.recommendations:
            assert recommendation["hypothesis_id"] in accepted_ids
        for claim in run.unverified_claims:
            assert claim["dropped_because"]

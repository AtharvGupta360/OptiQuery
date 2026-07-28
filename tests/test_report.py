"""Rendering. No database, no model -- a run artifact in, two documents out.

The load-bearing test in this file is
`test_rejected_hypotheses_are_not_inside_a_collapsible_block`. Everything else
here is formatting; that one is the project's central claim rendered as
structure. A reader who has to expand something to discover the agent's
failures is reading a different report than the one this project promises.
"""

from __future__ import annotations

import json

import pytest

from app.report import (
    best_accepted,
    headline,
    plan_summary,
    render_html,
    render_markdown,
    summarize_for_terminal,
    write_reports,
)

ACCEPTED_DDL = "CREATE INDEX ix_orders_user_id ON orders (user_id)"
PRODUCTION_DDL = "CREATE INDEX CONCURRENTLY ix_orders_user_id ON orders (user_id)"
REJECTED_DDL = "CREATE INDEX ix_orders_status ON orders (status)"
BAD_REWRITE = "SELECT u.id FROM users u LIMIT 5"


def bench(median: float, rows: int, checksum: str) -> dict:
    return {
        "runs": 5,
        "median_ms": median,
        "mean_ms": median * 1.02,
        "min_ms": median * 0.97,
        "max_ms": median * 1.09,
        "stdev_ms": median * 0.03,
        "samples_ms": [median] * 5,
        "discarded_warmup_ms": median * 1.4,
        "row_count": rows,
        "checksum": checksum,
        "checksum_order_sensitive": checksum[::-1],
        "checksum_stable_across_runs": True,
    }


@pytest.fixture()
def run() -> dict:
    """A run shaped exactly like AgentRun.to_json(): one win, two rejections."""
    return {
        "query_name": "q2_unindexed_fk_join",
        "original_sql": "SELECT u.id, count(*)\nFROM users u JOIN orders o ON o.user_id = u.id\nGROUP BY u.id",
        "model": "claude-sonnet-5",
        "started_at": "2026-07-28T09:00:00+00:00",
        "finished_at": "2026-07-28T09:04:12+00:00",
        "stop_reason": "finished",
        "partial": False,
        "iterations_used": 6,
        "max_iterations": 12,
        "tokens": {"input_tokens": 40000, "output_tokens": 2000, "total_tokens": 42000},
        "token_budget": 400000,
        "diagnosis": "orders is sequentially scanned to satisfy the join on user_id.",
        "baseline": bench(984.6, 25, "a" * 64),
        "baseline_drift": {
            "first_median_ms": 984.6,
            "final_median_ms": 1002.1,
            "drift_pct": 1.78,
            "checksum_unchanged": True,
        },
        "accepted": [
            {
                "hypothesis": {
                    "hypothesis_id": "h2",
                    "kind": "index",
                    "summary": "Index orders(user_id) so the join can use it",
                    "index_ddls": [ACCEPTED_DDL],
                    "rewritten_sql": None,
                },
                "verdict": "ACCEPTED",
                "baseline": bench(984.6, 25, "a" * 64),
                "optimized": bench(31.2, 25, "a" * 64),
                "indexes": [
                    {
                        "index_name": "ix_orders_user_id",
                        "table": "orders",
                        "ddl": ACCEPTED_DDL,
                        "production_ddl": PRODUCTION_DDL,
                        "build_ms": 812.0,
                        "size_bytes": 22000000,
                        "size_pretty": "21 MB",
                        "table_size_bytes": 640000000,
                        "table_size_pretty": "610 MB",
                        "pct_of_table": 3.44,
                        "oversized": False,
                        "write_amplification": "Every INSERT into orders must now also maintain this index.",
                    }
                ],
                "checksum_match": True,
                "ordered_checksum_match": True,
                "improvement_pct": 96.83,
                "speedup": 31.56,
                "min_improvement_pct": 20.0,
                "reasons": [],
                "flags": [],
                "shadow_reset": {"dropped": ["ix_orders_user_id"], "matches_baseline": True},
            }
        ],
        "rejected": [
            {
                "hypothesis": {
                    "hypothesis_id": "h1",
                    "kind": "index",
                    "summary": "Index orders(status)",
                    "index_ddls": [REJECTED_DDL],
                    "rewritten_sql": None,
                },
                "verdict": "REJECTED",
                "baseline": bench(984.6, 25, "a" * 64),
                "optimized": bench(961.4, 25, "a" * 64),
                "indexes": [],
                "checksum_match": True,
                "ordered_checksum_match": True,
                "improvement_pct": 2.36,
                "speedup": 1.02,
                "min_improvement_pct": 20.0,
                "reasons": ["improvement 2.4% is below the required 20.0%"],
                "flags": [],
                "shadow_reset": {"dropped": ["ix_orders_status"], "matches_baseline": True},
            },
            {
                "hypothesis": {
                    "hypothesis_id": "h3",
                    "kind": "rewrite",
                    "summary": "Drop the join entirely",
                    "index_ddls": [],
                    "rewritten_sql": BAD_REWRITE,
                },
                "verdict": "REJECTED",
                "baseline": bench(984.6, 25, "a" * 64),
                "optimized": bench(0.8, 5, "b" * 64),
                "indexes": [],
                "checksum_match": False,
                "ordered_checksum_match": False,
                "improvement_pct": 99.92,
                "speedup": 1230.75,
                "min_improvement_pct": 20.0,
                "reasons": ["checksum differs: 25 rows became 5"],
                "flags": [],
                "shadow_reset": {"dropped": [], "matches_baseline": True},
            },
        ],
        "recommendations": [
            {
                "kind": "index",
                "summary": "Index orders(user_id) so the join can use it",
                "ddl": ACCEPTED_DDL,
                "hypothesis_id": "h2",
                "before_ms": 984.6,
                "after_ms": 31.2,
                "improvement_pct": 96.83,
                "speedup": 31.56,
                "production_ddl": [PRODUCTION_DDL],
            }
        ],
        "unverified_claims": [
            {
                "kind": "index",
                "summary": "Also index users(country)",
                "dropped_because": "no accepted hypothesis backs this recommendation; it was never measured",
            }
        ],
        "trace": [
            {
                "iteration": 1,
                "thinking": "The join has no index on the foreign key.",
                "text": "",
                "tool_calls": [
                    {
                        "tool_use_id": "tu_1",
                        "name": "explain_query",
                        "arguments": {
                            "sql": "SELECT u.id, count(*) FROM users u JOIN orders o ON o.user_id = u.id GROUP BY u.id",
                            "analyze": True,
                        },
                        "observation": {
                            "summary": {
                                "node_count": 7,
                                "sequential_scans": [
                                    {
                                        "relation": "orders",
                                        "actual_rows": 1000000,
                                        "rows_removed_by_filter": 0,
                                    }
                                ],
                                "rows_read_by_sequential_scans": 1000000,
                                "rows_discarded_by_sequential_scans": 0,
                            }
                        },
                        "duration_ms": 4360.2,
                        "is_error": False,
                    }
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 2100, "output_tokens": 90},
            }
        ],
        "error": None,
    }


# ---------------------------------------------------------------------------
# The central structural claim
# ---------------------------------------------------------------------------


class TestRejectedIsNeverHidden:
    def test_rejected_hypotheses_are_not_inside_a_collapsible_block(self, run: dict) -> None:
        """The rejected table must render before any <details> element.

        This is the project's thesis expressed as document structure: the
        evidence that the agent measured instead of guessing is its failures,
        so they cannot sit behind a disclosure triangle. The reasoning trace
        may, because it is merely long.
        """
        document = render_html(run)
        first_details = document.index("<details")
        rejected_heading = document.index("Rejected hypotheses")
        assert rejected_heading < first_details

        # Every rejected id, and the numbers that killed it, are in the open.
        visible = document[:first_details]
        for identifier in ("h1", "h3"):
            assert identifier in visible
        assert "checksum differs" in visible
        assert "below the required" in visible

    def test_rejected_rows_carry_the_measurements_that_killed_them(self, run: dict) -> None:
        document = render_markdown(run)
        assert "961.4 ms" in document, "the measured 'after' of a rejected index"
        assert "+2.4%" in document, "the improvement that fell short"
        assert "**DIFFERS**" in document, "the checksum failure"
        assert "25 rows became 5" in document

    def test_a_faster_but_wrong_rewrite_is_reported_as_rejected(self, run: dict) -> None:
        """1230x faster and still rejected -- the headline case for the verifier."""
        for document in (render_markdown(run), render_html(run)):
            assert BAD_REWRITE in document
        markdown = render_markdown(run)
        # It never reaches the recommendations.
        recommendations = markdown.split("## Rejected")[0]
        assert BAD_REWRITE not in recommendations

    def test_rejected_count_is_in_the_heading(self, run: dict) -> None:
        assert "Rejected hypotheses (2)" in render_markdown(run)
        assert "Rejected hypotheses (2)" in render_html(run)


# ---------------------------------------------------------------------------
# Accepted work
# ---------------------------------------------------------------------------


class TestRecommendations:
    def test_the_deployable_ddl_is_the_concurrent_form(self, run: dict) -> None:
        """Shadow builds without CONCURRENTLY; production must not."""
        for document in (render_markdown(run), render_html(run)):
            assert PRODUCTION_DDL in document

    def test_index_size_and_write_cost_are_stated(self, run: dict) -> None:
        document = render_markdown(run)
        assert "21 MB" in document
        assert "3.44% of the 610 MB heap" in document
        assert "Every INSERT into orders" in document

    def test_the_headline_reports_both_medians_and_the_row_count(self, run: dict) -> None:
        line = headline(run)
        assert "984.6 ms" in line and "31.2 ms" in line
        assert "31.6x faster" in line
        assert "25 rows" in line

    def test_best_accepted_picks_the_largest_improvement(self, run: dict) -> None:
        run["accepted"].append(
            {**run["accepted"][0], "improvement_pct": 10.0, "hypothesis": {"hypothesis_id": "h9"}}
        )
        assert best_accepted(run)["hypothesis"]["hypothesis_id"] == "h2"

    def test_no_accepted_change_is_stated_plainly(self, run: dict) -> None:
        run["accepted"] = []
        run["recommendations"] = []
        assert "none met the acceptance bar" in headline(run)
        document = render_markdown(run)
        assert "Recommendations (0)" in document
        # The rejections are still there in full.
        assert "Rejected hypotheses (2)" in document


class TestUnverifiedClaims:
    def test_a_dropped_claim_is_printed_not_deleted(self, run: dict) -> None:
        for document in (render_markdown(run), render_html(run)):
            assert "Also index users(country)" in document
            assert "never measured" in document


# ---------------------------------------------------------------------------
# Provenance and honesty
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_every_report_names_the_functions_behind_its_numbers(self, run: dict) -> None:
        for document in (render_markdown(run), render_html(run)):
            assert "verifier.benchmark" in document
            assert "verifier.result_checksum" in document
            assert "tools.explain_query" in document

    def test_the_write_path_limitation_is_stated(self, run: dict) -> None:
        for document in (render_markdown(run), render_html(run)):
            assert "no production write traffic" in document
            assert "not\nmeasured" in document or "not measured" in document

    def test_explain_timings_are_excluded_with_a_reason(self, run: dict) -> None:
        document = render_markdown(run)
        assert "per-tuple instrumentation" in document
        # The inflated 4360ms EXPLAIN figure must not be presented as a runtime.
        assert "4,360.2 ms" not in document.split("Reasoning trace")[0]

    def test_baseline_drift_is_reported(self, run: dict) -> None:
        document = render_markdown(run)
        assert "+1.8%" in document
        assert "not distinguishable from the machine" in document


class TestPlanSummary:
    def test_the_plan_shown_is_the_one_for_the_original_query(self, run: dict) -> None:
        summary, describes = plan_summary(run)
        assert describes == "the original query"
        assert summary["rows_read_by_sequential_scans"] == 1000000

    def test_a_plan_for_a_different_query_is_labelled_as_such(self, run: dict) -> None:
        """Otherwise a rewrite's scan counts get attributed to the original."""
        run["trace"][0]["tool_calls"][0]["arguments"]["sql"] = "SELECT 1"
        _, describes = plan_summary(run)
        assert describes == "a candidate query, not the original"

    def test_a_run_with_no_explain_still_renders(self, run: dict) -> None:
        run["trace"] = []
        assert plan_summary(run) == (None, "a candidate query, not the original")
        assert render_markdown(run)
        assert render_html(run)


# ---------------------------------------------------------------------------
# Degraded runs
# ---------------------------------------------------------------------------


class TestPartialRuns:
    def test_a_partial_run_says_so_without_hiding_its_measurements(self, run: dict) -> None:
        run["partial"] = True
        run["stop_reason"] = "iteration_limit"
        for document in (render_markdown(run), render_html(run)):
            assert "Partial run" in document
            assert "iteration_limit" in document
            assert PRODUCTION_DDL in document, "measured work survives a partial run"

    def test_an_error_is_surfaced(self, run: dict) -> None:
        run["error"] = "LLMError: provider unreachable"
        assert "provider unreachable" in render_markdown(run)
        assert "provider unreachable" in render_html(run)

    def test_a_recovered_recommendation_is_labelled(self, run: dict) -> None:
        run["recommendations"][0]["source"] = "recovered_from_accepted_hypotheses"
        assert "Recovered from an accepted hypothesis" in render_markdown(run)

    def test_an_empty_run_renders_without_raising(self) -> None:
        """A run that died on its first request still has to produce a document."""
        minimal = {"query_name": "q", "original_sql": "SELECT 1", "stop_reason": "error"}
        assert "OptiQuery" in render_markdown(minimal)
        assert "<!doctype html>" in render_html(minimal)


# ---------------------------------------------------------------------------
# HTML safety
# ---------------------------------------------------------------------------


class TestHtmlSafety:
    def test_sql_and_summaries_are_escaped(self, run: dict) -> None:
        """SQL is full of angle brackets, and summaries are model output."""
        run["original_sql"] = "SELECT * FROM t WHERE a < 5 AND b > 2"
        run["rejected"][0]["hypothesis"]["summary"] = "<script>alert(1)</script>"
        document = render_html(run)
        assert "<script>alert(1)</script>" not in document
        assert "&lt;script&gt;" in document
        assert "a &lt; 5 AND b &gt; 2" in document

    def test_the_document_is_self_contained(self, run: dict) -> None:
        """No external asset: the report has to open from a file:// URL."""
        document = render_html(run)
        assert "<style>" in document
        for marker in ("http://", "https://", "<script"):
            assert marker not in document

    def test_markdown_table_cells_cannot_break_the_table(self, run: dict) -> None:
        run["rejected"][0]["hypothesis"]["summary"] = "index on a|b columns"
        document = render_markdown(run)
        assert r"index on a\|b columns" in document


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


class TestFiles:
    def test_three_artifacts_are_written(self, run: dict, tmp_path) -> None:
        paths = write_reports(run, tmp_path)
        assert set(paths) == {"markdown", "html", "json"}
        for path in paths.values():
            assert path.exists() and path.stat().st_size > 0
        assert paths["markdown"].name == "q2_unindexed_fk_join.md"

    def test_the_json_round_trips_unchanged(self, run: dict, tmp_path) -> None:
        """The rendered numbers are rounded; the artifact is the source of truth."""
        paths = write_reports(run, tmp_path)
        assert json.loads(paths["json"].read_text(encoding="utf-8")) == run

    def test_the_terminal_summary_shows_wins_and_losses(self, run: dict) -> None:
        text = summarize_for_terminal(run)
        assert "1 accepted, 2 rejected" in text
        assert PRODUCTION_DDL in text
        assert "h3" in text and "DIFFERS" in text

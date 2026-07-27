"""benchmark() methodology and the accept/reject rules, against real data.

The tests that matter most here are the ones where the verifier says NO to
something fast. A system that only ever confirms the model's proposals is not a
verifier.
"""

from __future__ import annotations

import statistics

import pytest

from app.db import SqlGuardError
from app.shadow import ShadowDatabase
from app.tools import ToolContext
from app.verifier import (
    MIN_IMPROVEMENT_PCT,
    BenchmarkError,
    Hypothesis,
    Verdict,
    Verifier,
    benchmark,
)
from seed.measure_baseline import NamedQuery

pytestmark = pytest.mark.db


@pytest.fixture()
def verifier(ctx: ToolContext) -> Verifier:
    # 3 runs rather than 5 to keep the suite tractable; the methodology under
    # test is identical and the seed queries are stable to within a few percent.
    return Verifier(ctx.shadow, runs=3)


class TestBenchmarkMethodology:
    def test_discards_the_warm_up_and_reports_it_separately(
        self, ctx: ToolContext, seed_queries: dict[str, NamedQuery]
    ) -> None:
        """The discarded run is real work, reported but never counted."""
        result = benchmark(ctx.shadow, seed_queries["q2_unindexed_fk_join"].sql, runs=3)
        assert len(result.samples_ms) == 3
        assert result.discarded_warmup_ms > 0
        assert result.discarded_warmup_ms not in result.samples_ms

    def test_reports_the_median_not_the_mean(self, ctx: ToolContext) -> None:
        result = benchmark(ctx.shadow, "SELECT count(*) FROM users", runs=5)
        assert result.median_ms == pytest.approx(sorted(result.samples_ms)[2])

    def test_median_resists_a_single_outlier(self, ctx: ToolContext) -> None:
        """Why the median and not the mean.

        Takes the real samples, replaces the slowest with one 100x worse -- a
        checkpoint, a noisy neighbour, a stop-the-world pause -- and shows the
        median does not move while the mean is destroyed. On a 20% acceptance
        threshold, that difference decides whether a recommendation ships.
        """
        result = benchmark(ctx.shadow, "SELECT count(*) FROM users", runs=5)
        samples = sorted(result.samples_ms)
        poisoned = samples[:-1] + [samples[-1] * 100]

        assert statistics.median(poisoned) == result.median_ms
        assert statistics.fmean(poisoned) > result.mean_ms * 2

    def test_returns_row_count_and_checksum(self, ctx: ToolContext) -> None:
        result = benchmark(ctx.shadow, "SELECT id FROM users WHERE id <= 500", runs=2)
        assert result.row_count == 500
        assert len(result.checksum) == 64
        assert result.checksum_stable_across_runs is True

    def test_detects_a_nondeterministic_query(self, ctx: ToolContext) -> None:
        """LIMIT without a total ordering can return different rows each run.

        If it does, a checksum matching the original is coincidence, and the
        verifier must not treat it as evidence.
        """
        sql = (
            "SELECT id FROM order_items TABLESAMPLE SYSTEM (0.01) "
            "ORDER BY quantity LIMIT 5"
        )
        result = benchmark(ctx.shadow, sql, runs=4)
        assert result.checksum_stable_across_runs is False

    def test_rejects_a_mutation(self, ctx: ToolContext) -> None:
        with pytest.raises(SqlGuardError):
            benchmark(ctx.shadow, "UPDATE users SET is_active = false", runs=1)

    def test_refuses_to_checksum_more_rows_than_it_can_hold(
        self, ctx: ToolContext
    ) -> None:
        """Truncating would compare two prefixes and call them equal."""
        with pytest.raises(BenchmarkError, match="checksum limit"):
            benchmark(ctx.shadow, "SELECT id FROM order_items", runs=1, max_rows=1000)

    def test_statement_timeout_surfaces_as_an_error(self, ctx: ToolContext) -> None:
        """No silent excepts: a cancelled query is reported, not returned as 0ms."""
        impatient = ShadowDatabase(ctx.shadow._dsn, statement_timeout_ms=50)
        try:
            with pytest.raises(BenchmarkError, match="statement timeout"):
                benchmark(impatient, "SELECT count(*) FROM order_items", runs=1)
        finally:
            impatient.close()


class TestAcceptance:
    def test_accepts_a_real_index_win(
        self, verifier: Verifier, seed_queries: dict[str, NamedQuery]
    ) -> None:
        sql = seed_queries["q1_seq_scan_high_cardinality"].sql
        result = verifier.evaluate(
            sql,
            Hypothesis(
                hypothesis_id="h1",
                kind="index",
                summary="Index the high-cardinality sku column",
                index_ddls=("CREATE INDEX ix_v_oi_sku ON order_items (sku)",),
            ),
        )

        assert result.verdict is Verdict.ACCEPTED, result.reasons
        assert result.checksum_match is True
        assert result.improvement_pct > 95
        assert result.speedup > 50
        assert result.reasons == []

    def test_accepted_index_carries_size_and_write_amplification(
        self, verifier: Verifier, seed_queries: dict[str, NamedQuery]
    ) -> None:
        sql = seed_queries["q1_seq_scan_high_cardinality"].sql
        result = verifier.evaluate(
            sql,
            Hypothesis(
                hypothesis_id="h1",
                kind="index",
                summary="Index sku",
                index_ddls=("CREATE INDEX ix_v_oi_sku ON order_items (sku)",),
            ),
        )
        report = result.index_reports[0]
        assert report.size_bytes > 1_000_000
        assert report.build_ms > 0
        assert 0 < report.pct_of_table < 15
        assert report.oversized is False
        assert "must now also maintain this index" in report.write_amplification
        assert "ESTIMATED" in report.write_amplification
        assert report.production_ddl == (
            "CREATE INDEX CONCURRENTLY ix_v_oi_sku ON order_items (sku)"
        )

    def test_accepts_the_union_all_rewrite(
        self, verifier: Verifier, seed_queries: dict[str, NamedQuery]
    ) -> None:
        sql = seed_queries["q4_or_across_columns"].sql
        rewrite = """
            SELECT o.id, o.created_at, o.status, o.total_amount, u.email, oi.sku, oi.quantity
            FROM orders o
            JOIN users u ON u.id = o.user_id
            JOIN order_items oi ON oi.order_id = o.id
            WHERE u.email = 'Kiran.Dubois89332@Example.com'
            UNION ALL
            SELECT o.id, o.created_at, o.status, o.total_amount, u.email, oi.sku, oi.quantity
            FROM orders o
            JOIN users u ON u.id = o.user_id
            JOIN order_items oi ON oi.order_id = o.id
            WHERE o.tracking_number = 'TRK-911393315'
              AND u.email IS DISTINCT FROM 'Kiran.Dubois89332@Example.com'
            ORDER BY 1, 6
        """
        result = verifier.evaluate(
            sql,
            Hypothesis(
                hypothesis_id="h4",
                kind="index+rewrite",
                summary="Split the cross-table OR into two sargable arms",
                index_ddls=(
                    "CREATE INDEX ix_v_u_email ON users (email)",
                    "CREATE INDEX ix_v_o_track ON orders (tracking_number)",
                    "CREATE INDEX ix_v_o_user ON orders (user_id)",
                    "CREATE INDEX ix_v_oi_order ON order_items (order_id)",
                ),
                rewritten_sql=rewrite,
            ),
        )

        assert result.verdict is Verdict.ACCEPTED, result.reasons
        assert result.checksum_match is True
        assert len(result.index_reports) == 4


class TestRejection:
    def test_rejects_a_faster_rewrite_that_returns_different_rows(
        self, verifier: Verifier, seed_queries: dict[str, NamedQuery]
    ) -> None:
        """The headline case.

        Dropping lower() and indexing email_snapshot directly is ~1000x faster
        and returns zero rows instead of 15, because the column is stored mixed
        case. Runtime alone would score this as a triumph.
        """
        sql = seed_queries["q3_non_sargable_lower"].sql
        broken = sql.replace(
            "lower(o.email_snapshot) = 'paulo.lindqvist138671@example.com'",
            "o.email_snapshot = 'paulo.lindqvist138671@example.com'",
        )
        assert broken != sql

        result = verifier.evaluate(
            sql,
            Hypothesis(
                hypothesis_id="h3-bad",
                kind="index+rewrite",
                summary="Drop the lower() call and index the column directly",
                index_ddls=(
                    "CREATE INDEX ix_v_o_email ON orders (email_snapshot)",
                    "CREATE INDEX ix_v_oi_order ON order_items (order_id)",
                ),
                rewritten_sql=broken,
            ),
        )

        assert result.verdict is Verdict.REJECTED
        assert result.checksum_match is False
        # It really was enormously faster. That is exactly why runtime alone is
        # not a sufficient test.
        assert result.improvement_pct > MIN_IMPROVEMENT_PCT
        assert any("checksum differs" in reason for reason in result.reasons)
        assert result.baseline is not None and result.baseline.row_count == 15
        assert result.optimized is not None and result.optimized.row_count == 0

    def test_rejects_an_index_that_does_not_help_enough(
        self, verifier: Verifier, seed_queries: dict[str, NamedQuery]
    ) -> None:
        """Correct results, no meaningful speedup. Still a rejection."""
        sql = seed_queries["q1_seq_scan_high_cardinality"].sql
        result = verifier.evaluate(
            sql,
            Hypothesis(
                hypothesis_id="h-useless",
                kind="index",
                summary="Index a column the query never filters on",
                index_ddls=("CREATE INDEX ix_v_oi_wh ON order_items (warehouse_code)",),
            ),
        )

        assert result.verdict is Verdict.REJECTED
        assert result.checksum_match is True
        assert result.improvement_pct < MIN_IMPROVEMENT_PCT
        assert any("below the" in reason for reason in result.reasons)

    def test_rejects_a_union_all_rewrite_that_duplicates_rows(
        self, ctx: ToolContext, verifier: Verifier
    ) -> None:
        """Overlapping OR arms without an anti-predicate emit rows twice.

        Built against a tracking number belonging to the same user the other arm
        matches, so the two arms genuinely overlap.
        """
        row = ctx.primary.run_read_only(
            "SELECT o.tracking_number FROM orders o JOIN users u ON u.id = o.user_id "
            "WHERE u.email = 'Kiran.Dubois89332@Example.com' "
            "AND o.tracking_number IS NOT NULL ORDER BY o.id LIMIT 1"
        )
        assert row, "expected this user to have at least one shipped order"
        tracking = row[0][0]

        original = f"""
            SELECT o.id, o.total_amount, u.email
            FROM orders o JOIN users u ON u.id = o.user_id
            WHERE u.email = 'Kiran.Dubois89332@Example.com'
               OR o.tracking_number = '{tracking}'
            ORDER BY o.id
        """
        naive = f"""
            SELECT o.id, o.total_amount, u.email
            FROM orders o JOIN users u ON u.id = o.user_id
            WHERE u.email = 'Kiran.Dubois89332@Example.com'
            UNION ALL
            SELECT o.id, o.total_amount, u.email
            FROM orders o JOIN users u ON u.id = o.user_id
            WHERE o.tracking_number = '{tracking}'
            ORDER BY 1
        """
        result = verifier.evaluate(
            original,
            Hypothesis(
                hypothesis_id="h-dup",
                kind="index+rewrite",
                summary="UNION ALL without an anti-predicate on the second arm",
                index_ddls=(
                    "CREATE INDEX ix_v_u_email ON users (email)",
                    "CREATE INDEX ix_v_o_track ON orders (tracking_number)",
                    "CREATE INDEX ix_v_o_user ON orders (user_id)",
                ),
                rewritten_sql=naive,
            ),
        )

        assert result.verdict is Verdict.REJECTED
        assert result.checksum_match is False
        assert result.optimized is not None and result.baseline is not None
        assert result.optimized.row_count > result.baseline.row_count

    def test_flags_an_oversized_index_without_rejecting_it(
        self, verifier: Verifier
    ) -> None:
        """Size is a warning for whoever deploys it, not a veto."""
        sql = "SELECT id, email FROM users WHERE email = 'Kiran.Dubois89332@Example.com'"
        result = verifier.evaluate(
            sql,
            Hypothesis(
                hypothesis_id="h-big",
                kind="index",
                summary="Index users.email",
                index_ddls=("CREATE INDEX ix_v_u_email ON users (email)",),
            ),
        )

        assert result.verdict is Verdict.ACCEPTED, result.reasons
        report = result.index_reports[0]
        assert report.pct_of_table > 15
        assert report.oversized is True
        assert any("above the 15% threshold" in flag for flag in result.flags)
        assert report.write_amplification.startswith("OVERSIZED")

    def test_malformed_ddl_yields_an_error_verdict_not_a_crash(
        self, verifier: Verifier
    ) -> None:
        result = verifier.evaluate(
            "SELECT count(*) FROM users",
            Hypothesis(
                hypothesis_id="h-bad-ddl",
                kind="index",
                summary="Index a column that does not exist",
                index_ddls=("CREATE INDEX ix_v_nope ON users (no_such_column)",),
            ),
        )
        assert result.verdict is Verdict.ERROR
        assert any("CREATE INDEX failed" in reason for reason in result.reasons)


class TestHypothesisValidation:
    @pytest.mark.parametrize(
        "kwargs,message",
        [
            (dict(kind="index", summary="s"), "requires at least one index DDL"),
            (dict(kind="rewrite", summary="s"), "requires rewritten_sql"),
            (dict(kind="index", summary=" "), "summary is required"),
            (
                dict(kind="index", summary="s", index_ddls=("CREATE INDEX i ON t (c)",),
                     rewritten_sql="SELECT 1"),
                "must not carry a rewrite",
            ),
            (
                dict(kind="rewrite", summary="s", rewritten_sql="SELECT 1",
                     index_ddls=("CREATE INDEX i ON t (c)",)),
                "must not carry index DDL",
            ),
        ],
    )
    def test_rejects_incoherent_hypotheses(self, kwargs: dict, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            Hypothesis(hypothesis_id="h", **kwargs)


class TestBaselineDrift:
    def test_reports_drift_between_first_and_final_measurement(
        self, verifier: Verifier
    ) -> None:
        """The baseline is cached; drift turns that risk into a number."""
        sql = "SELECT count(*) FROM orders WHERE status = 'delivered'"
        verifier.baseline_for(sql)
        drift = verifier.check_baseline_drift(sql)

        assert drift["checksum_unchanged"] is True
        assert abs(drift["drift_pct"]) < 40  # generous; this is a smoke check

    def test_drift_without_a_baseline_is_an_error(self, verifier: Verifier) -> None:
        with pytest.raises(BenchmarkError, match="no cached baseline"):
            verifier.check_baseline_drift("SELECT 1")

"""Proof that shadow returns to its baseline between hypotheses.

If an index built for hypothesis 2 survives into hypothesis 3, hypothesis 3's
numbers describe a database nobody proposed. It gets reported as a win, and a
production deploy of just that change reproduces none of it.

Catalog bookkeeping alone is not proof -- bookkeeping can agree with itself
while being wrong. `test_a_leaked_index_would_change_the_measurement` establishes
the property behaviourally: it measures the same query before and after a
hypothesis that made it 3000x faster, and requires the second measurement to be
slow again.
"""

from __future__ import annotations

import psycopg
import pytest

from app.shadow import ShadowDatabase, ShadowIsolationError, fetch_index_records
from app.tools import ToolContext, list_indexes
from app.verifier import Hypothesis, Verifier, benchmark
from seed.measure_baseline import NamedQuery

pytestmark = pytest.mark.db

SKU_INDEX = "CREATE INDEX ix_iso_oi_sku ON order_items (sku)"


@pytest.fixture()
def verifier(ctx: ToolContext) -> Verifier:
    return Verifier(ctx.shadow, runs=3)


class TestResetRestoresTheBaseline:
    def test_list_indexes_matches_the_baseline_after_a_hypothesis(
        self, ctx: ToolContext, verifier: Verifier, seed_queries: dict[str, NamedQuery]
    ) -> None:
        """Asserted through the same tool the agent sees, not a private helper."""
        before = {index["name"] for index in list_indexes(ctx, database="shadow")["indexes"]}

        verifier.evaluate(
            seed_queries["q1_seq_scan_high_cardinality"].sql,
            Hypothesis(
                hypothesis_id="iso-1",
                kind="index",
                summary="Index sku",
                index_ddls=(SKU_INDEX,),
            ),
        )

        after = {index["name"] for index in list_indexes(ctx, database="shadow")["indexes"]}
        assert after == before

    def test_reset_runs_even_when_the_hypothesis_errors(
        self, ctx: ToolContext, verifier: Verifier
    ) -> None:
        """A failure partway through must not leave the first index behind."""
        before = {index["name"] for index in list_indexes(ctx, database="shadow")["indexes"]}

        result = verifier.evaluate(
            "SELECT count(*) FROM order_items",
            Hypothesis(
                hypothesis_id="iso-err",
                kind="index",
                summary="One good index, then one that cannot be built",
                index_ddls=(SKU_INDEX, "CREATE INDEX ix_iso_bad ON order_items (nope)"),
            ),
        )

        assert result.verdict.value == "ERROR"
        assert result.reset["dropped"] == ["ix_iso_oi_sku"]
        after = {index["name"] for index in list_indexes(ctx, database="shadow")["indexes"]}
        assert after == before

    def test_multiple_hypotheses_each_start_clean(
        self, ctx: ToolContext, verifier: Verifier, seed_queries: dict[str, NamedQuery]
    ) -> None:
        sql = seed_queries["q2_unindexed_fk_join"].sql
        first = verifier.evaluate(
            sql,
            Hypothesis(
                hypothesis_id="iso-a",
                kind="index",
                summary="orders(user_id) only",
                index_ddls=("CREATE INDEX ix_iso_o_user ON orders (user_id)",),
            ),
        )
        second = verifier.evaluate(
            sql,
            Hypothesis(
                hypothesis_id="iso-b",
                kind="index",
                summary="order_items(order_id) only",
                index_ddls=("CREATE INDEX ix_iso_oi_order ON order_items (order_id)",),
            ),
        )

        assert first.reset["matches_baseline"] is True
        assert second.reset["matches_baseline"] is True
        # Each hypothesis carries only its own index.
        assert [r.name for r in first.index_reports] == ["ix_iso_o_user"]
        assert [r.name for r in second.index_reports] == ["ix_iso_oi_order"]
        # Both were measured against the same cached baseline.
        assert first.baseline is not None and second.baseline is not None
        assert first.baseline.median_ms == second.baseline.median_ms


class TestLeakWouldBeVisible:
    def test_a_leaked_index_would_change_the_measurement(
        self, ctx: ToolContext, verifier: Verifier, seed_queries: dict[str, NamedQuery]
    ) -> None:
        """The behavioural proof, not a catalog comparison.

        Measure, run a hypothesis that makes the query ~3000x faster, reset,
        then measure again. If the index leaked, the second measurement would
        come back fast and every later hypothesis would inherit a head start it
        never earned.
        """
        sql = seed_queries["q1_seq_scan_high_cardinality"].sql

        before = benchmark(ctx.shadow, sql, runs=3)
        result = verifier.evaluate(
            sql,
            Hypothesis(
                hypothesis_id="iso-leak",
                kind="index",
                summary="Index sku",
                index_ddls=(SKU_INDEX,),
            ),
        )
        after = benchmark(ctx.shadow, sql, runs=3)

        assert result.optimized is not None
        assert result.speedup > 50, "the hypothesis should have been dramatically faster"
        # Back to the original cost, not the optimised one.
        assert after.median_ms > result.optimized.median_ms * 50
        assert after.median_ms == pytest.approx(before.median_ms, rel=0.35)
        assert after.checksum == before.checksum

    def test_reset_drops_an_index_it_never_created(self, ctx: ToolContext) -> None:
        """Reset works off the catalog, not off its own bookkeeping.

        A CREATE INDEX that failed after the index materialised -- a
        server-side cancellation, say -- leaves an index behind that the
        tracking list never recorded. Dropping only what was tracked would miss
        exactly the case that motivates the reset.
        """
        ctx.shadow.connect().execute("CREATE INDEX ix_iso_untracked ON users (city)")
        assert "ix_iso_untracked" not in ctx.shadow.created_index_names

        report = ctx.shadow.reset()

        assert report.dropped == ["ix_iso_untracked"]
        assert report.leaked == ["ix_iso_untracked"]
        assert report.matches_baseline is True
        names = {index["name"] for index in list_indexes(ctx, database="shadow")["indexes"]}
        assert "ix_iso_untracked" not in names


class TestResetFailsLoudly:
    def test_missing_baseline_index_raises(self, ctx: ToolContext, config) -> None:
        """A baseline index disappearing is unrecoverable, not a warning.

        Uses its own ShadowDatabase whose baseline includes a disposable index,
        so the shared session baseline is untouched.
        """
        private = ShadowDatabase(config.shadow_dsn, config.statement_timeout_ms)
        try:
            private.connect().execute("CREATE INDEX ix_iso_baseline ON users (city)")
            private.capture_baseline()
            assert "ix_iso_baseline" in private.baseline_index_names()

            private.connect().execute("DROP INDEX ix_iso_baseline")

            with pytest.raises(ShadowIsolationError, match="does not match its baseline"):
                private.reset()
        finally:
            with private.connect().cursor() as cur:
                cur.execute("DROP INDEX IF EXISTS ix_iso_baseline")
            private.close()

    def test_refuses_to_drop_a_constraint_backed_index(self, ctx: ToolContext) -> None:
        with pytest.raises(ShadowIsolationError, match="baseline"):
            ctx.shadow.drop_index("users_pkey")

    def test_verifier_requires_a_captured_baseline(self, config) -> None:
        """Without a baseline there is no definition of 'clean' to reset to."""
        private = ShadowDatabase(config.shadow_dsn, config.statement_timeout_ms)
        try:
            with pytest.raises(ValueError, match="baseline must be captured"):
                Verifier(private)
        finally:
            private.close()


class TestPrimaryUntouched:
    def test_primary_still_has_only_primary_keys(self, ctx: ToolContext) -> None:
        """After everything above, production must be exactly as it started."""
        indexes = list_indexes(ctx, database="primary")["indexes"]
        assert len(indexes) == 5
        assert all(index["is_primary"] for index in indexes)

    def test_shadow_and_primary_agree_at_rest(self, ctx: ToolContext) -> None:
        primary = {record.identity() for record in fetch_index_records(ctx.primary)}
        shadow = {record.identity() for record in fetch_index_records(ctx.shadow)}
        assert primary == shadow

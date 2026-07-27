"""Each tool, called directly. No agent involved.

Every test asserts on a value the tool actually read out of Postgres, not on the
shape of the dict. A test that only checks keys exist would pass against a tool
that returned constants.
"""

from __future__ import annotations

import json

import pytest

from app.db import SqlGuardError
from app.shadow import ShadowIsolationError
from app.tools import (
    TOOLS,
    ToolContext,
    ToolError,
    ToolRegistry,
    benchmark,
    check_shadow_parity,
    create_index_on_shadow,
    drop_index_on_shadow,
    explain_query,
    finish,
    get_schema,
    list_indexes,
)
from seed.measure_baseline import NamedQuery

pytestmark = pytest.mark.db

TEST_INDEX = "ix_test_oi_sku"


class TestGetSchema:
    def test_returns_all_seed_tables(self, ctx: ToolContext) -> None:
        result = get_schema(ctx)
        assert result["table_count"] == 5
        names = {table["table"] for table in result["tables"]}
        assert names == {"users", "products", "orders", "order_items", "reviews"}

    def test_row_counts_are_close_to_reality(self, ctx: ToolContext) -> None:
        result = get_schema(ctx, table="order_items")
        estimate = result["tables"][0]["estimated_row_count"]
        # reltuples after VACUUM ANALYZE; allow 1% drift, but not an order of
        # magnitude -- a wrong row count sends every later cost judgement astray.
        assert 2_970_000 <= estimate <= 3_030_000

    def test_reconstructed_ddl_includes_columns_and_primary_key(self, ctx: ToolContext) -> None:
        ddl = get_schema(ctx, table="orders")["tables"][0]["ddl"]
        assert ddl.startswith("CREATE TABLE orders (")
        assert "user_id bigint NOT NULL" in ddl
        assert "PRIMARY KEY (id)" in ddl

    def test_exposes_n_distinct_and_correlation(self, ctx: ToolContext) -> None:
        """The pg_stats numbers are the reason this tool exists."""
        columns = {
            column["name"]: column
            for column in get_schema(ctx, table="order_items")["tables"][0]["columns"]
        }

        sku_stats = columns["sku"]["stats"]
        # ~50k distinct SKUs. Postgres may report this as an absolute count or
        # as a negative ratio of the row count; both are meaningful, neither is
        # "unique".
        assert sku_stats["n_distinct"] > 1000 or sku_stats["n_distinct"] < -0.001
        assert sku_stats["null_frac"] == 0.0

        # order_id is written in order, so physical and logical order agree.
        # That near-1.0 correlation is exactly why an index on it is cheap.
        assert columns["order_id"]["stats"]["correlation"] > 0.99

    def test_column_types_are_reported(self, ctx: ToolContext) -> None:
        columns = {
            column["name"]: column
            for column in get_schema(ctx, table="users")["tables"][0]["columns"]
        }
        assert columns["email"]["type"] == "text"
        assert columns["country"]["type"] == "character(2)"
        assert columns["lifetime_value"]["type"] == "numeric(12,2)"
        assert columns["id"]["not_null"] is True

    def test_unknown_table_raises(self, ctx: ToolContext) -> None:
        with pytest.raises(ToolError, match="does not exist"):
            get_schema(ctx, table="no_such_table")

    def test_table_argument_is_validated_as_an_identifier(self, ctx: ToolContext) -> None:
        with pytest.raises(SqlGuardError):
            get_schema(ctx, table="users; DROP TABLE users")

    def test_unknown_database_raises(self, ctx: ToolContext) -> None:
        with pytest.raises(ToolError, match="unknown database"):
            get_schema(ctx, database="staging")


class TestListIndexes:
    def test_primary_has_primary_keys_only(self, ctx: ToolContext) -> None:
        """The seed ships without secondary indexes; that is what makes it slow."""
        result = list_indexes(ctx)
        assert result["index_count"] == 5
        assert all(index["is_primary"] for index in result["indexes"])

    def test_reports_size_and_usage_counters(self, ctx: ToolContext) -> None:
        index = list_indexes(ctx, table="order_items")["indexes"][0]
        assert index["name"] == "order_items_pkey"
        assert index["size_bytes"] > 1_000_000
        assert index["size_pretty"].endswith(("kB", "MB", "GB"))
        assert index["idx_scan"] >= 0  # from pg_stat_user_indexes
        assert index["backs_constraint"] is True

    def test_table_filter_narrows_results(self, ctx: ToolContext) -> None:
        result = list_indexes(ctx, table="users")
        assert result["index_count"] == 1
        assert result["indexes"][0]["table"] == "users"

    def test_shadow_matches_primary_at_baseline(self, ctx: ToolContext) -> None:
        parity = check_shadow_parity(ctx)
        assert parity["in_parity"] is True, parity


class TestExplainQuery:
    def test_analyze_returns_measured_times(
        self, ctx: ToolContext, seed_queries: dict[str, NamedQuery]
    ) -> None:
        sql = seed_queries["q1_seq_scan_high_cardinality"].sql
        result = explain_query(ctx, sql, analyze=True)
        summary = result["summary"]

        assert result["analyzed"] is True
        assert summary["execution_time_ms"] > 100
        assert summary["planning_time_ms"] is not None

    def test_analyze_exposes_buffers(
        self, ctx: ToolContext, seed_queries: dict[str, NamedQuery]
    ) -> None:
        sql = seed_queries["q1_seq_scan_high_cardinality"].sql
        nodes = explain_query(ctx, sql, analyze=True)["summary"]["nodes"]
        assert any("shared_hit_blocks" in node or "shared_read_blocks" in node for node in nodes)

    def test_identifies_the_wasteful_sequential_scan(
        self, ctx: ToolContext, seed_queries: dict[str, NamedQuery]
    ) -> None:
        """The diagnosis, in the planner's own numbers."""
        sql = seed_queries["q1_seq_scan_high_cardinality"].sql
        summary = explain_query(ctx, sql, analyze=True)["summary"]

        scans = summary["sequential_scans"]
        assert [scan["relation"] for scan in scans] == ["order_items", "order_items"]
        # Two full scans of a 3M-row table to return a few dozen rows.
        assert summary["rows_read_by_sequential_scans"] > 5_900_000
        assert summary["rows_discarded_by_sequential_scans"] > 5_900_000

    def test_counts_rows_read_by_filterless_scans(
        self, ctx: ToolContext, seed_queries: dict[str, NamedQuery]
    ) -> None:
        """A seq scan feeding a hash join removes no rows but still reads them all.

        Judging waste by rows_removed alone scores that scan as harmless.
        """
        sql = seed_queries["q2_unindexed_fk_join"].sql
        summary = explain_query(ctx, sql, analyze=True)["summary"]

        relations = {scan["relation"] for scan in summary["sequential_scans"]}
        assert {"orders", "order_items"} <= relations
        # orders (1M) + order_items (3M) + users (200k), none of it needed.
        assert summary["rows_read_by_sequential_scans"] > 4_000_000
        assert summary["rows_discarded_by_sequential_scans"] < 1_000_000

    def test_without_analyze_returns_estimates_only(
        self, ctx: ToolContext, seed_queries: dict[str, NamedQuery]
    ) -> None:
        sql = seed_queries["q3_non_sargable_lower"].sql
        summary = explain_query(ctx, sql, analyze=False)["summary"]
        assert summary["execution_time_ms"] is None
        assert summary["nodes"][0]["total_cost"] > 0

    def test_runs_on_shadow_not_primary(
        self, ctx: ToolContext, seed_queries: dict[str, NamedQuery]
    ) -> None:
        sql = seed_queries["q1_seq_scan_high_cardinality"].sql
        assert explain_query(ctx, sql, analyze=False)["database"] == "shadow"

    def test_rejects_a_mutation(self, ctx: ToolContext) -> None:
        with pytest.raises(SqlGuardError):
            explain_query(ctx, "DELETE FROM users WHERE id = -1", analyze=False)

    def test_rejects_a_data_modifying_cte(self, ctx: ToolContext) -> None:
        with pytest.raises(SqlGuardError):
            explain_query(
                ctx,
                "WITH g AS (DELETE FROM users WHERE id = -1 RETURNING id) SELECT * FROM g",
                analyze=False,
            )


class TestCreateAndDropIndexOnShadow:
    def test_reports_build_time_and_size(self, ctx: ToolContext) -> None:
        result = create_index_on_shadow(
            ctx, f"CREATE INDEX {TEST_INDEX} ON order_items (sku)"
        )
        assert result["index_name"] == TEST_INDEX
        assert result["table"] == "order_items"
        assert result["build_ms"] > 100  # a 3M row index build is not instant
        assert result["size_bytes"] > 1_000_000
        assert 0 < result["pct_of_table"] < 100
        assert TEST_INDEX in ctx.shadow.created_index_names

    def test_index_actually_exists_afterwards(self, ctx: ToolContext) -> None:
        create_index_on_shadow(ctx, f"CREATE INDEX {TEST_INDEX} ON order_items (sku)")
        names = {i["name"] for i in list_indexes(ctx, database="shadow")["indexes"]}
        assert TEST_INDEX in names

    def test_duplicate_name_is_refused(self, ctx: ToolContext) -> None:
        ddl = f"CREATE INDEX {TEST_INDEX} ON order_items (sku)"
        create_index_on_shadow(ctx, ddl)
        with pytest.raises(SqlGuardError, match="already exists"):
            create_index_on_shadow(ctx, ddl)

    def test_unknown_table_is_refused(self, ctx: ToolContext) -> None:
        with pytest.raises(SqlGuardError, match="does not exist"):
            create_index_on_shadow(ctx, "CREATE INDEX ix_nope ON no_such_table (x)")

    def test_invalid_column_surfaces_the_server_error(self, ctx: ToolContext) -> None:
        with pytest.raises(SqlGuardError, match="CREATE INDEX failed"):
            create_index_on_shadow(ctx, "CREATE INDEX ix_nope ON users (no_such_column)")

    def test_drop_removes_the_index(self, ctx: ToolContext) -> None:
        create_index_on_shadow(ctx, f"CREATE INDEX {TEST_INDEX} ON order_items (sku)")
        assert drop_index_on_shadow(ctx, TEST_INDEX) == {
            "index_name": TEST_INDEX,
            "dropped": True,
        }
        names = {i["name"] for i in list_indexes(ctx, database="shadow")["indexes"]}
        assert TEST_INDEX not in names

    def test_dropping_a_missing_index_is_reported_not_raised(self, ctx: ToolContext) -> None:
        result = drop_index_on_shadow(ctx, "ix_never_created")
        assert result["dropped"] is False
        assert "does not exist" in result["reason"]

    def test_refuses_to_drop_a_baseline_index(self, ctx: ToolContext) -> None:
        """Dropping a primary key would make shadow stop resembling primary."""
        with pytest.raises(ShadowIsolationError, match="baseline"):
            drop_index_on_shadow(ctx, "order_items_pkey")

    def test_reset_restores_the_baseline(self, ctx: ToolContext) -> None:
        before = {i["name"] for i in list_indexes(ctx, database="shadow")["indexes"]}
        create_index_on_shadow(ctx, f"CREATE INDEX {TEST_INDEX} ON order_items (sku)")
        report = ctx.shadow.reset()
        after = {i["name"] for i in list_indexes(ctx, database="shadow")["indexes"]}

        assert report.dropped == [TEST_INDEX]
        assert report.matches_baseline is True
        assert after == before


class TestBenchmarkTool:
    """The tool wrapper. The methodology itself is tested in test_verifier.py."""

    def test_returns_median_row_count_and_checksum(self, ctx: ToolContext) -> None:
        result = benchmark(ctx, "SELECT id, email FROM users WHERE id <= 100", runs=3)
        assert result["runs"] == 3
        assert len(result["samples_ms"]) == 3
        assert result["row_count"] == 100
        assert len(result["checksum"]) == 64
        assert result["discarded_warmup_ms"] > 0

    def test_rejects_a_mutation(self, ctx: ToolContext) -> None:
        with pytest.raises(SqlGuardError):
            benchmark(ctx, "DELETE FROM users WHERE id = -1")


class TestFinish:
    def test_accepts_a_valid_recommendation_set(self, ctx: ToolContext) -> None:
        result = finish(
            ctx,
            [
                {
                    "kind": "index",
                    "summary": "Index order_items.sku",
                    "ddl": "CREATE INDEX ix_oi_sku ON order_items (sku)",
                },
                {
                    "kind": "rewrite",
                    "summary": "UNION ALL instead of cross-table OR",
                    "rewritten_sql": "SELECT 1",
                },
            ],
        )
        assert result["status"] == "finished"
        assert result["recommendation_count"] == 2

    def test_empty_is_a_legitimate_answer(self, ctx: ToolContext) -> None:
        assert finish(ctx, [])["recommendation_count"] == 0

    @pytest.mark.parametrize(
        "bad,message",
        [
            ([{"kind": "vibes", "summary": "s"}], "kind must be one of"),
            ([{"kind": "index", "ddl": "CREATE INDEX ..."}], "summary is required"),
            ([{"kind": "index", "summary": "s"}], "no ddl"),
            ([{"kind": "rewrite", "summary": "s"}], "no rewritten_sql"),
            (["not an object"], "not an object"),
        ],
    )
    def test_rejects_malformed_recommendations(
        self, ctx: ToolContext, bad: list, message: str
    ) -> None:
        with pytest.raises(ToolError, match=message):
            finish(ctx, bad)


class TestRegistryContract:
    def test_every_tool_output_is_json_serialisable(self, ctx: ToolContext) -> None:
        registry = ToolRegistry(ctx)
        for name, arguments in [
            ("get_schema", {"table": "users"}),
            ("list_indexes", {"table": "users"}),
            ("explain_query", {"sql": "SELECT count(*) FROM users", "analyze": True}),
            ("finish", {"recommendations": []}),
        ]:
            payload = registry.call(name, arguments)
            assert json.loads(json.dumps(payload, default=str)) is not None

    def test_unknown_tool_raises(self, ctx: ToolContext) -> None:
        with pytest.raises(ToolError, match="unknown tool"):
            ToolRegistry(ctx).call("rm_rf")

    def test_call_json_serialises_errors_instead_of_raising(self, ctx: ToolContext) -> None:
        """Phase 4 needs failures as observations, not as exceptions.

        A rejected DDL that never re-enters the message history is a rejection
        the model cannot learn from, so it proposes the same thing again.
        """
        payload = json.loads(
            ToolRegistry(ctx).call_json(
                "create_index_on_shadow", {"ddl": "CREATE INDEX ix ON nope (x)"}
            )
        )
        assert payload["error"] == "SqlGuardError"
        assert "does not exist" in payload["message"]

    def test_finish_is_the_only_terminal_tool(self, ctx: ToolContext) -> None:
        registry = ToolRegistry(ctx)
        assert registry.is_terminal("finish") is True
        assert [name for name in TOOLS if registry.is_terminal(name)] == ["finish"]

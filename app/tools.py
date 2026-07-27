"""The tool surface the agent is allowed to touch.

Every function here returns a plain JSON-serialisable dict (there is a test that
asserts `json.dumps` accepts each one). `ToolRegistry.call` is the single
dispatch point that turns a name plus arguments into a JSON string for the
Anthropic tool-use loop in Phase 4.

Read/write separation is enforced by the type system, not by convention:

    get_schema, list_indexes      -> PrimaryDatabase or ShadowDatabase (reads)
    explain_query                 -> ShadowDatabase only
    create_index_on_shadow        -> ShadowDatabase only
    drop_index_on_shadow          -> ShadowDatabase only

`PrimaryDatabase` has no method that executes arbitrary SQL and connects as a
SELECT-only role, so there is no argument the agent could construct that would
route a write to production.

Why EXPLAIN runs on shadow and never on primary: shadow is a logical clone with
the same rows, the same statistics and the same planner settings, so its plan is
primary's plan. Running EXPLAIN ANALYZE on primary would execute the slow query
against production for no additional information. During a hypothesis, shadow
also carries the candidate index -- which is the whole point of asking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from app.db import (
    DatabaseConfig,
    PrimaryDatabase,
    SqlGuardError,
    _Database,
    assert_read_only,
    open_primary,
    validate_identifier,
)
from app.shadow import (
    ShadowDatabase,
    compare_index_sets,
    fetch_index_records,
    open_shadow,
)
from app.verifier import DEFAULT_RUNS, BenchmarkError
from app.verifier import benchmark as run_benchmark

# Tools whose return value ends the agent loop.
TERMINAL_TOOLS = frozenset({"finish"})


class ToolError(RuntimeError):
    """A tool was called with arguments it cannot honour.

    Raised rather than returned so the agent loop can decide whether to feed the
    message back as an observation (it does) or abort. Never swallowed.
    """


@dataclass
class ToolContext:
    """The two database handles every tool is bound to."""

    primary: PrimaryDatabase
    shadow: ShadowDatabase

    @classmethod
    def open(cls, config: DatabaseConfig | None = None) -> "ToolContext":
        config = config or DatabaseConfig.from_env()
        return cls(primary=open_primary(config), shadow=open_shadow(config))

    def close(self) -> None:
        self.primary.close()
        self.shadow.close()

    def resolve(self, database: str) -> _Database:
        """Pick a handle for a read-only tool by name."""
        if database == "primary":
            return self.primary
        if database == "shadow":
            return self.shadow
        raise ToolError(f"unknown database {database!r}; expected 'primary' or 'shadow'")

    def __enter__(self) -> "ToolContext":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Catalog queries
# ---------------------------------------------------------------------------

_TABLES_QUERY = """
SELECT c.relname
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' AND c.relkind = 'r'
  AND (%(table)s::text IS NULL OR c.relname = %(table)s)
ORDER BY c.relname
"""

_COLUMNS_QUERY = """
SELECT a.attname,
       format_type(a.atttypid, a.atttypmod) AS data_type,
       a.attnotnull,
       pg_get_expr(d.adbin, d.adrelid)      AS column_default,
       a.attnum
FROM pg_attribute a
JOIN pg_class c     ON c.oid = a.attrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum
WHERE n.nspname = 'public' AND c.relname = %s
  AND a.attnum > 0 AND NOT a.attisdropped
ORDER BY a.attnum
"""

_CONSTRAINTS_QUERY = """
SELECT con.conname, pg_get_constraintdef(con.oid)
FROM pg_constraint con
JOIN pg_class c     ON c.oid = con.conrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' AND c.relname = %s
ORDER BY con.contype, con.conname
"""

# reltuples is the planner's own row estimate. It is what every cost decision in
# this project is actually based on, so it is the honest number to show the
# agent -- and it is accurate to a fraction of a percent right after the
# VACUUM ANALYZE the seed loader runs. It is labelled as an estimate in the
# output because an exact count(*) on order_items costs ~600ms per call.
_TABLE_STATS_QUERY = """
SELECT c.reltuples::bigint                       AS estimated_rows,
       c.relpages                                AS heap_pages,
       pg_relation_size(c.oid)                   AS heap_bytes,
       pg_size_pretty(pg_relation_size(c.oid))   AS heap_pretty,
       pg_indexes_size(c.oid)                    AS index_bytes,
       pg_size_pretty(pg_indexes_size(c.oid))    AS index_pretty,
       pg_total_relation_size(c.oid)             AS total_bytes,
       pg_size_pretty(pg_total_relation_size(c.oid)) AS total_pretty
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' AND c.relname = %s
"""

# n_distinct: positive = absolute count of distinct values; negative = a ratio
# of the row count (-1 means every row is unique). correlation: how closely
# physical row order tracks the column's sort order, which is what decides
# whether an index scan on that column does sequential or random I/O.
_COLUMN_STATS_QUERY = """
SELECT attname, null_frac, avg_width, n_distinct, correlation,
       left(most_common_vals::text, 200) AS most_common_vals_truncated,
       most_common_freqs[1:5]            AS most_common_freqs_top5
FROM pg_stats
WHERE schemaname = 'public' AND tablename = %s
ORDER BY attname
"""


def _reconstruct_ddl(
    table: str,
    columns: Sequence[tuple],
    constraints: Sequence[tuple],
) -> str:
    """Rebuild a readable CREATE TABLE.

    Postgres has no server-side DDL dump function (pg_dump is a client), so
    this is assembled from pg_attribute and pg_constraint. It is for the agent
    to read, not to execute.
    """
    lines: list[str] = []
    for name, data_type, not_null, default, _ in columns:
        parts = [f"    {name} {data_type}"]
        if not_null:
            parts.append("NOT NULL")
        if default:
            parts.append(f"DEFAULT {default}")
        lines.append(" ".join(parts))
    for conname, condef in constraints:
        lines.append(f"    CONSTRAINT {conname} {condef}")
    body = ",\n".join(lines)
    return f"CREATE TABLE {table} (\n{body}\n);"


def get_schema(ctx: ToolContext, table: str | None = None, database: str = "primary") -> dict[str, Any]:
    """Reconstructed DDL, column types, row counts and planner statistics.

    The pg_stats numbers are the point of this tool. n_distinct and correlation
    are what separate "an index on this column will be selective and cheap to
    scan" from "an index here will be ignored by the planner" -- and the agent
    cannot derive either from the DDL.
    """
    db = ctx.resolve(database)
    if table is not None:
        validate_identifier(table)

    table_rows = db.fetch_all(_TABLES_QUERY, {"table": table})  # type: ignore[arg-type]
    if table is not None and not table_rows:
        raise ToolError(f"table {table!r} does not exist in schema 'public'")

    tables: list[dict[str, Any]] = []
    for (table_name,) in table_rows:
        columns = db.fetch_all(_COLUMNS_QUERY, (table_name,))
        constraints = db.fetch_all(_CONSTRAINTS_QUERY, (table_name,))
        stats_row = db.fetch_one(_TABLE_STATS_QUERY, (table_name,))
        assert stats_row is not None, f"pg_class row vanished for {table_name}"
        column_stats = db.fetch_all(_COLUMN_STATS_QUERY, (table_name,))

        stats_by_column = {
            row[0]: {
                "null_frac": float(row[1]),
                "avg_width_bytes": int(row[2]),
                "n_distinct": float(row[3]),
                "correlation": float(row[4]) if row[4] is not None else None,
                "most_common_vals_truncated": row[5],
                "most_common_freqs_top5": [float(f) for f in (row[6] or [])],
            }
            for row in column_stats
        }

        tables.append(
            {
                "table": table_name,
                "ddl": _reconstruct_ddl(table_name, columns, constraints),
                "estimated_row_count": int(stats_row[0]),
                "estimated_row_count_note": "planner estimate (pg_class.reltuples), not an exact count",
                "heap_pages": int(stats_row[1]),
                "heap_bytes": int(stats_row[2]),
                "heap_size": stats_row[3],
                "index_bytes": int(stats_row[4]),
                "index_size": stats_row[5],
                "total_bytes": int(stats_row[6]),
                "total_size": stats_row[7],
                "columns": [
                    {
                        "name": name,
                        "type": data_type,
                        "not_null": bool(not_null),
                        "default": default,
                        "position": int(position),
                        "stats": stats_by_column.get(
                            name, {"note": "no pg_stats row; run ANALYZE on this table"}
                        ),
                    }
                    for name, data_type, not_null, default, position in columns
                ],
                "constraints": [
                    {"name": conname, "definition": condef} for conname, condef in constraints
                ],
            }
        )

    return {"database": database, "table_count": len(tables), "tables": tables}


def list_indexes(ctx: ToolContext, table: str | None = None, database: str = "primary") -> dict[str, Any]:
    """Existing indexes with sizes and usage counters from pg_stat_user_indexes.

    idx_scan is included so the agent can see which indexes are already earning
    their keep. A proposal duplicating an index that is already there is a
    rejection the agent should be able to make for itself.
    """
    db = ctx.resolve(database)
    records = fetch_index_records(db, table)
    return {
        "database": database,
        "table_filter": table,
        "index_count": len(records),
        "total_index_bytes": sum(record.size_bytes for record in records),
        "indexes": [record.to_json() for record in records],
    }


# ---------------------------------------------------------------------------
# EXPLAIN
# ---------------------------------------------------------------------------

def _walk_plan(node: dict[str, Any], depth: int, out: list[dict[str, Any]]) -> None:
    entry: dict[str, Any] = {
        "depth": depth,
        "node_type": node.get("Node Type"),
        "relation": node.get("Relation Name"),
        "index_name": node.get("Index Name"),
        "plan_rows": node.get("Plan Rows"),
        "actual_rows": node.get("Actual Rows"),
        "loops": node.get("Actual Loops"),
        "actual_total_ms": node.get("Actual Total Time"),
        "total_cost": node.get("Total Cost"),
        "shared_hit_blocks": node.get("Shared Hit Blocks"),
        "shared_read_blocks": node.get("Shared Read Blocks"),
        "filter": node.get("Filter"),
        "index_cond": node.get("Index Cond"),
        "hash_cond": node.get("Hash Cond"),
        "rows_removed_by_filter": node.get("Rows Removed by Filter"),
    }
    out.append({key: value for key, value in entry.items() if value is not None})
    for child in node.get("Plans", []):
        _walk_plan(child, depth + 1, out)


def _summarize_plan(root: dict[str, Any]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    _walk_plan(root["Plan"], 0, nodes)

    seq_scans = [
        {
            "relation": node.get("relation"),
            "actual_rows": node.get("actual_rows"),
            "rows_removed_by_filter": node.get("rows_removed_by_filter", 0),
            "actual_total_ms": node.get("actual_total_ms"),
            "filter": node.get("filter"),
        }
        for node in nodes
        if node.get("node_type") == "Seq Scan"
    ]
    # The headline diagnostic. Both totals are needed: rows_removed counts what
    # a filter threw away, but a sequential scan feeding a hash join has no
    # filter and removes nothing while still reading the entire table. Reporting
    # only the discarded count would score that scan as harmless.
    rows_read = sum(
        int(scan.get("actual_rows") or 0) + int(scan.get("rows_removed_by_filter") or 0)
        for scan in seq_scans
    )
    rows_discarded = sum(int(scan.get("rows_removed_by_filter") or 0) for scan in seq_scans)

    return {
        "planning_time_ms": root.get("Planning Time"),
        "execution_time_ms": root.get("Execution Time"),
        "node_count": len(nodes),
        "nodes": nodes,
        "sequential_scans": seq_scans,
        "rows_read_by_sequential_scans": rows_read,
        "rows_discarded_by_sequential_scans": rows_discarded,
    }


def explain_query(ctx: ToolContext, sql: str, analyze: bool = True) -> dict[str, Any]:
    """EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) on shadow.

    With analyze=True the query really runs, so the returned times are measured
    rather than estimated -- but they are measured with per-tuple instrumentation
    attached, and that is not free. Seed query 2 executes in ~960ms and reports
    ~4360ms under EXPLAIN ANALYZE: a 4.5x inflation, concentrated in the nodes
    that touch the most tuples, which is exactly where an optimisation would show
    up. Use this to find out WHERE the time goes and WHICH plan Postgres chose.
    Never use it to decide whether something got faster -- that is benchmark()'s
    job, and it times the query with no instrumentation attached.
    """
    assert_read_only(sql)

    # BUFFERS requires ANALYZE in Postgres 16; asking for it without ANALYZE is
    # an error rather than a no-op.
    options = "ANALYZE, BUFFERS, COSTS, TIMING, FORMAT JSON" if analyze else "COSTS, FORMAT JSON"
    row = ctx.shadow.fetch_one(f"EXPLAIN ({options}) {sql}")
    if row is None:
        raise ToolError("EXPLAIN returned no rows")

    plan_root = row[0][0]
    return {
        "database": "shadow",
        "analyzed": analyze,
        "summary": _summarize_plan(plan_root),
        "plan_json": plan_root,
    }


# ---------------------------------------------------------------------------
# Shadow mutation
# ---------------------------------------------------------------------------

def create_index_on_shadow(ctx: ToolContext, ddl: str) -> dict[str, Any]:
    """Build an index on shadow. Returns build time and resulting size.

    Both numbers feed the report. Build time is the closest honest proxy this
    project has for what deploying the index costs; size is checked against the
    table in Phase 3, and flagged when it exceeds 15%.
    """
    result = ctx.shadow.create_index(ddl)
    payload = result.to_json()
    payload["note"] = (
        "built on shadow without CONCURRENTLY. A production deploy should use "
        "CREATE INDEX CONCURRENTLY, which takes longer but does not hold a write "
        "lock on the table."
    )
    return payload


def drop_index_on_shadow(ctx: ToolContext, name: str) -> dict[str, Any]:
    """Drop an index from shadow. Baseline indexes are refused."""
    return ctx.shadow.drop_index(name)


def benchmark(ctx: ToolContext, sql: str, runs: int = DEFAULT_RUNS) -> dict[str, Any]:
    """Time a query on shadow and fingerprint what it returned.

    Thin wrapper over app.verifier.benchmark, which is where the methodology
    lives: the first run is executed and discarded because it pays for reading
    the heap off disk; the reported figure is the median of the runs after it;
    and the returned checksum is what makes "same rows" checkable rather than
    assertable.

    Whatever indexes currently exist on shadow are in effect. Calling this after
    create_index_on_shadow measures the query WITH the candidate index, which is
    the point -- but the comparison against the original is only meaningful if
    shadow was at its baseline when the original was measured.
    """
    result = run_benchmark(ctx.shadow, sql, runs=runs)
    return result.to_json()


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------

_VALID_RECOMMENDATION_KINDS = frozenset({"index", "rewrite"})


def finish(ctx: ToolContext, recommendations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """End the agent loop and hand back the final recommendation list.

    Validated rather than trusted. An unparseable recommendation reaching the
    report is worse than an empty list: the empty list is an honest answer, and
    "this query is already about as fast as it gets" is a legitimate outcome.
    """
    if not isinstance(recommendations, Sequence) or isinstance(recommendations, (str, bytes)):
        raise ToolError("recommendations must be a list of objects")

    cleaned: list[dict[str, Any]] = []
    for position, item in enumerate(recommendations):
        if not isinstance(item, dict):
            raise ToolError(f"recommendation[{position}] is not an object")
        kind = item.get("kind")
        if kind not in _VALID_RECOMMENDATION_KINDS:
            raise ToolError(
                f"recommendation[{position}].kind must be one of "
                f"{sorted(_VALID_RECOMMENDATION_KINDS)}, got {kind!r}"
            )
        if not item.get("summary"):
            raise ToolError(f"recommendation[{position}].summary is required")
        if kind == "index" and not item.get("ddl"):
            raise ToolError(f"recommendation[{position}] is an index but has no ddl")
        if kind == "rewrite" and not item.get("rewritten_sql"):
            raise ToolError(f"recommendation[{position}] is a rewrite but has no rewritten_sql")
        cleaned.append(dict(item))

    return {
        "status": "finished",
        "recommendation_count": len(cleaned),
        "recommendations": cleaned,
    }


# ---------------------------------------------------------------------------
# Auxiliary (not exposed to the agent)
# ---------------------------------------------------------------------------

def check_shadow_parity(ctx: ToolContext) -> dict[str, Any]:
    """Confirm shadow is still a faithful stand-in for primary.

    Not a tool the agent calls; the runner calls it before trusting any number
    that came off shadow.
    """
    return compare_index_sets(fetch_index_records(ctx.primary), fetch_index_records(ctx.shadow))


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

ToolFunction = Callable[..., dict[str, Any]]

TOOLS: dict[str, ToolFunction] = {
    "get_schema": get_schema,
    "list_indexes": list_indexes,
    "explain_query": explain_query,
    "create_index_on_shadow": create_index_on_shadow,
    "drop_index_on_shadow": drop_index_on_shadow,
    "benchmark": benchmark,
    "finish": finish,
}


@dataclass
class ToolRegistry:
    """Name plus arguments in, JSON string out."""

    ctx: ToolContext

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        function = TOOLS.get(name)
        if function is None:
            raise ToolError(f"unknown tool {name!r}; available: {sorted(TOOLS)}")
        return function(self.ctx, **(arguments or {}))

    def call_json(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """As `call`, but returns the string the agent loop actually consumes.

        Errors are serialised rather than raised: a rejected DDL is information
        the model needs in its next turn, not a reason to tear down the run.
        Phase 4 depends on this -- an observation that never re-enters the
        message history is a mistake the model repeats.
        """
        try:
            return json.dumps(self.call(name, arguments), default=str)
        except (ToolError, SqlGuardError, BenchmarkError, TypeError) as exc:
            return json.dumps({"error": type(exc).__name__, "message": str(exc)})

    def is_terminal(self, name: str) -> bool:
        return name in TERMINAL_TOOLS

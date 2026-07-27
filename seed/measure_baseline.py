"""Measure the seed workload against the primary database.

This is a Phase 1 sanity check, not the verifier. Its only job is to prove the
four seed queries are genuinely slow before anything gets built on top of them.
The real measurement path -- checksums, index isolation, accept/reject rules --
lands in app/verifier.py in Phase 3, and this file will keep only its narrower
role of confirming the seed data still produces the plans it is supposed to.

It does already follow the two timing rules that matter, because a baseline
measured any other way would be worthless as a comparison point:

  * the first execution is discarded (it reads from disk; every later one hits
    shared_buffers, so run 1 vs run 2 measures the page cache, not the query)
  * the reported number is the MEDIAN of the remaining runs, not the mean
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import time
from pathlib import Path
from typing import NamedTuple, Sequence

import psycopg

SEED_DIR = Path(__file__).resolve().parent
QUERIES_PATH = SEED_DIR / "slow_queries.sql"

NAME_MARKER = re.compile(r"^--\s*name:\s*(\S+)\s*$", re.MULTILINE)

SLOW_THRESHOLD_MS = 800.0


class NamedQuery(NamedTuple):
    name: str
    sql: str


class Timing(NamedTuple):
    name: str
    median_ms: float
    min_ms: float
    max_ms: float
    runs_ms: list[float]
    row_count: int
    plan_summary: str


def load_named_queries(path: Path = QUERIES_PATH) -> list[NamedQuery]:
    """Split slow_queries.sql on its `-- name:` markers."""
    text = path.read_text(encoding="utf-8")
    matches = list(NAME_MARKER.finditer(text))
    if not matches:
        raise ValueError(f"{path} contains no '-- name:' markers")

    queries: list[NamedQuery] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[start:end]
        # Strip comment lines; keep the statement itself.
        statement = "\n".join(
            line for line in block.splitlines() if not line.lstrip().startswith("--")
        ).strip()
        if statement.endswith(";"):
            statement = statement[:-1]
        if not statement:
            raise ValueError(f"query block '{match.group(1)}' is empty")
        queries.append(NamedQuery(name=match.group(1), sql=statement))
    return queries


def plan_summary(conn: psycopg.Connection, sql: str) -> str:
    """One-line description of the top-level plan shape, for the report table."""
    with conn.cursor() as cur:
        cur.execute(f"EXPLAIN (FORMAT JSON) {sql}")
        row = cur.fetchone()
        assert row is not None
        plan = row[0][0]["Plan"]

    nodes: list[str] = []

    def walk(node: dict) -> None:
        label = node["Node Type"]
        relation = node.get("Relation Name")
        if relation:
            label = f"{label} on {relation}"
        nodes.append(label)
        for child in node.get("Plans", []):
            walk(child)

    walk(plan)
    interesting = [n for n in nodes if "Seq Scan" in n or "Hash Join" in n or "Nested Loop" in n]
    return " + ".join(interesting) if interesting else nodes[0]


def time_query(conn: psycopg.Connection, query: NamedQuery, runs: int) -> Timing:
    def execute_once() -> tuple[float, int]:
        with conn.cursor() as cur:
            started = time.perf_counter()
            cur.execute(query.sql)
            rows = cur.fetchall()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
        return elapsed_ms, len(rows)

    # Discarded warm-up: this run pays for reading the heap off disk.
    _, row_count = execute_once()

    samples: list[float] = []
    for _ in range(runs):
        elapsed_ms, row_count = execute_once()
        samples.append(elapsed_ms)

    return Timing(
        name=query.name,
        median_ms=statistics.median(samples),
        min_ms=min(samples),
        max_ms=max(samples),
        runs_ms=samples,
        row_count=row_count,
        plan_summary=plan_summary(conn, query.sql),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Time the seed workload.")
    parser.add_argument("--runs", type=int, default=5, help="timed runs after the discarded warm-up")
    parser.add_argument("--dsn", default=None, help="override PRIMARY_DSN")
    args = parser.parse_args(argv)

    dsn = args.dsn or os.environ.get(
        "PRIMARY_DSN", "postgresql://optiquery:optiquery@localhost:55432/optiquery"
    )

    queries = load_named_queries()
    results: list[Timing] = []

    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '120s'")
            cur.execute("SHOW max_parallel_workers_per_gather")
            row = cur.fetchone()
            assert row is not None
            parallel = row[0]
        print(f"max_parallel_workers_per_gather = {parallel}")
        print(f"1 discarded warm-up run + {args.runs} timed runs per query, median reported\n")

        for query in queries:
            results.append(time_query(conn, query, args.runs))

    header = f"{'query':<34}{'median':>10}{'min':>10}{'max':>10}{'rows':>8}  plan"
    print(header)
    print("-" * (len(header) + 34))
    for result in results:
        print(
            f"{result.name:<34}"
            f"{result.median_ms:>9.1f}ms"
            f"{result.min_ms:>9.1f}ms"
            f"{result.max_ms:>9.1f}ms"
            f"{result.row_count:>8}  {result.plan_summary}"
        )

    print()
    too_fast = [r for r in results if r.median_ms < SLOW_THRESHOLD_MS]
    for result in results:
        verdict = "OK " if result.median_ms >= SLOW_THRESHOLD_MS else "FAST"
        print(
            f"  [{verdict}] {result.name:<34} {result.median_ms:>8.1f}ms "
            f"(threshold {SLOW_THRESHOLD_MS:.0f}ms)  samples="
            + ", ".join(f"{s:.0f}" for s in result.runs_ms)
        )

    if too_fast:
        print(
            f"\n{len(too_fast)} of {len(results)} seed queries are under the "
            f"{SLOW_THRESHOLD_MS:.0f}ms threshold; they are not worth optimising."
        )
        return 1
    print(f"\nall {len(results)} seed queries exceed {SLOW_THRESHOLD_MS:.0f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

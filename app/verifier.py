"""The verifier. The LLM proposes; this module decides.

Nothing reaches the report without passing through here, and everything here is
deterministic. There are three ideas in this file and each exists because the
obvious alternative is quietly wrong.

**Timing.** The first execution of a query reads pages off disk that every later
execution finds in shared_buffers. Comparing run 1 against run 2 measures the
page cache, not the query, and reliably produces a 3-10x "speedup" from doing
nothing at all. So the first run is executed and thrown away. The reported
number is the MEDIAN of the runs that follow, not the mean: one scheduler
hiccup in five samples moves a mean enough to flip a 20% threshold, and moves a
median not at all.

Timing is wall-clock around execute+fetch, with no instrumentation attached.
EXPLAIN ANALYZE is not used here -- Phase 2 measured it inflating seed query 2
from 959ms to 4363ms, concentrated in exactly the high-tuple-count nodes an
optimisation would change.

**Equivalence.** Row counts are not equivalence. `SELECT * FROM orders WHERE
lower(email) = 'x'` and `SELECT * FROM orders WHERE email = 'x'` can both return
15 rows and disagree about which 15. Every row is serialised to a canonical,
unambiguously-framed string; the strings are sorted so that a plan producing the
same rows in a different order is not a false failure; the concatenation is
sha256'd. A rewrite that changes results is a bug, not an optimisation.

**Attribution.** An index built for one hypothesis that survives into the next
makes the next hypothesis's numbers describe a database nobody proposed. After
every hypothesis, shadow is returned to its baseline index set and the catalog
is re-read to prove it -- see app/shadow.py.
"""

from __future__ import annotations

import hashlib
import statistics
import time
from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Literal, Sequence
from uuid import UUID

import psycopg

from app.db import SqlGuardError, assert_read_only
from app.shadow import IndexBuildResult, ShadowDatabase

# A recommendation must beat the original by at least this much to ship.
MIN_IMPROVEMENT_PCT = 20.0

# An index larger than this share of its table's heap gets flagged in the
# report. Not a rejection -- a 20% index that turns 1.5s into 1ms is usually
# still worth it -- but it is a number the person deploying it should see.
OVERSIZED_INDEX_PCT = 15.0

DEFAULT_RUNS = 5

# Checksumming requires every row in memory at once. Rather than silently
# truncating (which would compare two prefixes and call them equal) or streaming
# without sorting (which would make ordering differences fail), refuse loudly.
DEFAULT_MAX_ROWS = 250_000


class BenchmarkError(RuntimeError):
    """A benchmark could not be completed. Never swallowed."""


class Verdict(str, Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"


# ---------------------------------------------------------------------------
# Canonical row serialisation
# ---------------------------------------------------------------------------

def _canonical_decimal(value: Decimal) -> str:
    """Scale-independent text for a numeric.

    Decimal('10.00') and Decimal('10.0') are the same number carrying different
    scale, and a rewrite can legitimately change scale -- summing in a different
    order, or a numeric widened by a join. Comparing their raw repr would reject
    a correct rewrite. normalize() collapses the scale; format 'f' keeps the
    result out of exponent notation, so 1E+2 renders as 100.
    """
    if value.is_nan():
        return "NaN"
    if value.is_infinite():
        return "Infinity" if value > 0 else "-Infinity"
    return format(value.normalize(), "f")


def canonical_value(value: Any) -> str:
    """One value as a tagged, type-stable string.

    The tag matters as much as the text. Without it, the string 'NULL', the
    integer 0 and the boolean False could all serialise to text that collides,
    and two different result sets would hash the same. NULL gets its own tag and
    an empty body, so it cannot be spelled by any non-NULL value.
    """
    if value is None:
        return "N:"
    # bool before int: bool IS an int in Python, and True would serialise as 1.
    if isinstance(value, bool):
        return f"B:{'1' if value else '0'}"
    if isinstance(value, int):
        return f"I:{value:d}"
    if isinstance(value, Decimal):
        return f"D:{_canonical_decimal(value)}"
    if isinstance(value, float):
        # repr() round-trips exactly. Note that float8 aggregates are not
        # associative, so a plan change that reorders a sum() over float8 can
        # produce a genuinely different value here. That is reported as a
        # mismatch, which is the correct answer -- the results really do differ.
        return f"F:{value!r}"
    if isinstance(value, str):
        return f"S:{value}"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"X:{bytes(value).hex()}"
    if isinstance(value, datetime):
        return f"T:{value.isoformat()}"
    if isinstance(value, date):
        return f"d:{value.isoformat()}"
    if isinstance(value, dt_time):
        return f"t:{value.isoformat()}"
    if isinstance(value, timedelta):
        return f"i:{value.total_seconds()!r}"
    if isinstance(value, UUID):
        return f"U:{value}"
    if isinstance(value, (list, tuple)):
        return "A:" + "".join(_frame(canonical_value(item)) for item in value)
    if isinstance(value, dict):
        # Sorted keys: jsonb does not preserve key order, and two equal jsonb
        # values must not hash differently because psycopg handed them back in a
        # different order.
        return "J:" + "".join(
            _frame(canonical_value(key)) + _frame(canonical_value(value[key]))
            for key in sorted(value, key=str)
        )
    # Postgres types without dedicated handling (ranges, network addresses,
    # composites). str() is stable for all of psycopg's built-in adapters. The
    # type name is included so two different types with the same text do not
    # collide.
    return f"?{type(value).__name__}:{value}"


def _frame(encoded: str) -> str:
    """Length-prefix a fragment so concatenation is unambiguous.

    Without framing, the row ('a', 'b') and the row ('ab',) serialise to the
    same characters under any fixed separator that could itself occur in the
    data. A length prefix removes the question.
    """
    return f"{len(encoded)}|{encoded}"


def canonical_row(row: Sequence[Any]) -> str:
    return "".join(_frame(canonical_value(value)) for value in row)


@dataclass(frozen=True)
class Checksum:
    """Two digests over the same rows.

    `sorted_digest` is the equivalence test: rows sorted before hashing, so a
    different plan emitting the same rows in a different order is not a false
    failure.

    `ordered_digest` is reported but never gates acceptance. When the original
    query has an ORDER BY, its output order is part of what the caller asked
    for, and a rewrite that scrambles it is worth knowing about even though the
    row set is identical.
    """

    sorted_digest: str
    ordered_digest: str
    row_count: int

    def to_json(self) -> dict[str, Any]:
        return {
            "sha256": self.sorted_digest,
            "sha256_order_sensitive": self.ordered_digest,
            "row_count": self.row_count,
        }


def result_checksum(rows: Sequence[Sequence[Any]]) -> Checksum:
    serialized = [canonical_row(row) for row in rows]

    ordered = hashlib.sha256()
    for line in serialized:
        ordered.update(f"{len(line)}#{line}".encode("utf-8"))

    sorted_hash = hashlib.sha256()
    for line in sorted(serialized):
        sorted_hash.update(f"{len(line)}#{line}".encode("utf-8"))

    return Checksum(
        sorted_digest=sorted_hash.hexdigest(),
        ordered_digest=ordered.hexdigest(),
        row_count=len(rows),
    )


# ---------------------------------------------------------------------------
# benchmark()
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkResult:
    sql: str
    runs: int
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float
    samples_ms: tuple[float, ...]
    discarded_warmup_ms: float
    row_count: int
    checksum: str
    ordered_checksum: str
    checksum_stable_across_runs: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "runs": self.runs,
            "median_ms": round(self.median_ms, 2),
            "mean_ms": round(self.mean_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "stdev_ms": round(self.stdev_ms, 2),
            "samples_ms": [round(sample, 2) for sample in self.samples_ms],
            "discarded_warmup_ms": round(self.discarded_warmup_ms, 2),
            "row_count": self.row_count,
            "checksum": self.checksum,
            "checksum_order_sensitive": self.ordered_checksum,
            "checksum_stable_across_runs": self.checksum_stable_across_runs,
            "method": (
                "first run discarded (page cache), median of the remaining runs, "
                "wall-clock around execute+fetch with no EXPLAIN instrumentation"
            ),
        }


def benchmark(
    db: ShadowDatabase,
    sql: str,
    runs: int = DEFAULT_RUNS,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> BenchmarkResult:
    """Time a query honestly and fingerprint what it returned.

    The warm-up execution is real work whose timing is discarded: it pays for
    reading the heap off disk, and every run after it finds those pages in
    shared_buffers. Reporting it, or averaging it in, turns "the page cache
    warmed up" into "the query got 4x faster".

    Each timed run's rows are fingerprinted *after* the clock stops, so
    serialisation cost never lands inside a measurement. Comparing those
    per-run digests catches a query that is not deterministic -- one whose
    checksum matching the original is luck rather than equivalence.
    """
    assert_read_only(sql)
    if runs < 1:
        raise BenchmarkError(f"runs must be >= 1, got {runs}")

    conn = db.connect()

    def execute_once() -> tuple[float, list[tuple]]:
        with conn.cursor() as cur:
            started = time.perf_counter()
            try:
                cur.execute(sql)  # type: ignore[arg-type]
                rows = cur.fetchall()
            except psycopg.errors.QueryCanceled as exc:
                raise BenchmarkError(
                    f"query exceeded the {db._statement_timeout_ms}ms statement "
                    f"timeout and was cancelled: {exc}"
                ) from exc
            except psycopg.Error as exc:
                raise BenchmarkError(f"query failed during benchmark: {exc}") from exc
            elapsed_ms = (time.perf_counter() - started) * 1000.0
        return elapsed_ms, rows

    warmup_ms, warmup_rows = execute_once()
    if len(warmup_rows) > max_rows:
        raise BenchmarkError(
            f"query returned {len(warmup_rows):,} rows, above the {max_rows:,} row "
            "checksum limit. Every row must be held in memory to be sorted and "
            "hashed; truncating would compare two prefixes and call them equal. "
            "Add a LIMIT, or raise max_rows deliberately."
        )

    digests = [result_checksum(warmup_rows)]
    samples: list[float] = []
    last_rows = warmup_rows

    for _ in range(runs):
        elapsed_ms, rows = execute_once()
        samples.append(elapsed_ms)
        # Outside the timing window on purpose.
        digests.append(result_checksum(rows))
        last_rows = rows

    stable = all(digest.sorted_digest == digests[0].sorted_digest for digest in digests)

    return BenchmarkResult(
        sql=sql,
        runs=runs,
        median_ms=statistics.median(samples),
        mean_ms=statistics.fmean(samples),
        min_ms=min(samples),
        max_ms=max(samples),
        stdev_ms=statistics.stdev(samples) if len(samples) > 1 else 0.0,
        samples_ms=tuple(samples),
        discarded_warmup_ms=warmup_ms,
        row_count=len(last_rows),
        checksum=digests[-1].sorted_digest,
        ordered_checksum=digests[-1].ordered_digest,
        checksum_stable_across_runs=stable,
    )


# ---------------------------------------------------------------------------
# Hypotheses
# ---------------------------------------------------------------------------

HypothesisKind = Literal["index", "rewrite", "index+rewrite"]


@dataclass(frozen=True)
class Hypothesis:
    """One proposed change, to be applied and measured as a unit."""

    hypothesis_id: str
    kind: HypothesisKind
    summary: str
    index_ddls: tuple[str, ...] = ()
    rewritten_sql: str | None = None

    def __post_init__(self) -> None:
        if not self.summary.strip():
            raise ValueError("hypothesis summary is required")
        if self.kind in ("index", "index+rewrite") and not self.index_ddls:
            raise ValueError(f"kind={self.kind!r} requires at least one index DDL")
        if self.kind in ("rewrite", "index+rewrite") and not self.rewritten_sql:
            raise ValueError(f"kind={self.kind!r} requires rewritten_sql")
        if self.kind == "index" and self.rewritten_sql:
            raise ValueError("kind='index' must not carry a rewrite; use 'index+rewrite'")
        if self.kind == "rewrite" and self.index_ddls:
            raise ValueError("kind='rewrite' must not carry index DDL; use 'index+rewrite'")

    def to_json(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "kind": self.kind,
            "summary": self.summary,
            "index_ddls": list(self.index_ddls),
            "rewritten_sql": self.rewritten_sql,
        }


@dataclass(frozen=True)
class IndexReport:
    name: str
    table: str
    ddl: str
    production_ddl: str
    build_ms: float
    size_bytes: int
    size_pretty: str
    table_size_bytes: int
    table_size_pretty: str
    pct_of_table: float
    oversized: bool
    write_amplification: str

    def to_json(self) -> dict[str, Any]:
        return {
            "index_name": self.name,
            "table": self.table,
            "ddl": self.ddl,
            "production_ddl": self.production_ddl,
            "build_ms": round(self.build_ms, 1),
            "size_bytes": self.size_bytes,
            "size_pretty": self.size_pretty,
            "table_size_bytes": self.table_size_bytes,
            "table_size_pretty": self.table_size_pretty,
            "pct_of_table": round(self.pct_of_table, 2),
            "oversized": self.oversized,
            "oversize_threshold_pct": OVERSIZED_INDEX_PCT,
            "write_amplification": self.write_amplification,
        }


def _index_report(build: IndexBuildResult) -> IndexReport:
    oversized = build.pct_of_table > OVERSIZED_INDEX_PCT

    # Stated, not estimated with an invented multiplier. Shadow holds production
    # data volume but takes no production write traffic, so there is no honest
    # way to measure the ongoing maintenance cost here -- only to name it and
    # quantify the parts that were measured.
    amplification = (
        f"Every INSERT into {build.table}, every DELETE from it, and every UPDATE "
        f"touching the indexed expression must now also maintain this index. It "
        f"adds {build.size_pretty} ({build.pct_of_table:.1f}% of the "
        f"{build.table_size_pretty} heap) to the write path and to every backup. "
        f"Build took {build.build_ms:.0f}ms on shadow with an exclusive lock; use "
        f"CREATE INDEX CONCURRENTLY in production, which takes longer but does not "
        f"block writes. This cost is ESTIMATED from size and build time, not "
        f"measured: shadow has production data volume but no production write "
        f"concurrency."
    )
    if oversized:
        amplification = (
            f"OVERSIZED: this index is {build.pct_of_table:.1f}% of its table, above "
            f"the {OVERSIZED_INDEX_PCT:.0f}% threshold. " + amplification
        )

    return IndexReport(
        name=build.name,
        table=build.table,
        ddl=build.ddl,
        production_ddl=_to_production_ddl(build.ddl),
        build_ms=build.build_ms,
        size_bytes=build.size_bytes,
        size_pretty=build.size_pretty,
        table_size_bytes=build.table_size_bytes,
        table_size_pretty=build.table_size_pretty,
        pct_of_table=build.pct_of_table,
        oversized=oversized,
        write_amplification=amplification,
    )


def _to_production_ddl(ddl: str) -> str:
    """Add CONCURRENTLY back for the deployable form.

    Shadow builds without it because it is faster and shadow has no concurrent
    traffic to protect. Production does.
    """
    lowered = ddl.lower()
    marker = "unique index " if "create unique index" in lowered else "index "
    position = lowered.index(marker) + len(marker)
    return f"{ddl[:position]}CONCURRENTLY {ddl[position:]}"


@dataclass
class HypothesisResult:
    hypothesis: Hypothesis
    verdict: Verdict
    baseline: BenchmarkResult | None
    optimized: BenchmarkResult | None
    index_reports: list[IndexReport] = field(default_factory=list)
    checksum_match: bool = False
    ordered_checksum_match: bool = False
    improvement_pct: float = 0.0
    speedup: float = 0.0
    reasons: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    reset: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.verdict is Verdict.ACCEPTED

    def to_json(self) -> dict[str, Any]:
        return {
            "hypothesis": self.hypothesis.to_json(),
            "verdict": self.verdict.value,
            "baseline": self.baseline.to_json() if self.baseline else None,
            "optimized": self.optimized.to_json() if self.optimized else None,
            "indexes": [report.to_json() for report in self.index_reports],
            "checksum_match": self.checksum_match,
            "ordered_checksum_match": self.ordered_checksum_match,
            "improvement_pct": round(self.improvement_pct, 2),
            "speedup": round(self.speedup, 2),
            "min_improvement_pct": MIN_IMPROVEMENT_PCT,
            "reasons": self.reasons,
            "flags": self.flags,
            "shadow_reset": self.reset,
        }


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class Verifier:
    """Applies hypotheses to shadow, measures them, and rules on them.

    Holds no opinions the model can argue with. A hypothesis is accepted only
    when the numbers say so.
    """

    def __init__(
        self,
        shadow: ShadowDatabase,
        runs: int = DEFAULT_RUNS,
        max_rows: int = DEFAULT_MAX_ROWS,
    ) -> None:
        if not shadow.baseline_captured:
            raise ValueError(
                "shadow baseline must be captured before verification; without it "
                "there is no definition of 'clean' to reset to"
            )
        self.shadow = shadow
        self.runs = runs
        self.max_rows = max_rows
        self._baselines: dict[str, BenchmarkResult] = {}

    # -- baseline ----------------------------------------------------------

    def baseline_for(self, sql: str) -> BenchmarkResult:
        """Benchmark the original query with shadow at its baseline index set.

        Cached: the baseline cannot change between hypotheses, because every
        hypothesis is required to hand shadow back in baseline state. That
        requirement is asserted here rather than assumed.
        """
        self.shadow.assert_matches_baseline()
        cached = self._baselines.get(sql)
        if cached is None:
            cached = benchmark(self.shadow, sql, self.runs, self.max_rows)
            self._baselines[sql] = cached
        return cached

    def check_baseline_drift(self, sql: str) -> dict[str, Any]:
        """Re-measure the cached baseline to expose machine drift.

        The baseline is measured once and reused, so a machine that slows down
        over a long run would make later hypotheses look better than they are.
        Calling this at the end of a run turns that risk into a reported number
        instead of a silent bias.
        """
        original = self._baselines.get(sql)
        if original is None:
            raise BenchmarkError("no cached baseline for this SQL")
        self.shadow.assert_matches_baseline()
        again = benchmark(self.shadow, sql, self.runs, self.max_rows)
        drift = (again.median_ms - original.median_ms) / original.median_ms * 100.0
        return {
            "first_median_ms": round(original.median_ms, 2),
            "final_median_ms": round(again.median_ms, 2),
            "drift_pct": round(drift, 2),
            "checksum_unchanged": again.checksum == original.checksum,
        }

    # -- evaluation --------------------------------------------------------

    def evaluate(self, original_sql: str, hypothesis: Hypothesis) -> HypothesisResult:
        """Apply, measure, judge, and unconditionally reset shadow.

        The reset in the `finally` block is allowed to raise. If shadow cannot
        be returned to its baseline, every measurement taken after this point
        would be unattributable, so failing the whole run is the correct
        outcome -- louder and cheaper than a report full of numbers that
        describe a database nobody proposed.
        """
        baseline = self.baseline_for(original_sql)

        index_reports: list[IndexReport] = []
        optimized: BenchmarkResult | None = None
        error: str | None = None

        try:
            for ddl in hypothesis.index_ddls:
                index_reports.append(_index_report(self.shadow.create_index(ddl)))
            target_sql = hypothesis.rewritten_sql or original_sql
            optimized = benchmark(self.shadow, target_sql, self.runs, self.max_rows)
        except (SqlGuardError, BenchmarkError, ValueError) as exc:
            # Recorded as a rejection reason and fed back to the agent, not
            # swallowed and not fatal: a malformed DDL is something the model
            # can fix on its next turn if it can see what went wrong.
            error = f"{type(exc).__name__}: {exc}"
        finally:
            reset_report = self.shadow.reset()

        result = HypothesisResult(
            hypothesis=hypothesis,
            verdict=Verdict.ERROR,
            baseline=baseline,
            optimized=optimized,
            index_reports=index_reports,
            reset=reset_report.to_json(),
        )

        if error is not None or optimized is None:
            result.reasons.append(error or "hypothesis produced no measurement")
            return result

        result.checksum_match = optimized.checksum == baseline.checksum
        result.ordered_checksum_match = optimized.ordered_checksum == baseline.ordered_checksum
        result.improvement_pct = (
            (baseline.median_ms - optimized.median_ms) / baseline.median_ms * 100.0
        )
        result.speedup = baseline.median_ms / optimized.median_ms if optimized.median_ms else 0.0

        # Rule 1: identical results. Checked first because a query that returns
        # the wrong rows is not a fast query, it is a broken one, and its
        # runtime is irrelevant.
        if not result.checksum_match:
            result.reasons.append(
                f"result checksum differs: original {baseline.checksum[:16]}... over "
                f"{baseline.row_count} rows, optimised {optimized.checksum[:16]}... over "
                f"{optimized.row_count} rows. The rewrite does not return the same data."
            )

        # Rule 2: a real, measured improvement.
        if result.improvement_pct < MIN_IMPROVEMENT_PCT:
            result.reasons.append(
                f"median improved by {result.improvement_pct:.1f}% "
                f"({baseline.median_ms:.1f}ms -> {optimized.median_ms:.1f}ms), "
                f"below the {MIN_IMPROVEMENT_PCT:.0f}% threshold."
            )

        # Determinism: if the query does not return the same rows each time it
        # is run, a matching checksum is coincidence rather than evidence.
        if not optimized.checksum_stable_across_runs:
            result.reasons.append(
                "the optimised query returned different rows on different runs, so "
                "equivalence cannot be established (an unstable ORDER BY with LIMIT?)"
            )
        if not baseline.checksum_stable_across_runs:
            result.reasons.append(
                "the original query returned different rows on different runs, so "
                "there is no stable result to compare against"
            )

        # Rule 3: index size is a flag, not a veto. A 20%-of-table index that
        # turns 1.5s into 1ms is often still the right call -- but whoever
        # deploys it should be told.
        for report in result.index_reports:
            if report.oversized:
                result.flags.append(
                    f"{report.name} is {report.pct_of_table:.1f}% of {report.table} "
                    f"({report.size_pretty} on {report.table_size_pretty}), above the "
                    f"{OVERSIZED_INDEX_PCT:.0f}% threshold"
                )
        if result.checksum_match and not result.ordered_checksum_match:
            result.flags.append(
                "same rows, different order. If the caller relies on the original "
                "ORDER BY, add an equivalent one to the rewrite."
            )

        result.verdict = Verdict.ACCEPTED if not result.reasons else Verdict.REJECTED
        return result

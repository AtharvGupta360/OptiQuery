"""The shadow database: index bookkeeping and isolation between hypotheses.

Shadow is a logical clone of primary and the only database the agent may write
to. Its job is to answer "what would this index actually do?" -- which is only
worth anything if the answer is attributable to exactly one hypothesis.

That is what the baseline snapshot is for. Shadow's index set is captured once,
at startup, and every hypothesis must return it to that exact state. If an index
built for hypothesis 2 survives into hypothesis 3, then hypothesis 3's numbers
describe a database nobody proposed, and every result after it is unattributable
noise -- reported as a win that a production deploy would not reproduce.

`reset()` therefore does not merely drop what it *thinks* it created. It drops
everything absent from the baseline, then re-reads the catalog and asserts the
result matches. Bookkeeping that verifies itself against the catalog is the only
kind worth having here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import psycopg

from app.db import (
    DatabaseConfig,
    SqlGuardError,
    _Database,
    parse_create_index,
    validate_identifier,
)


class ShadowIsolationError(RuntimeError):
    """Shadow's index set diverged from its baseline."""


# Every index in the `public` schema, with the statistics the report needs.
# `backs_constraint` matters because a PRIMARY KEY's index cannot be dropped
# directly -- attempting it is an error, and it must never be attempted anyway,
# since primary keys are part of the baseline.
_INDEX_QUERY = """
SELECT ci.relname                                  AS index_name,
       ct.relname                                  AS table_name,
       pg_get_indexdef(i.indexrelid)               AS definition,
       i.indisunique                               AS is_unique,
       i.indisprimary                              AS is_primary,
       (con.conindid IS NOT NULL)                  AS backs_constraint,
       pg_relation_size(i.indexrelid)              AS size_bytes,
       pg_size_pretty(pg_relation_size(i.indexrelid)) AS size_pretty,
       coalesce(s.idx_scan, 0)                     AS idx_scan,
       coalesce(s.idx_tup_read, 0)                 AS idx_tup_read,
       coalesce(s.idx_tup_fetch, 0)                AS idx_tup_fetch
FROM pg_index i
JOIN pg_class ci      ON ci.oid = i.indexrelid
JOIN pg_class ct      ON ct.oid = i.indrelid
JOIN pg_namespace n   ON n.oid = ci.relnamespace
LEFT JOIN pg_constraint con ON con.conindid = i.indexrelid
LEFT JOIN pg_stat_user_indexes s ON s.indexrelid = i.indexrelid
WHERE n.nspname = 'public'
  AND (%(table)s::text IS NULL OR ct.relname = %(table)s)
ORDER BY ct.relname, ci.relname
"""


@dataclass(frozen=True)
class IndexRecord:
    name: str
    table: str
    definition: str
    is_unique: bool
    is_primary: bool
    backs_constraint: bool
    size_bytes: int
    size_pretty: str
    idx_scan: int
    idx_tup_read: int
    idx_tup_fetch: int

    def identity(self) -> tuple[str, str, str]:
        """The parts that define *what* the index is.

        Sizes and scan counters are excluded on purpose: they drift as queries
        run, and a baseline comparison that included them would report a
        violation every time a benchmark touched an index.
        """
        return (self.table, self.name, self.definition)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "table": self.table,
            "definition": self.definition,
            "is_unique": self.is_unique,
            "is_primary": self.is_primary,
            "backs_constraint": self.backs_constraint,
            "size_bytes": self.size_bytes,
            "size_pretty": self.size_pretty,
            "idx_scan": self.idx_scan,
            "idx_tup_read": self.idx_tup_read,
            "idx_tup_fetch": self.idx_tup_fetch,
        }


def fetch_index_records(db: _Database, table: str | None = None) -> list[IndexRecord]:
    """Read the live index set from the catalog of whichever database is passed.

    Takes the base `_Database` type because reading indexes is safe on primary
    and shadow alike; only the write paths below narrow to `ShadowDatabase`.
    """
    if table is not None:
        validate_identifier(table)
    rows = db.fetch_all(_INDEX_QUERY, {"table": table})  # type: ignore[arg-type]
    return [
        IndexRecord(
            name=row[0],
            table=row[1],
            definition=row[2],
            is_unique=row[3],
            is_primary=row[4],
            backs_constraint=row[5],
            size_bytes=int(row[6]),
            size_pretty=row[7],
            idx_scan=int(row[8]),
            idx_tup_read=int(row[9]),
            idx_tup_fetch=int(row[10]),
        )
        for row in rows
    ]


@dataclass(frozen=True)
class IndexBuildResult:
    name: str
    table: str
    ddl: str
    build_ms: float
    size_bytes: int
    size_pretty: str
    table_size_bytes: int
    table_size_pretty: str
    pct_of_table: float

    def to_json(self) -> dict[str, Any]:
        return {
            "index_name": self.name,
            "table": self.table,
            "ddl": self.ddl,
            "build_ms": round(self.build_ms, 1),
            "size_bytes": self.size_bytes,
            "size_pretty": self.size_pretty,
            "table_size_bytes": self.table_size_bytes,
            "table_size_pretty": self.table_size_pretty,
            "pct_of_table": round(self.pct_of_table, 2),
        }


@dataclass
class ResetReport:
    dropped: list[str] = field(default_factory=list)
    leaked: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    matches_baseline: bool = True

    def to_json(self) -> dict[str, Any]:
        return {
            "dropped": self.dropped,
            "leaked_beyond_tracking": self.leaked,
            "missing_from_baseline": self.missing,
            "matches_baseline": self.matches_baseline,
        }


class ShadowDatabase(_Database):
    """Read-write handle on the shadow database.

    Every mutating tool in the project takes this type. `PrimaryDatabase` is a
    separate class with no write path, so passing primary where a hypothesis
    gets applied does not type-check and cannot happen by accident.
    """

    role_label = "shadow"

    def __init__(self, dsn: str, statement_timeout_ms: int) -> None:
        super().__init__(dsn, statement_timeout_ms)
        self._baseline: dict[tuple[str, str, str], IndexRecord] | None = None
        self._created: list[str] = []

    # -- baseline ----------------------------------------------------------

    def capture_baseline(self) -> list[IndexRecord]:
        """Snapshot the index set that every hypothesis must return shadow to."""
        records = fetch_index_records(self)
        self._baseline = {record.identity(): record for record in records}
        self._created.clear()
        return records

    @property
    def baseline_captured(self) -> bool:
        return self._baseline is not None

    def _require_baseline(self) -> dict[tuple[str, str, str], IndexRecord]:
        if self._baseline is None:
            raise ShadowIsolationError(
                "shadow baseline was never captured; call capture_baseline() "
                "before creating or dropping anything, or reset() has no "
                "definition of 'clean' to restore"
            )
        return self._baseline

    def baseline_index_names(self) -> set[str]:
        return {record.name for record in self._require_baseline().values()}

    @property
    def created_index_names(self) -> list[str]:
        """Indexes this process built, in creation order."""
        return list(self._created)

    # -- mutation ----------------------------------------------------------

    def create_index(self, ddl: str) -> IndexBuildResult:
        parsed = parse_create_index(ddl)
        self._require_baseline()

        if parsed.name in self.baseline_index_names():
            raise SqlGuardError(
                f"index {parsed.name!r} is part of the shadow baseline and cannot "
                "be recreated; choose a different name"
            )

        existing = self.fetch_one(
            "SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'public' AND c.relname = %s AND c.relkind = 'i'",
            (parsed.name,),
        )
        if existing is not None:
            raise SqlGuardError(
                f"index {parsed.name!r} already exists on shadow. Reusing a name "
                "would benchmark the old index and attribute it to this hypothesis."
            )

        table_exists = self.fetch_one(
            "SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'public' AND c.relname = %s AND c.relkind = 'r'",
            (parsed.table,),
        )
        if table_exists is None:
            raise SqlGuardError(f"table {parsed.table!r} does not exist on shadow")

        conn = self.connect()
        started = time.perf_counter()
        try:
            conn.execute(parsed.ddl)  # type: ignore[arg-type]
        except psycopg.Error as exc:
            # Not swallowed: the agent needs the server's own words to fix its DDL.
            raise SqlGuardError(f"CREATE INDEX failed on shadow: {exc}") from exc
        build_ms = (time.perf_counter() - started) * 1000.0

        self._created.append(parsed.name)

        row = self.fetch_one(
            """
            SELECT pg_relation_size(i.oid),
                   pg_size_pretty(pg_relation_size(i.oid)),
                   pg_relation_size(t.oid),
                   pg_size_pretty(pg_relation_size(t.oid))
            FROM pg_class i, pg_class t
            WHERE i.relname = %s AND t.relname = %s
            """,
            (parsed.name, parsed.table),
        )
        assert row is not None, "index vanished immediately after being created"
        index_bytes, index_pretty, table_bytes, table_pretty = row

        return IndexBuildResult(
            name=parsed.name,
            table=parsed.table,
            ddl=parsed.ddl,
            build_ms=build_ms,
            size_bytes=int(index_bytes),
            size_pretty=index_pretty,
            table_size_bytes=int(table_bytes),
            table_size_pretty=table_pretty,
            # Guarded against a zero-byte table so an empty-table test cannot
            # divide by zero and mask the real result.
            pct_of_table=(100.0 * int(index_bytes) / int(table_bytes)) if table_bytes else 0.0,
        )

    def drop_index(self, name: str) -> dict[str, Any]:
        validate_identifier(name)
        self._require_baseline()

        if name in self.baseline_index_names():
            raise ShadowIsolationError(
                f"refusing to drop {name!r}: it belongs to the shadow baseline. "
                "Dropping it would make shadow structurally different from "
                "primary, and every subsequent measurement would describe a "
                "database that does not exist."
            )

        record = self.fetch_one(
            "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'public' AND c.relname = %s AND c.relkind = 'i'",
            (name,),
        )
        if record is None:
            return {"index_name": name, "dropped": False, "reason": "index does not exist"}

        conn = self.connect()
        try:
            conn.execute(f'DROP INDEX "{name}"')  # type: ignore[arg-type]
        except psycopg.Error as exc:
            raise SqlGuardError(f"DROP INDEX failed on shadow: {exc}") from exc

        if name in self._created:
            self._created.remove(name)
        return {"index_name": name, "dropped": True}

    # -- isolation ---------------------------------------------------------

    def diff_against_baseline(self) -> tuple[list[IndexRecord], list[IndexRecord]]:
        """(extra, missing) relative to the captured baseline."""
        baseline = self._require_baseline()
        current = {record.identity(): record for record in fetch_index_records(self)}
        extra = [record for key, record in current.items() if key not in baseline]
        missing = [record for key, record in baseline.items() if key not in current]
        return extra, missing

    def reset(self) -> ResetReport:
        """Return shadow to its baseline index set, then prove it.

        Drops by catalog difference rather than by the tracked list. Tracking
        can be wrong -- a CREATE INDEX that timed out server-side still leaves
        an index behind, and the exception path never recorded it -- and an
        untracked leak is precisely the failure this method exists to prevent.
        """
        report = ResetReport()
        extra, _ = self.diff_against_baseline()

        for record in extra:
            if record.backs_constraint:
                # Cannot be dropped directly, and should never be extra: it
                # would mean someone added a constraint to shadow.
                raise ShadowIsolationError(
                    f"index {record.name!r} is not in the baseline but backs a "
                    "constraint; shadow's schema has been altered and it can no "
                    "longer stand in for primary. Reseed it."
                )
            if record.name not in self._created:
                report.leaked.append(record.name)
            self.drop_index(record.name)
            report.dropped.append(record.name)

        extra_after, missing_after = self.diff_against_baseline()
        report.missing = [record.name for record in missing_after]
        report.matches_baseline = not extra_after and not missing_after
        self._created.clear()

        if not report.matches_baseline:
            raise ShadowIsolationError(
                "shadow does not match its baseline after reset: "
                f"extra={[r.name for r in extra_after]} missing={report.missing}. "
                "Every measurement taken from here on would be unattributable."
            )
        return report

    def assert_matches_baseline(self) -> None:
        extra, missing = self.diff_against_baseline()
        if extra or missing:
            raise ShadowIsolationError(
                f"shadow index set diverged from baseline: "
                f"extra={[r.name for r in extra]} missing={[r.name for r in missing]}"
            )


def open_shadow(config: DatabaseConfig, capture_baseline: bool = True) -> ShadowDatabase:
    shadow = ShadowDatabase(config.shadow_dsn, config.statement_timeout_ms)
    if capture_baseline:
        shadow.capture_baseline()
    return shadow


def compare_index_sets(
    primary_records: Sequence[IndexRecord], shadow_records: Sequence[IndexRecord]
) -> dict[str, Any]:
    """Check shadow is still a faithful stand-in for primary.

    Recommendations are made about primary but measured on shadow. If the two
    disagree about which indexes exist, the measurement answers a question
    nobody asked.
    """
    primary_ids = {record.identity() for record in primary_records}
    shadow_ids = {record.identity() for record in shadow_records}
    only_primary = sorted(name for _, name, _ in primary_ids - shadow_ids)
    only_shadow = sorted(name for _, name, _ in shadow_ids - primary_ids)
    return {
        "in_parity": not only_primary and not only_shadow,
        "only_on_primary": only_primary,
        "only_on_shadow": only_shadow,
    }

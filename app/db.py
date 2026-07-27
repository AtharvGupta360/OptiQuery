"""Connections, statement guards, and the read-only guarantee for `primary`.

The agent must be structurally incapable of modifying the primary database.
"Structurally" is doing real work in that sentence, so this module stacks four
independent layers rather than trusting any one of them:

  1. **Privileges (the actual guarantee).** `primary` is reached as the
     `optiquery_ro` role, which holds SELECT and nothing else. INSERT, UPDATE,
     DELETE, CREATE INDEX and friends fail with `permission denied` inside
     Postgres, no matter what Python does. A bug in the layers above cannot
     defeat this one.
  2. **Session read-only.** `default_transaction_read_only=on` is set on the
     role and again per session. On its own this is *not* a guarantee -- it is
     not a superuser-only GUC, so a determined caller could reset it -- which is
     exactly why layer 1 exists underneath it.
  3. **Type separation.** `PrimaryDatabase` and `ShadowDatabase` are distinct
     types. Every mutating tool signature names `ShadowDatabase`, so wiring
     primary into one is a type error rather than a runtime incident.
  4. **Statement whitelist.** Read-only entry points reject anything that is not
     a bare SELECT/WITH/TABLE/VALUES, and reject data-modifying CTEs
     (`WITH x AS (DELETE ... RETURNING *) SELECT ...`) which pass a naive
     "starts with SELECT" check. This layer exists to fail *early and legibly*,
     with a message the agent can act on, instead of surfacing a Postgres
     permission error ten frames down.

Every connection also carries a hard 30s statement_timeout.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Sequence

import psycopg
from psycopg import sql as pgsql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

DEFAULT_STATEMENT_TIMEOUT_MS = 30_000

DEFAULT_PRIMARY_DSN = "postgresql://optiquery:optiquery@localhost:55432/optiquery"
DEFAULT_SHADOW_DSN = "postgresql://optiquery:optiquery@localhost:55433/optiquery"

READONLY_ROLE = "optiquery_ro"


class SqlGuardError(ValueError):
    """A statement was rejected before it reached Postgres."""


class ReadOnlyViolation(SqlGuardError):
    """A statement that could modify data was aimed at the primary database."""


# ---------------------------------------------------------------------------
# Statement classification
# ---------------------------------------------------------------------------

# Statements a read-only entry point will accept as the outermost keyword.
READ_ONLY_LEADING_KEYWORDS = frozenset({"SELECT", "WITH", "TABLE", "VALUES"})

# Keywords that must not appear anywhere in a statement submitted as read-only,
# once comments and string literals have been stripped. Checking the whole
# statement rather than just the first token is what catches data-modifying
# CTEs, `SELECT ... INTO newtable`, and `SELECT ... FOR UPDATE`.
FORBIDDEN_IN_READ_ONLY = frozenset(
    {
        "INSERT", "UPDATE", "DELETE", "MERGE", "TRUNCATE",
        "CREATE", "DROP", "ALTER", "RENAME",
        "GRANT", "REVOKE", "COMMENT", "SECURITY",
        "COPY", "CALL", "DO", "PREPARE", "EXECUTE", "DEALLOCATE",
        "VACUUM", "ANALYZE", "REINDEX", "CLUSTER", "REFRESH",
        "LOCK", "SET", "RESET", "DISCARD",
        "LISTEN", "NOTIFY", "UNLISTEN",
        "INTO",
        "EXPLAIN",
    }
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")

_CREATE_INDEX_RE = re.compile(
    r"""^\s*CREATE\s+
        (?P<unique>UNIQUE\s+)?
        INDEX\s+
        (?P<concurrently>CONCURRENTLY\s+)?
        (?P<if_not_exists>IF\s+NOT\s+EXISTS\s+)?
        (?P<name>"[^"]+"|[A-Za-z_][A-Za-z0-9_$]*)\s+
        ON\s+
        (?:ONLY\s+)?
        (?P<table>"[^"]+"|[A-Za-z_][A-Za-z0-9_$]*(?:\.(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_$]*))?)
        \s*(?:USING\s+\w+\s*)?\(""",
    re.IGNORECASE | re.VERBOSE,
)


def strip_sql_noise(sql: str) -> str:
    """Remove comments and string/identifier literals, preserving structure.

    Keyword scanning is only trustworthy if it cannot be fooled by a literal.
    Without this, `SELECT 'drop table users'` trips the guard and
    `WITH t AS (DELETE FROM users RETURNING 1) SELECT * FROM t` does not --
    both backwards. Removed spans are replaced by a space so adjacent tokens do
    not fuse into one word.
    """
    out: list[str] = []
    index = 0
    length = len(sql)

    while index < length:
        char = sql[index]

        # -- line comment
        if char == "-" and sql.startswith("--", index):
            newline = sql.find("\n", index)
            index = length if newline == -1 else newline
            out.append(" ")
            continue

        # /* block comment */ -- nests in Postgres
        if char == "/" and sql.startswith("/*", index):
            depth = 1
            index += 2
            while index < length and depth:
                if sql.startswith("/*", index):
                    depth += 1
                    index += 2
                elif sql.startswith("*/", index):
                    depth -= 1
                    index += 2
                else:
                    index += 1
            out.append(" ")
            continue

        # 'string literal', with '' escaping
        if char == "'":
            index += 1
            while index < length:
                if sql[index] == "'":
                    if index + 1 < length and sql[index + 1] == "'":
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            out.append(" ")
            continue

        # "quoted identifier", with "" escaping
        if char == '"':
            index += 1
            while index < length:
                if sql[index] == '"':
                    if index + 1 < length and sql[index + 1] == '"':
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            out.append(" ")
            continue

        # $tag$ dollar quoted $tag$
        if char == "$":
            match = re.match(r"\$([A-Za-z_][A-Za-z0-9_]*)?\$", sql[index:])
            if match:
                tag = match.group(0)
                close = sql.find(tag, index + len(tag))
                index = length if close == -1 else close + len(tag)
                out.append(" ")
                continue

        out.append(char)
        index += 1

    return "".join(out)


def _statement_count(stripped: str) -> int:
    """Number of statements, given literal-stripped SQL."""
    body = stripped.strip()
    if body.endswith(";"):
        body = body[:-1]
    return body.count(";") + 1 if body else 0


def leading_keyword(sql: str) -> str:
    stripped = strip_sql_noise(sql)
    match = _WORD_RE.search(stripped)
    return match.group(0).upper() if match else ""


def assert_read_only(sql: str) -> None:
    """Reject anything that is not a single, side-effect-free read.

    Layer 4 of the primary-safety stack. This runs before every EXPLAIN and
    every benchmark, on shadow as well as primary -- a rewrite that silently
    mutates rows would corrupt the shadow baseline and poison every subsequent
    hypothesis, so it is not only primary that needs protecting.
    """
    if not sql or not sql.strip():
        raise SqlGuardError("empty statement")

    stripped = strip_sql_noise(sql)

    count = _statement_count(stripped)
    if count != 1:
        raise SqlGuardError(
            f"expected exactly 1 statement, found {count}. "
            "Multi-statement input is rejected: only the first would be measured, "
            "and the rest would run unmeasured."
        )

    keyword = leading_keyword(sql)
    if keyword not in READ_ONLY_LEADING_KEYWORDS:
        if keyword == "EXPLAIN":
            raise ReadOnlyViolation(
                "pass the query itself, not an EXPLAIN of it. The tool adds "
                "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) around it."
            )
        raise ReadOnlyViolation(
            f"statement starts with {keyword or '<nothing>'}; read-only entry "
            f"points accept only {sorted(READ_ONLY_LEADING_KEYWORDS)}"
        )

    words = {word.upper() for word in _WORD_RE.findall(stripped)}
    offending = sorted(words & FORBIDDEN_IN_READ_ONLY)
    if offending:
        raise ReadOnlyViolation(
            f"statement contains {offending}, which can have side effects even "
            "inside a statement that starts with SELECT or WITH (a data-modifying "
            "CTE, a SELECT INTO, or a locking clause). Rejected."
        )


@dataclass(frozen=True)
class ParsedCreateIndex:
    name: str
    table: str
    unique: bool
    ddl: str


def parse_create_index(ddl: str) -> ParsedCreateIndex:
    """Validate a CREATE INDEX and pull out the pieces the tools need.

    An explicit index name is mandatory. Shadow isolation works by dropping
    every index created during a hypothesis *by name*; an index Postgres named
    for us could not be reliably attributed to the hypothesis that created it.
    """
    if not ddl or not ddl.strip():
        raise SqlGuardError("empty DDL")

    stripped = strip_sql_noise(ddl)
    count = _statement_count(stripped)
    if count != 1:
        raise SqlGuardError(f"expected exactly 1 statement, found {count}")

    keyword = leading_keyword(ddl)
    if keyword != "CREATE":
        raise SqlGuardError(
            f"expected a CREATE INDEX statement, got {keyword or '<nothing>'}"
        )

    match = _CREATE_INDEX_RE.match(ddl)
    if match is None:
        raise SqlGuardError(
            "not a recognised CREATE INDEX statement. Required shape: "
            "CREATE [UNIQUE] INDEX <name> ON <table> [USING <method>] (<columns>) "
            "[WHERE <predicate>]. The index name is required."
        )
    if match.group("concurrently"):
        raise SqlGuardError(
            "CONCURRENTLY is rejected on shadow: it is several times slower to "
            "build and cannot run inside a transaction, and shadow has no "
            "concurrent traffic to protect. The production DDL emitted in the "
            "report adds CONCURRENTLY back."
        )
    if match.group("if_not_exists"):
        raise SqlGuardError(
            "IF NOT EXISTS is rejected: it would silently skip the build and "
            "report a benchmark against a pre-existing index as if it were new."
        )

    return ParsedCreateIndex(
        name=match.group("name").strip('"'),
        table=match.group("table").strip('"'),
        unique=bool(match.group("unique")),
        ddl=ddl.strip().rstrip(";"),
    )


def validate_identifier(name: str) -> str:
    """Accept a bare SQL identifier; reject anything needing quoting."""
    if not name or not _IDENTIFIER_RE.match(name):
        raise SqlGuardError(
            f"{name!r} is not a bare identifier (expected [A-Za-z_][A-Za-z0-9_$]*)"
        )
    if len(name) > 63:
        raise SqlGuardError(f"identifier {name!r} exceeds Postgres' 63 byte limit")
    return name


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatabaseConfig:
    """DSNs and the read-only credentials derived from them."""

    primary_admin_dsn: str
    shadow_dsn: str
    readonly_user: str = READONLY_ROLE
    readonly_password: str = "optiquery_ro"
    statement_timeout_ms: int = DEFAULT_STATEMENT_TIMEOUT_MS

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        return cls(
            primary_admin_dsn=os.environ.get("PRIMARY_DSN", DEFAULT_PRIMARY_DSN),
            shadow_dsn=os.environ.get("SHADOW_DSN", DEFAULT_SHADOW_DSN),
            readonly_user=os.environ.get("READONLY_USER", READONLY_ROLE),
            readonly_password=os.environ.get("READONLY_PASSWORD", "optiquery_ro"),
            statement_timeout_ms=int(
                os.environ.get("STATEMENT_TIMEOUT_MS", DEFAULT_STATEMENT_TIMEOUT_MS)
            ),
        )

    @property
    def primary_readonly_dsn(self) -> str:
        parts = conninfo_to_dict(self.primary_admin_dsn)
        parts["user"] = self.readonly_user
        parts["password"] = self.readonly_password
        return make_conninfo(**parts)


def ensure_readonly_role(config: DatabaseConfig) -> None:
    """Create/refresh the SELECT-only role on primary. Idempotent.

    This is the one place that connects to primary with write privileges, and
    it runs once at startup, before any tool exists to be called. It is not
    reachable from the agent.
    """
    role = validate_identifier(config.readonly_user)

    with psycopg.connect(config.primary_admin_dsn, autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)
        ).fetchone()
        if exists is None:
            conn.execute(
                pgsql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                    pgsql.Identifier(role), pgsql.Literal(config.readonly_password)
                )
            )
        else:
            conn.execute(
                pgsql.SQL("ALTER ROLE {} LOGIN PASSWORD {}").format(
                    pgsql.Identifier(role), pgsql.Literal(config.readonly_password)
                )
            )

        database = conn.execute("SELECT current_database()").fetchone()
        assert database is not None

        statements = [
            pgsql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(
                pgsql.Identifier(role)
            ),
            pgsql.SQL("ALTER ROLE {} SET statement_timeout = {}").format(
                pgsql.Identifier(role), pgsql.Literal(f"{config.statement_timeout_ms}ms")
            ),
            pgsql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                pgsql.Identifier(database[0]), pgsql.Identifier(role)
            ),
            pgsql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(pgsql.Identifier(role)),
            pgsql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA public TO {}").format(
                pgsql.Identifier(role)
            ),
            pgsql.SQL(
                "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO {}"
            ).format(pgsql.Identifier(role)),
            # Belt and braces: Postgres 15+ already withholds CREATE on the
            # public schema from PUBLIC, but say so explicitly rather than
            # depending on a default that changed within living memory.
            pgsql.SQL("REVOKE CREATE ON SCHEMA public FROM {}").format(
                pgsql.Identifier(role)
            ),
        ]
        for statement in statements:
            conn.execute(statement)


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------

class _Database:
    """A single long-lived autocommit connection with a statement timeout."""

    role_label: str = "database"

    def __init__(self, dsn: str, statement_timeout_ms: int) -> None:
        self._dsn = dsn
        self._statement_timeout_ms = statement_timeout_ms
        self._lock = threading.Lock()
        self._conn: psycopg.Connection | None = None

    def connect(self) -> psycopg.Connection:
        with self._lock:
            if self._conn is None or self._conn.closed:
                self._conn = psycopg.connect(self._dsn, autocommit=True)
                self._configure(self._conn)
            return self._conn

    def _configure(self, conn: psycopg.Connection) -> None:
        conn.execute(f"SET statement_timeout = {int(self._statement_timeout_ms)}")

    def close(self) -> None:
        with self._lock:
            if self._conn is not None and not self._conn.closed:
                self._conn.close()
            self._conn = None

    def fetch_all(self, query: str, params: Sequence[Any] | None = None) -> list[tuple]:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

    def fetch_one(self, query: str, params: Sequence[Any] | None = None) -> tuple | None:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchone()

    def __enter__(self) -> "_Database":
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class PrimaryDatabase(_Database):
    """Read-only handle on the production database.

    `fetch_all` and `fetch_one` are overridden to run `assert_read_only` on
    every statement. Without the override, the inherited implementations would
    execute arbitrary SQL and layer 4 could be bypassed simply by calling the
    base class -- the privilege layer would still stop the write, but the
    project would be relying on one layer while claiming four.

    Internal introspection that is not a plain SELECT (`SHOW ...`) goes through
    the unguarded base methods explicitly, at a call site that can be read.
    """

    role_label = "primary"

    def _configure(self, conn: psycopg.Connection) -> None:
        super()._configure(conn)
        conn.execute("SET default_transaction_read_only = on")

    def fetch_all(self, query: str, params: Sequence[Any] | None = None) -> list[tuple]:
        assert_read_only(query)
        return super().fetch_all(query, params)

    def fetch_one(self, query: str, params: Sequence[Any] | None = None) -> tuple | None:
        assert_read_only(query)
        return super().fetch_one(query, params)

    def run_read_only(self, sql: str, params: Sequence[Any] | None = None) -> list[tuple]:
        return self.fetch_all(sql, params)

    def is_read_only(self) -> bool:
        row = super().fetch_one("SHOW default_transaction_read_only")
        return bool(row and row[0] == "on")

    def statement_timeout_ms(self) -> int:
        row = super().fetch_one("SHOW statement_timeout")
        assert row is not None
        raw = str(row[0])
        return int(raw[:-2]) if raw.endswith("ms") else int(raw[:-1]) * 1000

    def current_user(self) -> str:
        row = self.fetch_one("SELECT current_user")
        assert row is not None
        return str(row[0])


def open_primary(config: DatabaseConfig, bootstrap: bool = True) -> PrimaryDatabase:
    """Open primary as the SELECT-only role, creating that role if needed."""
    if bootstrap:
        ensure_readonly_role(config)
    return PrimaryDatabase(config.primary_readonly_dsn, config.statement_timeout_ms)

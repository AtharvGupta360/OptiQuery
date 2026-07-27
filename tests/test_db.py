"""The primary database must be unwritable. These tests prove each layer.

The important test here is `test_writes_fail_even_with_read_only_disabled`. It
turns off the session flag that everything else relies on and shows the write
still fails, which is what makes the claim "structurally incapable" true rather
than aspirational.
"""

from __future__ import annotations

import psycopg
import pytest

from app.db import DatabaseConfig, ReadOnlyViolation, open_primary
from app.tools import ToolContext

pytestmark = pytest.mark.db


class TestPrimaryIsReadOnly:
    def test_connects_as_the_restricted_role(self, ctx: ToolContext, config: DatabaseConfig) -> None:
        assert ctx.primary.current_user() == config.readonly_user

    def test_session_is_read_only(self, ctx: ToolContext) -> None:
        assert ctx.primary.is_read_only() is True

    def test_statement_timeout_is_capped(self, ctx: ToolContext, config: DatabaseConfig) -> None:
        assert ctx.primary.statement_timeout_ms() == config.statement_timeout_ms
        assert config.statement_timeout_ms <= 30_000

    def test_guard_rejects_writes_before_they_reach_postgres(self, ctx: ToolContext) -> None:
        with pytest.raises(ReadOnlyViolation):
            ctx.primary.run_read_only("DELETE FROM users WHERE id = -1")

    def test_base_class_read_path_is_also_guarded(self, ctx: ToolContext) -> None:
        """fetch_all is overridden on primary; calling it directly must not bypass."""
        with pytest.raises(ReadOnlyViolation):
            ctx.primary.fetch_all("UPDATE users SET is_active = false WHERE id = -1")

    def test_reads_still_work(self, ctx: ToolContext) -> None:
        rows = ctx.primary.run_read_only("SELECT count(*) FROM users")
        assert rows[0][0] == 200_000


class TestPrivilegesAreTheRealGuarantee:
    """Layer 1, tested without any of the layers above it.

    These use a raw psycopg connection with the read-only DSN, so nothing in
    app/db.py is in the call path. If these pass, a bug anywhere in the Python
    guards still cannot damage primary.
    """

    @pytest.fixture()
    def raw_readonly_conn(self, config: DatabaseConfig):
        with psycopg.connect(config.primary_readonly_dsn, autocommit=True) as conn:
            yield conn

    def test_insert_is_denied(self, raw_readonly_conn: psycopg.Connection) -> None:
        with pytest.raises(psycopg.Error):
            raw_readonly_conn.execute(
                "INSERT INTO users (id, email, full_name, country, city, signup_ts, "
                "loyalty_tier, lifetime_value, is_active) VALUES "
                "(-1, 'x', 'x', 'XX', 'x', now(), 'bronze', 0, true)"
            )

    def test_create_index_is_denied(self, raw_readonly_conn: psycopg.Connection) -> None:
        with pytest.raises(psycopg.Error):
            raw_readonly_conn.execute("CREATE INDEX ix_should_not_exist ON users (email)")

    def test_writes_fail_even_with_read_only_disabled(
        self, raw_readonly_conn: psycopg.Connection
    ) -> None:
        """default_transaction_read_only is not a security boundary; privileges are.

        The role can turn the flag off -- it is not superuser-only. It gains
        nothing by doing so, because it holds SELECT and no other privilege.
        """
        raw_readonly_conn.execute("SET default_transaction_read_only = off")
        assert raw_readonly_conn.execute(
            "SHOW default_transaction_read_only"
        ).fetchone() == ("off",)

        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            raw_readonly_conn.execute("DELETE FROM users WHERE id = -1")

    def test_role_cannot_create_objects_in_public_schema(
        self, raw_readonly_conn: psycopg.Connection
    ) -> None:
        raw_readonly_conn.execute("SET default_transaction_read_only = off")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            raw_readonly_conn.execute("CREATE TABLE public.evil (id int)")

    def test_primary_still_has_only_its_primary_keys(self, ctx: ToolContext) -> None:
        """Nothing above left an index behind on production."""
        rows = ctx.primary.run_read_only(
            "SELECT ci.relname FROM pg_index i "
            "JOIN pg_class ci ON ci.oid = i.indexrelid "
            "JOIN pg_namespace n ON n.oid = ci.relnamespace "
            "WHERE n.nspname = 'public' AND NOT i.indisprimary"
        )
        assert rows == []


class TestBootstrapIsIdempotent:
    def test_open_primary_twice(self, config: DatabaseConfig) -> None:
        first = open_primary(config)
        second = open_primary(config)
        try:
            assert first.current_user() == second.current_user()
        finally:
            first.close()
            second.close()

"""Statement guard unit tests. No database required.

These cover the two ways a keyword-based guard is normally wrong: it trips on a
keyword that is only text inside a literal, and it misses a keyword that is real
but not in the first position.
"""

from __future__ import annotations

import pytest

from app.db import (
    ReadOnlyViolation,
    SqlGuardError,
    assert_read_only,
    leading_keyword,
    parse_create_index,
    strip_sql_noise,
    validate_identifier,
)


class TestStripSqlNoise:
    def test_removes_line_comments(self) -> None:
        assert "secret" not in strip_sql_noise("SELECT 1 -- secret\nFROM t")

    def test_removes_nested_block_comments(self) -> None:
        stripped = strip_sql_noise("SELECT /* a /* b */ c */ 1 FROM t")
        assert "a" not in stripped and "b" not in stripped
        assert "FROM" in stripped

    def test_removes_string_literals(self) -> None:
        assert "DROP" not in strip_sql_noise("SELECT 'DROP TABLE users' AS x")

    def test_handles_escaped_quote_inside_literal(self) -> None:
        stripped = strip_sql_noise("SELECT 'it''s DELETE' AS x FROM t")
        assert "DELETE" not in stripped
        assert "FROM" in stripped

    def test_removes_quoted_identifiers(self) -> None:
        assert "drop" not in strip_sql_noise('SELECT "drop" FROM t')

    def test_removes_dollar_quoted_blocks(self) -> None:
        assert "TRUNCATE" not in strip_sql_noise("SELECT $tag$TRUNCATE t$tag$ FROM t")

    def test_does_not_fuse_adjacent_tokens(self) -> None:
        # 'a'||'b' must not collapse into a single word.
        assert leading_keyword("SELECT 'a' 'b' FROM t") == "SELECT"


class TestAssertReadOnly:
    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT 1",
            "select id from users where id = 1",
            "WITH recent AS (SELECT * FROM orders) SELECT count(*) FROM recent",
            "TABLE users",
            "VALUES (1), (2)",
            "SELECT 1;",
            # A literal that merely contains dangerous words is not dangerous.
            "SELECT 'DROP TABLE users; DELETE FROM orders' AS payload",
            # Column names that merely start with a keyword.
            "SELECT deleted_at, update_count, into_bucket FROM t",
            # OFFSET must not be read as SET.
            "SELECT id FROM users ORDER BY id OFFSET 10 LIMIT 5",
        ],
    )
    def test_accepts_reads(self, sql: str) -> None:
        assert_read_only(sql)

    @pytest.mark.parametrize(
        "sql",
        [
            "DELETE FROM users",
            "UPDATE users SET is_active = false",
            "INSERT INTO users VALUES (1)",
            "TRUNCATE users",
            "DROP TABLE users",
            "CREATE INDEX ix ON users (email)",
            "GRANT ALL ON users TO public",
            "VACUUM users",
        ],
    )
    def test_rejects_obvious_writes(self, sql: str) -> None:
        with pytest.raises(ReadOnlyViolation):
            assert_read_only(sql)

    def test_rejects_data_modifying_cte(self) -> None:
        """The case a 'starts with SELECT/WITH' check waves through."""
        sql = "WITH gone AS (DELETE FROM users WHERE id = 1 RETURNING id) SELECT * FROM gone"
        with pytest.raises(ReadOnlyViolation, match="DELETE"):
            assert_read_only(sql)

    def test_rejects_select_into(self) -> None:
        with pytest.raises(ReadOnlyViolation, match="INTO"):
            assert_read_only("SELECT * INTO copy_of_users FROM users")

    def test_rejects_locking_clause(self) -> None:
        with pytest.raises(ReadOnlyViolation):
            assert_read_only("SELECT * FROM users FOR UPDATE")

    def test_rejects_multiple_statements(self) -> None:
        with pytest.raises(SqlGuardError, match="found 2"):
            assert_read_only("SELECT 1; DROP TABLE users")

    def test_semicolon_inside_literal_is_not_a_separator(self) -> None:
        assert_read_only("SELECT 'a;b' AS x FROM t")

    def test_rejects_explain_with_actionable_message(self) -> None:
        with pytest.raises(ReadOnlyViolation, match="pass the query itself"):
            assert_read_only("EXPLAIN ANALYZE SELECT 1")

    def test_rejects_empty(self) -> None:
        with pytest.raises(SqlGuardError):
            assert_read_only("   ")


class TestParseCreateIndex:
    def test_parses_simple_index(self) -> None:
        parsed = parse_create_index("CREATE INDEX ix_oi_sku ON order_items (sku)")
        assert parsed.name == "ix_oi_sku"
        assert parsed.table == "order_items"
        assert parsed.unique is False

    def test_parses_unique_expression_index_with_predicate(self) -> None:
        parsed = parse_create_index(
            "CREATE UNIQUE INDEX ix_o_email ON orders USING btree (lower(email_snapshot)) "
            "WHERE tracking_number IS NOT NULL"
        )
        assert parsed.name == "ix_o_email"
        assert parsed.table == "orders"
        assert parsed.unique is True

    def test_strips_trailing_semicolon(self) -> None:
        assert not parse_create_index("CREATE INDEX ix ON users (email);").ddl.endswith(";")

    def test_requires_an_explicit_name(self) -> None:
        """Shadow resets drop by name; an auto-named index is unattributable."""
        with pytest.raises(SqlGuardError, match="index name is required"):
            parse_create_index("CREATE INDEX ON order_items (sku)")

    def test_rejects_concurrently(self) -> None:
        with pytest.raises(SqlGuardError, match="CONCURRENTLY"):
            parse_create_index("CREATE INDEX CONCURRENTLY ix ON users (email)")

    def test_rejects_if_not_exists(self) -> None:
        with pytest.raises(SqlGuardError, match="IF NOT EXISTS"):
            parse_create_index("CREATE INDEX IF NOT EXISTS ix ON users (email)")

    def test_rejects_non_index_ddl(self) -> None:
        with pytest.raises(SqlGuardError):
            parse_create_index("CREATE TABLE evil (id int)")

    def test_rejects_trailing_statement(self) -> None:
        with pytest.raises(SqlGuardError, match="found 2"):
            parse_create_index("CREATE INDEX ix ON users (email); DROP TABLE users")


class TestValidateIdentifier:
    @pytest.mark.parametrize("name", ["ix_oi_sku", "_private", "a$b", "T1"])
    def test_accepts_bare_identifiers(self, name: str) -> None:
        assert validate_identifier(name) == name

    @pytest.mark.parametrize("name", ["", "1abc", "has space", 'quo"te', "drop;", "a-b"])
    def test_rejects_anything_needing_quoting(self, name: str) -> None:
        with pytest.raises(SqlGuardError):
            validate_identifier(name)

    def test_rejects_over_length(self) -> None:
        with pytest.raises(SqlGuardError, match="63 byte"):
            validate_identifier("x" * 64)

"""Shared fixtures.

The database fixtures are session-scoped because opening a connection and
bootstrapping the read-only role on every test would dominate the run time.
`shadow_is_reset` is function-scoped and autouse, so a test that leaves an index
behind fails immediately and loudly rather than silently corrupting whichever
test happens to run next.
"""

from __future__ import annotations

import pytest

from app.db import DatabaseConfig
from app.tools import ToolContext
from seed.measure_baseline import NamedQuery, load_named_queries

SEED_TABLES = {"users", "products", "orders", "order_items", "reviews"}


@pytest.fixture(scope="session")
def config() -> DatabaseConfig:
    return DatabaseConfig.from_env()


@pytest.fixture(scope="session")
def ctx(config: DatabaseConfig):
    with ToolContext.open(config) as tool_context:
        rows = tool_context.primary.run_read_only(
            "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'public' AND c.relkind = 'r'"
        )
        found = {row[0] for row in rows}
        missing = SEED_TABLES - found
        if missing:
            pytest.fail(
                f"primary is missing seed tables {sorted(missing)}. "
                "Run `make seed` (or `docker compose run --rm seeder`) first."
            )
        yield tool_context


@pytest.fixture(autouse=True)
def shadow_is_reset(request: pytest.FixtureRequest):
    """Return shadow to its baseline after every test that touched a database."""
    yield
    if "ctx" in request.fixturenames:
        tool_context = request.getfixturevalue("ctx")
        tool_context.shadow.reset()


@pytest.fixture(scope="session")
def seed_queries() -> dict[str, NamedQuery]:
    return {query.name: query for query in load_named_queries()}

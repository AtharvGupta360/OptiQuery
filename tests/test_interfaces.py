"""The CLI and the HTTP service.

Neither is tested against a live model -- that belongs in test_agent.py. What
matters here is everything that happens *before* a model is involved: that a
write statement is refused, that the read-only guard fires ahead of any
benchmarking, and that two concurrent optimisations cannot share one shadow
database.
"""

from __future__ import annotations

import pytest

import cli
from app.main import _RUN_LOCK, app

# ---------------------------------------------------------------------------
# CLI argument handling -- no database, no model
# ---------------------------------------------------------------------------


class TestSeedResolution:
    def test_a_prefix_resolves_to_one_query(self) -> None:
        seeds = cli._load_seed_queries()
        assert seeds, "seed/slow_queries.sql should parse"
        resolved = cli._resolve_seed("q2", seeds)
        assert resolved.name.startswith("q2")

    def test_an_unknown_name_lists_what_exists(self) -> None:
        seeds = cli._load_seed_queries()
        with pytest.raises(SystemExit) as excinfo:
            cli._resolve_seed("q9_nope", seeds)
        assert "no seed query matching" in str(excinfo.value)

    def test_an_ambiguous_prefix_is_refused_rather_than_guessed(self) -> None:
        """Silently picking one would benchmark a query nobody asked for."""
        seeds = cli._load_seed_queries()
        with pytest.raises(SystemExit) as excinfo:
            cli._resolve_seed("q", seeds)
        assert "ambiguous" in str(excinfo.value)

    def test_all_selects_every_seed_query(self) -> None:
        seeds = cli._load_seed_queries()
        args = cli._parse_args(["--all"])
        assert len(cli._targets(args, seeds)) == len(seeds)

    def test_no_target_is_an_error_not_a_no_op(self) -> None:
        with pytest.raises(SystemExit, match="nothing to do"):
            cli._targets(cli._parse_args([]), {})


class TestConfigOverrides:
    def test_a_cli_override_wins_over_the_environment(self, monkeypatch) -> None:
        """`--runs 3` and OPTIQUERY_BENCHMARK_RUNS both set the same field.

        Splatting the overrides alongside the environment-derived values made
        that a duplicate-keyword TypeError instead of the override winning --
        which crashed every `cli.py --runs N` invocation.
        """
        from app.agent import AgentConfig

        monkeypatch.setenv("OPTIQUERY_BENCHMARK_RUNS", "9")
        monkeypatch.setenv("OPTIQUERY_MAX_ITERATIONS", "7")

        assert AgentConfig.from_env().benchmark_runs == 9
        config = AgentConfig.from_env(model="m", benchmark_runs=3)
        assert config.benchmark_runs == 3
        assert config.max_iterations == 7, "unspecified fields still come from the env"

    def test_a_blank_environment_value_falls_back_to_the_default(self, monkeypatch) -> None:
        """env_file passes declared-but-empty keys through as empty strings."""
        from app.agent import DEFAULT_MAX_TOKENS, AgentConfig

        monkeypatch.setenv("OPTIQUERY_MAX_TOKENS", "")
        assert AgentConfig.from_env().max_tokens == DEFAULT_MAX_TOKENS


class TestArtifactNaming:
    def test_the_same_query_always_gets_the_same_name(self) -> None:
        """`make artifacts` re-runs must overwrite, not accumulate."""
        sql = "SELECT id FROM orders WHERE user_id = 5"
        assert cli._derive_name(sql) == cli._derive_name(sql)

    def test_different_queries_get_different_names(self) -> None:
        assert cli._derive_name("SELECT 1") != cli._derive_name("SELECT 2")

    def test_the_name_is_filesystem_safe(self) -> None:
        name = cli._derive_name("SELECT * FROM t WHERE x = 'a/b\\c' -- ../..")
        assert all(character.isalnum() or character == "_" for character in name)

    def test_an_explicit_name_wins(self) -> None:
        args = cli._parse_args(["SELECT 1", "--name", "custom"])
        assert cli._targets(args, {})[0][0] == "custom"


class TestReadOnlyGuard:
    @pytest.mark.parametrize(
        "statement",
        [
            "DELETE FROM orders",
            "UPDATE users SET country = 'ZZ'",
            "DROP TABLE users",
            "WITH gone AS (DELETE FROM orders RETURNING id) SELECT * FROM gone",
        ],
    )
    def test_a_write_is_refused_before_anything_runs(self, statement: str, capsys) -> None:
        """Exit 2 without opening a connection or calling a model.

        The guard is checked ahead of the provider and the database on purpose:
        finding this after the first query has been benchmarked would waste
        minutes to reject something knowable from the string alone.
        """
        assert cli.main([statement]) == 2
        assert "error:" in capsys.readouterr().err

    def test_a_select_passes_the_guard(self) -> None:
        """It fails later for want of a provider -- but not on the guard."""
        from app.db import assert_read_only

        assert_read_only("SELECT id FROM orders WHERE user_id = 5")


class TestListing:
    def test_list_prints_every_seed_query_and_exits_zero(self, capsys) -> None:
        assert cli.main(["--list"]) == 0
        out = capsys.readouterr().out
        for name in cli._load_seed_queries():
            assert name in out


# ---------------------------------------------------------------------------
# HTTP service
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.db
class TestService:
    def test_health_reports_the_guarantees_it_depends_on(self, client) -> None:
        payload = client.get("/health").json()
        assert payload["status"] == "ok"
        assert payload["primary_read_only"] is True
        assert payload["shadow_baseline_indexes"] == 5
        assert payload["busy"] is False

    def test_a_write_is_rejected_with_400(self, client) -> None:
        response = client.post("/optimize", json={"query": "DELETE FROM orders"})
        assert response.status_code == 400
        assert "read-only" in response.json()["detail"].lower()

    def test_a_malformed_body_is_rejected(self, client) -> None:
        assert client.post("/optimize", json={}).status_code == 422

    def test_out_of_range_tuning_is_rejected(self, client) -> None:
        response = client.post("/optimize", json={"query": "SELECT 1", "runs": 999})
        assert response.status_code == 422

    def test_a_second_run_is_refused_rather_than_queued(self, client) -> None:
        """Concurrent runs would reset one shadow database underneath each other.

        Both callers would get plausible-looking timings for a database neither
        one described, which is precisely the attribution the verifier exists to
        guarantee. Refusing is the only honest answer.
        """
        _RUN_LOCK.acquire()
        try:
            response = client.post("/optimize", json={"query": "SELECT 1"})
        finally:
            _RUN_LOCK.release()

        if response.status_code == 503:
            pytest.skip("no model provider configured; the lock is checked after that")
        assert response.status_code == 409
        assert "shadow" in response.json()["detail"]

    def test_health_reports_busy_while_the_lock_is_held(self, client) -> None:
        _RUN_LOCK.acquire()
        try:
            assert client.get("/health").json()["busy"] is True
        finally:
            _RUN_LOCK.release()

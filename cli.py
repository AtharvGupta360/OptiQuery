#!/usr/bin/env python3
"""OptiQuery command line.

    python cli.py "SELECT ..."           optimise a query given inline
    python cli.py --seed q2              optimise a named seed query
    python cli.py --all                  every seed query, one after another
    python cli.py --list                 what is available, and exit

Exit status is 0 when the run completed and 1 when it stopped early, so a
`make artifacts` that half-worked fails the build instead of quietly writing
partial reports.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from app.agent import AgentConfig, StopReason, optimize_query
from app.db import DatabaseConfig, SqlGuardError, assert_read_only
from app.llm import LLMError, build_client
from app.report import summarize_for_terminal, write_reports
from app.shadow import ShadowIsolationError
from app.tools import ToolContext
from seed.measure_baseline import NamedQuery, load_named_queries

DEFAULT_OUTPUT = Path("runs")


def _load_seed_queries() -> dict[str, NamedQuery]:
    try:
        return {query.name: query for query in load_named_queries()}
    except FileNotFoundError:
        return {}


def _resolve_seed(name: str, available: dict[str, NamedQuery]) -> NamedQuery:
    """Accept a full name or an unambiguous prefix: `q2` finds q2_unindexed_fk_join."""
    if name in available:
        return available[name]
    matches = [key for key in available if key.startswith(name)]
    if len(matches) == 1:
        return available[matches[0]]
    if not matches:
        raise SystemExit(
            f"no seed query matching {name!r}. Available: {', '.join(sorted(available)) or 'none'}"
        )
    raise SystemExit(f"{name!r} is ambiguous: {', '.join(sorted(matches))}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Diagnose a slow Postgres query, then measure every proposed fix on a shadow database.",
    )
    parser.add_argument("query", nargs="?", help="SQL to optimise.")
    parser.add_argument("--seed", metavar="NAME", help="Optimise a named seed query.")
    parser.add_argument("--all", action="store_true", help="Optimise every seed query.")
    parser.add_argument("--list", action="store_true", help="List seed queries and exit.")
    parser.add_argument("--provider", help="Override OPTIQUERY_PROVIDER.")
    parser.add_argument("--model", help="Override OPTIQUERY_MODEL.")
    parser.add_argument("--max-iterations", type=int, help="Override the iteration cap.")
    parser.add_argument("--runs", type=int, help="Timed benchmark runs per measurement.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Directory for .md/.html/.json artifacts (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--name",
        help="Artifact base name for an inline query (default: derived from the SQL).",
    )
    parser.add_argument("--quiet", action="store_true", help="Print paths only.")
    return parser.parse_args(argv)


def _derive_name(sql: str) -> str:
    """A short, filesystem-safe, stable name for an ad-hoc query."""
    from hashlib import sha256

    words = [word.lower() for word in sql.split() if word.isalpha()][:3]
    stem = "_".join(words) or "query"
    return f"{stem}_{sha256(sql.encode('utf-8')).hexdigest()[:8]}"


def _targets(args: argparse.Namespace, seeds: dict[str, NamedQuery]) -> list[tuple[str, str]]:
    if args.all:
        if not seeds:
            raise SystemExit("no seed queries found; run `make seed` first")
        return [(query.name, query.sql) for query in seeds.values()]
    if args.seed:
        query = _resolve_seed(args.seed, seeds)
        return [(query.name, query.sql)]
    if args.query:
        return [(args.name or _derive_name(args.query), args.query)]
    raise SystemExit("nothing to do: pass a query, --seed NAME, --all, or --list")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    seeds = _load_seed_queries()

    if args.list:
        if not seeds:
            print("No seed queries found. Run `make seed` first.")
            return 1
        print("Seed queries:")
        for name, query in seeds.items():
            first_line = " ".join(query.sql.split())[:88]
            print(f"  {name}\n      {first_line}...")
        return 0

    targets = _targets(args, seeds)

    # Rejected here rather than inside the loop: a guard violation is a mistake
    # in the invocation, and finding it after the first query has already been
    # benchmarked wastes minutes for no reason.
    for name, sql in targets:
        try:
            assert_read_only(sql)
        except SqlGuardError as exc:
            print(f"error: {name}: {exc}", file=sys.stderr)
            return 2

    try:
        bundle = build_client(provider=args.provider, model=args.model)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    overrides = {}
    if args.max_iterations is not None:
        overrides["max_iterations"] = args.max_iterations
    if args.runs is not None:
        overrides["benchmark_runs"] = args.runs
    config = AgentConfig.from_env(model=bundle.model, **overrides)

    if not args.quiet:
        print(
            f"provider {bundle.describe()} | {config.max_iterations} iterations max "
            f"| {config.benchmark_runs} timed runs per measurement"
        )

    incomplete = 0
    with ToolContext.open(DatabaseConfig.from_env()) as ctx:
        for name, sql in targets:
            if not args.quiet:
                print(f"\n=== {name} ===")
            try:
                run = optimize_query(ctx, bundle.client, name, sql, config)
            except ShadowIsolationError as exc:
                # Not caught and turned into a partial report: shadow can no
                # longer stand in for primary, so every later query in --all
                # would be measured against a database nobody described.
                print(f"error: shadow isolation failed on {name}: {exc}", file=sys.stderr)
                return 3

            paths = write_reports(run.to_json(), args.output)
            if run.stop_reason != StopReason.FINISHED.value:
                incomplete += 1

            if args.quiet:
                print(paths["markdown"])
            else:
                print(summarize_for_terminal(run.to_json()))
                print(f"\n  wrote {paths['markdown']}, {paths['html'].name}, {paths['json'].name}")

    if incomplete and not args.quiet:
        print(
            f"\n{incomplete} of {len(targets)} run(s) stopped early. "
            "The reports are marked partial and contain everything that was measured.",
            file=sys.stderr,
        )
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())

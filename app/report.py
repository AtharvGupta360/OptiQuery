"""Renders a run artifact as Markdown and as a self-contained HTML page.

Two rules govern the layout, and both are about what a reader is allowed to
miss.

**Rejected hypotheses are never collapsed.** They render as a full table with
the measured numbers that killed each one, at the same visual weight as the
accepted ones. A report that hides its failures is indistinguishable from a
report that never tested anything -- the rejected list is the evidence that the
agent measured rather than guessed. Only the reasoning trace is collapsible,
because it is long, not because it is unflattering.

**Every number names the function that produced it.** Runtimes come from
`verifier.benchmark`, index sizes and write-amplification text from
`verifier._index_report`, plan counts from `tools.explain_query`. The
methodology block at the bottom of every report states this, so a reader who
distrusts a figure knows exactly which code to go read.

Both renderers take the plain JSON artifact rather than the `AgentRun` object,
because a saved run reloaded from disk must render identically to one held in
memory -- and because Phase 6 consumes that same JSON.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

#: Digest prefix shown in tables. The full value stays in the JSON artifact.
CHECKSUM_CHARS = 12


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _ms(value: Any) -> str:
    if value is None:
        return "--"
    return f"{float(value):,.1f} ms"


def _pct(value: Any) -> str:
    if value is None:
        return "--"
    # Explicit sign: an improvement and a regression must not look alike.
    return f"{float(value):+.1f}%"


def _speedup(value: Any) -> str:
    if not value:
        return "--"
    return f"{float(value):.1f}x"


def _digest(value: Any) -> str:
    if not value:
        return "--"
    return str(value)[:CHECKSUM_CHARS]


def _int(value: Any) -> str:
    return "--" if value is None else f"{int(value):,}"


def _timestamp(value: Any) -> str:
    if not value:
        return "unknown"
    try:
        return datetime.fromisoformat(str(value)).strftime("%Y-%m-%d %H:%M:%S UTC")
    except ValueError:
        return str(value)


def _kind_label(hypothesis: dict[str, Any]) -> str:
    return str(hypothesis.get("kind") or "?")


# ---------------------------------------------------------------------------
# Derived views over the artifact
# ---------------------------------------------------------------------------


def best_accepted(run: dict[str, Any]) -> dict[str, Any] | None:
    """The accepted hypothesis with the largest measured improvement.

    The headline number has to come from a single hypothesis. Improvements from
    separate hypotheses are each measured against the same baseline on a clean
    shadow, so they are not additive and summing them would invent a speedup
    nobody measured.
    """
    accepted = run.get("accepted") or []
    if not accepted:
        return None
    return max(accepted, key=lambda result: float(result.get("improvement_pct") or 0.0))


def plan_summary(run: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """The plan the agent actually looked at, and which query it describes.

    Prefers an EXPLAIN of the original query. A run may also have explained a
    candidate rewrite, and labelling that as "the plan" would attribute the
    wrong scan counts to the query under test.
    """
    original = _normalize(run.get("original_sql", ""))
    fallback: dict[str, Any] | None = None

    for iteration in run.get("trace") or []:
        for call in iteration.get("tool_calls") or []:
            if call.get("name") != "explain_query" or call.get("is_error"):
                continue
            summary = (call.get("observation") or {}).get("summary")
            if not summary:
                continue
            if _normalize(str((call.get("arguments") or {}).get("sql", ""))) == original:
                return summary, "the original query"
            fallback = fallback or summary
    return fallback, "a candidate query, not the original"


def _normalize(sql: str) -> str:
    return " ".join(sql.split()).rstrip(";").lower()


def headline(run: dict[str, Any]) -> str:
    """One sentence a reader can stop after."""
    baseline = run.get("baseline") or {}
    winner = best_accepted(run)
    if winner is None:
        tested = len(run.get("accepted") or []) + len(run.get("rejected") or [])
        if tested == 0:
            return "No hypothesis was measured. Nothing is recommended."
        return (
            f"{tested} hypothesis(es) measured, none met the acceptance bar. "
            "The query is left as it is."
        )
    optimized = winner.get("optimized") or {}
    return (
        f"{_ms(baseline.get('median_ms'))} -> {_ms(optimized.get('median_ms'))} "
        f"({_speedup(winner.get('speedup'))} faster, {_pct(winner.get('improvement_pct'))}), "
        f"verified byte-identical over {_int(optimized.get('row_count'))} rows."
    )


def _rejection_reason(result: dict[str, Any]) -> str:
    reasons = result.get("reasons") or []
    if reasons:
        return str(reasons[0])
    return "no reason recorded"


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def render_markdown(run: dict[str, Any]) -> str:
    out: list[str] = []
    add = out.append

    baseline = run.get("baseline") or {}
    winner = best_accepted(run)
    accepted = run.get("accepted") or []
    rejected = run.get("rejected") or []

    add(f"# OptiQuery: `{run.get('query_name', 'query')}`")
    add("")
    add(f"**{headline(run)}**")
    add("")

    if run.get("partial"):
        add(
            f"> **Partial run.** Stopped because `{run.get('stop_reason')}`. "
            "Everything below was still measured; the agent simply did not get "
            "to propose more."
        )
        add("")
    if run.get("error"):
        # Fenced rather than quoted: provider errors arrive as multi-line JSON,
        # and a blockquote only marks its first line, so everything after it
        # renders as body text belonging to the report.
        add("**Error**")
        add("")
        add("```text")
        add(_truncate(str(run["error"]), 1200))
        add("```")
        add("")

    # -- before / after --------------------------------------------------
    add("## Measured result")
    add("")
    add("| | Median | Rows | Result checksum |")
    add("|---|---:|---:|---|")
    add(
        f"| Before | {_ms(baseline.get('median_ms'))} | {_int(baseline.get('row_count'))} "
        f"| `{_digest(baseline.get('checksum'))}` |"
    )
    if winner:
        optimized = winner.get("optimized") or {}
        add(
            f"| After | {_ms(optimized.get('median_ms'))} | {_int(optimized.get('row_count'))} "
            f"| `{_digest(optimized.get('checksum'))}` |"
        )
    else:
        add("| After | _no accepted change_ | | |")
    add("")
    if winner:
        add(
            "The two checksums are equal, which is what makes this a speedup "
            "rather than a different query."
        )
        add("")

    # -- diagnosis -------------------------------------------------------
    if run.get("diagnosis"):
        add("## Diagnosis")
        add("")
        add(str(run["diagnosis"]))
        add("")

    # -- the query -------------------------------------------------------
    add("## Original query")
    add("")
    add("```sql")
    add(str(run.get("original_sql", "")).strip())
    add("```")
    add("")

    summary, describes = plan_summary(run)
    if summary:
        add(f"### Plan ({describes})")
        add("")
        add(
            f"- {_int(summary.get('node_count'))} plan nodes, "
            f"{len(summary.get('sequential_scans') or [])} sequential scan(s)"
        )
        add(
            f"- Sequential scans read {_int(summary.get('rows_read_by_sequential_scans'))} rows "
            f"and discarded {_int(summary.get('rows_discarded_by_sequential_scans'))}"
        )
        for scan in summary.get("sequential_scans") or []:
            add(
                f"  - `{scan.get('relation')}`: read "
                f"{_int((scan.get('actual_rows') or 0) + (scan.get('rows_removed_by_filter') or 0))}, "
                f"kept {_int(scan.get('actual_rows'))}"
            )
        add("")
        add(
            "_Plan timings are omitted deliberately: EXPLAIN ANALYZE attaches "
            "per-tuple instrumentation that inflates them several-fold. Every "
            "runtime in this report comes from `verifier.benchmark` instead._"
        )
        add("")

    # -- recommendations -------------------------------------------------
    add(f"## Recommendations ({len(run.get('recommendations') or [])})")
    add("")
    recommendations = run.get("recommendations") or []
    if not recommendations:
        add(
            "None. No proposed change both preserved the result set and beat the "
            f"{_pct_threshold(rejected, accepted)} improvement threshold."
        )
        add("")
    for position, item in enumerate(recommendations, start=1):
        add(f"### {position}. {item.get('summary', 'change')}")
        add("")
        for ddl in _production_ddls(item):
            add("```sql")
            add(ddl)
            add("```")
        if item.get("rewritten_sql"):
            add("```sql")
            add(str(item["rewritten_sql"]).strip())
            add("```")
        add("")
        if item.get("before_ms") is not None:
            add(
                f"- Measured: {_ms(item.get('before_ms'))} -> {_ms(item.get('after_ms'))} "
                f"({_pct(item.get('improvement_pct'))}, {_speedup(item.get('speedup'))} faster)"
            )
        if item.get("source") == "recovered_from_accepted_hypotheses":
            add(
                "- Recovered from an accepted hypothesis: the run ended before the "
                "agent summarised it, but the measurement stands."
            )
        for report in _index_reports_for(run, item):
            add(
                f"- Index `{report.get('index_name')}` on `{report.get('table')}`: "
                f"{report.get('size_pretty')} ({report.get('pct_of_table')}% of the "
                f"{report.get('table_size_pretty')} heap), built in "
                f"{report.get('build_ms')} ms"
            )
            add(f"- Write cost: {report.get('write_amplification')}")
        add("")

    # -- rejected: never collapsed --------------------------------------
    add(f"## Rejected hypotheses ({len(rejected)})")
    add("")
    if not rejected:
        add("_None: every hypothesis the agent tested was accepted._")
        add("")
    else:
        add(
            "Each of these was applied to a clean shadow database and measured. "
            "They are listed because a report without its failures cannot be "
            "told apart from one that never tested anything."
        )
        add("")
        add("| ID | Kind | Proposal | Before | After | Change | Checksum | Why rejected |")
        add("|---|---|---|---:|---:|---:|---|---|")
        for result in rejected:
            hypothesis = result.get("hypothesis") or {}
            before = (result.get("baseline") or {}).get("median_ms")
            after = (result.get("optimized") or {}).get("median_ms")
            add(
                f"| `{hypothesis.get('hypothesis_id')}` "
                f"| {_kind_label(hypothesis)} "
                f"| {_escape_pipes(hypothesis.get('summary', ''))} "
                f"| {_ms(before)} | {_ms(after)} | {_pct(result.get('improvement_pct'))} "
                f"| {'match' if result.get('checksum_match') else '**DIFFERS**'} "
                f"| {_escape_pipes(_rejection_reason(result))} |"
            )
        add("")
        for result in rejected:
            ddls = (result.get("hypothesis") or {}).get("index_ddls") or []
            rewrite = (result.get("hypothesis") or {}).get("rewritten_sql")
            if not ddls and not rewrite:
                continue
            add(f"<details><summary>What <code>{(result.get('hypothesis') or {}).get('hypothesis_id')}</code> proposed</summary>")
            add("")
            for ddl in ddls:
                add("```sql")
                add(ddl)
                add("```")
            if rewrite:
                add("```sql")
                add(str(rewrite).strip())
                add("```")
            add("")
            add("</details>")
            add("")

    # -- unverified claims ----------------------------------------------
    unverified = run.get("unverified_claims") or []
    if unverified:
        add(f"## Unverified claims ({len(unverified)})")
        add("")
        add(
            "The agent asserted these without testing them, so they were dropped "
            "from the recommendations. They are printed rather than deleted so "
            "the omission is visible."
        )
        add("")
        for item in unverified:
            add(f"- **{item.get('summary', 'claim')}** -- {item.get('dropped_because')}")
        add("")

    # -- run metadata ----------------------------------------------------
    add("## Run")
    add("")
    tokens = run.get("tokens") or {}
    add(f"- Model: `{run.get('model')}`")
    add(f"- Iterations: {run.get('iterations_used')} of {run.get('max_iterations')}")
    add(f"- Stop reason: `{run.get('stop_reason')}`")
    add(f"- Tokens: {_int(tokens.get('total_tokens'))} of {_int(run.get('token_budget'))}")
    add(f"- Started: {_timestamp(run.get('started_at'))}")
    add(f"- Finished: {_timestamp(run.get('finished_at'))}")
    drift = run.get("baseline_drift") or {}
    if drift.get("drift_pct") is not None:
        add(
            f"- Baseline drift: {_pct(drift.get('drift_pct'))} between the first and "
            f"last measurement of the unmodified query "
            f"({_ms(drift.get('first_median_ms'))} -> {_ms(drift.get('final_median_ms'))}). "
            "Improvements smaller than this are not distinguishable from the machine."
        )
    elif drift.get("error"):
        add(f"- Baseline drift: could not be measured -- `{drift['error']}`")
    add("")

    add(_METHODOLOGY_MD)
    add("")

    # -- trace: the only collapsible section ----------------------------
    add("<details>")
    add("<summary>Reasoning trace (%d iterations)</summary>" % len(run.get("trace") or []))
    add("")
    for iteration in run.get("trace") or []:
        add(f"#### Iteration {iteration.get('iteration')}")
        add("")
        if iteration.get("thinking"):
            add("> " + str(iteration["thinking"]).replace("\n", "\n> "))
            add("")
        if iteration.get("text"):
            add(str(iteration["text"]))
            add("")
        for call in iteration.get("tool_calls") or []:
            flag = " [ERROR]" if call.get("is_error") else ""
            add(f"**`{call.get('name')}`**{flag} ({call.get('duration_ms')} ms)")
            add("")
            add("```json")
            add(json.dumps(call.get("arguments"), indent=2, default=str))
            add("```")
            add("")
            add("```json")
            add(_truncate(json.dumps(call.get("observation"), indent=2, default=str)))
            add("```")
            add("")
    add("</details>")
    add("")

    return "\n".join(out)


def _pct_threshold(rejected: Sequence[dict[str, Any]], accepted: Sequence[dict[str, Any]]) -> str:
    for result in list(rejected) + list(accepted):
        value = result.get("min_improvement_pct")
        if value is not None:
            return f"{float(value):.0f}%"
    return "required"


def _production_ddls(item: dict[str, Any]) -> list[str]:
    """The deployable form, preferring CONCURRENTLY where the verifier built it."""
    production = item.get("production_ddl")
    if isinstance(production, list) and production:
        return [str(ddl) for ddl in production]
    if isinstance(production, str) and production:
        return [production]
    return [str(item["ddl"])] if item.get("ddl") else []


def _index_reports_for(run: dict[str, Any], item: dict[str, Any]) -> list[dict[str, Any]]:
    """Size and write-cost detail, looked up from the hypothesis that was measured."""
    wanted = item.get("hypothesis_id")
    for result in run.get("accepted") or []:
        if (result.get("hypothesis") or {}).get("hypothesis_id") == wanted:
            return result.get("indexes") or []
    return []


def _escape_pipes(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [{len(text) - limit:,} more characters in the JSON artifact]"


_METHODOLOGY_MD = """\
### How these numbers were produced

- **Runtimes** -- `verifier.benchmark`: one warm-up execution whose timing is
  discarded so the comparison is not measuring the page cache, then the median
  of the remaining runs, timed around execute+fetch with no EXPLAIN
  instrumentation attached.
- **Equivalence** -- `verifier.result_checksum`: every row serialised to a
  type-tagged, length-framed canonical form, sorted, and hashed with sha256. A
  change that alters the result set is rejected regardless of how fast it is.
- **Index size and write cost** -- `verifier._index_report`, from
  `pg_relation_size` on shadow after the build.
- **Plan structure** -- `tools.explain_query`.
- **Attribution** -- every hypothesis is applied to a shadow database reset to
  its baseline index set, and shadow is reset again afterwards, so no
  measurement inherits an index from an earlier hypothesis.

Shadow carries production data volume but takes no production write traffic, so
write-path costs are **estimated from index size and build time, not
measured**."""


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_CSS = """\
:root {
  --bg: #ffffff; --fg: #15171a; --muted: #5b6470; --line: #e2e6ea;
  --panel: #f7f9fa; --accept: #0f7a3d; --accept-bg: #e8f6ed;
  --reject: #b3261e; --reject-bg: #fdecea; --warn: #8a5a00; --mono-bg: #f2f4f6;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1115; --fg: #e6e9ee; --muted: #9aa4b2; --line: #262c36;
    --panel: #161a21; --accept: #4ade80; --accept-bg: #10281a;
    --reject: #f87171; --reject-bg: #2a1414; --warn: #fbbf24; --mono-bg: #161a21;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.wrap { max-width: 1080px; margin: 0 auto; padding: 32px 20px 64px; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 32px 0 10px; padding-bottom: 6px;
     border-bottom: 1px solid var(--line); text-transform: uppercase;
     letter-spacing: .06em; color: var(--muted); }
h3 { font-size: 14px; margin: 20px 0 8px; }
code, pre, .num, td.num { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
pre { background: var(--mono-bg); border: 1px solid var(--line); border-radius: 4px;
      padding: 12px; overflow-x: auto; font-size: 12.5px; margin: 8px 0; }
code { background: var(--mono-bg); padding: 1px 4px; border-radius: 3px; font-size: 12.5px; }
pre code { background: none; padding: 0; }
.headline { font-size: 17px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            margin: 12px 0 4px; }
.sub { color: var(--muted); font-size: 12.5px; }
.banner { border-left: 3px solid var(--warn); background: var(--panel);
          padding: 10px 14px; margin: 16px 0; font-size: 13px; }
.banner.err { border-left-color: var(--reject); background: var(--reject-bg); }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 8px 0; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line);
         vertical-align: top; }
th { font-size: 11px; text-transform: uppercase; letter-spacing: .05em;
     color: var(--muted); font-weight: 600; white-space: nowrap; }
td.num, th.num { text-align: right; white-space: nowrap;
                 font-variant-numeric: tabular-nums; }
tr.accepted td:first-child { border-left: 3px solid var(--accept); }
tr.rejected td:first-child { border-left: 3px solid var(--reject); }
.tag { display: inline-block; padding: 1px 7px; border-radius: 3px; font-size: 11px;
       font-weight: 600; letter-spacing: .04em; }
.tag.accept { color: var(--accept); background: var(--accept-bg); }
.tag.reject { color: var(--reject); background: var(--reject-bg); }
.meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
        gap: 10px 20px; font-size: 13px; }
.meta div { border-left: 2px solid var(--line); padding-left: 10px; }
.meta .k { color: var(--muted); font-size: 11px; text-transform: uppercase;
           letter-spacing: .05em; }
.meta .v { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
details { border: 1px solid var(--line); border-radius: 4px; padding: 8px 12px; margin: 10px 0; }
summary { cursor: pointer; font-size: 13px; color: var(--muted); }
.iter { border-left: 2px solid var(--line); padding-left: 12px; margin: 14px 0; }
.think { color: var(--muted); font-style: italic; white-space: pre-wrap; font-size: 12.5px; }
.note { color: var(--muted); font-size: 12px; margin: 8px 0; }
"""


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _sql_block(sql: Any) -> str:
    return f"<pre><code>{_e(str(sql).strip())}</code></pre>"


def render_html(run: dict[str, Any]) -> str:
    baseline = run.get("baseline") or {}
    winner = best_accepted(run)
    rejected = run.get("rejected") or []
    recommendations = run.get("recommendations") or []
    parts: list[str] = []
    add = parts.append

    add(f"<h1>OptiQuery &mdash; <code>{_e(run.get('query_name'))}</code></h1>")
    add(f'<div class="headline">{_e(headline(run))}</div>')
    add(
        f'<div class="sub">{_e(run.get("model"))} &middot; '
        f'{_e(run.get("iterations_used"))}/{_e(run.get("max_iterations"))} iterations &middot; '
        f'{_timestamp(run.get("started_at"))}</div>'
    )

    if run.get("partial"):
        add(
            f'<div class="banner"><strong>Partial run.</strong> Stopped because '
            f'<code>{_e(run.get("stop_reason"))}</code>. Everything below was still '
            f'measured.</div>'
        )
    if run.get("error"):
        add(
            '<div class="banner err"><strong>Error</strong>'
            f'<pre><code>{_e(_truncate(str(run["error"]), 1200))}</code></pre></div>'
        )

    # -- before / after --------------------------------------------------
    add("<h2>Measured result</h2>")
    rows = [
        f'<tr><td>Before</td><td class="num">{_ms(baseline.get("median_ms"))}</td>'
        f'<td class="num">{_int(baseline.get("row_count"))}</td>'
        f'<td><code>{_digest(baseline.get("checksum"))}</code></td></tr>'
    ]
    if winner:
        optimized = winner.get("optimized") or {}
        rows.append(
            f'<tr class="accepted"><td>After</td>'
            f'<td class="num">{_ms(optimized.get("median_ms"))}</td>'
            f'<td class="num">{_int(optimized.get("row_count"))}</td>'
            f'<td><code>{_digest(optimized.get("checksum"))}</code></td></tr>'
        )
    else:
        rows.append('<tr><td>After</td><td class="num">--</td><td class="num">--</td><td>no accepted change</td></tr>')
    add(
        '<div class="scroll"><table><thead><tr><th></th><th class="num">Median</th>'
        '<th class="num">Rows</th><th>Result checksum</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )
    if winner:
        add(
            '<p class="note">The checksums are equal, which is what makes this a '
            "speedup rather than a different query.</p>"
        )

    if run.get("diagnosis"):
        add("<h2>Diagnosis</h2>")
        add(f"<p>{_e(run['diagnosis'])}</p>")

    add("<h2>Original query</h2>")
    add(_sql_block(run.get("original_sql", "")))

    summary, describes = plan_summary(run)
    if summary:
        add(f"<h3>Plan ({_e(describes)})</h3><ul>")
        add(
            f"<li>{_int(summary.get('node_count'))} plan nodes, "
            f"{len(summary.get('sequential_scans') or [])} sequential scan(s)</li>"
        )
        add(
            f"<li>Sequential scans read "
            f"<strong>{_int(summary.get('rows_read_by_sequential_scans'))}</strong> rows, "
            f"discarded {_int(summary.get('rows_discarded_by_sequential_scans'))}</li>"
        )
        for scan in summary.get("sequential_scans") or []:
            kept = scan.get("actual_rows") or 0
            read = kept + (scan.get("rows_removed_by_filter") or 0)
            add(
                f"<li><code>{_e(scan.get('relation'))}</code>: read {_int(read)}, "
                f"kept {_int(kept)}</li>"
            )
        add("</ul>")
        add(
            '<p class="note">Plan timings are omitted deliberately: EXPLAIN ANALYZE '
            "attaches per-tuple instrumentation that inflates them several-fold. "
            "Every runtime here comes from <code>verifier.benchmark</code>.</p>"
        )

    # -- recommendations -------------------------------------------------
    add(f"<h2>Recommendations ({len(recommendations)})</h2>")
    if not recommendations:
        add(
            '<p class="note">None. No proposed change both preserved the result set '
            "and beat the improvement threshold.</p>"
        )
    for position, item in enumerate(recommendations, start=1):
        add(f'<h3><span class="tag accept">ACCEPTED</span> {position}. {_e(item.get("summary"))}</h3>')
        for ddl in _production_ddls(item):
            add(_sql_block(ddl))
        if item.get("rewritten_sql"):
            add(_sql_block(item["rewritten_sql"]))
        if item.get("before_ms") is not None:
            add(
                f'<p class="sub">Measured {_ms(item.get("before_ms"))} &rarr; '
                f'{_ms(item.get("after_ms"))} ({_pct(item.get("improvement_pct"))}, '
                f'{_speedup(item.get("speedup"))} faster)</p>'
            )
        for report in _index_reports_for(run, item):
            add(
                f'<p class="sub">Index <code>{_e(report.get("index_name"))}</code> on '
                f'<code>{_e(report.get("table"))}</code>: {_e(report.get("size_pretty"))} '
                f'({_e(report.get("pct_of_table"))}% of the '
                f'{_e(report.get("table_size_pretty"))} heap), built in '
                f'{_e(report.get("build_ms"))} ms</p>'
            )
            add(f'<p class="note">{_e(report.get("write_amplification"))}</p>')

    # -- rejected: full table, never collapsed --------------------------
    add(f"<h2>Rejected hypotheses ({len(rejected)})</h2>")
    if not rejected:
        add('<p class="note">None: every hypothesis the agent tested was accepted.</p>')
    else:
        add(
            '<p class="note">Each was applied to a clean shadow database and measured. '
            "They are listed at full weight because a report without its failures "
            "cannot be told apart from one that never tested anything.</p>"
        )
        body = []
        for result in rejected:
            hypothesis = result.get("hypothesis") or {}
            before = (result.get("baseline") or {}).get("median_ms")
            after = (result.get("optimized") or {}).get("median_ms")
            checksum = (
                "match"
                if result.get("checksum_match")
                else '<strong style="color:var(--reject)">DIFFERS</strong>'
            )
            proposal = "".join(
                _sql_block(ddl) for ddl in (hypothesis.get("index_ddls") or [])
            )
            if hypothesis.get("rewritten_sql"):
                proposal += _sql_block(hypothesis["rewritten_sql"])
            body.append(
                f'<tr class="rejected">'
                f'<td><code>{_e(hypothesis.get("hypothesis_id"))}</code><br>'
                f'<span class="sub">{_e(hypothesis.get("kind"))}</span></td>'
                f"<td>{_e(hypothesis.get('summary'))}{proposal}</td>"
                f'<td class="num">{_ms(before)}</td>'
                f'<td class="num">{_ms(after)}</td>'
                f'<td class="num">{_pct(result.get("improvement_pct"))}</td>'
                f"<td>{checksum}</td>"
                f"<td>{_e(_rejection_reason(result))}</td></tr>"
            )
        add(
            '<div class="scroll"><table><thead><tr><th>ID</th><th>Proposal</th>'
            '<th class="num">Before</th><th class="num">After</th>'
            '<th class="num">Change</th><th>Checksum</th><th>Why rejected</th>'
            "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"
        )

    unverified = run.get("unverified_claims") or []
    if unverified:
        add(f"<h2>Unverified claims ({len(unverified)})</h2>")
        add(
            '<p class="note">Asserted without being tested, so dropped from the '
            "recommendations. Printed rather than deleted so the omission is "
            "visible.</p><ul>"
        )
        for item in unverified:
            add(f"<li><strong>{_e(item.get('summary'))}</strong> &mdash; {_e(item.get('dropped_because'))}</li>")
        add("</ul>")

    # -- run metadata ----------------------------------------------------
    tokens = run.get("tokens") or {}
    drift = run.get("baseline_drift") or {}
    add("<h2>Run</h2><div class='meta'>")
    for key, value in (
        ("Model", run.get("model")),
        ("Stop reason", run.get("stop_reason")),
        ("Iterations", f"{run.get('iterations_used')} / {run.get('max_iterations')}"),
        ("Tokens", f"{_int(tokens.get('total_tokens'))} / {_int(run.get('token_budget'))}"),
        ("Started", _timestamp(run.get("started_at"))),
        ("Finished", _timestamp(run.get("finished_at"))),
    ):
        add(f'<div><div class="k">{_e(key)}</div><div class="v">{_e(value)}</div></div>')
    add("</div>")
    if drift.get("drift_pct") is not None:
        add(
            f'<p class="note">Baseline drift {_pct(drift.get("drift_pct"))} between the '
            f'first and last measurement of the unmodified query '
            f'({_ms(drift.get("first_median_ms"))} &rarr; {_ms(drift.get("final_median_ms"))}). '
            "Improvements smaller than this are not distinguishable from the machine.</p>"
        )
    elif drift.get("error"):
        add(f'<p class="note">Baseline drift could not be measured: <code>{_e(drift["error"])}</code></p>')

    add(_METHODOLOGY_HTML)

    # -- trace: the only collapsible section ----------------------------
    trace = run.get("trace") or []
    add(f"<h2>Reasoning trace</h2>")
    add(f"<details><summary>{len(trace)} iteration(s) &mdash; click to expand</summary>")
    for iteration in trace:
        add(f'<div class="iter"><h3>Iteration {_e(iteration.get("iteration"))}</h3>')
        if iteration.get("thinking"):
            add(f'<div class="think">{_e(iteration["thinking"])}</div>')
        if iteration.get("text"):
            add(f"<p>{_e(iteration['text'])}</p>")
        for call in iteration.get("tool_calls") or []:
            flag = ' <span class="tag reject">ERROR</span>' if call.get("is_error") else ""
            add(
                f'<p><code>{_e(call.get("name"))}</code>{flag} '
                f'<span class="sub">{_e(call.get("duration_ms"))} ms</span></p>'
            )
            add(f"<pre><code>{_e(json.dumps(call.get('arguments'), indent=2, default=str))}</code></pre>")
            add(
                "<pre><code>"
                + _e(_truncate(json.dumps(call.get("observation"), indent=2, default=str)))
                + "</code></pre>"
            )
        add("</div>")
    add("</details>")

    title = f"OptiQuery: {run.get('query_name', 'query')}"
    return (
        "<!doctype html>\n<html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{_e(title)}</title><style>{_CSS}</style></head><body>"
        f'<div class="wrap">{"".join(parts)}</div></body></html>'
    )


_METHODOLOGY_HTML = """\
<h2>How these numbers were produced</h2>
<ul>
<li><strong>Runtimes</strong> &mdash; <code>verifier.benchmark</code>: one warm-up
execution whose timing is discarded so the comparison is not measuring the page
cache, then the median of the remaining runs, timed around execute+fetch with no
EXPLAIN instrumentation attached.</li>
<li><strong>Equivalence</strong> &mdash; <code>verifier.result_checksum</code>: every
row serialised to a type-tagged, length-framed canonical form, sorted, and hashed
with sha256. A change that alters the result set is rejected regardless of how
fast it is.</li>
<li><strong>Index size and write cost</strong> &mdash;
<code>verifier._index_report</code>, from <code>pg_relation_size</code> on shadow
after the build.</li>
<li><strong>Plan structure</strong> &mdash; <code>tools.explain_query</code>.</li>
<li><strong>Attribution</strong> &mdash; every hypothesis is applied to a shadow
database reset to its baseline index set, and shadow is reset again afterwards,
so no measurement inherits an index from an earlier hypothesis.</li>
</ul>
<p class="note">Shadow carries production data volume but takes
no production write traffic, so write-path costs are
<strong>estimated from index size and build time, not measured</strong>.</p>"""


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def write_reports(run: dict[str, Any], directory: Path | str) -> dict[str, Path]:
    """Write `<name>.md`, `<name>.html` and `<name>.json` side by side.

    The JSON goes out with them because every figure in the rendered reports is
    rounded for reading, and the artifact is what Phase 6 renders and what a
    reader checks the rounding against.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = str(run.get("query_name") or "run")

    paths = {
        "markdown": directory / f"{name}.md",
        "html": directory / f"{name}.html",
        "json": directory / f"{name}.json",
    }
    paths["markdown"].write_text(render_markdown(run), encoding="utf-8")
    paths["html"].write_text(render_html(run), encoding="utf-8")
    paths["json"].write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
    return paths


def summarize_for_terminal(run: dict[str, Any]) -> str:
    """Compact stdout summary for the CLI. The files hold the detail."""
    lines: list[str] = []
    accepted = run.get("accepted") or []
    rejected = run.get("rejected") or []

    lines.append(f"  {headline(run)}")
    lines.append("")
    lines.append(
        f"  {len(accepted)} accepted, {len(rejected)} rejected, "
        f"{run.get('iterations_used')}/{run.get('max_iterations')} iterations, "
        f"stop={run.get('stop_reason')}"
    )
    for item in run.get("recommendations") or []:
        for ddl in _production_ddls(item):
            lines.append(f"    + {ddl}")
        if item.get("rewritten_sql"):
            lines.append(f"    + rewrite: {' '.join(str(item['rewritten_sql']).split())[:90]}")
    for result in rejected:
        hypothesis = result.get("hypothesis") or {}
        lines.append(
            f"    - {hypothesis.get('hypothesis_id')}: "
            f"{_pct(result.get('improvement_pct'))}, "
            f"checksum {'match' if result.get('checksum_match') else 'DIFFERS'} "
            f"-- {_rejection_reason(result)[:80]}"
        )
    return "\n".join(lines)

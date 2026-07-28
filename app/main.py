"""HTTP surface: POST /optimize.

One design decision dominates this file: **optimisation runs are serialised**
behind a lock. Two concurrent requests would each apply indexes to the same
shadow database and reset it underneath the other, so both would report timings
for a database neither one described. The verifier's whole claim is that a
measurement is attributable to exactly the change proposed, and concurrency
would silently break that while every response still looked plausible.

Serialising costs throughput this service does not need. It is a local
inspection tool, not a multi-tenant API.

Run with:

    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from app.agent import AgentConfig, optimize_query
from app.db import DatabaseConfig, SqlGuardError, assert_read_only
from app.llm import LLMError, build_client
from app.report import render_html, render_markdown, write_reports
from app.shadow import ShadowIsolationError
from app.tools import ToolContext

#: Held for the entire duration of a run, including its benchmarks.
_RUN_LOCK = threading.Lock()

#: Populated at startup so the first request does not pay for connecting and
#: bootstrapping the read-only role.
_STATE: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = DatabaseConfig.from_env()
    context = ToolContext.open(config)
    _STATE["ctx"] = context.__enter__()
    try:
        yield
    finally:
        context.__exit__(None, None, None)
        _STATE.clear()


app = FastAPI(
    title="OptiQuery",
    description=(
        "An agent proposes Postgres query optimisations; a deterministic verifier "
        "applies each one to a shadow database, benchmarks it, and rejects anything "
        "that does not measurably win while returning byte-identical results."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class OptimizeRequest(BaseModel):
    query: str = Field(..., description="The SQL to optimise. Read-only statements only.")
    name: str | None = Field(None, description="Artifact name. Defaults to a hash of the query.")
    max_iterations: int | None = Field(None, ge=1, le=50)
    runs: int | None = Field(None, ge=1, le=25, description="Timed benchmark runs per measurement.")
    provider: str | None = None
    model: str | None = None
    format: Literal["json", "markdown", "html"] = "json"
    save: bool = Field(False, description="Also write .md/.html/.json into ./runs.")


@app.get("/health")
def health() -> dict[str, Any]:
    ctx: ToolContext | None = _STATE.get("ctx")
    if ctx is None:
        raise HTTPException(status_code=503, detail="database context not initialised")
    return {
        "status": "ok",
        "primary_read_only": ctx.primary.is_read_only(),
        "shadow_baseline_indexes": len(ctx.shadow.baseline_index_names()),
        "busy": _RUN_LOCK.locked(),
    }


@app.post("/optimize")
def optimize(request: OptimizeRequest):
    ctx: ToolContext | None = _STATE.get("ctx")
    if ctx is None:
        raise HTTPException(status_code=503, detail="database context not initialised")

    try:
        assert_read_only(request.query)
    except SqlGuardError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        bundle = build_client(provider=request.provider, model=request.model)
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    overrides: dict[str, Any] = {}
    if request.max_iterations is not None:
        overrides["max_iterations"] = request.max_iterations
    if request.runs is not None:
        overrides["benchmark_runs"] = request.runs
    config = AgentConfig.from_env(model=bundle.model, **overrides)

    name = request.name or _derive_name(request.query)

    # 409 rather than queueing: a run takes minutes, and a caller blocked on a
    # lock with no feedback cannot tell the difference between slow and hung.
    if not _RUN_LOCK.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail=(
                "another optimisation is running. Runs are serialised because they "
                "share one shadow database; concurrent runs would reset it underneath "
                "each other and produce unattributable measurements."
            ),
        )
    try:
        run = optimize_query(ctx, bundle.client, name, request.query, config)
    except ShadowIsolationError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"shadow database is no longer a faithful clone of primary: {exc}",
        ) from exc
    finally:
        _RUN_LOCK.release()

    payload = run.to_json()
    if request.save:
        write_reports(payload, "runs")

    if request.format == "markdown":
        return PlainTextResponse(render_markdown(payload), media_type="text/markdown")
    if request.format == "html":
        return HTMLResponse(render_html(payload))
    return payload


def _derive_name(sql: str) -> str:
    from hashlib import sha256

    words = [word.lower() for word in sql.split() if word.isalpha()][:3]
    stem = "_".join(words) or "query"
    return f"{stem}_{sha256(sql.encode('utf-8')).hexdigest()[:8]}"

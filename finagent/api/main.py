"""Minimal FastAPI wrapper around the orchestration loop.

This is deliberately thin and NOT Phase 5: the endpoint below runs the whole
pipeline synchronously and returns the full event trace + final result in one
JSON response once it's done, rather than streaming events as they happen.
Phase 5 replaces this handler's body with an SSE generator that yields each
event as `EventEmitter` produces it - the event schema and orchestration loop
underneath don't change, only how results reach the client.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from finagent.agents.analyst import Analyst
from finagent.agents.critic import Critic
from finagent.agents.planner import Planner
from finagent.config import get_settings
from finagent.data import stub_source
from finagent.orchestration.loop import run_research

app = FastAPI(title="FinAgent API")

_STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


class ResearchRequest(BaseModel):
    question: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.post("/api/research")
def research(request: ResearchRequest) -> dict[str, Any]:
    settings = get_settings()
    planner = Planner()
    analyst = Analyst(fetch=stub_source.fetch)
    critic = Critic()

    collected_events: list[dict[str, Any]] = []

    def collect(event) -> None:
        collected_events.append({"type": event.type, "data": event.data, "timestamp": event.timestamp})

    result = run_research(
        request.question,
        planner=planner,
        analyst=analyst,
        critic=critic,
        max_revisions=settings.max_revisions,
        event_sink=collect,
    )

    return {
        "events": collected_events,
        "final_outputs": [
            {
                "task_id": output.task_id,
                "summary": output.summary,
                "claims": [
                    {"text": c.text, "supporting_snippet_ids": c.supporting_snippet_ids}
                    for c in output.claims
                ],
            }
            for output in result.final_outputs
        ],
    }

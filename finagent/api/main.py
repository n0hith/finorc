"""FastAPI wrapper around the orchestration loop, streaming events via SSE.

This is still deliberately thin (no auth, no persistence, one in-memory
pipeline run per request) but is now real Phase 5: `POST /api/research`
streams each orchestration event to the client as it happens, rather than
collecting the whole run into one JSON response.

Bridging sync and async: `run_research()` and the agents underneath it make
blocking Gemini/RSS calls, so the pipeline runs in a background thread that
pushes each `Event` onto a `queue.Queue`. The response body is a plain
*synchronous* generator that blocks on `queue.get()` - Starlette detects this
and iterates it in a thread pool automatically, so the server's event loop is
never blocked waiting on the pipeline. This keeps `orchestration/loop.py`
exactly as it was for Phase 1-4 (still a plain callback-based `event_sink`)
rather than rewriting it as async.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from finagent.agents.analyst import Analyst
from finagent.agents.critic import Critic
from finagent.agents.planner import Planner
from finagent.config import get_settings
from finagent.data import rss_source
from finagent.orchestration.events import Event
from finagent.orchestration.loop import run_research

app = FastAPI(title="FinAgent API")

_STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

_STREAM_DONE = object()
"""Sentinel pushed onto the queue once the background thread finishes, success or not."""


class ResearchRequest(BaseModel):
    question: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


def _run_pipeline(question: str, events_queue: "queue.Queue[Event | object]") -> None:
    """Runs the full pipeline on a background thread, pushing each event onto the queue."""
    try:
        settings = get_settings()
        run_research(
            question,
            planner=Planner(),
            analyst=Analyst(fetch=rss_source.fetch),
            critic=Critic(),
            max_revisions=settings.max_revisions,
            event_sink=events_queue.put,
        )
    except Exception as exc:
        events_queue.put(Event(type="research_error", data={"message": str(exc)}, timestamp=time.time()))
    finally:
        events_queue.put(_STREAM_DONE)


def _stream_events(question: str) -> Iterator[str]:
    events_queue: "queue.Queue[Event | object]" = queue.Queue()
    thread = threading.Thread(target=_run_pipeline, args=(question, events_queue), daemon=True)
    thread.start()

    while True:
        item = events_queue.get()
        if item is _STREAM_DONE:
            return
        yield f"data: {item.to_json()}\n\n"


@app.post("/api/research")
def research(request: ResearchRequest) -> StreamingResponse:
    return StreamingResponse(_stream_events(request.question), media_type="text/event-stream")

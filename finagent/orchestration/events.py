"""Event schema for the orchestration loop.

Every meaningful step emits an `Event`. In Phase 1 events are just logged to
stdout, but the schema is designed so Phase 5 can stream the exact same
objects to a client over SSE: each event is a flat, JSON-serializable dict
with a `type` field, a `data` payload, and a timestamp. Nothing here assumes a
synchronous/CLI context.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Callable, Literal

EventType = Literal[
    "research_started",
    "planner_done",
    "analyst_step_started",
    "analyst_step_done",
    "verification_done",
    "critic_review",
    "revision_requested",
    "research_completed",
    "research_error",
]
"""research_error is emitted only by the SSE transport (finagent/api/main.py)
when the pipeline raises - run_research() itself never emits it, since a
synchronous caller (the CLI) just lets the exception propagate normally."""


@dataclass
class Event:
    type: EventType
    data: dict[str, Any]
    timestamp: float

    def to_json(self) -> str:
        return json.dumps({"type": self.type, "data": self.data, "timestamp": self.timestamp}, default=_default)


def _default(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


EventSink = Callable[[Event], None]


def print_sink(event: Event) -> None:
    """Default sink for Phase 1: print each event as a JSON line to stdout."""
    print(event.to_json(), file=sys.stderr)


class EventEmitter:
    """Collects events and forwards each to a sink (print, SSE queue, etc.)."""

    def __init__(self, sink: EventSink = print_sink) -> None:
        self._sink = sink
        self.history: list[Event] = []

    def emit(self, event_type: EventType, **data: Any) -> Event:
        event = Event(type=event_type, data=data, timestamp=time.time())
        self.history.append(event)
        self._sink(event)
        return event

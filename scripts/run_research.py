#!/usr/bin/env python
"""CLI entry point: run a research question end-to-end and print the trace.

Usage:
    python scripts/run_research.py "Why are AI data center buildouts accelerating?"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finagent.agents.analyst import Analyst
from finagent.agents.critic import Critic
from finagent.agents.planner import Planner
from finagent.config import get_settings
from finagent.data import rss_source
from finagent.orchestration.loop import run_research

DEFAULT_QUESTION = "Why are AI data center buildouts accelerating, and what could constrain them?"


def _print_event_sink(event):
    print(f"\n=== {event.type} ===")
    print(json.dumps(event.data, indent=2, default=str))


def main() -> None:
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    settings = get_settings()

    planner = Planner()
    analyst = Analyst(fetch=rss_source.fetch)
    critic = Critic()

    result = run_research(
        question,
        planner=planner,
        analyst=analyst,
        critic=critic,
        max_revisions=settings.max_revisions,
        event_sink=_print_event_sink,
    )

    print("\n\n=== FINAL RESULT ===")
    for output in result.final_outputs:
        print(f"\n[{output.task_id}] {output.summary}")
        for claim in output.claims:
            print(f"  - {claim.text} (cites: {', '.join(claim.supporting_snippet_ids)})")


if __name__ == "__main__":
    main()

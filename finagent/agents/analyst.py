"""Analyst agent: researches one sub-task using retrieved source data.

The Analyst never free-writes claims from memory - it is only given the
snippets returned by the data source (`finagent.data.stub_source` in Phase 1,
real RSS in Phase 3) and must ground each claim in specific snippet IDs. That
citation is what the Phase 2 groundedness verifier checks.
"""

from __future__ import annotations

from typing import Callable

from finagent.agents.base import BaseAgent
from finagent.orchestration.context import AnalystOutput, Claim, CriticReview, DataSnippet, Task

_TOOL_NAME = "submit_analysis"

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "A short synthesis answering the sub-task, in your own words.",
        },
        "claims": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "A single factual claim, phrased as one specific assertion.",
                    },
                    "supporting_snippet_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of the provided snippets that directly support this claim.",
                    },
                },
                "required": ["text", "supporting_snippet_ids"],
            },
        },
    },
    "required": ["summary", "claims"],
}

DataFetcher = Callable[[str], list[DataSnippet]]


class Analyst(BaseAgent):
    system_prompt = (
        "You are the Analyst in a financial research pipeline. You will be given "
        "a sub-task and a set of source snippets (news articles, filings "
        "excerpts). Analyze ONLY the provided snippets - do not draw on outside "
        "knowledge. Produce a short summary and a list of discrete claims. Every "
        "claim MUST cite the snippet ID(s) that support it in "
        "supporting_snippet_ids. If the snippets don't support a claim, don't "
        "make it. If you are given prior feedback, address it directly."
    )

    def __init__(self, *args, fetch: DataFetcher, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._fetch = fetch

    def run(self, task: Task, revision_feedback: CriticReview | None = None) -> AnalystOutput:
        snippets = self._fetch(task.description)
        snippet_block = "\n\n".join(
            f"[{s.id}] ({s.source}, {s.published_at}) {s.title}\n{s.text}" for s in snippets
        )

        user_message = f"Sub-task: {task.description}\n\nSource snippets:\n{snippet_block}"
        if revision_feedback is not None:
            issues = "\n".join(f"- {i.claim_text}: {i.problem}" for i in revision_feedback.issues)
            user_message += (
                f"\n\nYour previous analysis was sent back for revision. "
                f"Critic notes: {revision_feedback.notes}\nSpecific issues:\n{issues}"
            )

        result = self._call_tool(
            user_message=user_message,
            tool_name=_TOOL_NAME,
            tool_description="Submit the analysis for this sub-task.",
            input_schema=_INPUT_SCHEMA,
        )
        claims = [
            Claim(text=c["text"], supporting_snippet_ids=c["supporting_snippet_ids"])
            for c in result["claims"]
        ]
        return AnalystOutput(
            task_id=task.id,
            summary=result["summary"],
            claims=claims,
            snippets_used=snippets,
        )

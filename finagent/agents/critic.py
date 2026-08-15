"""Critic agent: reviews an Analyst output for gaps or weak reasoning.

Phase 1 scope: the Critic judges reasoning quality (does the summary follow
from the claims, are there obvious gaps or unsupported leaps). Phase 2 adds a
separate, non-LLM NLI groundedness check that runs before the Critic and feeds
its results in as additional signal - the Critic doesn't need to *detect*
hallucination itself once that exists, just judge overall quality.
"""

from __future__ import annotations

from finagent.agents.base import BaseAgent
from finagent.orchestration.context import AnalystOutput, CriticReview, Issue, Task

_TOOL_NAME = "submit_review"

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["approved", "revise"]},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_text": {"type": "string"},
                    "problem": {
                        "type": "string",
                        "description": "What's wrong with this claim: unsupported, overreaching, vague, etc.",
                    },
                },
                "required": ["claim_text", "problem"],
            },
        },
        "notes": {
            "type": "string",
            "description": "Overall feedback for the Analyst, especially if verdict is 'revise'.",
        },
    },
    "required": ["verdict", "issues", "notes"],
}


class Critic(BaseAgent):
    system_prompt = (
        "You are the Critic in a financial research pipeline. Review the "
        "Analyst's summary and claims against the source snippets they were "
        "given. Flag claims that overreach the source data, are vague, "
        "contradict each other, or leave an obvious gap relevant to the "
        "sub-task. Set verdict to 'revise' if there are issues worth fixing, "
        "or 'approved' if the analysis is well-supported and reasonably "
        "complete. Be a genuinely skeptical reviewer, not a rubber stamp."
    )

    def run(self, task: Task, output: AnalystOutput) -> CriticReview:
        snippet_block = "\n\n".join(
            f"[{s.id}] ({s.source}, {s.published_at}) {s.title}\n{s.text}" for s in output.snippets_used
        )
        claims_block = "\n".join(
            f"- {c.text} (cites: {', '.join(c.supporting_snippet_ids) or 'none'})" for c in output.claims
        )
        user_message = (
            f"Sub-task: {task.description}\n\n"
            f"Source snippets:\n{snippet_block}\n\n"
            f"Analyst summary: {output.summary}\n\n"
            f"Analyst claims:\n{claims_block}"
        )
        result = self._call_tool(
            user_message=user_message,
            tool_name=_TOOL_NAME,
            tool_description="Submit your review of this analysis.",
            input_schema=_INPUT_SCHEMA,
        )
        issues = [Issue(claim_text=i["claim_text"], problem=i["problem"]) for i in result["issues"]]
        return CriticReview(task_id=task.id, verdict=result["verdict"], issues=issues, notes=result["notes"])

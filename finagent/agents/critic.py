"""Critic agent: reviews an Analyst output for gaps or weak reasoning.

The Critic judges overall reasoning quality (does the summary follow from the
claims, are there obvious gaps). It does not have to *detect* hallucination
itself: a separate, non-LLM NLI groundedness check (finagent.verification)
runs before the Critic and its per-claim verdicts are handed in as evidence,
so the Critic can reason about *why* a claim was flagged rather than having to
notice unsupported claims unaided.
"""

from __future__ import annotations

from finagent.agents.base import BaseAgent
from finagent.orchestration.context import AnalystOutput, CriticReview, Issue, Task
from finagent.verification.groundedness import GroundednessReport

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
        "complete. Be a genuinely skeptical reviewer, not a rubber stamp.\n\n"
        "You will also receive an automated groundedness check: for each "
        "claim, whether an NLI model found its cited source snippet(s) to "
        "entail it, be neutral toward it, contradict it, or find it cited no "
        "snippet at all. Treat any claim NOT marked 'entailment' as a strong "
        "signal to flag - the NLI model is not perfect, so use judgment, but "
        "a claim it could not verify should not pass review without you "
        "independently confirming it against the source snippets yourself."
    )

    def run(
        self, task: Task, output: AnalystOutput, groundedness: GroundednessReport | None = None
    ) -> CriticReview:
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
        if groundedness is not None:
            verification_block = "\n".join(
                f"- \"{v.claim_text}\": {v.verdict} (confidence {v.score:.2f}, "
                f"best snippet cited: {v.best_snippet_id or 'none'})"
                for v in groundedness.claim_verifications
            )
            user_message += f"\n\nAutomated groundedness check:\n{verification_block}"
        result = self._call_tool(
            user_message=user_message,
            tool_name=_TOOL_NAME,
            tool_description="Submit your review of this analysis.",
            input_schema=_INPUT_SCHEMA,
        )
        issues = [Issue(claim_text=i["claim_text"], problem=i["problem"]) for i in result["issues"]]
        return CriticReview(task_id=task.id, verdict=result["verdict"], issues=issues, notes=result["notes"])

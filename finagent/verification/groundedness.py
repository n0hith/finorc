"""Groundedness verification: does the Analyst's cited source data actually
support each claim it made?

For every claim, this runs NLI between the claim (hypothesis) and each
snippet it cites as support (premise), and keeps the best result. A claim is
considered grounded only if at least one cited snippet entails it; if none of
its citations entail it (best case: neutral, worst case: contradiction, or it
cited nothing at all), the claim is flagged as unsupported/hallucinated.

This runs independently of the Critic - it's a deterministic, non-LLM check
that produces evidence the Critic then reasons over, rather than an LLM
grading another LLM's homework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from finagent.orchestration.context import AnalystOutput
from finagent.verification.nli import classify

Verdict = Literal["entailment", "neutral", "contradiction", "uncited"]


@dataclass
class ClaimVerification:
    claim_text: str
    verdict: Verdict
    score: float
    best_snippet_id: str | None
    flagged: bool
    """True unless the claim is clearly entailed by at least one snippet it cites."""


@dataclass
class GroundednessReport:
    task_id: str
    claim_verifications: list[ClaimVerification]

    @property
    def flagged(self) -> list[ClaimVerification]:
        return [c for c in self.claim_verifications if c.flagged]


def verify(output: AnalystOutput) -> GroundednessReport:
    snippets_by_id = {s.id: s for s in output.snippets_used}
    verifications: list[ClaimVerification] = []

    for claim in output.claims:
        cited = [snippets_by_id[sid] for sid in claim.supporting_snippet_ids if sid in snippets_by_id]
        if not cited:
            verifications.append(
                ClaimVerification(
                    claim_text=claim.text, verdict="uncited", score=0.0, best_snippet_id=None, flagged=True
                )
            )
            continue

        scored = [(snippet.id, classify(snippet.text, claim.text)) for snippet in cited]
        entailments = [(sid, result) for sid, result in scored if result.label == "entailment"]

        if entailments:
            best_id, best_result = max(entailments, key=lambda pair: pair[1].score)
            flagged = False
        else:
            best_id, best_result = max(scored, key=lambda pair: pair[1].score)
            flagged = True

        verifications.append(
            ClaimVerification(
                claim_text=claim.text,
                verdict=best_result.label,
                score=best_result.score,
                best_snippet_id=best_id,
                flagged=flagged,
            )
        )

    return GroundednessReport(task_id=output.task_id, claim_verifications=verifications)

"""Data models shared across the orchestration loop.

Design note: a single `ResearchContext` accumulates state (plan, analyst
outputs, critic reviews) and is passed by reference through Planner -> Analyst
-> Critic. This was chosen over an immutable event-sourced chain for
simplicity: agents can read `context.analyst_outputs` directly instead of
replaying an event log, and the whole run is trivially serializable for
logging/debugging. The event *log* still exists (see events.py) but it is a
side channel for observability, not the mechanism agents use to see state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SentimentScore:
    """FinBERT's sentiment classification of one snippet's text.

    `label` is FinBERT's own vocabulary (positive/negative/neutral), kept
    as-is rather than remapped, so a snippet's sentiment can be shown or
    reasoned about without translation.
    """

    label: Literal["positive", "negative", "neutral"]
    score: float
    """Softmax probability of the predicted label, in [0, 1]."""


@dataclass
class DataSnippet:
    """A single piece of source data (news article, filing excerpt, etc.).

    Shape is intentionally generic so a real RSS ingester (Phase 3) can
    produce these without the Analyst or verifier code changing. `sentiment`
    is optional and `None` for sources that don't score it (e.g. Phase 1's
    stub) - the Analyst treats a missing score as "no signal", not as neutral.
    """

    id: str
    source: str
    title: str
    text: str
    published_at: str
    url: str
    sentiment: SentimentScore | None = None


@dataclass
class Task:
    id: str
    description: str


@dataclass
class Plan:
    tasks: list[Task]


@dataclass
class Claim:
    """One factual assertion made by the Analyst, tied to the data it drew on.

    `supporting_snippet_ids` is what the Phase 2 groundedness verifier will
    check against — did the cited snippets actually support this claim.
    """

    text: str
    supporting_snippet_ids: list[str]


@dataclass
class AnalystOutput:
    task_id: str
    summary: str
    claims: list[Claim]
    snippets_used: list[DataSnippet]
    revision_of: int | None = None
    """Index into context.analyst_outputs this revises, if any."""


@dataclass
class Issue:
    claim_text: str
    problem: str


@dataclass
class CriticReview:
    task_id: str
    verdict: Literal["approved", "revise"]
    issues: list[Issue]
    notes: str


@dataclass
class ResearchContext:
    """Mutable state threaded through the orchestration loop for one research question."""

    question: str
    plan: Plan | None = None
    analyst_outputs: list[AnalystOutput] = field(default_factory=list)
    critic_reviews: list[CriticReview] = field(default_factory=list)
    revision_counts: dict[str, int] = field(default_factory=dict)
    """task_id -> number of revisions requested so far."""

    def latest_output_for(self, task_id: str) -> AnalystOutput | None:
        for output in reversed(self.analyst_outputs):
            if output.task_id == task_id:
                return output
        return None

    def latest_review_for(self, task_id: str) -> CriticReview | None:
        for review in reversed(self.critic_reviews):
            if review.task_id == task_id:
                return review
        return None

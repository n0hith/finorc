"""The hand-rolled orchestration loop: Planner -> Analyst -> Critic (-> revise) -> result.

No agent framework is used here on purpose - this loop is simple enough
(iterate over tasks, call three functions, check a verdict, bound the retries)
that a framework would add indirection without buying much. That tradeoff
stops holding once streaming, parallel sub-task execution, or more complex
routing enters the picture, which is exactly why this module is isolated from
the agents and the transport layer.
"""

from __future__ import annotations

from dataclasses import dataclass

from finagent.agents.analyst import Analyst
from finagent.agents.critic import Critic
from finagent.agents.planner import Planner
from finagent.orchestration.context import AnalystOutput, ResearchContext
from finagent.orchestration.events import EventEmitter, EventSink, print_sink


@dataclass
class ResearchResult:
    context: ResearchContext
    final_outputs: list[AnalystOutput]


def run_research(
    question: str,
    *,
    planner: Planner,
    analyst: Analyst,
    critic: Critic,
    max_revisions: int,
    event_sink: EventSink = print_sink,
) -> ResearchResult:
    events = EventEmitter(sink=event_sink)
    context = ResearchContext(question=question)

    events.emit("research_started", question=question)

    context.plan = planner.run(question)
    events.emit(
        "planner_done",
        tasks=[{"id": t.id, "description": t.description} for t in context.plan.tasks],
    )

    for task in context.plan.tasks:
        context.revision_counts.setdefault(task.id, 0)
        revision_feedback = None

        while True:
            events.emit("analyst_step_started", task_id=task.id, revision=context.revision_counts[task.id])
            output = analyst.run(task, revision_feedback=revision_feedback)
            context.analyst_outputs.append(output)
            events.emit(
                "analyst_step_done",
                task_id=task.id,
                summary=output.summary,
                claims=[{"text": c.text, "supporting_snippet_ids": c.supporting_snippet_ids} for c in output.claims],
            )

            review = critic.run(task, output)
            context.critic_reviews.append(review)
            events.emit(
                "critic_review",
                task_id=task.id,
                verdict=review.verdict,
                issues=[{"claim_text": i.claim_text, "problem": i.problem} for i in review.issues],
                notes=review.notes,
            )

            if review.verdict == "approved":
                break

            if context.revision_counts[task.id] >= max_revisions:
                events.emit(
                    "revision_requested",
                    task_id=task.id,
                    accepted_anyway=True,
                    reason=f"Reached max_revisions={max_revisions}; accepting latest output.",
                )
                break

            context.revision_counts[task.id] += 1
            events.emit("revision_requested", task_id=task.id, accepted_anyway=False, notes=review.notes)
            revision_feedback = review

    final_outputs = [context.latest_output_for(t.id) for t in context.plan.tasks]
    events.emit(
        "research_completed",
        question=question,
        task_count=len(context.plan.tasks),
        final_verdicts={t.id: context.latest_review_for(t.id).verdict for t in context.plan.tasks},
    )

    return ResearchResult(context=context, final_outputs=[o for o in final_outputs if o is not None])

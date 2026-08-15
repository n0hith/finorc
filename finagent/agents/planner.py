"""Planner agent: decomposes a research question into sub-tasks."""

from __future__ import annotations

from finagent.agents.base import BaseAgent
from finagent.orchestration.context import Plan, Task

_TOOL_NAME = "submit_plan"

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "minItems": 2,
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Short slug identifying this sub-task, e.g. 'demand_drivers'.",
                    },
                    "description": {
                        "type": "string",
                        "description": "A concrete, researchable sub-question the Analyst should investigate.",
                    },
                },
                "required": ["id", "description"],
            },
        }
    },
    "required": ["tasks"],
}


class Planner(BaseAgent):
    system_prompt = (
        "You are the Planner in a financial research pipeline. Given a research "
        "question, break it into 2-5 concrete, independently researchable "
        "sub-tasks that together would let an analyst build a well-supported "
        "answer. Each sub-task should be specific enough to search for "
        "(e.g. 'identify recent revenue guidance changes' rather than 'look at "
        "financials'). Do not answer the question yourself - only decompose it."
    )

    def run(self, question: str) -> Plan:
        result = self._call_tool(
            user_message=f"Research question: {question}",
            tool_name=_TOOL_NAME,
            tool_description="Submit the sub-task breakdown for this research question.",
            input_schema=_INPUT_SCHEMA,
        )
        tasks = [Task(id=t["id"], description=t["description"]) for t in result["tasks"]]
        return Plan(tasks=tasks)

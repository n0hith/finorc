"""Shared base class for the three agents.

Design note: rather than parsing free-text responses, every agent forces
Claude to respond through a single tool call (`tool_choice={"type": "tool",
"name": ...}`) whose input schema is the agent's structured output. This
avoids brittle prose-parsing and gives each agent a guaranteed, validated
shape to hand to the next stage of the pipeline.
"""

from __future__ import annotations

from typing import Any

import anthropic

from finagent.config import get_settings


class BaseAgent:
    system_prompt: str = ""

    def __init__(self, client: anthropic.Anthropic | None = None, model: str | None = None) -> None:
        settings = get_settings()
        self._client = client or anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = model or settings.model

    def _call_tool(
        self,
        *,
        user_message: str,
        tool_name: str,
        tool_description: str,
        input_schema: dict[str, Any],
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """Call Claude and force it to respond via the given tool, returning its input dict."""
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=self.system_prompt,
            tools=[
                {
                    "name": tool_name,
                    "description": tool_description,
                    "input_schema": input_schema,
                }
            ],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": user_message}],
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return block.input
        raise RuntimeError(f"Expected a '{tool_name}' tool call, got: {response.content}")

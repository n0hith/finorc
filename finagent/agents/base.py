"""Shared base class for the three agents.

Design note: rather than parsing free-text responses, every agent forces
Gemini to respond through a single function call (`function_calling_config`
mode "ANY" restricted to one allowed function) whose parameters schema is the
agent's structured output. This avoids brittle prose-parsing and gives each
agent a guaranteed, validated shape to hand to the next stage of the
pipeline. Gemini accepts plain JSON Schema dicts via `parameters_json_schema`,
so the schemas below are ordinary JSON Schema - no Anthropic/Gemini-specific
dialect to maintain.
"""

from __future__ import annotations

import re
import time
from typing import Any

from google import genai
from google.genai import errors, types

from finagent.config import get_settings

_MAX_RATE_LIMIT_RETRIES = 3


class BaseAgent:
    system_prompt: str = ""

    def __init__(self, client: genai.Client | None = None, model: str | None = None) -> None:
        settings = get_settings()
        self._client = client or genai.Client(api_key=settings.gemini_api_key)
        self._model = model or settings.model

    def _call_tool(
        self,
        *,
        user_message: str,
        tool_name: str,
        tool_description: str,
        input_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Call Gemini and force it to respond via the given function, returning its args dict."""
        function = types.FunctionDeclaration(
            name=tool_name,
            description=tool_description,
            parameters_json_schema=input_schema,
        )
        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            tools=[types.Tool(function_declarations=[function])],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.ANY,
                    allowed_function_names=[tool_name],
                )
            ),
        )
        response = self._generate_with_retry(user_message, config)
        for part in response.candidates[0].content.parts:
            if part.function_call and part.function_call.name == tool_name:
                return dict(part.function_call.args)
        raise RuntimeError(f"Expected a '{tool_name}' function call, got: {response.candidates[0].content.parts}")

    def _generate_with_retry(
        self, user_message: str, config: types.GenerateContentConfig
    ) -> types.GenerateContentResponse:
        """Retry once per rate-limit error, honoring the server's suggested retryDelay.

        The free tier's per-minute quota is easy to hit with a multi-agent pipeline
        that makes several calls per research question, so this is a practical
        necessity rather than generic defensive coding.
        """
        for attempt in range(_MAX_RATE_LIMIT_RETRIES + 1):
            try:
                return self._client.models.generate_content(
                    model=self._model, contents=user_message, config=config
                )
            except errors.ClientError as exc:
                if exc.code != 429 or attempt == _MAX_RATE_LIMIT_RETRIES:
                    raise
                delay = _parse_retry_delay_seconds(str(exc)) or 30
                time.sleep(delay)
        raise AssertionError("unreachable")


def _parse_retry_delay_seconds(message: str) -> float | None:
    match = re.search(r"'retryDelay': '(\d+(?:\.\d+)?)s'", message)
    return float(match.group(1)) if match else None

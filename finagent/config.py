"""Runtime configuration, loaded from environment variables (and .env in dev)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    model: str
    max_revisions: int

    @classmethod
    def load(cls) -> "Settings":
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        return cls(
            gemini_api_key=api_key,
            model=os.environ.get("FINAGENT_MODEL", "gemini-flash-lite-latest"),
            max_revisions=int(os.environ.get("FINAGENT_MAX_REVISIONS", "2")),
        )


def get_settings() -> Settings:
    """Load settings lazily so importing this module never requires an API key."""
    return Settings.load()

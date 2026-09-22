"""Minimal environment-based settings without a framework dependency."""

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    provider: str = "mock"
    model_name: str = ""
    base_url: str = ""
    max_steps: int = 8

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            provider=os.getenv("LLM_PROVIDER", "mock"),
            model_name=os.getenv("MODEL_NAME", ""),
            base_url=os.getenv("OPENAI_BASE_URL", ""),
            max_steps=int(os.getenv("MAX_AGENT_STEPS", "8")),
        )


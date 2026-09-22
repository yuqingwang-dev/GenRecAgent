"""Configuration-only OpenAI-compatible adapter stub.

Network transport is intentionally not implemented in Phase 0. Keeping the
provider settings here gives later implementations one stable seam.
"""

import os
from typing import Any

from genrecagent.llm.base import BaseLLM


class OpenAICompatibleLLM(BaseLLM):
    """Read provider configuration without making an API request."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url if base_url is not None else os.getenv("OPENAI_BASE_URL", "")
        self.model_name = model_name if model_name is not None else os.getenv("MODEL_NAME", "")

    def generate(self, prompt: str) -> str:
        del prompt
        raise NotImplementedError("Network LLM transport is reserved for a later phase.")

    def tool_call(self, prompt: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
        del prompt, tools
        raise NotImplementedError("Network tool calling is reserved for a later phase.")


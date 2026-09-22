"""Provider-neutral LLM interface."""

from abc import ABC, abstractmethod
from typing import Any


class BaseLLM(ABC):
    """Minimal interface for future Qwen, vLLM, or OpenAI-compatible backends."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""

    @abstractmethod
    def tool_call(self, prompt: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
        """Return a structured tool call."""


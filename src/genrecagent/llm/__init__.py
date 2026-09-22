"""LLM provider abstractions."""

from genrecagent.llm.base import BaseLLM
from genrecagent.llm.openai_compatible import OpenAICompatibleLLM

__all__ = ["BaseLLM", "OpenAICompatibleLLM"]


"""DeepSeek API client.

Wraps the OpenAI SDK for DeepSeek's OpenAI-compatible API.
Supports streaming responses for a better interactive experience.
"""

import logging
from typing import List, Optional

from openai import OpenAI

from agent.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are Genesis, a helpful AI assistant with access to tools for interacting with the user's system.
You can read files, write files, list directories, and run shell commands.
Always think step-by-step before using a tool. Explain what you're doing and why.
When writing files, produce complete, correct content — never truncate.
"""


class DeepSeekClient:
    """Thin wrapper around OpenAI SDK for DeepSeek API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ):
        self.api_key = api_key or settings.deepseek_api_key
        self.base_url = base_url or settings.deepseek_base_url
        self.model = model or settings.deepseek_model
        self.max_tokens = max_tokens or settings.max_tokens

        if not self.api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY is not set. "
                "Set it in .env or as an environment variable."
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def chat(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
    ) -> dict:
        """Send a chat completion request (non-streaming).

        Args:
            messages: Conversation messages in OpenAI format.
            tools: Tool definitions for function calling.

        Returns:
            Raw API response object.
        """
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        return self.client.chat.completions.create(**kwargs)

    def chat_stream(self, messages: List[dict], tools: Optional[List[dict]] = None):
        """Send a streaming chat completion request.

        Yields chunk objects from the API.
        """
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools

        return self.client.chat.completions.create(**kwargs)

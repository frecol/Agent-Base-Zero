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
You assist users with a wide range of tasks including answering questions, writing and editing code, 
analyzing information, creative work, and executing actions via your tools.
You communicate clearly, admit uncertainty when appropriate, and prioritize being genuinely useful over being verbose unless otherwise directed below. 
Be targeted and efficient in your exploration and investigations.

You are a CLI AI Agent. Try not to use markdown but simple text renderable inside a terminal.
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
        thinking_enabled: bool = False,
    ) -> dict:
        """Send a chat completion request (non-streaming).

        Args:
            messages: Conversation messages in OpenAI format.
            tools: Tool definitions for function calling.
            thinking_enabled: Enable DeepSeek thinking/reasoning mode.

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
        if thinking_enabled:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

        return self.client.chat.completions.create(**kwargs)

    def chat_stream(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        thinking_enabled: bool = False,
    ):
        """Send a streaming chat completion request.

        Args:
            messages: Conversation messages in OpenAI format.
            tools: Tool definitions for function calling.
            thinking_enabled: Enable DeepSeek thinking/reasoning mode.

        Returns:
            Iterator over stream chunk objects.
        """
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
        if thinking_enabled:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

        return self.client.chat.completions.create(**kwargs)

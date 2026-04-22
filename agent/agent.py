"""Core agent loop.

Manages conversation history, calls the LLM, and executes tool calls
in a loop until the model returns a plain text response.
"""

import json
import logging
from typing import Callable, List, Optional

from agent.client import DeepSeekClient, SYSTEM_PROMPT
from tools.registry import registry

logger = logging.getLogger(__name__)

# Maximum tool-calling iterations per user turn to prevent infinite loops.
MAX_TOOL_ITERATIONS = 50


class Agent:
    """An LLM-powered agent that can use tools to accomplish tasks."""

    def __init__(
        self,
        client: Optional[DeepSeekClient] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = MAX_TOOL_ITERATIONS,
    ):
        self.client = client or DeepSeekClient()
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.max_iterations = max_iterations
        self.messages: List[dict] = []

    def _build_messages(self) -> List[dict]:
        """Prepend system prompt to conversation history."""
        return [{"role": "system", "content": self.system_prompt}] + self.messages

    def _get_tools(self) -> Optional[List[dict]]:
        """Return tool definitions for the API call."""
        names = registry.get_all_names()
        return registry.get_definitions(names) if names else None

    def _execute_tool_calls(
        self,
        tool_calls: list,
        on_tool_call: Optional[Callable] = None,
        on_tool_result: Optional[Callable] = None,
    ) -> None:
        """Execute all tool calls and append results to messages.

        Each tool call produces:
        1. The assistant message (with tool_calls) — already appended by caller.
        2. A tool result message for each tool call.
        """
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            if on_tool_call:
                on_tool_call(name, args)

            result = registry.dispatch(name, args)

            # Truncate very long results for display
            display = result[:200] + "..." if len(result) > 200 else result
            if on_tool_result:
                on_tool_result(name, display)

            self.messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    def run_turn(
        self,
        user_input: str,
        on_tool_call: Optional[Callable] = None,
        on_tool_result: Optional[Callable] = None,
    ) -> str:
        """Process one user turn: call LLM, handle tool calls, return final response.

        Args:
            user_input: The user's text input.
            on_tool_call: Optional callback(name, args) when a tool is invoked.
            on_tool_result: Optional callback(name, display_text) for tool results.

        Returns:
            The assistant's final text response.
        """
        self.messages.append({"role": "user", "content": user_input})

        tools = self._get_tools()

        for _ in range(self.max_iterations):
            api_messages = self._build_messages()
            response = self.client.chat(api_messages, tools=tools)

            choice = response.choices[0]
            assistant_msg = choice.message

            # Build the message dict to append to history
            msg_dict: dict = {"role": "assistant", "content": assistant_msg.content or ""}

            # Check for tool calls
            if assistant_msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_msg.tool_calls
                ]
                self.messages.append(msg_dict)
                self._execute_tool_calls(
                    assistant_msg.tool_calls, on_tool_call, on_tool_result
                )
                # Continue the loop — send tool results back to the model
                continue

            # No tool calls — this is the final response
            self.messages.append(msg_dict)
            return msg_dict["content"]

        # Exhausted iterations
        return "[Agent reached maximum tool iterations without a final response.]"

    def reset(self) -> None:
        """Clear conversation history."""
        self.messages.clear()

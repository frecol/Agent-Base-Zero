"""Core agent loop.

Manages conversation history, calls the LLM, and executes tool calls
in a loop until the model returns a plain text response.
Supports both streaming and non-streaming modes with optional thinking.
"""

import json
import logging
import types
from dataclasses import dataclass, field
from typing import Callable, Generator, List, Optional

from agent.client import DeepSeekClient, SYSTEM_PROMPT
from tools.registry import registry

logger = logging.getLogger(__name__)

# Maximum tool-calling iterations per user turn to prevent infinite loops.
MAX_TOOL_ITERATIONS = 50


@dataclass
class StreamEvent:
    """A structured event yielded during streaming."""

    type: str  # "thinking" | "content" | "tool_start" | "tool_result" | "done"
    data: dict = field(default_factory=dict)


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

    def _build_messages(self, thinking_enabled: bool = False) -> List[dict]:
        """Prepend system prompt to conversation history.

        Args:
            thinking_enabled: If True, keep reasoning_content in messages
                (needed for multi-step tool call reasoning within a turn).
                If False, strip reasoning_content to save bandwidth.
        """
        result = [{"role": "system", "content": self.system_prompt}]
        for msg in self.messages:
            if not thinking_enabled and "reasoning_content" in msg:
                msg = {k: v for k, v in msg.items() if k != "reasoning_content"}
            result.append(msg)
        return result

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

    def run_turn_stream(
        self,
        user_input: str,
        thinking_enabled: bool = False,
        on_tool_call: Optional[Callable] = None,
        on_tool_result: Optional[Callable] = None,
    ) -> Generator[StreamEvent, None, None]:
        """Streaming version of run_turn. Yields StreamEvent objects.

        Args:
            user_input: The user's text input.
            thinking_enabled: Enable DeepSeek thinking/reasoning mode.
            on_tool_call: Optional callback(name, args) when a tool is invoked.
            on_tool_result: Optional callback(name, display_text) for tool results.

        Yields:
            StreamEvent objects for the CLI to display in real-time.
        """
        self.messages.append({"role": "user", "content": user_input})
        tools = self._get_tools()

        for _ in range(self.max_iterations):
            api_messages = self._build_messages(thinking_enabled=thinking_enabled)
            stream = self.client.chat_stream(
                api_messages, tools=tools, thinking_enabled=thinking_enabled
            )

            # Accumulators for this stream
            content_chunks: List[str] = []
            reasoning_chunks: List[str] = []
            tool_call_accs: dict = {}  # index -> {id, name, arguments}

            for chunk in stream:
                delta = chunk.choices[0].delta

                # 1. Reasoning/thinking content (arrives first)
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    reasoning_chunks.append(delta.reasoning_content)
                    yield StreamEvent(
                        type="thinking", data={"text": delta.reasoning_content}
                    )

                # 2. Regular content
                if delta.content:
                    content_chunks.append(delta.content)
                    yield StreamEvent(
                        type="content", data={"text": delta.content}
                    )

                # 3. Tool call deltas (fragmented, accumulate by index)
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_call_accs:
                            tool_call_accs[idx] = {
                                "id": "", "name": "", "arguments": ""
                            }
                        acc = tool_call_accs[idx]
                        if tc_delta.id:
                            acc["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                acc["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                acc["arguments"] += tc_delta.function.arguments

            # Stream ended — check what we accumulated
            reasoning_content = "".join(reasoning_chunks)
            content = "".join(content_chunks)

            if tool_call_accs:
                # Build assistant message with tool calls
                msg_dict: dict = {"role": "assistant", "content": content or ""}
                if reasoning_content:
                    msg_dict["reasoning_content"] = reasoning_content

                tc_list = []
                tc_objects = []
                for idx in sorted(tool_call_accs.keys()):
                    acc = tool_call_accs[idx]
                    tc_list.append({
                        "id": acc["id"],
                        "type": "function",
                        "function": {
                            "name": acc["name"],
                            "arguments": acc["arguments"],
                        },
                    })
                    tc_objects.append(_make_tool_call_namespace(acc))

                msg_dict["tool_calls"] = tc_list
                self.messages.append(msg_dict)
                self._execute_tool_calls(tc_objects, on_tool_call, on_tool_result)
                continue  # Loop back with tool results

            else:
                # Final response — no tool calls
                msg_dict = {"role": "assistant", "content": content}
                if reasoning_content:
                    msg_dict["reasoning_content"] = reasoning_content
                self.messages.append(msg_dict)
                yield StreamEvent(
                    type="done",
                    data={"content": content, "reasoning": reasoning_content},
                )
                return content

        yield StreamEvent(
            type="done",
            data={"content": "[Agent reached maximum tool iterations without a final response.]"},
        )

    def reset(self) -> None:
        """Clear conversation history."""
        self.messages.clear()


def _make_tool_call_namespace(acc: dict) -> types.SimpleNamespace:
    """Create a namespace object that mimics OpenAI's ChatCompletionMessageToolCall.

    This allows _execute_tool_calls to work with both real API objects
    and reconstructed tool calls from streaming deltas.
    """
    tc = types.SimpleNamespace()
    tc.id = acc["id"]
    tc.function = types.SimpleNamespace()
    tc.function.name = acc["name"]
    tc.function.arguments = acc["arguments"]
    return tc

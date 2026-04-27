"""Token estimation and context compression.

Provides rough token counting via chars/4 heuristic and
context compression when the conversation grows too long.
"""

import json
import logging
from typing import List, Optional

from agent.client import DeepSeekClient

logger = logging.getLogger(__name__)

# Rough chars-per-token ratio (works well for English + Chinese mixed text).
_CHARS_PER_TOKEN = 4


def estimate_text_tokens(text: str) -> int:
    """Estimate token count for a single text string."""
    return (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN


def estimate_tokens(messages: List[dict]) -> int:
    """Estimate total token count for a message list.

    Iterates all string values in each message dict (content, role,
    reasoning_content, function.name, function.arguments, etc.)
    and sums chars/4.
    """
    total_chars = 0
    for msg in messages:
        for value in msg.values():
            if isinstance(value, str):
                total_chars += len(value)
            elif isinstance(value, list):
                # tool_calls list
                for item in value:
                    if isinstance(item, dict):
                        for v in item.values():
                            if isinstance(v, str):
                                total_chars += len(v)
                            elif isinstance(v, dict):
                                for vv in v.values():
                                    if isinstance(vv, str):
                                        total_chars += len(vv)
    return (total_chars + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN


def get_token_usage(messages: List[dict], max_tokens: int) -> dict:
    """Return token usage stats.

    Returns:
        dict with keys: used, max, percent, remaining.
    """
    used = estimate_tokens(messages)
    return {
        "used": used,
        "max": max_tokens,
        "percent": (used / max_tokens * 100) if max_tokens > 0 else 0,
        "remaining": max(0, max_tokens - used),
    }


def needs_compression(
    messages: List[dict], max_tokens: int, threshold: float = 0.95
) -> bool:
    """Check if messages exceed the compression threshold."""
    used = estimate_tokens(messages)
    return used >= max_tokens * threshold


# ---------------------------------------------------------------------------
# Context compression
# ---------------------------------------------------------------------------

COMPRESSION_PROMPT = """Summarize the following conversation segment into a concise summary \
that preserves all information needed to continue the conversation seamlessly. \
Use this structure:

## Goal
What the user is trying to accomplish (1-2 sentences)

## Key Actions Taken
List of significant actions already performed (file reads, command outputs, tool uses)

## Current State
Where things stand right now: what has been completed, what remains

## Important Decisions
Any choices or decisions made that affect the direction

## Technical Details
Specific values, paths, names, or data that must be preserved exactly

## User Preferences Noted
Any preferences the user expressed during this segment

Conversation to summarize:
{conversation}

Provide ONLY the summary text, no meta-commentary."""


def _format_messages_for_compression(messages: List[dict]) -> str:
    """Format a message list into readable text for the compression prompt."""
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                lines.append(f"[{role}] tool_call: {fn.get('name', '?')}({fn.get('arguments', '')})")
        elif role == "tool":
            tc_id = msg.get("tool_call_id", "")
            preview = content[:300] + "..." if len(content) > 300 else content
            lines.append(f"[tool result {tc_id}] {preview}")
        else:
            text = content[:500] + "..." if len(content) > 500 else content
            lines.append(f"[{role}] {text}")
    return "\n".join(lines)


def compress_messages(
    client: DeepSeekClient,
    messages: List[dict],
    head_keep: int = 3,
    tail_keep: int = 20,
) -> List[dict]:
    """Compress the middle section of messages into a single summary.

    Keeps the first *head_keep* and last *tail_keep* messages intact,
    and asks the LLM to summarize everything in between.

    Returns:
        New message list with middle replaced by a summary system message.
    """
    total = len(messages)
    min_required = head_keep + tail_keep

    if total <= min_required:
        logger.warning(
            "Not enough messages to compress (%d msgs, need > %d).",
            total, min_required,
        )
        return messages

    head = messages[:head_keep]
    middle = messages[head_keep:-tail_keep]
    tail = messages[-tail_keep:]

    conversation_text = _format_messages_for_compression(middle)
    prompt = COMPRESSION_PROMPT.format(conversation=conversation_text)

    try:
        response = client.chat(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            thinking_enabled=False,
        )
        summary = response.choices[0].message.content or ""
    except Exception as e:
        logger.warning("Compression LLM call failed: %s", e)
        return messages

    summary_msg = {
        "role": "system",
        "content": f"[Conversation Summary]\n{summary}",
    }

    compressed = head + [summary_msg] + tail
    logger.info(
        "Compressed %d messages -> %d messages",
        total, len(compressed),
    )
    return compressed


def run_memory_check_then_compress(
    client: DeepSeekClient,
    messages: List[dict],
    head_keep: int = 3,
    tail_keep: int = 20,
) -> List[dict]:
    """Compress messages when the conversation grows too long.

    Note: Memory saving is now handled by the memory_save tool
    (LLM decides when to save). This function only handles compression.

    Returns:
        New (potentially compressed) message list.
    """
    return compress_messages(client, messages, head_keep, tail_keep)

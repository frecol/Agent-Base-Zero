# v0.2 Deep Dive: Streaming, Thinking Mode & Web Search

v0.2 builds on the v0.1 "agent loop + tool registry" foundation with three major additions: **streaming output**, **thinking/reasoning mode**, and **web search tools**, along with a polished CLI experience.

---

## Table of Contents

- [Change Overview](#change-overview)
- [New Feature: Streaming Output](#new-feature-streaming-output)
  - [StreamEvent Model](#streamevent-model)
  - [The run_turn_stream Method](#the-run_turn_stream-method)
  - [CLI Streaming Renderer](#cli-streaming-renderer)
- [New Feature: Thinking Mode](#new-feature-thinking-mode)
  - [DeepSeek Thinking API](#deepseek-thinking-api)
  - [reasoning_content Handling Strategy](#reasoning_content-handling-strategy)
- [New Feature: Web Search Tools](#new-feature-web-search-tools)
  - [web_search](#web_search)
  - [current_time](#current_time)
- [Configuration & Dependency Changes](#configuration--dependency-changes)
- [New CLI Commands](#new-cli-commands)
- [System Prompt Update](#system-prompt-update)
- [File Change Summary](#file-change-summary)

---

## Change Overview

| Dimension | v0.1 | v0.2 |
|-----------|------|------|
| Response mode | Non-streaming only (wait for full response) | Streaming + non-streaming dual mode |
| Thinking chain | None | DeepSeek Thinking support |
| Tool count | 4 | 6 (+web_search, current_time) |
| max_tokens | 4096 | 8192 |
| CLI commands | /help /clear /exit /tools | +/stream /think /status |
| Dependencies | openai, pydantic-settings, rich, python-dotenv | +ddgs, tzdata |

---

## New Feature: Streaming Output

In v0.1, every agent response required waiting for the full LLM output before displaying anything — a noticeable latency. v0.2 introduces streaming so users see tokens in real time.

### StreamEvent Model

A new `StreamEvent` dataclass in `agent/agent.py` serves as the unified event carrier:

```python
# agent/agent.py

@dataclass
class StreamEvent:
    """A structured event yielded during streaming."""
    type: str  # "thinking" | "content" | "tool_start" | "tool_result" | "done"
    data: dict = field(default_factory=dict)
```

Instead of returning a single string, the streaming method yields events through a Generator:

| Event Type | Meaning | data |
|------------|---------|------|
| `"thinking"` | A fragment of the LLM's reasoning process | `{"text": "..."}` |
| `"content"` | A fragment of the formal response text | `{"text": "..."}` |
| `"done"` | Turn complete | `{"content": "...", "reasoning": "..."}` |

### The run_turn_stream Method

The `Agent` class gains a new `run_turn_stream()` method, mirroring v0.1's `run_turn()`:

```python
# agent/agent.py — run_turn_stream() core logic (simplified)

def run_turn_stream(self, user_input, thinking_enabled=False,
                    on_tool_call=None, on_tool_result=None):
    self.messages.append({"role": "user", "content": user_input})
    tools = self._get_tools()

    for _ in range(self.max_iterations):
        stream = self.client.chat_stream(
            api_messages, tools=tools, thinking_enabled=thinking_enabled
        )

        # Accumulators: collect stream fragments
        content_chunks = []
        reasoning_chunks = []
        tool_call_accs = {}  # tool calls are fragmented, accumulate by index

        for chunk in stream:
            delta = chunk.choices[0].delta

            # 1. Thinking/reasoning content (arrives first)
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                reasoning_chunks.append(delta.reasoning_content)
                yield StreamEvent(type="thinking", data={"text": delta.reasoning_content})

            # 2. Formal response content
            if delta.content:
                content_chunks.append(delta.content)
                yield StreamEvent(type="content", data={"text": delta.content})

            # 3. Tool call deltas (fragmented — accumulate by index)
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    # ...accumulate id, name, arguments...
                    pass

        # Stream ended — check if tool calls were accumulated
        if tool_call_accs:
            # Tool calls found → execute and continue the loop
            self._execute_tool_calls(...)
            continue
        else:
            # No tool calls → final answer
            yield StreamEvent(type="done", ...)
            return content
```

Key points:
- Tool calls arrive as **fragments** in streaming mode and must be accumulated by `index` before execution
- Reconstructed tool calls are wrapped in `SimpleNamespace` objects (via `_make_tool_call_namespace()`) to reuse the existing `_execute_tool_calls()` logic
- The overall loop structure (LLM → tool → LLM → ...) remains identical to v0.1

### CLI Streaming Renderer

A new `_stream_response()` function in `agent/cli.py` handles terminal rendering in streaming mode:

```
After user input, the terminal displays in real time:

───── Thinking ──────────────────────────────
The user wants me to read config.py, I need to use read_file...
─────────────────────────────────────────────

┌─ read_file ─────────┐
│ path='agent/config.py' │
└──────────────────────┘
  Result: {"success": true, "content": "..."}
─────────────────────────────────────────────

───── Response ──────────────────────────────
Here is the content of config.py: from pydantic_settings...
───── Response Finish ───────────────────────
```

Design highlights:
- A `_Phase` enum tracks the current display phase (IDLE → THINKING → CONTENT), avoiding duplicate section headers
- Thinking text uses ANSI escape codes (`\033[2m`) for a dimmed appearance, writing directly to stdout to bypass Rich's per-token rendering overhead
- Tool call display upgraded to `Panel` components for clearer presentation than v0.1's single-line format

---

## New Feature: Thinking Mode

v0.2 integrates DeepSeek's Thinking capability, allowing the LLM to produce an "inner monologue" before answering, improving reasoning quality.

### DeepSeek Thinking API

Both `chat()` and `chat_stream()` in `agent/client.py` now accept a `thinking_enabled` parameter:

```python
# agent/client.py

def chat(self, messages, tools=None, thinking_enabled=False):
    kwargs = {
        "model": self.model,
        "messages": messages,
        "max_tokens": self.max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
    if thinking_enabled:
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
    return self.client.chat.completions.create(**kwargs)
```

When thinking is enabled, the API response `delta` includes a `reasoning_content` field containing the LLM's chain-of-thought text.

### reasoning_content Handling Strategy

Reasoning content must be stored in `messages` so the LLM can continue its reasoning across multi-turn tool calls. However, it can be very long and increase token costs.

`_build_messages()` now includes conditional filtering:

```python
# agent/agent.py — _build_messages()

def _build_messages(self, thinking_enabled=False):
    result = [{"role": "system", "content": self.system_prompt}]
    for msg in self.messages:
        if not thinking_enabled and "reasoning_content" in msg:
            # Non-thinking mode: strip reasoning_content to save bandwidth
            msg = {k: v for k, v in msg.items() if k != "reasoning_content"}
        result.append(msg)
    return result
```

Strategy:
- **Thinking enabled**: Keep `reasoning_content` so the LLM can reference previous reasoning
- **Thinking disabled**: Strip `reasoning_content` to reduce token overhead

---

## New Feature: Web Search Tools

v0.2 adds two tools that break v0.1's local-only limitation.

### web_search

`tools/web_search.py` uses [DuckDuckGo Search](https://pypi.org/project/ddgs/) for web searches:

```python
# tools/web_search.py

SCHEMA = {
    "name": "web_search",
    "description": "Search the web using DuckDuckGo. "
                   "Returns a list of results with title, URL, and a short snippet.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "max_results": {"type": "integer", "description": "Max results (default 5)."},
        },
        "required": ["query"],
    },
}

def handler(args: dict) -> str:
    with DDGS() as ddgs:
        results = [{"title": r["title"], "href": r["href"], "body": r["body"]}
                    for r in ddgs.text(query, max_results=max_results)]
    return json.dumps({"success": True, "results": results})
```

Highlights:
- No API key required — searches directly via DuckDuckGo
- Returns title, URL, and snippet for each result
- The SCHEMA description hints the LLM to resolve relative time terms (like "today") to concrete dates before searching

### current_time

`tools/current_time.py` provides the current time query, designed to work alongside web_search:

```python
# tools/current_time.py

def handler(args: dict) -> str:
    tz_name = args.get("timezone") or settings.default_timezone
    tz = ZoneInfo(tz_name)
    now = datetime.now(tz)
    return json.dumps({
        "success": True,
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": tz_name,
        "weekday": now.strftime("%A"),
    })
```

Design intent: LLMs don't know the current time. When a user asks "what's in the news today", the agent first uses `current_time` to get the date, then `web_search` to search. Together they form a complete web-access capability chain.

Default timezone is set via the `default_timezone` config, defaulting to `Asia/Shanghai`.

---

## Configuration & Dependency Changes

### New Configuration Items

`agent/config.py` adds three settings:

```python
# agent/config.py

class Settings(BaseSettings):
    # ... existing config ...
    max_tokens: int = 8192            # raised from 4096

    # Streaming & Thinking (v0.2)
    stream_enabled: bool = True       # streaming on by default
    thinking_enabled: bool = True     # thinking mode on by default

    # Timezone
    default_timezone: str = "Asia/Shanghai"
```

All settings can be overridden via `.env` file or environment variables.

### New Dependencies

`pyproject.toml` adds two dependencies:

| Dependency | Version | Purpose |
|------------|---------|---------|
| `ddgs` | >=7.0.0 | DuckDuckGo search client |
| `tzdata` | >=2024.1 | Timezone data (needed on Windows) |

---

## New CLI Commands

v0.2 adds three built-in commands:

| Command | Action |
|---------|--------|
| `/stream` | Toggle streaming / non-streaming mode |
| `/think` | Toggle thinking mode on / off |
| `/status` | Display current settings (streaming, thinking, model) |

`run_cli()` uses local variables `stream_enabled` and `thinking_enabled` to control mode switching. Each conversation turn selects `run_turn()` or `run_turn_stream()` based on the current state.

---

## System Prompt Update

The system prompt evolves from v0.1's brief description to a more complete role definition:

```
v0.1:
  "You can read files, write files, list directories, and run shell commands.
   Always think step-by-step before using a tool."

v0.2:
  "You assist users with a wide range of tasks including answering questions,
   writing and editing code, analyzing information, creative work...
   You are a CLI AI Agent. Try not to use markdown but simple text
   renderable inside a terminal."
```

Key changes:
- Upgraded from "file operations assistant" to "general-purpose task assistant"
- Explicitly identifies as a CLI Agent, prompting the LLM to avoid complex Markdown (limited terminal rendering)
- Emphasizes efficiency and clear communication

---

## File Change Summary

| File | Change | Description |
|------|--------|-------------|
| `agent/agent.py` | Modified | Added `StreamEvent`, `run_turn_stream()`, `_make_tool_call_namespace()`; `_build_messages()` adds thinking filter |
| `agent/cli.py` | Modified | Added `_stream_response()`, `_Phase` enum, `/stream` `/think` `/status` commands; improved tool call display |
| `agent/client.py` | Modified | `chat()` and `chat_stream()` gain `thinking_enabled` parameter |
| `agent/config.py` | Modified | Added `stream_enabled`, `thinking_enabled`, `default_timezone`; `max_tokens` raised to 8192 |
| `tools/web_search.py` | **New** | DuckDuckGo web search tool |
| `tools/current_time.py` | **New** | Current time query tool |
| `pyproject.toml` | Modified | Version bumped to 0.2.0, added ddgs and tzdata dependencies |
| `.env.example` | Modified | Added v0.2 config examples |
| `README.md` / `README_zh.md` | Modified | Progress marker updated to v0.2 |

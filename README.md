# Agent-Base-Zero

**One commit per agent version — learn to build a general-purpose AI agent from scratch.**

Agent-Base-Zero is an open-source project that teaches you how to build a general-purpose AI agent powered by a single LLM (DeepSeek), step by step from simple to complex.
The entire codebase evolves through git commits — each version introduces a new concept or feature, from a bare-minimum chat loop to a full-featured autonomous agent.

**How to learn:** `git log --oneline` to see every evolution stage, then `git checkout <commit>` to jump to any version and read the code.

## Evolution Roadmap

| Version | Focus | Key Concepts |
|---------|-------|-------------|
| **v0.1** | Minimum viable agent | Agent loop, tool calling, CLI, tool registry |
| **v0.2** | Streaming & thinking | Streaming output, DeepSeek thinking mode, web search, current_time tool |
| v0.3 | Memory system | Session persistence, long-term memory, context compression |
| v0.4 | Skill system | Skill registration, Prompt templates, tool composition |
| v0.5 | Planning & execution | Task decomposition, multi-step planning, self-reflection |
| v0.6 | Social media integration | API integrations, async operations |
| v0.7 | Multi-agent collaboration | Agent cooperation, task routing |
Current progress: v0.2

## Quick Start

> **Prerequisites:** [uv](https://docs.astral.sh/uv/) (Python package manager), Python >= 3.11, a [DeepSeek API key](https://platform.deepseek.com/)

```bash
# 1. Clone the repo
git clone https://github.com/frecol/Agent-Base-Zero.git
cd Agent-Base-Zero

# 2. Install dependencies
uv sync

# 3. Set your API key
echo "DEEPSEEK_API_KEY=your-key-here" > .env

# 4. Run
uv run genesis
```

## v0.1 — Minimum Viable Agent

A working agent in ~300 lines of Python.

**What it does:**
- Interactive CLI with Rich-powered display
- Multi-turn conversation with tool-calling loop
- 4 built-in tools: `read_file`, `write_file`, `list_dir`, `run_command`
- Auto-discovered tool registry (drop a file in `tools/` to add a tool)

**How the agent loop works:**

```
User Input → Agent → LLM
                      ↓
                 Tool Calls?
                  ├─ Yes → Execute Tool → Append Result → Back to LLM
                  └─ No  → Return response to user
```

The loop continues until the LLM responds with plain text (no more tool calls), up to 50 iterations.

## v0.2 — Streaming, Thinking & Web Search

The current version. Built on top of v0.1 with three major additions:

**Streaming output** — Real-time token-by-token rendering. The new `run_turn_stream()` method yields structured `StreamEvent` objects (`thinking` / `content` / `done`), giving the CLI full control over display phases.

**Thinking mode** — Integrates DeepSeek's reasoning API. The LLM produces a visible chain-of-thought before answering. `reasoning_content` is conditionally preserved or stripped in `_build_messages()` to balance reasoning continuity against token cost.

**Web search tools** — Two new tools break the local-only limitation:
- `web_search` — DuckDuckGo-powered web search, no API key required
- `current_time` — Current date/time query, designed to work with web_search (e.g. "today's news" → get date → search)

Other improvements: `max_tokens` raised to 8192, new CLI commands (`/stream`, `/think`, `/status`), refined system prompt, and enhanced tool call display with Rich `Panel` components.

## Project Structure

```
Agent-Base-Zero/
├── agent/
│   ├── config.py          # Pydantic Settings, reads from .env
│   ├── client.py          # DeepSeek API client (OpenAI SDK, streaming + thinking)
│   ├── agent.py           # Core agent loop + StreamEvent streaming
│   └── cli.py             # Interactive CLI (Rich, streaming renderer)
├── tools/
│   ├── registry.py        # Tool registry (register + dispatch)
│   ├── read_file.py       # Read file contents
│   ├── write_file.py      # Write to files
│   ├── list_dir.py        # List directory entries
│   ├── run_command.py     # Execute shell commands
│   ├── web_search.py      # Web search via DuckDuckGo
│   └── current_time.py    # Current date/time query
├── docs/
│   ├── v01/               # v0.1 architecture docs (zh + en)
│   └── v02/               # v0.2 architecture docs (zh + en)
├── main.py                # Entry point (python main.py)
├── pyproject.toml         # Project config & dependencies
└── .env                   # Your API key (not tracked)
```

## Adding a New Tool

Each tool file self-registers at import time:

```python
# tools/my_tool.py
import json
from tools.registry import registry

SCHEMA = {
    "name": "my_tool",
    "description": "What this tool does",
    "parameters": {
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "Some input"},
        },
        "required": ["input"],
    },
}

def handler(args: dict) -> str:
    result = do_something(args["input"])
    return json.dumps({"success": True, "result": result})

registry.register("my_tool", SCHEMA, handler)
```

Drop the file in `tools/` — it's auto-discovered on startup.

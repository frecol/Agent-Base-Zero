# Agent-Base-Zero

**One commit per agent version — learn to build a general-purpose AI agent from scratch.**

Agent-Base-Zero is an open-source project that teaches you how to build a general-purpose AI agent powered by a single LLM (DeepSeek), step by step from simple to complex.
The entire codebase evolves through git commits — each version introduces a new concept or feature, from a bare-minimum chat loop to a full-featured autonomous agent.

**How to learn:** `git log --oneline` to see every evolution stage, then `git checkout <commit>` to jump to any version and read the code.

## Evolution Roadmap

| Version | Focus | Key Concepts |
|---------|-------|-------------|
| **v0.1** | Minimum viable agent | Agent loop, tool calling, CLI, tool registry |
| v0.2 | Streaming output | SSE streaming, real-time display |
| v0.3 | Memory system | Session persistence, long-term memory, context compression |
| v0.4 | Skill system | Skill registration, Prompt templates, tool composition |
| v0.5 | Planning & execution | Task decomposition, multi-step planning, self-reflection |
| v0.6 | Social media integration | API integrations, async operations |
| v0.7 | Multi-agent collaboration | Agent cooperation, task routing |
Current progress: v0.1

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

The current version. A working agent in ~300 lines of Python.

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

## Project Structure

```
Agent-Base-Zero/
├── agent/
│   ├── config.py          # Pydantic Settings, reads from .env
│   ├── client.py          # DeepSeek API client (OpenAI SDK)
│   ├── agent.py           # Core agent loop (LLM ↔ Tools)
│   └── cli.py             # Interactive CLI (Rich)
├── tools/
│   ├── registry.py        # Tool registry (register + dispatch)
│   ├── read_file.py       # Read file contents
│   ├── write_file.py      # Write to files
│   ├── list_dir.py        # List directory entries
│   └── run_command.py     # Execute shell commands
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

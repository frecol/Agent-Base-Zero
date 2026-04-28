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
| **v0.3** | Memory system | Session persistence, long-term memory, context compression |
| **v0.4** | Skill system | Skill registration, PromptManager, LLM auto-activation, composite tools |
| **v0.5** | Plan Mode | Plan/Normal dual mode, read-only exploration, structured planning, step-by-step execution |
| v0.6 | Social media integration | API integrations, async operations |
| v0.7 | Multi-agent collaboration | Agent cooperation, task routing |
Current progress: v0.5

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

Built on top of v0.1 with three major additions:

**Streaming output** — Real-time token-by-token rendering. The new `run_turn_stream()` method yields structured `StreamEvent` objects (`thinking` / `content` / `done`), giving the CLI full control over display phases.

**Thinking mode** — Integrates DeepSeek's reasoning API. The LLM produces a visible chain-of-thought before answering. `reasoning_content` is conditionally preserved or stripped in `_build_messages()` to balance reasoning continuity against token cost.

**Web search tools** — Two new tools break the local-only limitation:
- `web_search` — DuckDuckGo-powered web search, no API key required
- `current_time` — Current date/time query, designed to work with web_search (e.g. "today's news" → get date → search)

Other improvements: `max_tokens` raised to 8192, new CLI commands (`/stream`, `/think`, `/status`), refined system prompt, and enhanced tool call display with Rich `Panel` components.

## v0.3 — Memory System

Built on top of v0.2 with session persistence, long-term memory, and context compression.

**Session persistence** — Every conversation is automatically saved to `.genesis/sessions/{session_id}.json`. Sessions include metadata (title, timestamps, message count) and use atomic writes to prevent corruption.

**Session management commands:**
- `/sessions` — List all saved sessions with timestamps and titles
- `/resume <id>` — Load a previous session to continue where you left off
- `/new` — Start a fresh session (long-term memory is preserved across sessions)

**Long-term memory** — The LLM can save durable facts about the user using the `memory_save` tool. Memories persist across all sessions and are loaded at startup as a system prompt. The LLM decides what to remember — user preferences, environment details, and stable conventions.

**Session search** — The `session_search` tool allows the LLM to search past conversations by keyword, retrieving relevant snippets without manual file inspection.

**Context compression** — When the conversation grows too long (default: >60,800 tokens), the middle of the conversation is automatically summarized by the LLM into a structured summary. The `/compact` command triggers this manually. Key messages at the head (3) and tail (20) are preserved intact.

**Token estimation** — Rough chars/4 heuristic estimates token usage, displayed in `/status`. This enables proactive context management without an external tokenizer.

| Feature | Description |
|---------|-------------|
| `/status` | Now shows session ID, message count, token usage |
| `/sessions` | List all saved conversations |
| `/resume <id>` | Continue a previous session |
| `/new` | Start a fresh session |
| `/compact` | Manually compress conversation history |

## v0.4 — Skill System

Built on top of v0.3 with skill registration, PromptManager, LLM-driven skill activation, and composite tools.

**Skill system** — Each skill is a folder in `skills/` containing a `SKILL.md` file (YAML frontmatter + Markdown instructions). Skills are auto-discovered at startup. The SKILL.md format is compatible with the standard skill format used across the ecosystem.

**PromptManager** — Replaces the hardcoded `SYSTEM_PROMPT` in `client.py`. Loads the base prompt from `agent/system_prompt.md` and appends a skills index (all skill names + descriptions) at startup. The system prompt stays stable throughout the session, preserving prompt prefix caching.

**LLM auto-activation** — The LLM sees the skills index in the system prompt and autonomously calls `activate_skill(name)` when a task matches a skill. The skill's detailed instructions are returned as a tool result (not injected into the system prompt), so the prompt prefix cache is never broken. The LLM can also call `deactivate_skill` to return to base mode.

**Composite tools** — A new `research_topic` tool demonstrates tool composition: its handler internally chains `web_search` + `fetch_url` via `registry.dispatch()` to provide a one-call research capability.

**Cache-friendly design:**
```
System prompt (stable prefix):
  base prompt + skills index → never changes → cache hits preserved

Skill instructions:
  activate_skill tool result → enters conversation history
  system prompt unchanged
```

**Skill commands:**
- `/skills` — List all available skills (marks active, shows auto-only)
- `/skill <name>` — Manually activate a skill (respects `user_invocable`)
- `/unskill` — Deactivate current skill

**Example skills:**
- `research` — Web research specialist with search strategy and source citation guidelines
- `code_assistant` — Code assistant specialist with debugging and code style guidelines

| Feature | Description |
|---------|-------------|
| `/skills` | List all available skills (auto-only marked) |
| `/skill <name>` | Activate a skill (respects `user_invocable`) |
| `/unskill` | Deactivate current skill |
| `/status` | Now shows active skill |

## v0.5 — Plan Mode (Current)

Built on top of v0.4 with Plan/Normal dual mode, tool classification, structured planning, interactive review, and step-by-step execution.

**Plan Mode** — A dual-mode workflow that separates read-only exploration/planning from execution. Press **Shift+Tab** or type `/plan` to toggle between Normal mode (all tools available) and Plan mode (read-only tools only). In Plan mode, the Agent explores the codebase, analyzes the task, and generates a structured plan for user review before any changes are made.

**Tool classification** — All tools are classified as either `read_only` or `write` in `tools/registry.py`. In Plan mode, the LLM only receives read-only tool definitions (read_file, list_dir, grep_search, etc.), and a hard guard at the execution layer blocks any write tool calls — even if the model "hallucinates" them from prior Normal-mode conversation history.

**Structured planning** — The LLM generates plans in a fixed format (Goal + numbered Steps). The plan parser supports multiple format variations (numbering styles, separators). A loose fallback parser detects plans without formal section headers.

**Interactive review** — After the Agent generates a plan, a magenta-bordered panel displays the goal and proposed steps. The user can Accept (execute), Modify (provide feedback for revision, up to 3 rounds), or Cancel.

**Step-by-step execution** — Once accepted, the Agent switches to Normal mode and executes each step silently. A live progress display with shimmer animation shows real-time status for each step (Pending → In Progress → Done/Failed/Skipped). On failure, the user can choose to continue or stop.

**Visual enhancements** — `prompt_toolkit` input handler with Shift+Tab detection, Rich Live progress bar with 8fps auto-refresh, shimmer light-sweep animation on "In Progress" text, and a completion summary panel.

**Plan Mode commands:**
- `/plan` or **Shift+Tab** — Toggle Plan / Normal mode
- `/status` — Now shows current mode (Normal / Plan)

| Feature | Description |
|---------|-------------|
| `/plan` | Toggle Plan / Normal mode |
| Shift+Tab | Toggle mode during input |
| Plan Review | Accept / Modify / Cancel plan before execution |
| Live Progress | Real-time step status with shimmer animation |
| Write Guard | Execution-layer blocks write tools in Plan mode |

## Project Structure

```
Agent-Base-Zero/
├── agent/
│   ├── config.py          # Pydantic Settings, reads from .env
│   ├── client.py          # DeepSeek API client (OpenAI SDK, streaming + thinking)
│   ├── prompt.py          # PromptManager: base prompt + plan mode + skills index (v0.4+)
│   ├── prompts/           # Prompt templates directory (v0.5)
│   │   ├── system_prompt.md   # Base system prompt (v0.4)
│   │   └── plan_prompt.md     # Plan Mode system prompt (v0.5)
│   ├── plan.py            # Plan data structures: PlanPhase, StepStatus, PlanStep, Plan (v0.5)
│   ├── plan_parser.py     # Parse structured plan output from LLM responses (v0.5)
│   ├── plan_input.py      # InputHandler with Shift+Tab key binding (v0.5)
│   ├── plan_renderer.py   # Plan review + live progress bar rendering (v0.5)
│   ├── shimmer.py         # Shared shimmer animation position calculator (v0.5)
│   ├── agent.py           # Core agent loop + StreamEvent streaming + Plan mode
│   ├── session.py         # Session persistence and recovery (v0.3)
│   ├── memory.py          # Long-term memory storage (v0.3)
│   ├── tokens.py          # Token estimation and context compression (v0.3)
│   └── cli.py             # Interactive CLI (Rich, streaming renderer, Plan mode)
├── tools/
│   ├── registry.py        # Tool registry (register + dispatch + read_only/write categories)
│   ├── read_file.py       # Read file contents
│   ├── write_file.py      # Write to files
│   ├── list_dir.py        # List directory entries
│   ├── run_command.py     # Execute shell commands
│   ├── web_search.py      # Web search via DuckDuckGo
│   ├── current_time.py    # Current date/time query
│   ├── edit_file.py       # Edit files by string replacement (v0.3)
│   ├── grep_search.py     # Search file contents by regex/keyword (v0.3)
│   ├── fetch_url.py       # Fetch web pages and extract text (v0.3)
│   ├── tree.py            # Display recursive directory tree (v0.3)
│   ├── find_file.py       # Find files by glob pattern (v0.3)
│   ├── file_delete.py     # Delete a specified file (v0.3)
│   ├── system_info.py     # Get system runtime information (v0.3)
│   ├── memory_save.py     # Save facts to long-term memory (v0.3)
│   ├── session_search.py  # Search past conversations (v0.3)
│   └── research_topic.py  # Composite tool: search + fetch (v0.4)
├── skills/
│   ├── registry.py        # Skill registry, SKILL.md parsing, skill tools (v0.4)
│   ├── research/          # Web research skill (v0.4)
│   │   └── SKILL.md
│   └── code_assistant/    # Code assistant skill (v0.4)
│       └── SKILL.md
├── docs/
│   ├── v01/               # v0.1 architecture docs (zh + en)
│   ├── v02/               # v0.2 architecture docs (zh + en)
│   ├── v03/               # v0.3 architecture docs (zh + en)
│   ├── v04/               # v0.4 architecture docs (zh + en)
│   └── v05/               # v0.5 architecture docs (zh + en)
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

## Adding a New Skill

Each skill is a folder with a `SKILL.md` file:

```markdown
<!-- skills/my_skill/SKILL.md -->
---
name: my_skill
description: "What this skill does and when to activate it."
user_invocable: true
---

# My Skill

Detailed instructions for the LLM when this skill is active...
```

Drop the folder in `skills/` — it's auto-discovered on startup. The `name` and `description` appear in the system prompt's skills index, and the LLM can activate the skill by calling `activate_skill("my_skill")`. Set `user_invocable: false` to restrict activation to LLM-only (hidden from `/skill` CLI command, shown as `(auto-only)` in `/skills`).

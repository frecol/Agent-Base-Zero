# v0.1 Architecture Deep Dive: The Agent Loop & Tool System

This document provides an in-depth walkthrough of the two core mechanisms in Agent-Base-Zero v0.1: **the Agent Loop** and **the Tool Registry**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [The Agent Loop](#the-agent-loop)
  - [Overall Flow](#overall-flow)
  - [Message Format](#message-format)
  - [Core Code Walkthrough](#core-code-walkthrough)
- [Tool Registration & Dispatch](#tool-registration--dispatch)
  - [The Registry: ToolRegistry](#the-registry-toolregistry)
  - [Auto-Discovery](#auto-discovery)
  - [Dispatching](#dispatching)
- [How to Write a New Tool](#how-to-write-a-new-tool)
- [Built-in Tools](#built-in-tools)
- [The CLI Layer](#the-cli-layer)
- [Summary](#summary)

---

## Project Overview

v0.1 is the most minimal starting point of the project. It does exactly three things:

1. **Talk to an LLM** — via the OpenAI-compatible protocol (DeepSeek API)
2. **Call tools** — the LLM autonomously decides when and which tool to invoke
3. **Terminal interaction** — the user chats with the agent through a command-line interface

### Directory Structure

```
Agent-Base-Zero/
├── main.py                  # Entry point: launches the CLI
├── agent/
│   ├── config.py            # Configuration (API key, model name, etc.)
│   ├── client.py            # DeepSeek API client wrapper
│   ├── agent.py             # Core agent loop
│   └── cli.py               # Terminal CLI interface
├── tools/
│   ├── __init__.py          # Auto-discovers and registers all tools
│   ├── registry.py          # Tool registry (the core)
│   ├── read_file.py         # Tool: read a file
│   ├── write_file.py        # Tool: write a file
│   ├── list_dir.py          # Tool: list directory contents
│   └── run_command.py       # Tool: execute a shell command
└── docs/v01/                # This document
```

The entire v0.1 is roughly **650 lines of Python** — the perfect first step for learning how to build an agent.

---

## The Agent Loop

The agent loop is the heart of the system. It governs the entire process from "the user says something" to "the agent delivers a final answer."

### Overall Flow

```
User Input
  │
  ▼
┌──────────────────────────────────────┐
│ 1. Append user message to history     │
│ 2. Gather all registered tool schemas │
└──────────────────┬───────────────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Call LLM API       │ ◄──────────────────────┐
        │  (history + tools)  │                        │
        └──────────┬──────────┘                        │
                   │                                   │
                   ▼                                   │
         LLM returned tool calls?                      │
           ┌─────┴─────┐                              │
           │ Yes       │ No                            │
           ▼           ▼                               │
    Execute tools   Return final                       │
    Append results  text response                      │
    to messages      (loop ends)                       │
           │                                           │
           └───────────────────────────────────────────┘
              (continue loop — let the LLM see tool results)
```

Key insight: **this is not a single API call.** When the LLM decides to use a tool, the tool's result is appended to the conversation history, and the **LLM is called again** so it can decide what to do next — either invoke more tools or produce a final answer.

To prevent infinite loops, a safety cap is set at `MAX_TOOL_ITERATIONS = 50`.

### Message Format

The conversation history `self.messages` is a list of dictionaries following the OpenAI message format. A complete conversation that includes a tool call looks like this:

```python
messages = [
    # System prompt (not stored in messages; injected by _build_messages())
    {"role": "system", "content": "You are Genesis, a helpful AI assistant..."}

    # User input
    {"role": "user", "content": "Read the contents of config.py for me"},

    # LLM decides to call a tool (assistant message with tool_calls)
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": "{\"path\": \"agent/config.py\"}"
            }
        }]
    },

    # Tool execution result (role "tool", must carry tool_call_id)
    {
        "role": "tool",
        "tool_call_id": "call_abc123",
        "content": "{\"success\": true, \"content\": \"from pydantic_settings...\"}"
    },

    # LLM generates the final answer based on the tool result
    {"role": "assistant", "content": "Here is the content of config.py:\n..."}
]
```

Note: the `tool_call_id` in the `tool` message must match the `id` in the corresponding `assistant.tool_calls` entry. This is how the LLM associates a tool result with its originating call.

### Core Code Walkthrough

The core of the agent loop lives in `Agent.run_turn()` in `agent/agent.py`:

```python
# agent/agent.py — Agent.run_turn()

def run_turn(self, user_input, on_tool_call=None, on_tool_result=None) -> str:
    # Step 1: append the user message
    self.messages.append({"role": "user", "content": user_input})
    tools = self._get_tools()

    # Step 2: call the LLM in a loop
    for _ in range(self.max_iterations):
        # Each LLM call prepends the system prompt and sends the full
        # conversation history (user messages, assistant replies, tool results...)
        api_messages = self._build_messages()
        response = self.client.chat(api_messages, tools=tools)
        assistant_msg = response.choices[0].message

        msg_dict = {"role": "assistant", "content": assistant_msg.content or ""}

        if assistant_msg.tool_calls:
            # The LLM wants to call a tool
            msg_dict["tool_calls"] = [...]  # serialize tool call info
            self.messages.append(msg_dict)
            self._execute_tool_calls(assistant_msg.tool_calls, on_tool_call, on_tool_result)
            continue  # continue the loop — send tool results back to the LLM
        else:
            # The LLM produced a final answer
            self.messages.append(msg_dict)
            return msg_dict["content"]

    return "[Agent reached maximum tool iterations without a final response.]"
```

Tool execution happens in `_execute_tool_calls()`:

```python
# agent/agent.py — Agent._execute_tool_calls()

def _execute_tool_calls(self, tool_calls, on_tool_call=None, on_tool_result=None):
    for tc in tool_calls:
        name = tc.function.name
        args = json.loads(tc.function.arguments)

        # Callback to notify the CLI for display
        if on_tool_call:
            on_tool_call(name, args)

        # Dispatch through the registry
        result = registry.dispatch(name, args)

        # Append tool result to conversation history
        self.messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result,
        })
```

---

## Tool Registration & Dispatch

v0.1's tool system uses a **centralized registry + auto-discovery** design. The mechanism spans just two files:

- `tools/registry.py` — the registry itself
- `tools/__init__.py` — auto-discovery entry point

### The Registry: ToolRegistry

`ToolRegistry` is a simple dictionary-backed container that stores each tool's **schema** (its description) and **handler** (its execution function):

```python
# tools/registry.py

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, dict] = {}
        # Each entry: {"schema": dict, "handler": Callable}

    def register(self, name: str, schema: dict, handler: Callable) -> None:
        """Register a tool. Called automatically when a tool module is imported."""
        self._tools[name] = {"schema": schema, "handler": handler}

    def get_definitions(self, names=None) -> List[dict]:
        """Return tool definitions in OpenAI format, ready for the LLM API."""
        result = []
        for name in sorted(names or self._tools.keys()):
            entry = self._tools[name]
            result.append({
                "type": "function",
                "function": entry["schema"],
            })
        return result

    def dispatch(self, name: str, args: dict) -> str:
        """Execute the handler for the named tool. Returns a result string."""
        entry = self._tools.get(name)
        if not entry:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            return entry["handler"](args)
        except Exception as e:
            return json.dumps({"error": f"Tool execution failed: {e}"})
```

A **global singleton** `registry` is created at module load time, and all other modules import from this single instance:

```python
# Bottom of tools/registry.py
registry = ToolRegistry()
```

### Auto-Discovery

Tools don't need to be manually imported. The `discover_tools()` function in `tools/__init__.py` automatically scans the `tools/` directory and dynamically imports every `.py` file:

```python
# tools/registry.py — discover_tools()

def discover_tools(tools_dir=None) -> List[str]:
    tools_path = tools_dir or Path(__file__).resolve().parent
    module_names = [
        f"tools.{p.stem}"
        for p in sorted(tools_path.glob("*.py"))
        if p.name not in {"__init__.py", "registry.py"}  # skip itself
    ]
    for mod_name in module_names:
        importlib.import_module(mod_name)  # importing triggers registration
    return imported
```

```python
# tools/__init__.py — executed on package import
from tools.registry import registry, discover_tools
discover_tools()
```

**How it works:** Every tool file calls `registry.register()` at module level (the bottom of the file). When `discover_tools()` imports the module via `importlib.import_module()`, the module-level code runs automatically, and the tool is registered.

This means **adding a new tool only requires creating a new `.py` file in the `tools/` directory** — no other code needs to change.

### Dispatching

When the agent receives a tool call request from the LLM, it executes the tool via `registry.dispatch(name, args)`:

```python
# agent/agent.py — inside _execute_tool_calls()
result = registry.dispatch(name, args)
```

The `dispatch` method looks up the handler by name and calls it. The return value is always a string (typically JSON). If the tool is unknown or execution fails, a JSON string with an `"error"` field is returned instead.

---

## How to Write a New Tool

Using `tools/read_file.py` as an example, here is the standard three-part structure:

```python
# tools/read_file.py

import json
from pathlib import Path
from tools.registry import registry

# ---- Part 1: SCHEMA ----
# Tells the LLM what this tool is called, what it does, and what arguments it needs
SCHEMA = {
    "name": "read_file",
    "description": "Read the contents of a file at the given path.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read.",
            },
        },
        "required": ["path"],
    },
}

# ---- Part 2: handler ----
# Receives an arguments dict, returns a result string (JSON)
def handler(args: dict) -> str:
    path = args.get("path", "")
    try:
        content = Path(path).read_text(encoding="utf-8")
        return json.dumps({"success": True, "content": content})
    except FileNotFoundError:
        return json.dumps({"success": False, "error": f"File not found: {path}"})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

# ---- Part 3: register ----
# Runs automatically when the module is imported
registry.register("read_file", SCHEMA, handler)
```

**Three steps to add a new tool:**

1. Create a new `.py` file under `tools/`
2. Define a `SCHEMA` (OpenAI Function Calling format) and a `handler` function
3. Call `registry.register()` at the bottom of the file

Restart the program and the tool is available — no other changes required.

---

## Built-in Tools

v0.1 ships with 4 tools covering basic file-system and command operations:

| Tool | File | Description | Required Params |
|------|------|-------------|-----------------|
| `read_file` | `tools/read_file.py` | Read file contents | `path` |
| `write_file` | `tools/write_file.py` | Write to a file (auto-creates parent dirs) | `path`, `content` |
| `list_dir` | `tools/list_dir.py` | List directory entries (name, type, size) | `path` (optional, defaults to `.`) |
| `run_command` | `tools/run_command.py` | Execute a shell command | `command` (optional: `timeout`) |

All tools return JSON strings with a `success` field:

```json
// Success
{"success": true, "content": "file content here..."}

// Failure
{"success": false, "error": "File not found: xxx"}
```

---

## The CLI Layer

The CLI layer (`agent/cli.py`) uses [Rich](https://rich.readthedocs.io/) for terminal formatting and serves as the outermost layer of the agent.

### Execution Flow

```
main.py → run_cli() → create Agent instance → enter interactive loop
```

```python
# agent/cli.py — run_cli() (simplified)

def run_cli():
    _print_banner()         # display welcome banner
    agent = Agent()         # create the agent

    while True:
        user_input = Prompt.ask("[bold green]You[/bold green]")
        # handle built-in commands or forward to the agent
        response = agent.run_turn(
            user_input,
            on_tool_call=_print_tool_call,      # callback for tool invocations
            on_tool_result=_print_tool_result,   # callback for tool results
        )
        # render the answer with Rich Panel + Markdown
```

### Built-in Commands

| Command | Action |
|---------|--------|
| `/help` | Show help text |
| `/clear` | Clear conversation history |
| `/exit` | Quit the program |
| `/tools` | List all registered tools |

### Callback Mechanism

`run_turn()` accepts two optional callbacks:

- `on_tool_call(name, args)` — fired when a tool is invoked; used by the CLI to display tool call info
- `on_tool_result(name, display)` — fired when a tool returns; used to display results (truncated to 200 chars)

This callback design **decouples agent logic from UI presentation** — the agent doesn't care how results are displayed, and the CLI doesn't care how tools are executed.

---

## Summary

v0.1's architecture can be summed up in one phrase: **one loop + one registry**.

```
┌─────────────────────────────────────────────────┐
│                    CLI (cli.py)                  │
│            User input / Rich-rendered output     │
├─────────────────────────────────────────────────┤
│                Agent (agent.py)                  │
│    Agent loop: LLM ↔ tool calls → final answer   │
├──────────────────┬──────────────────────────────┤
│  Client          │        Registry              │
│  (client.py)     │    (tools/registry.py)        │
│  API communication│ Registry + auto-discovery    │
├──────────────────┼──────────────────────────────┤
│                  │  read_file / write_file /     │
│                  │  list_dir / run_command       │
└──────────────────┴──────────────────────────────┘
```

Key design principles:

- **Loop-driven**: The agent calls the LLM in a `for` loop until the LLM stops requesting tool calls
- **Registry pattern**: Tools self-register via auto-discovery; adding a tool requires zero changes to existing code
- **Callback decoupling**: Agent logic and UI presentation are separated through callback functions
- **OpenAI-compatible**: Uses the standard Function Calling format, compatible with any OpenAI API-compatible LLM service

This is the foundation for building more advanced agent capabilities. Future versions will add memory management, a skill system, social media integrations, and more — but the core "loop + registry" pattern remains the backbone.

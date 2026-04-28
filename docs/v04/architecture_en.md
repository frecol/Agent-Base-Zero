# v0.4 Deep Dive: Skill System, PromptManager & Composite Tools

v0.4 adds a **Skill System** on top of v0.3's session persistence, long-term memory, and context compression. The core idea is giving the Agent "role-switching" capabilities — different tasks activate different specialized skills, each with its own instructions and domain expertise. Three major features: **Skill auto-discovery & registration**, **PromptManager for dynamic prompt assembly**, and **LLM-driven skill activation + composite tools**.

---

## Table of Contents

- [Change Overview](#change-overview)
- [Feature 1: Skill Auto-Discovery & Registration](#feature-1-skill-auto-discovery--registration)
  - [SKILL.md Standard Format](#skillmd-standard-format)
  - [SkillInfo Data Class](#skillinfo-data-class)
  - [SkillRegistry](#skillregistry)
  - [discover_skills Process](#discover_skills-process)
  - [build_skills_index](#build_skills_index)
- [Feature 2: PromptManager](#feature-2-promptmanager)
  - [From Hardcoded to File-Based](#from-hardcoded-to-file-based)
  - [Prompt Assembly Logic](#prompt-assembly-logic)
  - [Cache-Friendly Design](#cache-friendly-design)
- [Feature 3: LLM-Driven Skill Activation](#feature-3-llm-driven-skill-activation)
  - [activate_skill Tool](#activate_skill-tool)
  - [deactivate_skill Tool](#deactivate_skill-tool)
  - [Activation Flow](#activation-flow)
- [Feature 4: Composite Tools](#feature-4-composite-tools)
  - [research_topic Composite Tool](#research_topic-composite-tool)
  - [Tool Orchestration Pattern](#tool-orchestration-pattern)
- [Example Skills](#example-skills)
  - [research — Web Research Specialist](#research--web-research-specialist)
  - [code_assistant — Code Assistant Specialist](#code_assistant--code-assistant-specialist)
- [New CLI Commands](#new-cli-commands)
- [File Change Summary](#file-change-summary)

---

## Change Overview

| Dimension | v0.3 | v0.4 |
|-----------|------|------|
| Prompt management | Hardcoded in `client.py` | `PromptManager` loads from file, supports dynamic assembly |
| Skill system | None | SKILL.md standard format, auto-discovery, LLM auto-activation |
| Composite tools | None | `research_topic` chains multiple tools via dispatch |
| CLI Commands | 11 | +3 new (`/skills`, `/skill`, `/unskill`) |
| /status | Session + Token | Now shows Active Skill |
| New directories | 0 | 1 (`skills/`) |
| New files | 0 | 6 (2 core + 2 skills + 1 tool + 1 prompt) |

---

## Feature 1: Skill Auto-Discovery & Registration

### SKILL.md Standard Format

Each skill is a subdirectory of `skills/` containing a `SKILL.md` file. Uses the **YAML frontmatter** format (delimited by `---`) compatible with Claude Code and other ecosystems:

```markdown
---
name: research
description: "Web research specialist, excels at search, extraction, and synthesis."
user_invocable: true
---

You are now operating in Research Mode...
(full instructions)
```

**Frontmatter fields:**
- `name` (required): Skill identifier
- `description` (required): Skill description, included in system prompt for LLM to decide when to activate
- `user_invocable` (optional, default: true): Whether users can invoke the skill directly

**Content after the frontmatter** is the skill's detailed instructions (Markdown), returned to the LLM as a tool result when the skill is activated.

### SkillInfo Data Class

```python
# skills/registry.py
@dataclass
class SkillInfo:
    name: str
    description: str
    prompt_text: str         # Full content after frontmatter in SKILL.md
    user_invocable: bool
    skill_dir: Optional[Path]
```

### SkillRegistry

Follows the existing registry pattern (symmetric with `ToolRegistry`):

```python
class SkillRegistry:
    _skills: Dict[str, SkillInfo]
    _active_skill: Optional[str]
    _on_activate: Optional[Callable]

    def register(skill_info)           # Register a skill
    def activate(name) -> str          # Activate, returns JSON (includes prompt_text)
    def deactivate() -> str            # Deactivate
    def get_active() -> SkillInfo?     # Get currently active skill
    def get_all_names() -> List[str]   # All registered skill names
    def get_skill(name) -> SkillInfo?  # Lookup by name
    def set_on_activate(callback)      # Register state change callback
```

Global singleton: `skill_registry`.

### discover_skills Process

```python
def discover_skills(skills_dir=None) -> List[str]:
    skills_path = skills_dir or Path(__file__).resolve().parent
    for item in sorted(skills_path.iterdir()):
        if not item.is_dir(): continue
        if item.name.startswith(("_", ".")): continue

        skill_md = item / "SKILL.md"
        if not skill_md.exists(): continue

        info = _parse_skill_md(skill_md)   # Parse frontmatter + content
        skill_registry.register(info)

    _register_skill_tools()  # Register activate_skill / deactivate_skill
```

**Frontmatter parsing**: Uses regex `^---\s*\n(.*?)\n---\s*\n` to extract the YAML block, then parses `key: value` pairs line by line. Supports quoted strings and booleans — no `pyyaml` dependency needed.

### build_skills_index

```python
def build_skills_index() -> str:
    # Output format:
    # ## Available Skills
    # When a task matches a skill below, call activate_skill(name)...
    #
    # - research: Web research specialist...
    # - code_assistant: Code assistant specialist...
```

Called at startup and appended to the system prompt.

---

## Feature 2: PromptManager

### From Hardcoded to File-Based

v0.3's `SYSTEM_PROMPT` was a Python string constant in `agent/client.py`. v0.4 extracts it to an independent file:

```
agent/client.py (SYSTEM_PROMPT constant) → agent/system_prompt.md (standalone file)
                                        + agent/prompt.py (PromptManager class)
```

`client.py` no longer holds any prompt knowledge — it is now a pure API wrapper.

### Prompt Assembly Logic

```python
class PromptManager:
    def __init__(self, base_prompt_path=None):
        self._base_prompt = read("agent/system_prompt.md")
        self._skills_index = ""

    def update_skills_index(self, skills_index):
        # Called once at startup
        self._skills_index = skills_index

    def get_system_prompt(self) -> str:
        return self._base_prompt + "\n\n" + self._skills_index
```

**Assembled result:**
```
[base prompt from system_prompt.md]

## Available Skills
When a task matches a skill below, call activate_skill(name)...

- research: Web research specialist...
- code_assistant: Code assistant specialist...
```

### Cache-Friendly Design

**Key principle: the system prompt prefix stays stable throughout the entire session.**

1. **At startup**: base prompt + skills index are assembled → prefix is fixed
2. **On skill activation**: skill instructions are returned as `activate_skill`'s **tool result**, entering conversation history
3. **System prompt unchanged** → prompt prefix cache always hits

```
System prompt (stable prefix, never changes with skill switching):
┌─────────────────────────────┐
│ base prompt                 │ ← agent/system_prompt.md
│ + skills index (name+desc)  │ ← Loaded at startup
├─────────────────────────────┤ ← Cache boundary
│ memory system message       │ ← v0.3 existing
│ + conversation history      │ ← Grows dynamically
│ + skill tool results        │ ← Skill instructions enter here
└─────────────────────────────┘
```

---

## Feature 3: LLM-Driven Skill Activation

### activate_skill Tool

```python
SCHEMA = {
    "name": "activate_skill",
    "description": "Activate a skill by name. Returns the skill's detailed instructions.",
    "parameters": {
        "type": "object",
        "properties": {
            "skill_name": {"type": "string", "description": "Name of the skill to activate"}
        },
        "required": ["skill_name"],
    },
}
```

**Handler logic:**
1. Call `skill_registry.activate(name)` to set activation state
2. Trigger callback to notify Agent (state change)
3. Return the skill's full `prompt_text` as **tool result**

### deactivate_skill Tool

```python
SCHEMA = {
    "name": "deactivate_skill",
    "description": "Deactivate the currently active skill and return to base mode.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}
```

### Activation Flow

```
User: "Search for the latest Python version"
  → _build_messages(): system prompt unchanged (includes skills index)
  → LLM sees "research: Web research specialist..." in skills index
  → LLM calls activate_skill("research")
  → handler returns full content of research/SKILL.md as tool result
  → tool result enters conversation history (system prompt unchanged, cache preserved)
  → LLM follows skill instructions to call web_search, etc.
  → Final response to user
```

**Callback mechanism**: The tool handler calls `skill_registry._on_activate(name, prompt_text)`, which Agent registers at init. The current implementation has an empty callback body (pass), since `_get_tools()` reads `skill_registry`'s latest state on every call. The callback mechanism preserves an extension point for future versions.

**CLI manual activation**: When the user activates a skill via `/skill <name>`, `Agent.activate_skill()` constructs a simulated assistant tool_call + tool result message pair and injects it into `self.messages`, ensuring the LLM receives the skill instructions. This mirrors the LLM-driven `activate_skill` tool path — instructions enter conversation history via tool result, keeping the system prompt unchanged. Skills with `user_invocable: false` are rejected by the CLI, allowing LLM-only activation.

---

## Feature 4: Composite Tools

### research_topic Composite Tool

A composite tool is a regular tool in the `tools/` directory whose handler orchestrates multiple existing tools via `registry.dispatch()`:

```python
# tools/research_topic.py
def handler(args: dict) -> str:
    query = args.get("query", "")
    max_sources = args.get("max_sources", 3)

    # Step 1: Search
    search_raw = registry.dispatch("web_search", {"query": query})
    search_result = json.loads(search_raw)

    # Step 2: Fetch full content
    for r in search_result.get("results", [])[:max_sources]:
        fetch_raw = registry.dispatch("fetch_url", {"url": r["href"]})
        # ... consolidate results

    return json.dumps({"success": True, "sources": fetched, ...})
```

### Tool Orchestration Pattern

The composite tool pattern:

1. Define `SCHEMA` and `handler` — identical to any regular tool
2. Handler internally calls `registry.dispatch("tool_name", args)` to invoke other tools
3. Consolidate results from multiple tools into a higher-level return value
4. Register via `registry.register()` — to the LLM it appears as a new tool

**Advantage**: No new abstractions or frameworks needed. Reuses the existing tool registration and dispatch mechanism. Developers only need to write a handler function that orchestrates existing tools.

---

## Example Skills

### research — Web Research Specialist

```
skills/research/
└── SKILL.md
```

**Instruction highlights:**
- Search strategy (broad first, then narrow; no repeated queries)
- Information extraction and synthesis methods
- Source citation standards
- When to search vs. use existing knowledge
- Recommends `research_topic` composite tool for complex research tasks

### code_assistant — Code Assistant Specialist

```
skills/code_assistant/
└── SKILL.md
```

**Instruction highlights:**
- Read-before-modify workflow
- Debugging guide (understand error → locate code → verify fix)
- Code style (follow existing project conventions)
- Tool preferences (edit_file > write_file, tree for structure, etc.)

---

## New CLI Commands

| Command | Description |
|---------|-------------|
| `/skills` | List all available skills (marks active, shows `(auto-only)` for `user_invocable: false`) |
| `/skill <name>` | Manually activate a skill (rejects `user_invocable: false` skills) |
| `/unskill` | Deactivate current skill, return to base mode |

`/status` enhanced: New `Active Skill` line shows the currently active skill name.

---

## File Change Summary

| File | Action | Description |
|------|--------|-------------|
| `agent/prompt.py` | **New** | PromptManager: loads base prompt from file + appends skills index |
| `agent/system_prompt.md` | **New** | Base system prompt (extracted from client.py) |
| `skills/__init__.py` | **New** | Auto-discovery entry point, calls discover_skills() |
| `skills/registry.py` | **New** | SkillRegistry, SKILL.md parsing, activate_skill/deactivate_skill tools |
| `skills/research/SKILL.md` | **New** | Web research skill |
| `skills/code_assistant/SKILL.md` | **New** | Code assistant skill |
| `tools/research_topic.py` | **New** | Composite tool: web_search + fetch_url chained orchestration |
| `agent/client.py` | Modified | Removed SYSTEM_PROMPT constant |
| `agent/agent.py` | Modified | PromptManager integration, skill callback, activate_skill/deactivate_skill methods (CLI activation injects instructions into conversation) |
| `agent/cli.py` | Modified | +/skills /skill /unskill commands, /status shows Active Skill, version v0.4 |
| `main.py` | Modified | Added `import tools` / `import skills` for auto-discovery |
| `pyproject.toml` | Modified | Version 0.4.0, package discovery includes skills* |

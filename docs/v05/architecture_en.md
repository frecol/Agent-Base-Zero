# v0.5 Deep Dive: Plan Mode — Separate Planning from Execution

v0.5 adds **Plan Mode** on top of v0.4's skill system, PromptManager, and composite tools. The core idea is splitting the Agent's work into two phases: **read-only exploration + planning** and **step-by-step execution**. Users first let the Agent analyze a task and generate a structured plan in a controlled environment, then confirm and execute step by step. Six new components: **Plan data structures**, **tool classification & read-only filtering**, **plan parser**, **Plan Mode prompt**, **interactive input handler**, and **visual renderer**.

---

## Table of Contents

- [Change Overview](#change-overview)
- [Feature 1: Plan Mode Entry & Toggle](#feature-1-plan-mode-entry--toggle)
  - [Shift+Tab Key Binding](#shifttab-key-binding)
  - [/plan Command](#plan-command)
  - [Mode State Synchronization](#mode-state-synchronization)
- [Feature 2: Tool Classification & Read-Only Filtering](#feature-2-tool-classification--read-only-filtering)
  - [Tool Categories in ToolRegistry](#tool-categories-in-toolregistry)
  - [Agent._get_tools Filtering](#agent_get_tools-filtering)
  - [Execution-Layer Hard Guard](#execution-layer-hard-guard)
- [Feature 3: Plan Data Structures & Lifecycle](#feature-3-plan-data-structures--lifecycle)
  - [PlanPhase & StepStatus](#planphase--stepstatus)
  - [PlanStep & Plan](#planstep--plan)
- [Feature 4: Plan Parser](#feature-4-plan-parser)
  - [LLM Output Format](#llm-output-format)
  - [parse_plan Flow](#parse_plan-flow)
  - [Loose Parsing Fallback](#loose-parsing-fallback)
- [Feature 5: Plan Mode Prompt](#feature-5-plan-mode-prompt)
  - [PromptManager Conditional Injection](#promptmanager-conditional-injection)
  - [plan_prompt.md Contents](#plan_promptmd-contents)
- [Feature 6: Interactive Review & Visualization](#feature-6-interactive-review--visualization)
  - [Plan Review Panel](#plan-review-panel)
  - [Modification Flow](#modification-flow)
  - [Live Execution Progress](#live-execution-progress)
  - [Shimmer Animation](#shimmer-animation)
  - [Completion Summary](#completion-summary)
- [Plan Mode End-to-End Flow](#plan-mode-end-to-end-flow)
  - [1. Planning Phase](#1-planning-phase)
  - [2. Review Phase](#2-review-phase)
  - [3. Execution Phase](#3-execution-phase)
  - [4. Completion Phase](#4-completion-phase)
- [CLI Changes](#cli-changes)
- [File Change Summary](#file-change-summary)

---

## Change Overview

| Dimension | v0.4 | v0.5 |
|-----------|------|------|
| Work mode | Single mode (direct execution) | Normal + Plan dual mode (plan then execute) |
| Tool safety | All tools always available | Plan mode exposes read-only tools only + execution-layer hard guard |
| Prompt | base + skills index | base + **plan prompt** (conditionally injected) + skills index |
| Task execution | LLM completes in one go | Structured plan → user confirmation → step-by-step execution |
| Input handling | Basic `input()` | prompt_toolkit input with Shift+Tab toggle |
| Visualization | Rich panels | New Plan Review panel, live progress bar, shimmer animation |
| New files | 0 | 6 (5 core + 1 prompt) |
| New dependencies | 0 | 1 (`prompt-toolkit`) |

---

## Feature 1: Plan Mode Entry & Toggle

### Shift+Tab Key Binding

v0.5 introduces `prompt_toolkit` to replace standard `input()` for user input, enabling special key detection:

```python
# agent/plan_input.py
class InputHandler:
    def _setup_keybindings(self) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add("s-tab")
        def _toggle_mode(event):
            self._plan_mode = not self._plan_mode
            event.app.exit(result="")  # Exit prompt immediately, CLI detects toggle

        return bindings
```

When Shift+Tab is pressed, `_plan_mode` flips and `get_input()` returns an empty string. The CLI main loop detects the mode change and synchronizes the Agent state.

### /plan Command

The mode can also be toggled via the `/plan` CLI command:

```python
elif cmd == "/plan":
    input_handler.plan_mode = not input_handler.plan_mode
    agent.set_plan_mode(input_handler.plan_mode)
```

### Mode State Synchronization

Mode toggles must synchronize three locations:

1. `InputHandler._plan_mode` — UI layer (determines prompt style)
2. `Agent._plan_mode` — Agent layer (determines tool filtering + prompt injection)
3. `PromptManager._plan_mode_enabled` — Prompt layer (conditional plan prompt injection)

```python
# agent/agent.py
def set_plan_mode(self, enabled: bool) -> None:
    self._plan_mode = enabled
    self.prompt_manager.set_plan_mode(enabled)  # Triggers prompt update
```

**Prompt styles:**
- Normal mode: `You> ` (green)
- Plan mode: `Plan> ` (magenta)

---

## Feature 2: Tool Classification & Read-Only Filtering

### Tool Categories in ToolRegistry

`tools/registry.py` adds a `_TOOL_CATEGORIES` dictionary that classifies all tools into two groups:

```python
_TOOL_CATEGORIES: Dict[str, str] = {
    # Read-only tools — available in Plan Mode
    "read_file": "read_only",
    "list_dir": "read_only",
    "tree": "read_only",
    "find_file": "read_only",
    "grep_search": "read_only",
    "web_search": "read_only",
    "fetch_url": "read_only",
    "research_topic": "read_only",
    "current_time": "read_only",
    "system_info": "read_only",
    "session_search": "read_only",
    # Write tools — blocked in Plan Mode
    "write_file": "write",
    "edit_file": "write",
    "file_delete": "write",
    "run_command": "write",
    "memory_save": "write",
    "activate_skill": "write",
    "deactivate_skill": "write",
}
```

New methods:
- `get_read_only_names()` — Returns all read-only tool names (for Plan Mode)
- `is_write_tool(name)` — Checks if a tool is classified as write (for execution-layer guard)

### Agent._get_tools Filtering

```python
def _get_tools(self) -> Optional[List[dict]]:
    if self._plan_mode:
        names = registry.get_read_only_names()  # Only read-only tools
    else:
        names = registry.get_all_names()         # All tools
    return registry.get_definitions(names) if names else None
```

At the API level: write tool definitions are not sent to the LLM in Plan Mode.

### Execution-Layer Hard Guard

Even if the LLM "hallucinates" a write tool call in Plan Mode (e.g., the conversation history from Normal Mode contains `tool_calls` for write tools, and the model may still attempt to call them), `_execute_tool_calls()` intercepts at the execution layer:

```python
if self._plan_mode and registry.is_write_tool(name):
    rejected = json.dumps({"error": f"Tool '{name}' is not available in Plan Mode."})
    self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": rejected})
    continue  # Skip execution, return error to LLM
```

**Dual protection**: API-level tool filtering + execution-layer rejection ensures Plan Mode's read-only safety.

---

## Feature 3: Plan Data Structures & Lifecycle

### PlanPhase & StepStatus

```python
class PlanPhase(Enum):
    PLANNING    = auto()  # Agent explores with read-only tools and generates plan
    REVIEWING   = auto()  # User reviews plan (accept/modify/cancel)
    EXECUTING   = auto()  # Plan accepted, executing step by step
    COMPLETED   = auto()  # All steps finished (success or failure)

class StepStatus(Enum):
    PENDING     = "pending"
    IN_PROGRESS = "in_progress"
    DONE        = "done"
    FAILED      = "failed"
    SKIPPED     = "skipped"
```

### PlanStep & Plan

```python
@dataclass
class PlanStep:
    index: int              # 1-based step number
    title: str              # Short title (one line)
    description: str        # Detailed description
    status: StepStatus = StepStatus.PENDING

@dataclass
class Plan:
    goal: str                               # User's original goal
    steps: List[PlanStep]                   # Step list
    phase: PlanPhase = PlanPhase.PLANNING   # Current phase
    raw_plan_text: str = ""                 # Raw LLM output
```

**Lifecycle:**
```
User input (Plan Mode)
    ↓
PLANNING: Agent explores codebase with read-only tools, generates structured plan
    ↓
REVIEWING: parse_plan() extracts steps → render_plan_review() displays plan panel
    ↓                                   User selects Accept / Modify / Cancel
    ↓ (Accept)
EXECUTING: Switch to Normal mode → execute steps one by one → PlanProgressLive real-time updates
    ↓
COMPLETED: render_plan_summary() shows final statistics
```

---

## Feature 4: Plan Parser

### LLM Output Format

The LLM is instructed to produce plans in this format in Plan Mode:

```markdown
## Plan
**Goal**: <one-sentence summary>

### Steps
1. **<title>** -- <detailed description>
2. **<title>** -- <detailed description>
...
```

### parse_plan Flow

```python
# agent/plan_parser.py
def parse_plan(llm_output: str, user_goal: str = "") -> Optional[Plan]:
    # 1. Look for "## Plan" or "## Execution Plan" section
    plan_match = re.search(r"(?:##\s*Plan|##\s*Execution\s*Plan)\s*\n(.*?)(?:\n##\s|\Z)", ...)

    if not plan_match:
        return _try_parse_loose(llm_output, user_goal)

    # 2. Extract Goal if present
    goal_match = re.search(r"\*\*Goal\*\*:\s*(.+?)(?:\n|$)", ...)

    # 3. Extract Steps
    steps = _parse_steps(plan_text)
```

**Step regex** supports multiple format variations:
- Numbering: `1.` or `1)`
- Title: `**bold**` or plain text
- Separator: `--`, `-`, `:`, `—` (em dash), `–` (en dash)

### Loose Parsing Fallback

If no formal `## Plan` section is found, `_try_parse_loose()` searches the entire text for at least 2 numbered steps. A minimum of 2 steps is required — a single numbered line doesn't constitute a plan.

---

## Feature 5: Plan Mode Prompt

### PromptManager Conditional Injection

v0.5 extends `PromptManager` with conditional Plan Mode prompt injection:

```python
class PromptManager:
    def __init__(self):
        self._base_prompt = read("agent/prompts/system_prompt.md")
        self._skills_index = ""
        # v0.5: Plan Mode prompt
        self._plan_mode_enabled = False
        self._plan_mode_prompt = read("agent/prompts/plan_prompt.md")

    def get_system_prompt(self) -> str:
        parts = [self._base_prompt]
        if self._plan_mode_enabled and self._plan_mode_prompt:
            parts.append(self._plan_mode_prompt)  # Conditional injection
        if self._skills_index:
            parts.append(self._skills_index)
        return "\n\n".join(parts)
```

**Prompt layering:**
```
┌──────────────────────┐
│ Base Prompt          │ ← Always present
├──────────────────────┤
│ Plan Mode Prompt     │ ← Only injected in Plan Mode
├──────────────────────┤
│ Skills Index         │ ← Always present
└──────────────────────┘
```

### plan_prompt.md Contents

The Plan Mode prompt enforces a strict two-phase process and information-completeness:

**Phase 1 — Explore & Collect Information:**
- Use read-only tools (read_file, list_dir, grep_search, etc.) to thoroughly explore the codebase
- Read all relevant files, search for existing patterns, trace dependencies
- Note exact file paths, line numbers, function names, and variable names
- Identify every file that will need to be created or modified
- If the user's request has ambiguity, make a reasonable assumption based on code context

**Phase 2 — Generate the Plan:**
- Only output the plan after Phase 1 is complete
- Every step description MUST include: specific file paths, line numbers or function names, exact changes to make, and expected result
- FORBIDDEN: steps containing "confirm", "ask user", "depending on user choice", "TBD" — any wording that requires runtime interaction
- Steps: 3–10, ordered by dependency, titles under 60 chars
- Include a verification step at the end
- If information is genuinely insufficient, state the assumption explicitly rather than deferring to the user

---

## Feature 6: Interactive Review & Visualization

### Plan Review Panel

After the plan is generated, `render_plan_review()` displays a magenta-bordered panel:

```
┌─────────── Plan Mode ───────────┐
│ Goal: Refactor auth module      │
│                                  │
│ Proposed Steps:                  │
│   1. Analyze existing auth code │
│      Read all files in auth/ dir │
│   2. Extract auth logic          │
│      Create auth/handler.py      │
│   ...                            │
└──────────────────────────────────┘
```

The user selects **Accept / Modify / Cancel** using arrow keys.

### Modification Flow

When Modify is selected, the user provides feedback, and the original plan + feedback is sent to the LLM to produce a revised plan. Up to **3 modification rounds** are supported (prevents infinite loops).

### Live Execution Progress

The execution phase uses a `PlanProgressLive` context manager based on Rich Live with 8fps auto-refresh:

```
○  Step 1: Analyze project structure       Pending
●  Step 2: Review existing code            In Progress...
✓  Step 3: Implement changes               Done
```

**Status icons:**
- ○ Pending (dim)
- ● In Progress (yellow pulsing animation + shimmer sweep effect)
- ✓ Done (green)
- ✗ Failed (red)
- — Skipped (dim)

**Step failure handling:** The user can choose Continue (skip and proceed) or Stop (skip all remaining steps).

### Shimmer Animation

`agent/shimmer.py` provides a shared light-sweep animation effect. A bright spot travels left-to-right across text:

```python
def shimmer_positions(text: str, frame: int) -> list[int]:
    """Return brightness level (0-3) for each character. 3 = peak."""
    cycle = length + 8
    pos = frame % cycle - 4
    return [max(0, 3 - abs(i - pos)) for i in range(length)]
```

Two usage scenarios in the CLI:
1. **Streaming Thinking animation** — ANSI escape codes
2. **Progress bar In Progress text** — Rich Text styles

### Completion Summary

After execution, `render_plan_summary()` displays statistics:

```
┌────────── Plan Complete ──────────┐
│ 3 completed, 0 failed (of 3 total)│
└───────────────────────────────────┘
```

Border color: green if no failures, yellow if any failures.

---

## Plan Mode End-to-End Flow

### 1. Planning Phase

```
User: /plan (or Shift+Tab)
  → Agent enters Plan Mode
  → PromptManager injects plan_prompt.md
  → _get_tools() returns read-only tools only
  → _execute_tool_calls() blocks write tool calls

User: "Refactor the auth module"
  → Agent explores codebase with read-only tools (read_file, grep_search, tree, etc.)
  → LLM generates structured plan (Goal + Steps)
```

### 2. Review Phase

```
  → parse_plan() parses LLM output into a Plan object
  → render_plan_review() displays plan panel
  → User selects:
     ├─ Accept → proceed to execution
     ├─ Modify → user provides feedback → LLM revises → re-review (max 3 rounds)
     └─ Cancel → discard plan
```

### 3. Execution Phase

```
  → Switch to Normal mode (all tools available)
  → PlanProgressLive starts real-time progress display
  → For each Step:
     → Build step_prompt (includes title, description, context)
     → agent.run_turn(step_prompt) executes silently
     → Update step status (PENDING → IN_PROGRESS → DONE/FAILED)
     → On failure, user chooses Continue or Stop
```

### 4. Completion Phase

```
  → render_plan_summary() displays statistics
  → plan.phase = COMPLETED
```

---

## CLI Changes

### New Commands

| Command | Description |
|---------|-------------|
| `/plan` | Toggle Plan / Normal mode |
| Shift+Tab | Same (instant toggle during input) |

### Changed Commands

| Command | Change |
|---------|--------|
| `/status` | New Plan Mode row showing current mode |
| `/help` | New `/plan` entry |

### Banner Enhancement

When Plan Mode is active, the banner displays `Mode: PLAN (read-only)`.

---

## File Change Summary

| File | Action | Description |
|------|--------|-------------|
| `agent/plan.py` | **New** | Plan data structures (PlanPhase, StepStatus, PlanStep, Plan) |
| `agent/plan_parser.py` | **New** | Plan parser, extracts structured plans from LLM output |
| `agent/prompts/plan_prompt.md` | **New** | Plan Mode system prompt (instructs LLM to generate plan format) |
| `agent/prompts/system_prompt.md` | Moved | Relocated to prompts/ directory (originally in agent/) |
| `agent/plan_input.py` | **New** | InputHandler with prompt_toolkit + Shift+Tab toggle |
| `agent/plan_renderer.py` | **New** | Plan Review renderer + live progress bar |
| `agent/shimmer.py` | **New** | Shared light-sweep animation position calculator |
| `agent/agent.py` | Modified | New `_plan_mode` state, `set_plan_mode()`, execution-layer write tool guard |
| `agent/cli.py` | Modified | `/plan` command, Plan Mode prompt, plan review & execution flow |
| `agent/prompt.py` | Modified | `set_plan_mode()` for conditional plan_prompt injection |
| `tools/registry.py` | Modified | `_TOOL_CATEGORIES` dict, `get_read_only_names()`, `is_write_tool()` |
| `pyproject.toml` | Modified | Version 0.5.0, added `prompt-toolkit` dependency |
| `main.py` | Modified | Version comment updated to v0.5 |

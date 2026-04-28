# Agent Runtime Logic Deep Dive

This document details the runtime logic of Agent-Base-Zero v0.5, including the main loop flow, mode toggle mechanism, and the response paths for Normal / Plan dual modes.

---

## Main Loop Flow

The CLI entry point is the `run_cli()` function in `agent/cli.py`. Three core objects are initialized at startup:

```
run_cli() starts
  ├── PromptManager    loads base prompt + plan_prompt.md + skills index
  ├── Agent            core agent, holds conversation history and LLM client
  ├── InputHandler     input handler, holds _plan_mode state (default False = Normal)
  │
  ├── stream_enabled   = settings.stream_enabled   (default True)
  ├── thinking_enabled = settings.thinking_enabled  (default True)
  │
  └── while True:  main loop
```

### 6 Steps Per Loop Iteration

```
while True:

  ① Record current mode
     mode_before = input_handler.plan_mode

  ② Wait for user input
     user_input = input_handler.get_input()
       ├── prompt_toolkit renders prompt:
       │     Normal → "You> "  (green)
       │     Plan   → "Plan> " (magenta)
       └── If user pressed Shift+Tab:
             _plan_mode flips
             get_input() returns ""

  ③ Detect mode change (Shift+Tab trigger)
     if input_handler.plan_mode != mode_before:
       → agent.set_plan_mode(...)
       → Print mode toggle notification
       → continue  ← Skip this round, no input processed

  ④ Empty input check
     if not user_input: continue

  ⑤ Built-in command handling
     /exit, /help, /plan, /status ... each handled with continue

  ⑥ Conversation handling (core branch, see below)
```

**Mode detection is not active polling.** Shift+Tab causes `get_input()` to return early with an empty string, and step ③ catches the change.

---

## Tool Filtering Mechanism

`Agent._get_tools()` determines which tools are sent to the LLM based on the current mode:

```
_get_tools()
  │
  ├── _plan_mode = True  (Plan Mode)
  │   └── registry.get_read_only_names()
  │       → read_file, list_dir, tree, find_file, grep_search
  │       → web_search, fetch_url, research_topic
  │       → current_time, system_info, session_search
  │       (11 read-only tools total)
  │
  └── _plan_mode = False (Normal Mode)
      └── registry.get_all_names()
          → above 11 + write_file, edit_file, file_delete
          → run_command, memory_save, activate_skill, deactivate_skill
          (18 tools total)
```

**Dual protection**: In addition to not sending write tool definitions at the API level, `_execute_tool_calls()` has a hard guard at the execution layer — write tool calls are intercepted in Plan Mode and an error is returned to the LLM.

---

## Prompt Assembly

`PromptManager.get_system_prompt()` conditionally assembles the system prompt based on mode:

```
Normal Mode:
  [base prompt from system_prompt.md]
  [skills index]

Plan Mode:
  [base prompt from system_prompt.md]
  [plan_prompt.md]              ← Conditionally injected, instructs LLM to generate structured plan
  [skills index]
```

---

## Normal Mode Response Logic

Reference code: `cli.py` lines 639-660

```
Normal Mode
  │
  ├── stream_enabled = True:
  │   └── _stream_response(agent, user_input, thinking_enabled)
  │       │
  │       └── agent.run_turn_stream(user_input, ...)
  │           │
  │           │  Setup:
  │           │  ├── messages.append({"role": "user", "content": user_input})
  │           │  ├── tools = _get_tools()            → all 18 tools
  │           │  └── api_messages = _build_messages() → base prompt + skills index
  │           │
  │           │  Enter iteration loop (max 50 times):
  │           │  │
  │           │  ├── client.chat_stream(api_messages, tools, thinking_enabled)
  │           │  │
  │           │  │  Accumulate streaming chunks:
  │           │  │  ├── reasoning_content  → yield thinking event
  │           │  │  │   CLI shows: shimmer "Thinking..." animation
  │           │  │  │
  │           │  │  ├── content            → yield content event
  │           │  │  │   CLI outputs token by token to terminal
  │           │  │  │
  │           │  │  └── tool_calls deltas  → accumulate complete tool call
  │           │  │
  │           │  ├── After stream ends, check:
  │           │  │   │
  │           │  │   ├── Has tool_calls:
  │           │  │   │   ├── messages.append(assistant_msg + tool_calls)
  │           │  │   │   ├── _execute_tool_calls(...)
  │           │  │   │   │   For each tool_call:
  │           │  │   │   │   ├── on_tool_call(name, args)  → CLI shows tool call panel
  │           │  │   │   │   ├── registry.dispatch(name, args) → execute tool
  │           │  │   │   │   └── messages.append(tool_result)
  │           │  │   │   └── continue → back to loop start, send tool results to LLM
  │           │  │   │
  │           │  │   └── No tool_calls (plain text):
  │           │  │       ├── messages.append(assistant_msg)
  │           │  │       ├── _post_turn() → compression check + session save
  │           │  │       └── yield done event → finish
  │           │
  │           └── Reached 50 iteration limit:
  │               └── yield done (with max iteration message)
  │
  └── stream_enabled = False:
      └── agent.run_turn(user_input, ...)
          │  Blocking call to client.chat()
          │  Same tool_calls loop logic, but non-streaming
          └── Print Panel(Markdown(response), title="Assistant", border="blue")
```

---

## Plan Mode Response Logic

Reference code: `cli.py` lines 603-638

Plan Mode has three phases: **Planning** → **Review** → **Execution**.

### Phase 1: Planning (Plan Mode, Read-Only Exploration)

```
Plan Mode — Planning Phase
  │
  ├── stream_enabled = True:
  │   └── _stream_response(agent, user_input, thinking_enabled)
  │       └── agent.run_turn_stream(user_input, ...)
  │           │  tools = _get_tools()            → only 11 read-only tools
  │           │  api_messages = _build_messages() → base + plan_prompt + skills index
  │           │  _execute_tool_calls hard guard   → write tool calls intercepted
  │           │
  │           └── LLM explores codebase with read_file, grep_search, etc.
  │              Final output is text containing a structured plan
  │
  │   response = _get_last_assistant_content(agent.messages)
  │
  ├── stream_enabled = False:
  │   └── agent.run_turn(user_input, ...)
  │       └── Print Panel(Markdown(response), title="Plan", border="magenta")
  │
  └── Parse plan:
      plan = parse_plan(response, user_input)
```

### Phase 2: Review (User Interaction)

```
parse_plan has result?
  │
  ├── Yes (parsed structured plan with steps):
  │   └── _handle_plan_review(plan, input_handler, agent, ...)
  │       │
  │       ├── render_plan_review(plan, console)
  │       │   → Display magenta Plan panel (Goal + Steps)
  │       │
  │       └── select_option(["Accept", "Modify", "Cancel"])
  │           │
  │           ├── Accept (0):
  │           │   └── return plan  → proceed to execution
  │           │
  │           ├── Modify (1):
  │           │   ├── Get user feedback text
  │           │   ├── Build revise_prompt = original plan + user feedback
  │           │   ├── agent.run_turn(revise_prompt)
  │           │   │   → LLM generates revised plan
  │           │   ├── parse_plan(revised response)
  │           │   └── Recursive _handle_plan_review(depth + 1)
  │           │       → Max 3 modification rounds, prevents infinite loop
  │           │
  │           └── Cancel (2) / Ctrl+C:
  │               └── return None  → plan discarded
  │
  └── No (no plan parsed):
      └── Silently skip (no warning shown)
```

### Phase 3: Execution (Switches Back to Normal Mode)

```
plan accepted?
  │
  ├── Yes:
  │   └── _execute_plan(plan, agent, thinking_enabled, input_handler)
  │       │
  │       ├── Switch back to Normal Mode:
  │       │   input_handler.plan_mode = False
  │       │   agent.set_plan_mode(False)    → restore all tools
  │       │   plan.phase = EXECUTING
  │       │
  │       ├── Print "Executing Plan" green rule
  │       │
  │       └── with PlanProgressLive(plan):  ← Rich Live 8fps progress bar
  │           │
  │           └── for step in plan.steps:
  │               │
  │               ├── step.status = IN_PROGRESS
  │               │
  │               ├── Build step_prompt:
  │               │   "Execute Step {n}: {title}
  │               │    Details: {description}
  │               │    Context: step {n} of {total} in plan to '{goal}'
  │               │    Perform ONLY this step. Be concise."
  │               │
  │               ├── agent.run_turn(step_prompt, thinking_enabled)
  │               │   ← Note: no on_tool_call / on_tool_result callbacks
  │               │   ← Tool calls execute silently, user only sees progress bar
  │               │   ← _get_tools() restored to return all 18 tools
  │               │
  │               ├── Success:
  │               │   step.status = DONE
  │               │   Print one-line result summary (truncated to 120 chars)
  │               │
  │               └── Failure:
  │                   step.status = FAILED
  │                   select_option(["Continue", "Stop"])
  │                   ├── Continue → skip current step, proceed to next
  │                   └── Stop → all PENDING steps marked SKIPPED, break
  │
  └── render_plan_summary(plan, console)
      → "X completed, Y failed, Z skipped (of N total)"
      → No failures: green border
      → With failures: yellow border
```

---

## Overall Flow Diagram

```
                       ┌──────────────────────┐
                       │    while True loop    │
                       └──────────┬───────────┘
                                  │
                     ① mode_before = plan_mode
                     ② user_input = get_input()
                                  │
                     ③ Mode changed? ──Yes──→ Sync Agent state, continue
                                  │ No
                     ④ Empty input? ────Yes──→ continue
                                  │ No
                     ⑤ /command? ────Yes──→ Handle command, continue
                                  │ No
                     ⑥ plan_mode?
                       ┌──────────┴──────────┐
                       │                     │
                    False                  True
                       │                     │
              ┌────────┴────────┐   ┌────────┴─────────────────┐
              │   Normal Mode   │   │   Plan Mode               │
              │                 │   │                           │
              │  _get_tools():  │   │  _get_tools():            │
              │  All 18 tools   │   │  Only 11 read-only tools   │
              │                 │   │  + Execution-layer guard   │
              │  Prompt:        │   │                           │
              │  base + skills  │   │  Prompt:                  │
              │                 │   │  base + plan_prompt        │
              │                 │   │  + skills                  │
              │                 │   │                           │
              │  run_turn /     │   │  run_turn / stream         │
              │  stream         │   │  (LLM read-only explore    │
              │                 │   │   + generate plan)          │
              │  Live display:  │   │         ↓                  │
              │  thinking anim  │   │  parse_plan()              │
              │  tool call panel│   │         ↓                  │
              │  token output   │   │  review: Accept / Modify   │
              │  response panel │   │         ↓ (Accept)         │
              │                 │   │  Switch to Normal Mode     │
              │                 │   │  _get_tools() → all tools  │
              │                 │   │         ↓                  │
              │                 │   │  for step in plan.steps:   │
              │                 │   │    run_turn(step) silent    │
              │                 │   │    progress bar live update │
              │                 │   │         ↓                  │
              │                 │   │  summary panel             │
              └─────────────────┘   └────────────────────────────┘
```

---

## Key Design Decisions

1. **Mode toggle timing**: Shift+Tab is not actively polled. It causes `get_input()` to return early with an empty string, and step ③ catches the change at the top of the next loop iteration.
2. **Plan Mode read-only safety**: API-level tool filtering + execution-layer hard guard provides dual protection against write operations.
3. **Silent execution**: During Plan execution, each step's `run_turn()` has no `on_tool_call` callback. Tool calls are invisible to the user, who only sees the progress bar.
4. **Auto-restore after execution**: `_execute_plan()` switches back to Normal Mode at the start, making all tools available during execution.
5. **Iteration limit**: A single `run_turn` allows up to 50 tool-call loop iterations to prevent infinite loops.
6. **Plan modification limit**: The review phase allows up to 3 modification rounds to prevent infinite modification loops.

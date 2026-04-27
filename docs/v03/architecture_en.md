# v0.3 Deep Dive: Session Persistence, Long Memory & Context Compression

v0.3 adds six major capabilities on top of v0.2's "conversation loop + streaming + tools" foundation: **session persistence**, **session recovery**, **long-term memory tool**, **memory injection**, **token estimation & progress display**, and **context compression**. All data is stored as pure JSON files — no database required.

---

## Table of Contents

- [Change Overview](#change-overview)
- [Feature 1: Session Persistence](#feature-1-session-persistence)
  - [Session ID Generation](#session-id-generation)
  - [Storage Structure](#storage-structure)
  - [Atomic Writes](#atomic-writes)
- [Feature 2: Session Recovery](#feature-2-session-recovery)
  - [/sessions Command](#sessions-command)
  - [/resume Command](#resume-command)
  - [/new Command](#new-command)
- [Feature 3: Long-Term Memory Tool](#feature-3-long-term-memory-tool)
  - [Memory Storage Structure](#memory-storage-structure)
  - [memory_save Tool](#memory_save-tool)
  - [session_search Tool](#session_search-tool)
  - [Memory Loads Only at Startup](#memory-loads-only-at-startup)
- [Feature 4: Memory Injection](#feature-4-memory-injection)
- [Feature 5: Token Estimation & Progress Display](#feature-5-token-estimation--progress-display)
  - [chars/4 Estimation Algorithm](#chars4-estimation-algorithm)
  - [Enhanced /status Command](#enhanced-status-command)
- [Feature 6: Context Compression](#feature-6-context-compression)
  - [Auto-Compression Trigger](#auto-compression-trigger)
  - [Compression Algorithm](#compression-algorithm)
  - [Manual /compact Command](#manual-compact-command)
  - [Compression Prompt Design](#compression-prompt-design)
- [New Tools](#new-tools)
  - [grep_search — Text Search](#grep_search--text-search)
  - [fetch_url — Web Page Fetching](#fetch_url--web-page-fetching)
  - [tree — Directory Tree](#tree--directory-tree)
  - [find_file — File Search](#find_file--file-search)
  - [file_delete — File Deletion](#file_delete--file-deletion)
  - [system_info — System Information](#system_info--system-information)
- [Core Integration: The \_post_turn Hook](#core-integration-the-_post_turn-hook)
- [Configuration Changes](#configuration-changes)
- [New CLI Commands](#new-cli-commands)
- [File Change Summary](#file-change-summary)

---

## Change Overview

| Dimension | v0.2 | v0.3 |
|-----------|------|------|
| Sessions | In-memory, lost on exit | Persisted to `.genesis/sessions/` |
| Memory | None | Cross-session long memory in `.genesis/memory/` |
| Context | Unlimited | 64K token cap with auto-compression |
| CLI Commands | 7 | +4 new (`/sessions`, `/resume`, `/new`, `/compact`) |
| /status | Basic info | Shows token usage, session ID |
| New Files | 0 | 3 (`session.py`, `memory.py`, `tokens.py`) |

---

## Feature 1: Session Persistence

### Session ID Generation

A unique session ID is auto-generated when the Agent is created:

```python
# agent/session.py
def generate_session_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_part = secrets.token_hex(3)  # 6 hex chars
    return f"{timestamp}_{random_part}"
    # Example: "20260427_143025_a3f8b2"
```

### Storage Structure

Session files are stored at `.genesis/sessions/{session_id}.json`:

```json
{
  "session_id": "20260427_143025_a3f8b2",
  "created_at": "2026-04-27T14:30:25",
  "updated_at": "2026-04-27T15:12:03",
  "title": "User asked about Python decorators",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

- `title` is auto-generated from the first user message (truncated to 60 chars)
- `created_at` is set on first save and never updated
- `updated_at` is refreshed on every save

### Atomic Writes

Uses a write-then-rename strategy to prevent data corruption on crash:

```python
tmp_path = file_path.with_suffix(".tmp")
with open(tmp_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
os.replace(tmp_path, file_path)  # Atomic on same filesystem
```

---

## Feature 2: Session Recovery

### /sessions Command

Lists all saved sessions sorted by `updated_at` (newest first):

```
  20260427_143025_a3f8b2  2026-04-27T15:12:03  User asked about Python decorators  (12 msgs)
  20260426_091500_f1e2d3  2026-04-26T09:15:00  Help with git rebase                 (8 msgs)
```

### /resume Command

Loads a previous session's messages and continues the conversation:

```
/resume 20260427_143025_a3f8b2
→ Resumed session 20260427_143025_a3f8b2
→ 12 messages loaded
```

### /new Command

Starts a fresh session with a new session_id (long memory is preserved):

```
/new
→ New session started: 20260427_160000_b4c5d6
→ Previous session 20260427_143025_a3f8b2 saved
```

---

## Feature 3: Long-Term Memory Tool

### Memory Storage Structure

Memory file stored at `.genesis/memory/memory.json`:

```json
{
  "entries": [
    {
      "id": "mem_a3f8b2c1",
      "content": "User prefers Python over JavaScript for scripting",
      "created_at": "2026-04-27T14:35:00",
      "updated_at": "2026-04-27T14:35:00"
    }
  ]
}
```

Memory is shared across all sessions — they all read and write the same `memory.json`.

### memory_save Tool

Instead of automatic per-turn extraction, memory is now a tool (`memory_save`) that the LLM calls when it decides saving is warranted:

```python
# tools/memory_save.py
SCHEMA = {
    "name": "memory_save",
    "description": "Save a durable fact to persistent long-term memory...",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact to save..."}
        }
    }
}
```

**System Prompt guidance**:
- Save: user preferences, environment details, tool quirks, stable conventions
- Don't save: task progress, session outcomes, temporary state (use `session_search` instead)
- Prioritize memories that reduce future user corrections

**Design rationale**: Letting the LLM decide when to save is more accurate and token-efficient than forced per-turn extraction.

### session_search Tool

The LLM can search past session transcripts via the `session_search` tool:

```python
# tools/session_search.py
SCHEMA = {
    "name": "session_search",
    "description": "Search past conversation sessions by keyword...",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search keyword..."},
            "limit": {"type": "integer", "description": "Max results. Default: 10"}
        }
    }
}
```

Search logic: iterates all session files in `.genesis/sessions/`, performs keyword matching on message content, returns matching message previews with session IDs.

### Memory Loads Only at Startup

```python
# agent/agent.py
def load_memory(self):
    entries = load_mem()
    self._memory_text = format_memory_for_prompt(entries)
```

`_memory_text` is loaded once at session start and never updated mid-session. This is by design:

1. **Preserve prompt cache hits** — the system prompt prefix stays stable, allowing providers like DeepSeek to reuse cached prefix tokens, reducing cost
2. **Session stability** — mid-session memory changes could cause inconsistent LLM behavior
3. **New memories take effect next session** — memories saved via `memory_save` are loaded on next startup

---

## Feature 4: Memory Injection

When a new session starts, `agent.load_memory()` loads long-term memory and caches it:

```python
# agent/agent.py
def load_memory(self):
    entries = load_mem()
    self._memory_text = format_memory_for_prompt(entries)
```

Injected as a second system message in `_build_messages()`:

```python
result = [{"role": "system", "content": self.system_prompt}]
if self._memory_text:
    result.append({"role": "system", "content": self._memory_text})
# ... conversation messages
```

Formatted memory text:

```
[Long-term Memory - Information about the user from past conversations]
- User prefers Python over JavaScript for scripting
- User prefers concise explanations
- User's timezone is Asia/Shanghai
```

Memory is always enabled — no configuration needed.

---

## Feature 5: Token Estimation & Progress Display

### chars/4 Estimation Algorithm

Uses a simple chars-per-token heuristic, consistent with hermes-agent:

```python
_CHARS_PER_TOKEN = 4

def estimate_tokens(messages: list[dict]) -> int:
    total_chars = 0
    for msg in messages:
        for value in msg.values():
            if isinstance(value, str):
                total_chars += len(value)
            elif isinstance(value, list):  # tool_calls
                # Recursively count nested structures
    return (total_chars + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN
```

Iterates all string fields in messages (content, role, tool_calls, etc.), sums chars, divides by 4.

### Enhanced /status Command

```
/status
  Streaming:    on
  Thinking:     on
  Model:        deepseek-v4-flash
  Session:      20260427_143025_a3f8b2
  Messages:     12
  Token usage:  ~3,200 / 64,000 (5%)
```

---

## Feature 6: Context Compression

### Auto-Compression Trigger

Compression triggers automatically when token usage reaches `max_context_tokens * compression_threshold` (default: 64000 × 95% = 60,800 tokens).

### Compression Algorithm

```
Original messages: [msg0, msg1, msg2, ..., msgN-21, msgN-20, ..., msgN]

Split:
  head   = msgs[:3]        ← Keep first 3 messages
  middle = msgs[3:-20]     ← Compress this section
  tail   = msgs[-20:]      ← Keep last 20 messages

During compress: LLM generates structured summary of middle
After compress:  [head... + summary_msg + tail...]
```

### Manual /compact Command

Users can trigger compression manually:

- Checks if total messages > `head_keep + tail_keep` (default 23)
- Refuses with explanation if not enough messages
- Shows before/after comparison on success:

```
/compact
→ Compressed: 45 -> 24 messages, tokens ~58,000 -> ~12,000
```

### Compression Prompt Design

Uses a structured template to ensure no critical information is lost:

```
## Goal             — User's objective (1-2 sentences)
## Key Actions      — Actions already performed
## Current State    — Where things stand
## Decisions        — Important technical decisions
## Technical Details — Values that must be preserved exactly
## User Preferences — Preferences expressed by the user
```

---

## New Tools

v0.3 adds 6 utility tools to fill gaps in code search, web reading, project structure browsing, file lookup, file deletion, and system information. All tools follow the same `SCHEMA + handler + registry.register()` pattern — drop a file in `tools/` and it auto-registers.

### grep_search — Text Search

Search file contents by keyword or regex pattern recursively, replacing the need for LLM to manually construct `grep` commands.

```python
# tools/grep_search.py
SCHEMA = {
    "name": "grep_search",
    "description": "Search file contents by keyword or regex pattern...",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Search pattern (regex supported)"},
            "path": {"type": "string", "description": "Directory to search, defaults to current"},
            "file_pattern": {"type": "string", "description": "Glob filter (e.g., '*.py')"},
            "case_insensitive": {"type": "boolean", "description": "Ignore case, default false"},
            "max_results": {"type": "integer", "description": "Max matching lines, default 50"},
        },
        "required": ["pattern"],
    },
}
```

**Design notes:**
- Uses Python `re` module for regex matching — no external dependencies
- Automatically skips `.git`, `__pycache__`, `node_modules`, `.genesis`, `.venv` directories
- Detects binary files via 8KB sampling and skips them
- Returns a list of matches with `file`, `line_number`, and `line` fields

### fetch_url — Web Page Fetching

Fetch a web page and extract its text content. Works alongside `web_search`: search first to find URLs, then fetch to read full content.

```python
# tools/fetch_url.py
SCHEMA = {
    "name": "fetch_url",
    "description": "Fetch a web page and return its text content...",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to fetch"},
            "max_length": {"type": "integer", "description": "Max characters to return, default 10000"},
        },
        "required": ["url"],
    },
}
```

**Design notes:**
- Uses `urllib.request` with a reasonable User-Agent header
- Uses BeautifulSoup (requires `beautifulsoup4` dependency) to strip HTML tags and extract readable text
- Automatically removes `<script>`, `<style>`, `<nav>`, `<footer>` elements
- Reads up to 1MB; marks `truncated: true` if exceeded
- Non-HTML content (e.g., plain text) is decoded and returned directly

### tree — Directory Tree

Display project directory structure recursively, similar to the `tree` command — more efficient than multiple `list_dir` calls.

```python
# tools/tree.py
SCHEMA = {
    "name": "tree",
    "description": "Display the directory tree structure recursively...",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Root directory, defaults to current"},
            "max_depth": {"type": "integer", "description": "Max traversal depth, default 3"},
        },
        "required": [],
    },
}
```

**Design notes:**
- Uses `├──` / `└──` / `│  ` characters to draw tree lines
- Automatically skips `.git`, `__pycache__`, `node_modules`, `.genesis`, `.venv` and hidden files
- Entry limit of 500; marks `truncated: true` if exceeded
- Directories are suffixed with `/`

### find_file — File Search

Find files by name glob pattern recursively across the project.

```python
# tools/find_file.py
SCHEMA = {
    "name": "find_file",
    "description": "Find files by name pattern (glob) in a directory tree...",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern (e.g., '*.py', '**/test_*.json')"},
            "path": {"type": "string", "description": "Directory to search, defaults to current"},
        },
        "required": ["pattern"],
    },
}
```

**Design notes:**
- Uses `pathlib.Path.rglob()` for recursive matching
- Automatically skips `.git`, `__pycache__`, `.genesis` directories
- Returns a list of matching file paths and total count

### file_delete — File Deletion

Delete a file at the given path, completing the file operation suite (read, write, edit already exist).

```python
# tools/file_delete.py
SCHEMA = {
    "name": "file_delete",
    "description": "Delete a file at the given path...",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to delete"},
        },
        "required": ["path"],
    },
}
```

**Design notes:**
- Only allows deletion of regular files, not directories
- Checks file existence before deletion; returns clear error on permission denial

### system_info — System Information

Get key information about the current runtime environment to help the LLM make informed decisions.

```python
# tools/system_info.py
SCHEMA = {
    "name": "system_info",
    "description": "Get current system information including OS, CPU, memory, and disk...",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}
```

**Design notes:**
- Returns OS, version, hostname, Python version, CPU count, architecture, working directory
- Best-effort disk space via `shutil.disk_usage`; silently skipped on failure
- Pure standard library — no external dependencies

---

## Core Integration: The \_post_turn Hook

`_post_turn()` is the central integration point, automatically called after every turn:

```python
def _post_turn(self):
    # 1. Compression check — auto-compress if over threshold
    if needs_compression(self.messages, ...):
        self.messages = run_memory_check_then_compress(...)

    # 2. Session save — persist to file
    if settings.auto_save:
        save_session(self.session_id, self.messages, ...)
```

**Design points**:
- Memory saving is handled by the LLM via the `memory_save` tool, not in `_post_turn`
- Compression runs before session save, ensuring the saved state is post-compression
- Each step is wrapped in try-catch — a failure in one step does not block others

---

## Configuration Changes

5 new fields added to `agent/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `max_context_tokens` | 64000 | Maximum context token limit |
| `compression_threshold` | 0.95 | Auto-compression trigger threshold |
| `head_keep` | 3 | Messages to preserve at the head during compression |
| `tail_keep` | 20 | Messages to preserve at the tail during compression |
| `auto_save` | True | Enable/disable automatic session saving |

All can be overridden in the `.env` file. Long-term memory is always enabled — no configuration needed.

---

## New CLI Commands

| Command | Description |
|---------|-------------|
| `/sessions` | List all saved sessions |
| `/resume <id>` | Resume a previous session |
| `/new` | Start a new session (preserves memory) |
| `/compact` | Manually compress conversation history |

`/status` enhanced: Shows Session ID, message count, token usage.

---

## File Change Summary

| File | Action | Description |
|------|--------|-------------|
| `agent/session.py` | **New** | Session persistence and recovery |
| `agent/memory.py` | **New** | Long-term memory storage (file I/O only) |
| `agent/tokens.py` | **New** | Token estimation and context compression |
| `tools/memory_save.py` | **New** | Memory save tool (LLM-driven) |
| `tools/session_search.py` | **New** | Past session search tool |
| `tools/edit_file.py` | **New** | Edit files by string replacement |
| `tools/grep_search.py` | **New** | Search file contents by regex/keyword |
| `tools/fetch_url.py` | **New** | Fetch web pages and extract text |
| `tools/tree.py` | **New** | Display recursive directory tree |
| `tools/find_file.py` | **New** | Find files by glob pattern |
| `tools/file_delete.py` | **New** | Delete a specified file |
| `tools/system_info.py` | **New** | Get system runtime information |
| `agent/agent.py` | Modified | Integrated session_id, memory injection, _post_turn hook |
| `agent/cli.py` | Modified | +4 commands, enhanced /status, updated banner |
| `agent/config.py` | Modified | +5 configuration fields |
| `.gitignore` | Modified | Added `.genesis/` |
| `pyproject.toml` | Modified | Version bump to 0.3.0 |

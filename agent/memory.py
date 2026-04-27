"""Long-term memory storage.

Memory is stored as a JSON file at .genesis/memory/memory.json.
Each entry represents a fact about the user that persists across sessions.
The LLM decides when to save memories via the memory_save tool.
Memory is loaded once at session startup and never updated mid-session
to preserve prompt cache hits.
"""

import json
import logging
import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def get_memory_path() -> Path:
    """Return the .genesis/memory/memory.json path, creating dirs if needed."""
    mem_dir = Path.cwd() / ".genesis" / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    return mem_dir / "memory.json"


def load_memory() -> List[dict]:
    """Load all memory entries from disk.

    Returns an empty list if the file doesn't exist.
    Backs up and ignores corrupted files.
    """
    path = get_memory_path()
    if not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("entries", [])
    except json.JSONDecodeError:
        # Backup corrupted file and start fresh
        bak_path = path.with_suffix(".json.bak")
        logger.warning("Memory file corrupted, backing up to %s", bak_path)
        try:
            os.replace(str(path), str(bak_path))
        except OSError:
            pass
        return []


def save_memory(entries: List[dict]) -> None:
    """Write all memory entries to disk atomically."""
    path = get_memory_path()
    data = {"entries": entries}

    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def add_entry(content: str) -> dict:
    """Add a new memory entry and persist.

    Called by the memory_save tool handler.
    Returns the newly created entry dict.
    """
    entries = load_memory()
    now = datetime.now().isoformat()
    entry_id = f"mem_{secrets.token_hex(4)}"

    entry = {
        "id": entry_id,
        "content": content,
        "created_at": now,
        "updated_at": now,
    }
    entries.append(entry)
    save_memory(entries)
    return entry


def format_memory_for_prompt(entries: List[dict]) -> str:
    """Format memory entries as a text block for system prompt injection."""
    if not entries:
        return ""

    lines = ["[Long-term Memory - Information about the user from past conversations]"]
    for entry in entries:
        lines.append(f"- {entry['content']}")
    return "\n".join(lines)

"""Session persistence and recovery.

Sessions are stored as JSON files in .genesis/sessions/.
Each file contains the full conversation history and metadata.
"""

import json
import logging
import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def generate_session_id() -> str:
    """Generate a unique session ID.

    Format: YYYYMMDD_HHMMSS_xxxxxx (6 hex chars from random).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_part = secrets.token_hex(3)  # 6 hex chars
    return f"{timestamp}_{random_part}"


def get_sessions_dir() -> Path:
    """Return the .genesis/sessions/ directory, creating it if needed."""
    sessions_dir = Path.cwd() / ".genesis" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return sessions_dir


def save_session(
    session_id: str,
    messages: List[dict],
    title: str = "",
) -> None:
    """Save (or overwrite) a session file atomically.

    Preserves created_at if the file already exists.
    """
    sessions_dir = get_sessions_dir()
    file_path = sessions_dir / f"{session_id}.json"
    now = datetime.now().isoformat()

    # Preserve created_at from existing file
    created_at = now
    existing_title = title
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            created_at = existing.get("created_at", now)
            if not existing_title:
                existing_title = existing.get("title", "")
        except (json.JSONDecodeError, OSError):
            pass

    data = {
        "session_id": session_id,
        "created_at": created_at,
        "updated_at": now,
        "title": existing_title,
        "messages": messages,
    }

    # Atomic write: write to temp file, then rename
    tmp_path = file_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, file_path)


def load_session(session_id: str) -> dict:
    """Load a session file by ID.

    Returns:
        Full session dict with messages and metadata.

    Raises:
        FileNotFoundError: If session file doesn't exist.
        json.JSONDecodeError: If session file is corrupted.
    """
    sessions_dir = get_sessions_dir()
    file_path = sessions_dir / f"{session_id}.json"

    if not file_path.exists():
        raise FileNotFoundError(f"Session not found: {session_id}")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_sessions() -> List[dict]:
    """List all sessions sorted by updated_at (newest first).

    Each entry contains: session_id, title, message_count, updated_at.
    """
    sessions_dir = get_sessions_dir()
    sessions = []

    for file_path in sessions_dir.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            sessions.append({
                "session_id": data.get("session_id", file_path.stem),
                "title": data.get("title", ""),
                "message_count": len(data.get("messages", [])),
                "updated_at": data.get("updated_at", ""),
            })
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping corrupted session file %s: %s", file_path.name, e)

    sessions.sort(key=lambda s: s["updated_at"], reverse=True)
    return sessions


def generate_title(first_user_message: str) -> str:
    """Create a short title from the first user message.

    Truncates to 60 chars, stripping newlines.
    """
    if not first_user_message:
        return "Untitled session"
    cleaned = first_user_message.replace("\n", " ").strip()
    if len(cleaned) > 60:
        return cleaned[:57] + "..."
    return cleaned

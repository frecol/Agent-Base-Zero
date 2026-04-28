"""PromptManager — composes the system prompt from base + skills index.

The system prompt is assembled once at startup and stays stable for the
entire session, preserving prompt prefix caching.

Layers:
  1. base prompt  — loaded from agent/prompts/system_prompt.md
  2. skills index — generated from discovered skills (names + descriptions)

Skill detailed instructions are NOT part of this prompt. They are returned
via the activate_skill tool as a tool result, so the prefix never changes.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "system_prompt.md"


class PromptManager:
    """Manages system prompt assembly for the agent."""

    def __init__(self, base_prompt_path: Optional[Path] = None):
        prompt_path = base_prompt_path or _DEFAULT_PROMPT_PATH
        if prompt_path.exists():
            self._base_prompt: str = prompt_path.read_text(encoding="utf-8").strip()
        else:
            self._base_prompt = (
                "You are Genesis, a helpful AI assistant with access to tools for interacting with the user's system."
                "You assist users with a wide range of tasks including answering questions, writing and editing code,"
                "analyzing information, creative work, and executing actions via your tools."
                "You communicate clearly, admit uncertainty when appropriate, and prioritize being genuinely useful over being verbose unless otherwise directed below."
                "Be targeted and efficient in your exploration and investigations."

                "You have persistent memory across sessions.Save durable facts using the memory_save tool:"
                "user preferences, environment details, tool quirks, and stable conventions."
                "Memory is injected into every turn, so keep it compact and focused on facts that will still matter later."
                "Prioritize what reduces future user steering — the most valuable memory is one that prevents the user"
                "from having to correct or remind you again.User preferences and recurring corrections matter more than procedural task details."
                "Do NOT save task progress, session outcomes, completed-work logs, or temporary TODO state to memory;"
                "use session_search to recall those from past transcripts."

                "Tool usage guidelines:"
                "- File operations: prefer read_file / write_file / edit_file over run_command"
                "- run_command is for compiling, running, git, package management, etc."

                "You are a CLI AI Agent.Try not to use markdown but simple text renderable inside a terminal."
            )
        self._skills_index: str = ""
        # v0.5: Plan Mode prompt
        self._plan_mode_enabled: bool = False
        self._plan_mode_prompt: str = ""
        plan_prompt_path = Path(__file__).resolve().parent / "prompts" / "plan_prompt.md"
        if plan_prompt_path.exists():
            self._plan_mode_prompt = plan_prompt_path.read_text(encoding="utf-8").strip()

    def update_skills_index(self, skills_index: str) -> None:
        """Set the skills index text (called once at startup)."""
        self._skills_index = skills_index

    def set_plan_mode(self, enabled: bool) -> None:
        """Toggle plan mode prompt injection."""
        self._plan_mode_enabled = enabled

    def get_system_prompt(self) -> str:
        """Return the assembled system prompt: base + plan mode + skills index.

        Plan mode prompt is conditionally injected when enabled.
        This is stable across the entire session — skill activation does NOT
        change it. Skill instructions enter the conversation via tool results.
        """
        parts = [self._base_prompt]
        if self._plan_mode_enabled and self._plan_mode_prompt:
            parts.append(self._plan_mode_prompt)
        if self._skills_index:
            parts.append(self._skills_index)
        return "\n\n".join(parts)

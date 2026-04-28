"""Central skill registry.

Each skill is a subdirectory of skills/ containing a SKILL.md file with
YAML frontmatter (--- delimited) for metadata and Markdown content for
instructions. Compatible with the standard skill format used across the
ecosystem (e.g. Claude Code skills).

Skills are auto-discovered at startup via discover_skills(). The registry
also registers activate_skill / deactivate_skill as tools in the ToolRegistry
so the LLM can autonomously activate skills.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SkillInfo
# ---------------------------------------------------------------------------

@dataclass
class SkillInfo:
    """Metadata and content for a single skill."""

    name: str
    description: str
    prompt_text: str                       # full Markdown content after frontmatter
    user_invocable: bool = True
    skill_dir: Optional[Path] = None


# ---------------------------------------------------------------------------
# SkillRegistry
# ---------------------------------------------------------------------------

class SkillRegistry:
    """Registry that collects skills discovered from the skills/ directory."""

    def __init__(self) -> None:
        self._skills: Dict[str, SkillInfo] = {}
        self._active_skill: Optional[str] = None
        self._on_activate: Optional[Callable] = None

    # -- registration -------------------------------------------------------

    def register(self, skill_info: SkillInfo) -> None:
        if skill_info.name in self._skills:
            logger.warning("Skill '%s' already registered — overwriting.", skill_info.name)
        self._skills[skill_info.name] = skill_info

    # -- activation ---------------------------------------------------------

    def activate(self, name: str) -> str:
        """Activate a skill. Returns JSON with the skill's instructions."""
        skill = self._skills.get(name)
        if not skill:
            return json.dumps({"error": f"Unknown skill: {name}"})
        self._active_skill = name
        logger.info("Skill activated: %s", name)
        return json.dumps(
            {"success": True, "skill": name, "instructions": skill.prompt_text},
            ensure_ascii=False,
        )

    def deactivate(self) -> str:
        """Deactivate the current skill."""
        prev = self._active_skill
        self._active_skill = None
        logger.info("Skill deactivated: %s", prev)
        return json.dumps(
            {"success": True, "message": f"Skill '{prev}' deactivated. Back to base mode."}
        )

    # -- query --------------------------------------------------------------

    def get_active(self) -> Optional[SkillInfo]:
        if self._active_skill:
            return self._skills.get(self._active_skill)
        return None

    def get_all_names(self) -> List[str]:
        return sorted(self._skills.keys())

    def get_skill(self, name: str) -> Optional[SkillInfo]:
        return self._skills.get(name)

    # -- callback -----------------------------------------------------------

    def set_on_activate(self, callback: Callable) -> None:
        """Register a callback invoked when skill activation state changes."""
        self._on_activate = callback


# Global singleton.
skill_registry = SkillRegistry()


# ---------------------------------------------------------------------------
# Skill management tools (registered into ToolRegistry)
# ---------------------------------------------------------------------------

_ACTIVATE_SKILL_SCHEMA = {
    "name": "activate_skill",
    "description": (
        "Activate a skill by name. Returns the skill's detailed instructions. "
        "Use this when a task matches one of the available skills."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "skill_name": {
                "type": "string",
                "description": "Name of the skill to activate.",
            },
        },
        "required": ["skill_name"],
    },
}

_DEACTIVATE_SKILL_SCHEMA = {
    "name": "deactivate_skill",
    "description": (
        "Deactivate the currently active skill and return to base mode. "
        "Call this when the task no longer requires a specialized skill."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


def _activate_skill_handler(args: dict) -> str:
    name = args.get("skill_name", "").strip()
    if not name:
        return json.dumps({"error": "skill_name is required"})
    result = skill_registry.activate(name)
    if skill_registry._on_activate:
        skill = skill_registry.get_skill(name)
        if skill:
            skill_registry._on_activate(name, skill.prompt_text)
    return result


def _deactivate_skill_handler(args: dict) -> str:
    result = skill_registry.deactivate()
    if skill_registry._on_activate:
        skill_registry._on_activate("", "")
    return result


def _register_skill_tools() -> None:
    """Register activate_skill / deactivate_skill into the ToolRegistry."""
    from tools.registry import registry

    registry.register("activate_skill", _ACTIVATE_SKILL_SCHEMA, _activate_skill_handler)
    registry.register("deactivate_skill", _DEACTIVATE_SKILL_SCHEMA, _deactivate_skill_handler)


# ---------------------------------------------------------------------------
# SKILL.md parsing
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_skill_md(md_path: Path) -> SkillInfo:
    """Parse a SKILL.md file with YAML frontmatter.

    Format:
        ---
        name: skill-name
        description: "Description text"
        user_invocable: true
        ---
        # Skill instructions in Markdown...
    """
    content = md_path.read_text(encoding="utf-8")

    # Extract frontmatter
    match = _FRONTMATTER_RE.match(content)
    if not match:
        raise ValueError(f"No YAML frontmatter found in {md_path}")

    frontmatter_text = match.group(1)
    body = content[match.end():].strip()

    # Parse simple YAML key-value pairs (no nested structures needed)
    meta: dict = {}
    for line in frontmatter_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        # Strip quotes
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        elif value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        # Parse booleans
        if value.lower() in ("true", "yes"):
            value = True
        elif value.lower() in ("false", "no"):
            value = False
        meta[key] = value

    name = meta.get("name", md_path.parent.name)
    description = meta.get("description", "")
    user_invocable = meta.get("user_invocable", True)

    return SkillInfo(
        name=name,
        description=description,
        prompt_text=body,
        user_invocable=bool(user_invocable),
        skill_dir=md_path.parent,
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_skills(skills_dir: Optional[Path] = None) -> List[str]:
    """Scan skills/ directory for skill folders containing SKILL.md."""
    skills_path = skills_dir or Path(__file__).resolve().parent
    discovered: List[str] = []

    for item in sorted(skills_path.iterdir()):
        if not item.is_dir():
            continue
        if item.name.startswith("_") or item.name.startswith("."):
            continue

        skill_md = item / "SKILL.md"
        if not skill_md.exists():
            continue

        try:
            info = _parse_skill_md(skill_md)
            skill_registry.register(info)
            discovered.append(info.name)
            logger.info("Discovered skill: %s", info.name)
        except Exception as e:
            logger.warning("Could not load skill '%s': %s", item.name, e)

    # Register skill management tools
    _register_skill_tools()

    return discovered


def build_skills_index() -> str:
    """Build the skills index text for the system prompt."""
    names = skill_registry.get_all_names()
    if not names:
        return ""

    lines = [
        "## Available Skills",
        "When a task matches a skill below, call activate_skill(name) to load its detailed instructions.\n",
    ]
    for name in names:
        skill = skill_registry.get_skill(name)
        if skill:
            lines.append(f"- {name}: {skill.description}")
    return "\n".join(lines)

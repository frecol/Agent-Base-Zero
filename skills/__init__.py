"""Auto-discover and register all skills on package import."""

from skills.registry import skill_registry, discover_skills  # noqa: F401

discover_skills()

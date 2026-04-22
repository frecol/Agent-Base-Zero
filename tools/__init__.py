"""Auto-discover and import all tool modules on package import."""

from tools.registry import registry, discover_tools

# Import all tool modules — each module calls registry.register() at load time.
discover_tools()

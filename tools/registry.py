"""Central tool registry.

Each tool file calls ``registry.register()`` at module level to declare its
schema and handler.  The agent queries the registry for tool definitions
and dispatches tool calls by name.

Design follows Hermes-Agent's registry pattern, simplified for a single-provider setup.
"""

import importlib
import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry that collects tool schemas + handlers from tool files."""

    def __init__(self):
        self._tools: Dict[str, dict] = {}
        # Each entry: {"schema": dict, "handler": Callable}

    def register(self, name: str, schema: dict, handler: Callable) -> None:
        """Register a tool. Called at module-import time by each tool file.

        Args:
            name: Unique tool name (e.g. "read_file").
            schema: OpenAI function-calling schema dict with keys:
                    "name", "description", "parameters".
            handler: Callable(args: dict) -> str.  Receives parsed JSON
                     arguments, returns the tool result as a string.
        """
        if name in self._tools:
            logger.warning("Tool '%s' already registered — overwriting.", name)
        self._tools[name] = {"schema": schema, "handler": handler}

    def get_definitions(self, names: Optional[List[str]] = None) -> List[dict]:
        """Return OpenAI-format tool definitions.

        Args:
            names: If provided, only return definitions for these tools.
                   Otherwise, return all registered tools.
        """
        result = []
        targets = names if names is not None else list(self._tools.keys())
        for name in sorted(targets):
            entry = self._tools.get(name)
            if entry:
                result.append({
                    "type": "function",
                    "function": entry["schema"],
                })
        return result

    def dispatch(self, name: str, args: dict) -> str:
        """Execute a tool handler by name.

        Returns:
            Tool result as a string (usually JSON).
        """
        entry = self._tools.get(name)
        if not entry:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            return entry["handler"](args)
        except Exception as e:
            logger.exception("Tool %s execution failed", name)
            return json.dumps({"error": f"Tool execution failed: {e}"})

    def get_all_names(self) -> List[str]:
        """Return sorted list of all registered tool names."""
        return sorted(self._tools.keys())


# Global singleton — import this from everywhere.
registry = ToolRegistry()


def discover_tools(tools_dir: Optional[Path] = None) -> List[str]:
    """Import all tool modules in the tools/ directory.

    Each module calls registry.register() at import time, so importing
    is all that's needed to populate the registry.
    """
    tools_path = tools_dir or Path(__file__).resolve().parent
    module_names = [
        f"tools.{p.stem}"
        for p in sorted(tools_path.glob("*.py"))
        if p.name not in {"__init__.py", "registry.py"}
    ]

    imported: List[str] = []
    for mod_name in module_names:
        try:
            importlib.import_module(mod_name)
            imported.append(mod_name)
        except Exception as e:
            logger.warning("Could not import tool module %s: %s", mod_name, e)
    return imported

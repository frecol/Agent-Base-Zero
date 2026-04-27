"""Tool: tree — Display directory tree structure."""

import json
from pathlib import Path

from tools.registry import registry

SCHEMA = {
    "name": "tree",
    "description": (
        "Display the directory tree structure recursively. "
        "Returns a text-based tree similar to the 'tree' command.\n\n"
        "Guidelines:\n"
        "- Use this to quickly understand project layout.\n"
        "- Adjust 'max_depth' to control how deep to traverse (default 3)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Root directory path. Defaults to current directory.",
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum depth to traverse (default 3).",
            },
        },
        "required": [],
    },
}

_DEFAULT_SKIP = {".git", "__pycache__", ".genesis", "node_modules", ".venv", "venv", "__pypackages__"}
_MAX_ENTRIES = 500


def handler(args: dict) -> str:
    path = args.get("path", ".")
    max_depth = args.get("max_depth", 3)

    try:
        root = Path(path).resolve()
        if not root.is_dir():
            return json.dumps({"success": False, "error": f"Not a directory: {path}"})

        lines = [root.name + "/"]
        entry_count = [0]

        def _build_tree(directory: Path, prefix: str, depth: int):
            if depth > max_depth or entry_count[0] > _MAX_ENTRIES:
                return

            try:
                items = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except PermissionError:
                return

            for i, item in enumerate(items):
                if entry_count[0] > _MAX_ENTRIES:
                    lines.append(f"{prefix}... (truncated)")
                    return

                # Skip hidden and ignored directories
                if item.is_dir() and item.name in _DEFAULT_SKIP:
                    continue
                if item.name.startswith(".") and item.name not in {".env", ".gitignore", ".python-version"}:
                    continue

                entry_count[0] += 1
                is_last = i == len(items) - 1
                connector = "└── " if is_last else "├── "
                suffix = "/" if item.is_dir() else ""

                lines.append(f"{prefix}{connector}{item.name}{suffix}")

                if item.is_dir():
                    extension = "    " if is_last else "│   "
                    _build_tree(item, prefix + extension, depth + 1)

        _build_tree(root, "", 0)
        tree_text = "\n".join(lines)

        return json.dumps({
            "success": True,
            "tree": tree_text,
            "entries": entry_count[0],
            "truncated": entry_count[0] > _MAX_ENTRIES,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


registry.register("tree", SCHEMA, handler)

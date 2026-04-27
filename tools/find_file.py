"""Tool: find_file — Find files by name pattern."""

import json
from pathlib import Path

from tools.registry import registry

SCHEMA = {
    "name": "find_file",
    "description": (
        "Find files by name pattern (glob) in a directory tree. "
        "Returns a list of matching file paths.\n\n"
        "Guidelines:\n"
        "- Use glob patterns like '*.py', '**/*.json', 'test_*'.\n"
        "- Use this to locate files when you know the name or extension."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match file names (e.g., '*.py', '**/test_*.json').",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in. Defaults to current directory.",
            },
        },
        "required": ["pattern"],
    },
}

_SKIP_DIRS = {".git", "__pycache__", ".genesis", "node_modules", ".venv", "venv"}


def handler(args: dict) -> str:
    pattern = args.get("pattern", "")
    path = args.get("path", ".")

    if not pattern:
        return json.dumps({"success": False, "error": "No pattern provided"})

    try:
        root = Path(path).resolve()
        if not root.is_dir():
            return json.dumps({"success": False, "error": f"Not a directory: {path}"})

        matches = []
        for p in root.rglob(pattern):
            if not p.is_file():
                continue
            # Skip ignored directories
            if any(part in _SKIP_DIRS for part in p.relative_to(root).parts):
                continue
            matches.append(str(p))

        return json.dumps({
            "success": True,
            "files": matches,
            "count": len(matches),
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


registry.register("find_file", SCHEMA, handler)

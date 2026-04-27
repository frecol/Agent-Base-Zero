"""Tool: grep_search — Search file contents by keyword or regex."""

import json
import re
from pathlib import Path

from tools.registry import registry

SCHEMA = {
    "name": "grep_search",
    "description": (
        "Search file contents by keyword or regex pattern. "
        "Returns matching lines with file path and line number.\n\n"
        "Guidelines:\n"
        "- Use this to find where functions, classes, or variables are defined or used.\n"
        "- Supports full regex syntax (e.g., 'class \\w+' to find class definitions).\n"
        "- Use 'file_pattern' to narrow search to specific file types (e.g., '*.py')."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "The search pattern (regex supported).",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in. Defaults to current directory.",
            },
            "file_pattern": {
                "type": "string",
                "description": "Glob pattern to filter files (e.g., '*.py', '*.{js,ts}'). Defaults to all files.",
            },
            "case_insensitive": {
                "type": "boolean",
                "description": "Whether to ignore case. Defaults to false.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of matching lines to return (default 50).",
            },
        },
        "required": ["pattern"],
    },
}

_SKIP_DIRS = {".git", "__pycache__", ".genesis", "node_modules", ".venv", "venv"}


def _is_binary(file_path: Path) -> bool:
    """Check if a file appears to be binary."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(8192)
        return b"\x00" in chunk
    except OSError:
        return True


def handler(args: dict) -> str:
    pattern = args.get("pattern", "")
    path = args.get("path", ".")
    file_pattern = args.get("file_pattern", "*")
    case_insensitive = args.get("case_insensitive", False)
    max_results = args.get("max_results", 50)

    if not pattern:
        return json.dumps({"success": False, "error": "No pattern provided"})

    try:
        flags = re.IGNORECASE if case_insensitive else 0
        regex = re.compile(pattern, flags)
    except re.error as e:
        return json.dumps({"success": False, "error": f"Invalid regex: {e}"})

    try:
        root = Path(path).resolve()
        if not root.is_dir():
            return json.dumps({"success": False, "error": f"Not a directory: {path}"})

        matches = []
        for file_path in root.rglob(file_pattern):
            # Skip directories and ignored dirs
            if not file_path.is_file():
                continue
            if any(part in _SKIP_DIRS for part in file_path.relative_to(root).parts):
                continue
            if _is_binary(file_path):
                continue

            try:
                lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                for i, line in enumerate(lines, 1):
                    if regex.search(line):
                        matches.append({
                            "file": str(file_path),
                            "line_number": i,
                            "line": line.strip(),
                        })
                        if len(matches) >= max_results:
                            return json.dumps({
                                "success": True,
                                "matches": matches,
                                "truncated": True,
                            }, ensure_ascii=False)
            except OSError:
                continue

        return json.dumps({
            "success": True,
            "matches": matches,
            "truncated": False,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


registry.register("grep_search", SCHEMA, handler)

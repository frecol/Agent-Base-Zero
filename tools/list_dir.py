"""Tool: list_dir — List files and directories."""

import json
from pathlib import Path

from tools.registry import registry

SCHEMA = {
    "name": "list_dir",
    "description": "List files and directories at the given path. Returns names, types, and sizes.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list. Defaults to current directory.",
            },
        },
        "required": [],
    },
}


def handler(args: dict) -> str:
    path = args.get("path", ".")
    try:
        dir_path = Path(path)
        if not dir_path.is_dir():
            return json.dumps({"success": False, "error": f"Not a directory: {path}"})

        entries = []
        for item in sorted(dir_path.iterdir()):
            entries.append({
                "name": item.name,
                "type": "dir" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else None,
            })
        return json.dumps({"success": True, "entries": entries})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


registry.register("list_dir", SCHEMA, handler)

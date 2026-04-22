"""Tool: write_file — Write content to a file."""

import json
from pathlib import Path

from tools.registry import registry

SCHEMA = {
    "name": "write_file",
    "description": "Write content to a file. Creates parent directories if they don't exist.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write.",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file.",
            },
        },
        "required": ["path", "content"],
    },
}


def handler(args: dict) -> str:
    path = args.get("path", "")
    content = args.get("content", "")
    try:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return json.dumps({"success": True, "message": f"Wrote {len(content)} chars to {path}"})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


registry.register("write_file", SCHEMA, handler)

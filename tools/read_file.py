"""Tool: read_file — Read the contents of a file."""

import json
from pathlib import Path

from tools.registry import registry

SCHEMA = {
    "name": "read_file",
    "description": "Read the contents of a file at the given path. Returns the file content as a string.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read.",
            },
        },
        "required": ["path"],
    },
}


def handler(args: dict) -> str:
    path = args.get("path", "")
    try:
        content = Path(path).read_text(encoding="utf-8")
        return json.dumps({"success": True, "content": content})
    except FileNotFoundError:
        return json.dumps({"success": False, "error": f"File not found: {path}"})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


registry.register("read_file", SCHEMA, handler)

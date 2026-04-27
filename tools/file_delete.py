"""Tool: file_delete — Delete a file."""

import json
from pathlib import Path

from tools.registry import registry

SCHEMA = {
    "name": "file_delete",
    "description": (
        "Delete a file at the given path. Only works on regular files, not directories.\n\n"
        "Guidelines:\n"
        "- Use with caution — deletion is permanent.\n"
        "- Cannot delete directories; use run_command for that if needed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to delete.",
            },
        },
        "required": ["path"],
    },
}


def handler(args: dict) -> str:
    path = args.get("path", "")

    if not path:
        return json.dumps({"success": False, "error": "No path provided"})

    try:
        file_path = Path(path).resolve()

        if not file_path.exists():
            return json.dumps({"success": False, "error": f"File not found: {path}"})

        if file_path.is_dir():
            return json.dumps({"success": False, "error": f"Path is a directory, not a file: {path}"})

        file_path.unlink()
        return json.dumps({"success": True, "deleted": str(file_path)})
    except PermissionError:
        return json.dumps({"success": False, "error": f"Permission denied: {path}"})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


registry.register("file_delete", SCHEMA, handler)

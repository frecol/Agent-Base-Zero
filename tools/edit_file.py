"""Tool: edit_file — Edit file content by replacing a unique string."""

import json
from pathlib import Path

from tools.registry import registry

SCHEMA = {
    "name": "edit_file",
    "description": "Edit a file by replacing an existing string with a new string. "
                   "The old_string must be exactly unique in the file (only one occurrence). "
                   "Use this for targeted edits instead of rewriting the entire file with write_file.",
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to edit.",
            },
            "old_string": {
                "type": "string",
                "description": "The existing string to replace. Must be unique in the file.",
            },
            "new_string": {
                "type": "string",
                "description": "The new string to replace it with.",
            },
        },
        "required": ["file_path", "old_string", "new_string"],
    },
}


def handler(args: dict) -> str:
    file_path = args.get("file_path", "")
    old_string = args.get("old_string", "")
    new_string = args.get("new_string", "")

    if not old_string:
        return json.dumps({"success": False, "error": "old_string cannot be empty"})

    try:
        path = Path(file_path)
        if not path.exists():
            return json.dumps({"success": False, "error": f"File not found: {file_path}"})

        content = path.read_text(encoding="utf-8")
        count = content.count(old_string)

        if count == 0:
            return json.dumps({
                "success": False,
                "error": f"old_string not found in file: {file_path}",
            })
        elif count > 1:
            return json.dumps({
                "success": False,
                "error": f"old_string is not unique in file (found {count} occurrences): {file_path}",
            })

        # Exactly one occurrence — safe to replace
        new_content = content.replace(old_string, new_string, 1)
        path.write_text(new_content, encoding="utf-8")

        return json.dumps({
            "success": True,
            "message": f"Replaced 1 occurrence in {file_path}",
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


registry.register("edit_file", SCHEMA, handler)

"""Tool: memory_save — Save a fact to long-term memory."""

import json

from tools.registry import registry

SCHEMA = {
    "name": "memory_save",
    "description": (
        "Save a durable fact to persistent long-term memory. "
        "Memory is loaded into every future session automatically. "
        "Use for: user preferences, environment details, tool quirks, stable conventions, "
        "recurring corrections, or anything that reduces future user steering. "
        "Do NOT use for task progress, session outcomes, or temporary state."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": (
                    "The fact to save. Be concise and specific. "
                    "Example: 'User prefers Python with type hints and 4-space indentation'"
                ),
            },
        },
        "required": ["content"],
    },
}


def handler(args: dict) -> str:
    content = args.get("content", "").strip()
    if not content:
        return json.dumps({"success": False, "error": "Empty content"})

    from agent.memory import add_entry

    entry = add_entry(content)
    return json.dumps({"success": True, "id": entry["id"], "content": entry["content"]})


registry.register("memory_save", SCHEMA, handler)

"""Tool: session_search — Search past session transcripts."""

import json

from tools.registry import registry

SCHEMA = {
    "name": "session_search",
    "description": (
        "Search past conversation sessions by keyword. "
        "Returns matching message snippets with session metadata. "
        "Use when you need to recall what happened in previous conversations, "
        "find past solutions, or check historical context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search keyword or phrase to find in past sessions.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of matching messages to return. Default: 10.",
            },
        },
        "required": ["query"],
    },
}


def handler(args: dict) -> str:
    query = args.get("query", "").strip().lower()
    limit = args.get("limit", 10)

    if not query:
        return json.dumps({"success": False, "error": "Empty query"})

    from agent.session import list_sessions, load_session

    sessions = list_sessions()
    results = []

    for s in sessions:
        if len(results) >= limit:
            break
        try:
            data = load_session(s["session_id"])
            for msg in data.get("messages", []):
                if len(results) >= limit:
                    break
                content = msg.get("content", "")
                if query in content.lower():
                    preview = content[:200] + "..." if len(content) > 200 else content
                    results.append({
                        "session_id": s["session_id"],
                        "role": msg.get("role", ""),
                        "preview": preview,
                        "updated_at": s["updated_at"],
                    })
        except Exception:
            continue

    if not results:
        return json.dumps({
            "success": True,
            "query": query,
            "count": 0,
            "message": f"No results found for '{query}'",
        })

    return json.dumps({
        "success": True,
        "query": query,
        "count": len(results),
        "results": results,
    }, ensure_ascii=False)


registry.register("session_search", SCHEMA, handler)

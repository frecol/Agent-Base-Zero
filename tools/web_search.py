"""Tool: web_search — Search the web using DuckDuckGo."""

import json

from ddgs import DDGS

from tools.registry import registry

SCHEMA = {
    "name": "web_search",
    "description": (
        "Search the web using DuckDuckGo. "
        "Returns a list of results with title, URL, and a short snippet."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The search query. Must be a concrete, specific query — "
                    "resolve relative terms before searching. "
                    "For example, convert 'today' or 'this week' to the actual date "
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default 5).",
            },
        },
        "required": ["query"],
    },
}


def handler(args: dict) -> str:
    query = args.get("query", "")
    max_results = args.get("max_results", 5)
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "href": r.get("href", ""),
                    "body": r.get("body", ""),
                })
        return json.dumps({"success": True, "results": results}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


registry.register("web_search", SCHEMA, handler)

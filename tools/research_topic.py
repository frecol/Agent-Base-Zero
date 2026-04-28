"""Composite tool: research_topic — search and fetch in one call.

Demonstrates tool composition: chains web_search + fetch_url via
registry.dispatch() to provide a higher-level research capability.
"""

import json

from tools.registry import registry

SCHEMA = {
    "name": "research_topic",
    "description": (
        "Research a topic by searching the web and fetching the top results. "
        "Returns a consolidated summary with source URLs. "
        "Combines web_search + fetch_url into a single call."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The research query to investigate.",
            },
            "max_sources": {
                "type": "integer",
                "description": "Maximum number of sources to fetch in detail (default 3).",
            },
        },
        "required": ["query"],
    },
}


def handler(args: dict) -> str:
    query = args.get("query", "")
    max_sources = args.get("max_sources", 3)

    # Step 1: web search
    search_raw = registry.dispatch("web_search", {"query": query})
    search_result = json.loads(search_raw)
    if not search_result.get("success"):
        return search_raw

    # Step 2: fetch top results
    results = search_result.get("results", [])[:max_sources]
    fetched = []
    for r in results:
        url = r.get("href", "")
        if not url:
            continue
        fetch_raw = registry.dispatch("fetch_url", {"url": url})
        fetch_result = json.loads(fetch_raw)
        if fetch_result.get("success"):
            fetched.append({
                "title": r.get("title", ""),
                "url": url,
                "snippet": r.get("body", ""),
                "content": fetch_result.get("content", "")[:3000],
            })

    return json.dumps(
        {
            "success": True,
            "query": query,
            "sources": fetched,
            "source_count": len(fetched),
        },
        ensure_ascii=False,
    )


registry.register("research_topic", SCHEMA, handler)

"""Tool: fetch_url — Fetch and read web page content."""

import json
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from tools.registry import registry

SCHEMA = {
    "name": "fetch_url",
    "description": (
        "Fetch a web page and return its text content. "
        "Use this after web_search to read the full content of a found URL.\n\n"
        "Guidelines:\n"
        "- Use web_search first to find relevant URLs, then fetch_url to read them.\n"
        "- The returned text is stripped of HTML tags for readability."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch.",
            },
            "max_length": {
                "type": "integer",
                "description": "Maximum characters to return (default 10000).",
            },
        },
        "required": ["url"],
    },
}


def handler(args: dict) -> str:
    url = args.get("url", "")
    max_length = args.get("max_length", 10000)

    if not url:
        return json.dumps({"success": False, "error": "No URL provided"})

    if not url.startswith(("http://", "https://")):
        return json.dumps({"success": False, "error": "URL must start with http:// or https://"})

    try:
        req = Request(url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; Agent-Base-Zero/0.3)",
            "Accept": "text/html,application/xhtml+xml,text/plain",
        })
        with urlopen(req, timeout=30) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read(1024 * 1024)  # Read at most 1MB

        # Try to extract text from HTML
        if "text/html" in content_type:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(raw, "html.parser")
                # Remove script and style elements
                for tag in soup(["script", "style", "nav", "footer"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
            except ImportError:
                # Fallback: crude HTML tag stripping
                import re
                text = re.sub(r"<[^>]+>", "", raw.decode("utf-8", errors="ignore"))
        else:
            text = raw.decode("utf-8", errors="ignore")

        # Truncate if needed
        truncated = len(text) > max_length
        text = text[:max_length]

        return json.dumps({
            "success": True,
            "content": text,
            "truncated": truncated,
            "url": url,
        }, ensure_ascii=False)

    except HTTPError as e:
        return json.dumps({
            "success": False,
            "error": f"HTTP {e.code}: {e.reason}",
        })
    except URLError as e:
        return json.dumps({"success": False, "error": f"URL error: {e.reason}"})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


registry.register("fetch_url", SCHEMA, handler)

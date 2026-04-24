"""Tool: current_time — Get the current date and time."""

import json
from datetime import datetime
from zoneinfo import ZoneInfo

from agent.config import settings
from tools.registry import registry

SCHEMA = {
    "name": "current_time",
    "description": (
        "Get the current date and time. "
        "Use this tool whenever you need to know the current date or time, "
        "for example to resolve relative terms like 'today' or 'now'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": (
                    "IANA timezone name, e.g. 'Asia/Shanghai', 'America/New_York'. "
                    "Defaults to the configured timezone."
                ),
            },
        },
        "required": [],
    },
}


def handler(args: dict) -> str:
    tz_name = args.get("timezone") or settings.default_timezone
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        return json.dumps({"success": False, "error": f"Unknown timezone: {tz_name}"})
    now = datetime.now(tz)
    return json.dumps({
        "success": True,
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": tz_name,
        "weekday": now.strftime("%A"),
    }, ensure_ascii=False)


registry.register("current_time", SCHEMA, handler)

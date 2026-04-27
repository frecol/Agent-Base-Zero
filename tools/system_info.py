"""Tool: system_info — Get system information."""

import json
import os
import platform
import shutil

from tools.registry import registry

SCHEMA = {
    "name": "system_info",
    "description": (
        "Get current system information including OS, CPU, memory, and disk.\n\n"
        "Guidelines:\n"
        "- Use this to understand the runtime environment and its constraints.\n"
        "- No parameters needed."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


def handler(args: dict) -> str:
    try:
        info = {
            "os": platform.system(),
            "os_release": platform.release(),
            "os_version": platform.version(),
            "hostname": platform.node(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "architecture": platform.machine(),
            "cwd": os.getcwd(),
        }

        # Disk info — best effort, skip silently on failure
        try:
            disk_path = os.getcwd()
            disk = shutil.disk_usage(disk_path)
            info["disk_total_gb"] = round(disk.total / (1024 ** 3), 2)
            info["disk_free_gb"] = round(disk.free / (1024 ** 3), 2)
        except Exception:
            pass

        return json.dumps({"success": True, **info}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


registry.register("system_info", SCHEMA, handler)

"""Tool: run_command — Execute a shell command."""

import json
import subprocess

from tools.registry import registry

SCHEMA = {
    "name": "run_command",
    "description": "Execute a shell command and return its stdout, stderr, and exit code.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 30).",
            },
        },
        "required": ["command"],
    },
}


def handler(args: dict) -> str:
    command = args.get("command", "")
    timeout = args.get("timeout", 30)
    if not command:
        return json.dumps({"success": False, "error": "No command provided"})

    try:
        # NOTE: shell=True allows command injection. Safe for local dev, but a real
        # production agent should use subprocess without shell or add an allowlist.
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return json.dumps({
            "success": result.returncode == 0,
            "exit_code": result.returncode,
            "stdout": result.stdout[:10000],
            "stderr": result.stderr[:5000],
        })
    except subprocess.TimeoutExpired:
        return json.dumps({"success": False, "error": f"Command timed out after {timeout}s"})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


registry.register("run_command", SCHEMA, handler)

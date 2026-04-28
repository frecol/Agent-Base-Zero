"""Agent-Base-Zero v0.5 — Entry point.

A minimal AI agent powered by DeepSeek, with tool-calling support.
"""

import tools  # auto-discovers all tool modules
import skills  # auto-discovers all skill folders

from agent.cli import run_cli

if __name__ == "__main__":
    run_cli()

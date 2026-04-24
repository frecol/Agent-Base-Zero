"""Interactive CLI for the agent."""

import sys
from enum import Enum, auto

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.text import Text

from agent.agent import Agent
from agent.config import settings

console = Console()

# ANSI escape codes for streaming (bypasses Rich per-token overhead)
_ANSI_DIM = "\033[2m"
_ANSI_RESET = "\033[0m"

# Braille spinner animation frames
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class _Phase(Enum):
    IDLE = auto()
    THINKING = auto()
    CONTENT = auto()

HELP_TEXT = """
Available commands:
  /help    — Show this help message
  /clear   — Clear conversation history
  /exit    — Exit the program
  /tools   — List available tools
  /stream  — Toggle streaming mode on/off
  /think   — Toggle thinking mode on/off
  /status  — Show current settings
  <text>   — Send a message to the agent
"""


def _print_banner() -> None:
    title = Text()
    title.append("  Agent-Base-Zero v0.2\n", style="bold cyan")
    title.append("  DeepSeek-powered AI Agent", style="dim")
    console.print(Panel(title, border_style="cyan", padding=(1, 2)))
    console.print(
        "Type [bold]/help[/bold] for commands, [bold]/exit[/bold] to quit.",
        style="dim",
    )
    console.print()


def _format_tool_args(args: dict, max_val: int = 60) -> str:
    """Format tool arguments for display, truncating long values."""
    parts = []
    for k, v in args.items():
        s = repr(v)
        if len(s) > max_val:
            s = s[:max_val] + "...'"
        parts.append(f"{k}={s}")
    return ", ".join(parts)


def _print_tool_call(name: str, args: dict) -> None:
    console.print(Rule(style="dim"))
    formatted = _format_tool_args(args)
    console.print(
        Panel(
            Text(formatted),
            title=f"[bold yellow]{name}[/bold yellow]",
            border_style="yellow",
            padding=(0, 1),
            expand=False,
        )
    )


def _print_tool_result(name: str, display: str) -> None:
    console.print(f"  [dim]Result:[/dim] {display}")
    console.print(Rule(style="dim"))


def _tail_text(text: str, max_lines: int = 20) -> str:
    """Truncate text to the last *max_lines*, adding an overflow indicator."""
    lines = text.split('\n')
    if len(lines) <= max_lines:
        return text
    hidden = len(lines) - max_lines
    return f"... ({hidden} lines hidden) ...\n" + '\n'.join(lines[-max_lines:])


def _stream_response(
    agent: Agent,
    user_input: str,
    thinking_enabled: bool,
) -> None:
    """Handle a streaming turn with structured visual output."""
    phase = _Phase.IDLE
    thinking_buffer: list[str] = []
    spinner_idx = 0
    spinner_active = False

    def _clear_spinner() -> None:
        """Clear the spinner indicator line."""
        nonlocal spinner_active
        if spinner_active:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
            spinner_active = False

    def _flush_thinking_as_thinking() -> None:
        """Flush buffered thinking text with 'Thinking' label."""
        nonlocal thinking_buffer
        if thinking_buffer:
            console.print(Rule("Thinking", style="dim"))
            sys.stdout.write(_ANSI_DIM + "".join(thinking_buffer) + _ANSI_RESET + "\n")
            sys.stdout.flush()
            thinking_buffer.clear()

    def _on_tool_call(name: str, args: dict) -> None:
        nonlocal phase
        if phase == _Phase.THINKING:
            _clear_spinner()
            _flush_thinking_as_thinking()
        phase = _Phase.IDLE
        _print_tool_call(name, args)

    def _on_tool_result(name: str, display: str) -> None:
        _print_tool_result(name, display)

    for event in agent.run_turn_stream(
        user_input,
        thinking_enabled=thinking_enabled,
        on_tool_call=_on_tool_call,
        on_tool_result=_on_tool_result,
    ):
        if event.type == "thinking":
            if phase == _Phase.IDLE:
                phase = _Phase.THINKING
                console.print()
            thinking_buffer.append(event.data["text"])
            # Update spinner animation
            frame = _SPINNER_FRAMES[spinner_idx % len(_SPINNER_FRAMES)]
            sys.stdout.write(f"\r  {frame} Thinking...")
            sys.stdout.flush()
            spinner_idx += 1
            spinner_active = True

        elif event.type == "content":
            if phase == _Phase.THINKING:
                _clear_spinner()
                _flush_thinking_as_thinking()
                console.print()
            if phase != _Phase.CONTENT:
                phase = _Phase.CONTENT
                console.print(Rule("Response", style="blue"))
            sys.stdout.write(event.data["text"])
            sys.stdout.flush()

        elif event.type == "done":
            if phase == _Phase.THINKING and thinking_buffer:
                _clear_spinner()
                console.print(Rule("Response", style="blue"))
                sys.stdout.write("".join(thinking_buffer))
                sys.stdout.flush()
                thinking_buffer.clear()
                console.print()
            elif phase == _Phase.CONTENT:
                sys.stdout.write("\n")
                sys.stdout.flush()
            elif phase == _Phase.IDLE:
                console.print()
            console.print(Rule("Response Finish", style="blue"))
            break


def run_cli() -> None:
    """Start the interactive CLI loop."""
    _print_banner()

    agent = Agent()
    stream_enabled = settings.stream_enabled
    thinking_enabled = settings.thinking_enabled

    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye!", style="dim")
            break

        if not user_input:
            continue

        # Built-in commands
        if user_input.startswith("/"):
            cmd = user_input.lower()

            if cmd in ("/exit", "/quit"):
                console.print("Goodbye!", style="dim")
                break
            elif cmd == "/clear":
                agent.reset()
                console.print("[Conversation cleared]", style="dim")
                continue
            elif cmd == "/help":
                console.print(HELP_TEXT)
                continue
            elif cmd == "/tools":
                from tools.registry import registry

                names = registry.get_all_names()
                console.print(
                    f"Available tools ({len(names)}): {', '.join(names)}"
                )
                continue
            elif cmd == "/stream":
                stream_enabled = not stream_enabled
                state = "on" if stream_enabled else "off"
                console.print(f"Streaming: {state}", style="bold yellow")
                continue
            elif cmd == "/think":
                thinking_enabled = not thinking_enabled
                state = "on" if thinking_enabled else "off"
                console.print(f"Thinking mode: {state}", style="bold yellow")
                continue
            elif cmd == "/status":
                console.print(f"  Streaming: {'on' if stream_enabled else 'off'}")
                console.print(f"  Thinking:  {'on' if thinking_enabled else 'off'}")
                console.print(f"  Model:     {settings.deepseek_model}")
                continue
            else:
                console.print(
                    f"Unknown command: {user_input}. Type /help for available commands.",
                    style="bold red",
                )
                continue

        # Normal conversation turn
        try:
            if stream_enabled:
                _stream_response(agent, user_input, thinking_enabled)
            else:
                console.print()
                response = agent.run_turn(
                    user_input,
                    on_tool_call=_print_tool_call,
                    on_tool_result=_print_tool_result,
                )
                if response:
                    console.print()
                    console.print(
                        Panel(
                            Markdown(response),
                            title="[bold blue]Assistant[/bold blue]",
                            border_style="blue",
                        )
                    )
                else:
                    console.print("\n[dim]Assistant: [no response][/dim]\n")
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}\n")

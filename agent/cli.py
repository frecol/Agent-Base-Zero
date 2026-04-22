"""Interactive CLI for the agent."""

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from agent.agent import Agent

console = Console()

HELP_TEXT = """
Available commands:
  /help    — Show this help message
  /clear   — Clear conversation history
  /exit    — Exit the program
  /tools   — List available tools
  <text>   — Send a message to the agent
"""


def _print_banner() -> None:
    title = Text()
    title.append("  Agent-Base-Zero v0.1\n", style="bold cyan")
    title.append("  DeepSeek-powered AI Agent", style="dim")
    console.print(Panel(title, border_style="cyan", padding=(1, 2)))
    console.print(
        "Type [bold]/help[/bold] for commands, [bold]/exit[/bold] to quit.",
        style="dim",
    )
    console.print()


def _print_tool_call(name: str, args: dict) -> None:
    console.print()
    args_str = ", ".join(f"{k}=[bold]{v!r}[/bold]" for k, v in args.items())
    console.print(f"  [dim]Tool:[/dim] [bold yellow]{name}[/bold yellow]({args_str})")


def _print_tool_result(name: str, display: str) -> None:
    console.print(f"  [dim]Result:[/dim] {display}")


def run_cli() -> None:
    """Start the interactive CLI loop."""
    _print_banner()

    agent = Agent()

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
            else:
                console.print(
                    f"Unknown command: {user_input}. Type /help for available commands.",
                    style="bold red",
                )
                continue

        # Normal conversation turn
        try:
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

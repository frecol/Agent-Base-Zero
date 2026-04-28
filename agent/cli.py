"""Interactive CLI for the agent."""

import sys
from enum import Enum, auto
from typing import Optional

import tools  # auto-discovers all tool modules
import skills  # auto-discovers all skill folders

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from agent.agent import Agent
from agent.config import settings
from agent.plan import Plan, PlanPhase, StepStatus
from agent.plan_parser import parse_plan
from agent.plan_renderer import (
    PlanProgressLive,
    render_plan_review,
    render_plan_summary,
)

console = Console()

# ANSI escape codes for streaming (bypasses Rich per-token overhead)
_ANSI_DIM = "\033[2m"
_ANSI_RESET = "\033[0m"
_DIM_GRAY = "\033[2;37m"
_GRAY = "\033[37m"
_BOLD_BRIGHT_WHITE = "\033[1;97m"
_BOLD_WHITE = "\033[1;37m"


def _shimmer_ansi(text: str, frame: int) -> str:
    """Create a light-sweep shimmer on *text* using raw ANSI codes."""
    from agent.shimmer import shimmer_positions

    _STYLE_MAP = [_DIM_GRAY, _GRAY, _BOLD_WHITE, _BOLD_BRIGHT_WHITE]
    parts = [_STYLE_MAP[b] + ch for ch, b in zip(text, shimmer_positions(text, frame))]
    parts.append(_ANSI_RESET)
    return "".join(parts)


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
  /skills  — List available skills
  /skill <name> — Activate a skill
  /unskill — Deactivate current skill
  /stream  — Toggle streaming mode on/off
  /think   — Toggle thinking mode on/off
  /plan    — Toggle Plan Mode (or press Shift+Tab)
  /status  — Show current settings
  /sessions — List all saved sessions
  /resume <id> — Resume a previous session
  /new     — Start a new session
  /compact — Manually compress conversation history
  <text>   — Send a message to the agent
"""


def _print_banner(session_id: str = "", plan_mode: bool = False) -> None:
    title = Text()
    title.append("  Agent-Base-Zero v0.5\n", style="bold cyan")
    title.append("  DeepSeek-powered AI Agent\n", style="dim")
    if plan_mode:
        title.append("  Mode: ", style="dim")
        title.append("PLAN (read-only)", style="bold magenta")
        title.append("\n")
    if session_id:
        title.append(f"  Session: {session_id}", style="dim")
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
                spinner_idx = 0
                console.print()
            thinking_buffer.append(event.data["text"])
            # Shimmer animation (advance frame every 3 events for slower sweep)
            if spinner_idx % 5 == 0:
                sys.stdout.write(f"\r  {_shimmer_ansi('Thinking...', spinner_idx // 5)}")
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


def _get_last_assistant_content(messages: list) -> str:
    """Get the content of the last assistant message."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"]
    return ""


def _handle_plan_review(
    plan: Plan,
    input_handler,
    agent: Optional[Agent] = None,
    thinking_enabled: bool = False,
    _depth: int = 0,
) -> Optional[Plan]:
    """Display plan for user review with interactive option selector.

    When the user selects Modify, the original plan + user feedback is sent
    back to the LLM to produce a revised plan, which is then presented for
    review again (up to 3 rounds).
    """
    MAX_MODIFY_ROUNDS = 3

    plan.phase = PlanPhase.REVIEWING
    render_plan_review(plan, console)

    options = [
        ("Accept", "Execute the plan step by step"),
        ("Modify", "Request changes to the plan"),
        ("Cancel", "Discard the plan"),
    ]

    console.print()
    try:
        choice = input_handler.select_option(options)
    except (EOFError, KeyboardInterrupt):
        console.print("[dim]Plan cancelled.[/dim]")
        return None

    if choice == 0:
        console.print("[green]Plan accepted. Starting execution...[/green]")
        return plan
    elif choice == 1:
        if agent is None:
            console.print("[yellow]Modify not available — no agent context.[/yellow]")
            return None
        if _depth >= MAX_MODIFY_ROUNDS:
            console.print("[yellow]Maximum modification rounds reached.[/yellow]")
            return None

        try:
            feedback = input_handler.get_input_with_prompt("Modifications")
        except (EOFError, KeyboardInterrupt):
            console.print("[dim]Plan cancelled.[/dim]")
            return None

        if not feedback:
            console.print("[dim]No feedback provided. Plan unchanged.[/dim]")
            return None

        revise_prompt = (
            f"Revise the following plan based on user feedback.\n\n"
            f"Original plan:\n{plan.raw_plan_text}\n\n"
            f"User feedback: {feedback}\n\n"
            f"Produce a revised plan in the same format."
        )

        console.print("[dim]Generating revised plan...[/dim]")
        try:
            agent.run_turn(
                revise_prompt,
                on_tool_call=_print_tool_call,
                on_tool_result=_print_tool_result,
                thinking_enabled=thinking_enabled,
            )
            response = _get_last_assistant_content(agent.messages)
        except Exception as e:
            console.print(f"[bold red]Error generating revised plan: {e}[/bold red]")
            return None

        new_plan = parse_plan(response, plan.goal)
        if not new_plan:
            console.print(
                "[yellow]Could not parse a revised plan. "
                "Try describing your task again in Plan mode.[/yellow]"
            )
            return None

        return _handle_plan_review(
            new_plan, input_handler, agent, thinking_enabled, _depth + 1
        )
    else:
        console.print("[dim]Plan cancelled.[/dim]")
        return None


def _execute_plan(
    plan: Plan,
    agent: Agent,
    thinking_enabled: bool,
    input_handler,
) -> None:
    """Execute an accepted plan step-by-step with progress display.

    Steps execute silently (no tool-call/thinking output) — only the
    progress bar updates in real-time.  A brief result is shown after
    each step completes.
    """
    # Switch to Normal mode for execution
    input_handler.plan_mode = False
    agent.set_plan_mode(False)
    plan.phase = PlanPhase.EXECUTING

    console.print(Rule("[bold]Executing Plan[/bold]", style="green"))

    with PlanProgressLive(plan, console) as progress:
        for step in plan.steps:
            # Mark step as in progress
            step.status = StepStatus.IN_PROGRESS
            progress.update()

            step_prompt = (
                f"Execute Step {step.index}: {step.title}\n"
                f"Details: {step.description}\n"
                f"Context: This is step {step.index} of {len(plan.steps)} "
                f"in the plan to '{plan.goal}'.\n"
                f"Perform ONLY this step. Be concise in your response."
            )

            try:
                # Silent execution — no callbacks, no streaming output
                response = agent.run_turn(
                    step_prompt, thinking_enabled=thinking_enabled
                )
                step.status = StepStatus.DONE

                # Brief one-line result
                if response:
                    # Take first non-empty line, truncate
                    first_line = next(
                        (l for l in response.split("\n") if l.strip()), ""
                    )
                    if len(first_line) > 120:
                        first_line = first_line[:117] + "..."
                    console.print(
                        f"  [dim]Step {step.index} result:[/dim] {first_line}"
                    )

            except Exception as e:
                console.print(f"[bold red]Step {step.index} failed: {e}[/bold red]")
                step.status = StepStatus.FAILED

                options = [
                    ("Continue", "Skip and proceed with remaining steps"),
                    ("Stop", "Skip all remaining steps"),
                ]
                cont = input_handler.select_option(options)
                if cont == 1:  # Stop
                    for remaining in plan.steps:
                        if remaining.status == StepStatus.PENDING:
                            remaining.status = StepStatus.SKIPPED
                    break

            progress.update()

    # Show summary
    plan.phase = PlanPhase.COMPLETED
    render_plan_summary(plan, console)


def run_cli() -> None:
    """Start the interactive CLI loop."""
    from agent.prompt import PromptManager
    from agent.plan_input import InputHandler
    from skills.registry import build_skills_index

    prompt_mgr = PromptManager()
    prompt_mgr.update_skills_index(build_skills_index())
    agent = Agent(prompt_manager=prompt_mgr)
    agent.load_memory()

    input_handler = InputHandler()

    _print_banner(session_id=agent.session_id)

    stream_enabled = settings.stream_enabled
    thinking_enabled = settings.thinking_enabled

    while True:
        # Track mode before input to detect Shift+Tab toggle
        mode_before = input_handler.plan_mode

        try:
            user_input = input_handler.get_input()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye!", style="dim")
            break

        # Detect mode toggle via Shift+Tab (before empty-input check)
        if input_handler.plan_mode != mode_before:
            agent.set_plan_mode(input_handler.plan_mode)
            if input_handler.plan_mode:
                console.print(
                    Panel(
                        "[bold magenta]Plan Mode activated.[/bold magenta]\n"
                        "Only read-only tools available. "
                        "Describe your task and I will create a plan.\n"
                        "Press [bold]Shift+Tab[/bold] or type [bold]/plan[/bold] to exit.",
                        title="[bold]Mode: Plan[/bold]",
                        border_style="magenta",
                    )
                )
            else:
                console.print("[yellow]Switched back to Normal mode.[/yellow]")
            continue

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
            elif cmd == "/skills":
                from skills.registry import skill_registry

                names = skill_registry.get_all_names()
                active = skill_registry.get_active()
                if not names:
                    console.print("[dim]No skills available.[/dim]")
                else:
                    console.print(f"Available skills ({len(names)}):")
                    for name in names:
                        skill = skill_registry.get_skill(name)
                        marker = " [bold green]*(active)[/bold green]" if active and active.name == name else ""
                        auto_only = " [dim](auto-only)[/dim]" if not skill.user_invocable else ""
                        console.print(f"  [cyan]{name}[/cyan]{marker}{auto_only} - {skill.description}")
                continue
            elif cmd.startswith("/skill ") or cmd == "/skill":
                parts = user_input.strip().split(maxsplit=1)
                if len(parts) < 2:
                    console.print("[bold red]Usage: /skill <name>[/bold red]")
                    continue
                skill_name = parts[1].strip()
                msg = agent.activate_skill(skill_name)
                if msg.startswith("Unknown") or msg.startswith("Skill '") and "cannot" in msg:
                    console.print(f"[bold red]{msg}[/bold red]")
                else:
                    console.print(f"[green]{msg}[/green]")
                continue
            elif cmd == "/unskill":
                msg = agent.deactivate_skill()
                console.print(f"[yellow]{msg}[/yellow]")
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
            elif cmd == "/plan":
                input_handler.plan_mode = not input_handler.plan_mode
                agent.set_plan_mode(input_handler.plan_mode)
                if input_handler.plan_mode:
                    console.print(
                        Panel(
                            "[bold magenta]Plan Mode activated.[/bold magenta]\n"
                            "Only read-only tools available. "
                            "Describe your task and I will create a plan.\n"
                            "Press [bold]Shift+Tab[/bold] or type [bold]/plan[/bold] to exit.",
                            title="[bold]Mode: Plan[/bold]",
                            border_style="magenta",
                        )
                    )
                else:
                    console.print("[yellow]Switched back to Normal mode.[/yellow]")
                continue
            elif cmd == "/status":
                from agent.tokens import get_token_usage
                from skills.registry import skill_registry

                usage = get_token_usage(agent.messages, settings.max_context_tokens)
                active_skill = skill_registry.get_active()
                active_skill_name = active_skill.name if active_skill else "none"
                console.print(f"  Streaming:    {'on' if stream_enabled else 'off'}")
                console.print(f"  Thinking:     {'on' if thinking_enabled else 'off'}")
                console.print(f"  Plan Mode:    {'on' if agent._plan_mode else 'off'}")
                console.print(f"  Model:        {settings.deepseek_model}")
                console.print(f"  Session:      {agent.session_id}")
                console.print(f"  Active Skill: {active_skill_name}")
                console.print(f"  Messages:     {len(agent.messages)}")
                console.print(f"  Token usage:  ~{usage['used']:,} / {usage['max']:,} ({usage['percent']:.0f}%)")
                continue
            elif cmd == "/sessions":
                from agent.session import list_sessions

                sessions = list_sessions()
                if not sessions:
                    console.print("[dim]No saved sessions.[/dim]")
                else:
                    for s in sessions:
                        console.print(
                            f"  [cyan]{s['session_id']}[/cyan]  "
                            f"[dim]{s['updated_at'][:19]}[/dim]  "
                            f"{s['title'][:50]}  "
                            f"[dim]({s['message_count']} msgs)[/dim]"
                        )
                continue
            elif cmd.startswith("/resume"):
                parts = user_input.strip().split(maxsplit=1)
                if len(parts) < 2:
                    console.print("[bold red]Usage: /resume <session_id>[/bold red]")
                    continue
                target_id = parts[1].strip()
                try:
                    agent.load_session_data(target_id)
                    console.print(f"[green]Resumed session {target_id}[/green]")
                    console.print(f"  [dim]{len(agent.messages)} messages loaded[/dim]")
                except FileNotFoundError:
                    console.print(f"[bold red]Session not found: {target_id}[/bold red]")
                except Exception:
                    console.print(f"[bold red]Session file corrupted: {target_id}[/bold red]")
                continue
            elif cmd == "/new":
                old_id = agent.session_id
                new_id = agent.new_session()
                console.print(f"[green]New session started: {new_id}[/green]")
                console.print(f"  [dim]Previous session {old_id} preserved[/dim]")
                continue
            elif cmd == "/compact":
                from agent.tokens import compress_messages, get_token_usage

                total = len(agent.messages)
                min_required = settings.head_keep + settings.tail_keep
                if total <= min_required:
                    console.print(
                        f"[yellow]Not enough messages to compress "
                        f"({total} msgs, need > {min_required}).[/yellow]"
                    )
                    continue
                try:
                    usage_before = get_token_usage(agent.messages, settings.max_context_tokens)
                    agent.messages = compress_messages(
                        agent.client,
                        agent.messages,
                        head_keep=settings.head_keep,
                        tail_keep=settings.tail_keep,
                    )
                    usage_after = get_token_usage(agent.messages, settings.max_context_tokens)
                    console.print(
                        f"[green]Compressed: {total} -> {len(agent.messages)} messages, "
                        f"tokens ~{usage_before['used']} -> ~{usage_after['used']}[/green]"
                    )
                except Exception as e:
                    console.print(f"[bold red]Compression failed: {e}[/bold red]")
                continue
            else:
                console.print(
                    f"Unknown command: {user_input}. Type /help for available commands.",
                    style="bold red",
                )
                continue

        # Normal conversation turn
        try:
            if input_handler.plan_mode:
                # Plan Mode: generate plan with read-only tools
                if stream_enabled:
                    _stream_response(agent, user_input, thinking_enabled)
                    response = _get_last_assistant_content(agent.messages)
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
                                title="[bold magenta]Plan[/bold magenta]",
                                border_style="magenta",
                            )
                        )

                # Try to parse plan from response
                plan = parse_plan(response, user_input)
                if plan:
                    plan = _handle_plan_review(plan, input_handler, agent, thinking_enabled)
                    if plan:
                        _execute_plan(
                            plan, agent,
                            thinking_enabled, input_handler,
                        )
                else:
                    pass
                    # console.print(
                    #     "[yellow]No structured plan detected. "
                    #     "Ask the agent to create a plan with numbered steps.[/yellow]"
                    # )
            else:
                # Normal Mode: existing behavior
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

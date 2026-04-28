"""Plan progress bar rendering using Rich.

Displays step-by-step execution progress:
  [X] Step 1: Scan project structure (Done)
  [>] Step 2: Modify database config (In Progress...)
  [ ] Step 3: Run integration tests (Pending)

Uses Rich Live for real-time updates during plan execution.
"""

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent.plan import Plan, StepStatus

_STATUS_ICONS = {
    StepStatus.PENDING: ("○", "dim"),
    StepStatus.IN_PROGRESS: ("●", "bold yellow"),
    StepStatus.DONE: ("✓", "bold green"),
    StepStatus.FAILED: ("✗", "bold red"),
    StepStatus.SKIPPED: ("—", "dim"),
}

_STATUS_LABELS = {
    StepStatus.PENDING: "Pending",
    StepStatus.IN_PROGRESS: "In Progress...",
    StepStatus.DONE: "Done",
    StepStatus.FAILED: "Failed",
    StepStatus.SKIPPED: "Skipped",
}

# Pulsing color sequence for the IN_PROGRESS icon
_PULSE_COLORS = [
    "yellow", "yellow", "bright_yellow", "bold bright_yellow",
    "bold white", "bold bright_yellow", "bright_yellow", "yellow",
]


def _shimmer(text: str, frame: int) -> Text:
    """Animate text with a sweeping light-highlight effect.

    A bright spot travels left-to-right across the text, creating a
    glossy / shimmer appearance.
    """
    from agent.shimmer import shimmer_positions

    _STYLE_MAP = ["dim yellow", "yellow", "bold bright_yellow", "bold white"]
    result = Text()
    for ch, b in zip(text, shimmer_positions(text, frame)):
        result.append(ch, style=_STYLE_MAP[b])
    return result


def render_plan_review(plan: Plan, console: Console) -> None:
    """Render a plan for user review (before execution)."""
    lines = []
    lines.append(f"[bold]Goal:[/bold] {plan.goal}")
    lines.append("")
    lines.append("[bold]Proposed Steps:[/bold]")
    for step in plan.steps:
        lines.append(f"  [cyan]{step.index}.[/cyan] [bold]{step.title}[/bold]")
        lines.append(f"     [dim]{step.description}[/dim]")

    content = "\n".join(lines)
    console.print(
        Panel(
            content,
            title="[bold magenta]Plan Mode[/bold magenta]",
            border_style="magenta",
            padding=(1, 2),
        )
    )


def render_plan_progress(plan: Plan, frame: int = 0) -> Table:
    """Build a Rich Table showing plan execution progress.

    Used inside a Live context for real-time updates.
    ``frame`` drives the shimmer / pulse animation for IN_PROGRESS steps.
    """
    table = Table(
        show_header=False,
        box=None,
        padding=(0, 1, 0, 0),
        expand=False,
    )
    table.add_column("Status", width=1)
    table.add_column("Step", width=4)
    table.add_column("Title", min_width=20)
    table.add_column("State", min_width=14)

    for step in plan.steps:
        label = _STATUS_LABELS[step.status]

        if step.status == StepStatus.IN_PROGRESS:
            # Animated icon (color pulse) + shimmer text
            color = _PULSE_COLORS[frame % len(_PULSE_COLORS)]
            table.add_row(
                Text("●", style=color),
                f"[dim]Step {step.index}:[/dim]",
                step.title,
                _shimmer(label, frame),
            )
        else:
            icon, style = _STATUS_ICONS[step.status]
            table.add_row(
                f"[{style}]{icon}[/{style}]",
                f"[dim]Step {step.index}:[/dim]",
                step.title,
                f"[{style}]{label}[/{style}]",
            )

    return table


def render_plan_summary(plan: Plan, console: Console) -> None:
    """Render final plan summary after execution completes."""
    done = sum(1 for s in plan.steps if s.status == StepStatus.DONE)
    failed = sum(1 for s in plan.steps if s.status == StepStatus.FAILED)
    skipped = sum(1 for s in plan.steps if s.status == StepStatus.SKIPPED)
    total = len(plan.steps)

    parts = [f"[bold green]{done} completed[/bold green]"]
    if failed:
        parts.append(f"[bold red]{failed} failed[/bold red]")
    if skipped:
        parts.append(f"[dim]{skipped} skipped[/dim]")

    summary = ", ".join(parts) + f" [dim](of {total} total)[/dim]"

    console.print(
        Panel(
            summary,
            title="[bold]Plan Complete[/bold]",
            border_style="green" if failed == 0 else "yellow",
        )
    )


class PlanProgressLive:
    """Context manager for live-updating plan progress display.

    Uses ``get_renderable`` so the shimmer animation auto-refreshes
    without manual ``update()`` calls.  Step status changes are picked
    up automatically on the next frame because the render function
    reads from the live ``plan`` object.

    Usage:
        with PlanProgressLive(plan, console) as progress:
            # ... execute steps, mutate plan.step.status ...
            # No need to call progress.update()
    """

    def __init__(self, plan: Plan, console: Console):
        self.plan = plan
        self.console = console
        self._live: Live | None = None
        self._frame = 0

    def _render_frame(self) -> Table:
        self._frame += 1
        return render_plan_progress(self.plan, self._frame)

    def __enter__(self):
        self._live = Live(
            get_renderable=self._render_frame,
            console=self.console,
            refresh_per_second=8,
            vertical_overflow="visible",
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args):
        if self._live:
            self._live.__exit__(*args)

    def update(self) -> None:
        """No-op kept for API compatibility.  Auto-refreshed via get_renderable."""
        pass

    def stop(self) -> None:
        """Stop the live display (final state shown)."""
        if self._live:
            self._live.stop()

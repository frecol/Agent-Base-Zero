"""Plan Mode data structures and management.

A Plan represents a structured sequence of steps that the agent proposes
to accomplish a task. The user reviews and accepts/rejects/modifies the
plan before execution begins.

Plan lifecycle:
  1. PLANNING    -- Agent generates a plan (read-only tools only)
  2. REVIEWING   -- User reviews the plan, can accept/reject/modify
  3. EXECUTING   -- Plan accepted, executing step-by-step in Normal mode
  4. COMPLETED   -- All steps finished (success or failure)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


class PlanPhase(Enum):
    PLANNING = auto()
    REVIEWING = auto()
    EXECUTING = auto()
    COMPLETED = auto()


class StepStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """A single step in a plan."""

    index: int  # 1-based step number
    title: str  # Short description (one line)
    description: str  # Detailed description of what this step does
    status: StepStatus = StepStatus.PENDING


@dataclass
class Plan:
    """A structured execution plan with numbered steps."""

    goal: str  # User's original goal
    steps: List[PlanStep] = field(default_factory=list)
    phase: PlanPhase = PlanPhase.PLANNING
    raw_plan_text: str = ""  # Original LLM output for reference

    def get_step(self, index: int) -> Optional[PlanStep]:
        """Get a step by 1-based index."""
        for step in self.steps:
            if step.index == index:
                return step
        return None

    def current_step(self) -> Optional[PlanStep]:
        """Get the first in-progress step, or the first pending step."""
        for step in self.steps:
            if step.status == StepStatus.IN_PROGRESS:
                return step
        for step in self.steps:
            if step.status == StepStatus.PENDING:
                return step
        return None

    def mark_step(self, index: int, status: StepStatus) -> None:
        """Update the status of a step by 1-based index."""
        step = self.get_step(index)
        if step:
            step.status = status

    def all_done(self) -> bool:
        """Check if all steps are in a terminal state."""
        return all(
            s.status in (StepStatus.DONE, StepStatus.FAILED, StepStatus.SKIPPED)
            for s in self.steps
        )

    def summary_line(self) -> str:
        """One-line summary: '3/7 steps completed'."""
        done = sum(1 for s in self.steps if s.status == StepStatus.DONE)
        return f"{done}/{len(self.steps)} steps completed"

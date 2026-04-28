"""Parse structured plan output from LLM responses.

Expected LLM output format (instructed via system prompt):
  ## Plan
  **Goal**: <goal statement>

  ### Steps
  1. **<step title>** -- <step description>
  2. **<step title>** -- <step description>
  ...

The parser handles common variations:
  - Steps may use "N." or "N)" numbering
  - Bold markers (** **) around titles are optional
  - Separator between title and description can be " -- ", " - ", ": ", " — ", or " – "
"""

import re
from typing import List, Optional

from agent.plan import Plan, PlanStep


def parse_plan(llm_output: str, user_goal: str = "") -> Optional[Plan]:
    """Parse an LLM response into a Plan object.

    Returns None if no valid plan structure is detected.
    """
    # Look for a "## Plan" or "## Execution Plan" section
    plan_match = re.search(
        r"(?:##\s*Plan|##\s*Execution\s*Plan)\s*\n(.*?)(?:\n##\s|\Z)",
        llm_output,
        re.DOTALL | re.IGNORECASE,
    )

    if not plan_match:
        # Fallback: look for numbered steps anywhere in the text
        return _try_parse_loose(llm_output, user_goal)

    plan_text = plan_match.group(1)

    # Extract goal if present
    goal = user_goal
    goal_match = re.search(
        r"\*\*Goal\*\*:\s*(.+?)(?:\n|$)",
        plan_text,
        re.IGNORECASE,
    )
    if goal_match:
        goal = goal_match.group(1).strip()

    # Extract steps
    steps = _parse_steps(plan_text)

    if not steps:
        return None

    return Plan(
        goal=goal,
        steps=steps,
        raw_plan_text=llm_output,
    )


def _parse_steps(text: str) -> List[PlanStep]:
    """Extract numbered steps from text."""
    steps = []

    # Pattern matches:
    #   1. **Title** -- Description
    #   1. Title - Description
    #   1) Title: Description
    step_pattern = re.compile(
        r"(?:^|\n)\s*(\d+)[.)]\s+"  # Step number: "1." or "1)"
        r"(?:\*\*)?([^*\n]+?)(?:\*\*)?"  # Title (optional bold)
        r"\s*(?:--|-|:|—|–)\s*"  # Separator: --, -, :, em dash, en dash
        r"(.+?)(?=\n\s*\d+[.)]\s|\n\s*$|\Z)",  # Description
        re.DOTALL,
    )

    for match in step_pattern.finditer(text):
        num = int(match.group(1))
        title = match.group(2).strip()
        desc = match.group(3).strip()
        steps.append(
            PlanStep(
                index=num,
                title=title,
                description=desc,
            )
        )

    return steps


def _try_parse_loose(text: str, user_goal: str) -> Optional[Plan]:
    """Try to find numbered steps without a formal Plan section header."""
    steps = _parse_steps(text)
    if len(steps) >= 2:
        return Plan(
            goal=user_goal,
            steps=steps,
            raw_plan_text=text,
        )
    return None

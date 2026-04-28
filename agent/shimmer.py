"""Shared shimmer animation position calculator.

A bright spot sweeps left-to-right across text.  This module provides the
core position math; callers map the resulting brightness levels to their
own rendering format (ANSI codes, Rich Text styles, etc.).
"""


def shimmer_positions(text: str, frame: int) -> list[int]:
    """Return brightness level (0-3) for each character position.

    3 = peak (center of spotlight), 2 = near, 1 = close, 0 = dim.
    Returns an empty list when *text* is empty.
    """
    length = len(text)
    if length == 0:
        return []
    cycle = length + 8
    pos = frame % cycle - 4
    return [max(0, 3 - abs(i - pos)) for i in range(length)]

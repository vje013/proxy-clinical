"""Text anchors: the v2 output contract locates a mention by its exact string
and its occurrence index instead of character offsets.

Definition (used identically on the emit side and the relocation side):
    occurrence index n of a mention = 1 + the number of occurrences of the
    exact mention string that start before the mention's own start offset,
    counting every occurrence including overlapping ones.

``nth_occurrence`` inverts it. Both scan with ``str.find`` stepping one
character past each hit, so the pair is exact for any string, including
self-overlapping ones like "aa" in "aaa".
"""
from __future__ import annotations


def occurrence_index(text: str, sub: str, start: int) -> int:
    """1-based occurrence index of the occurrence of ``sub`` at ``start``.
    Raises if ``text[start:]`` does not begin with ``sub``."""
    if not sub or not text.startswith(sub, start):
        raise ValueError(f"{sub!r} does not occur at {start}")
    n = 1
    idx = text.find(sub)
    while idx != -1 and idx < start:
        n += 1
        idx = text.find(sub, idx + 1)
    return n


def nth_occurrence(text: str, sub: str, n: int) -> int | None:
    """Start offset of the n-th (1-based) occurrence of ``sub``; None if absent."""
    if not sub or n < 1:
        return None
    idx = text.find(sub)
    k = 1
    while idx != -1 and k < n:
        idx = text.find(sub, idx + 1)
        k += 1
    return idx if idx != -1 else None

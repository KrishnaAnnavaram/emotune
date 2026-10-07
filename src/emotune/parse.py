"""Strict answer parser.

An answer is valid only if its first line, without punctuation and case, is
exactly one label, or holds exactly one label as a whole word. Anything else
is ``invalid``. An invalid answer counts as a wrong prediction in every metric.
"""
from __future__ import annotations

import re

from .labels import INVALID, LABELS

WORD = re.compile(r"[a-z]+")


def parse_label(answer: str) -> str:
    first = (answer or "").strip().splitlines()[0] if (answer or "").strip() else ""
    words = WORD.findall(first.casefold())
    if not words:
        return INVALID
    if len(words) == 1 and words[0] in LABELS:
        return words[0]
    found = {w for w in words if w in LABELS}  # whole words only: "enjoy" is not "joy"
    return found.pop() if len(found) == 1 else INVALID

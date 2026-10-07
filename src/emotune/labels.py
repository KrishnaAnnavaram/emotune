"""The one label schema. Its order is the ``ClassLabel`` order of ``dair-ai/emotion``.

Every module gets names and ids from here. No module iterates a dict of labels.
"""
from __future__ import annotations

LABELS: tuple[str, ...] = ("sadness", "joy", "love", "anger", "fear", "surprise")
NAME_TO_ID: dict[str, int] = {name: i for i, name in enumerate(LABELS)}
INVALID = "invalid"  # a model answer that is not exactly one label


def to_id(name: str) -> int:
    try:
        return NAME_TO_ID[name]
    except KeyError as exc:
        raise ValueError(f"unknown label {name!r}; use one of {LABELS}") from exc


def to_name(idx: int) -> str:
    if not 0 <= int(idx) < len(LABELS):
        raise ValueError(f"label id {idx} is outside 0..{len(LABELS) - 1}")
    return LABELS[int(idx)]

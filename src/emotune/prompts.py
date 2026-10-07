"""Versioned prompts as chat messages. A backend applies the chat template of its model."""
from __future__ import annotations

from .data import Example
from .labels import LABELS

PROMPT_VERSION = "v2"

SYSTEM = ("You label the main emotion of a short English text. The possible labels are: "
          + ", ".join(LABELS) + ". Answer with exactly one label in lower case and nothing else.")


def zero_shot(text: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Text: {text}\nLabel:"}]


def few_shot(text: str, shots: list[Example]) -> list[dict[str, str]]:
    """Each example is one user turn and one assistant turn, so the answer format is shown, not described."""
    msgs = [{"role": "system", "content": SYSTEM}]
    for ex in shots:
        msgs.append({"role": "user", "content": f"Text: {ex.text}\nLabel:"})
        msgs.append({"role": "assistant", "content": LABELS[ex.label]})
    msgs.append({"role": "user", "content": f"Text: {text}\nLabel:"})
    return msgs


def flatten(messages: list[dict[str, str]]) -> str:
    """Plain-text fallback for a model without a chat template."""
    parts: list[str] = []
    for m in messages:
        if m["role"] == "assistant" and parts:
            parts[-1] = f"{parts[-1]} {m['content']}"
        else:
            parts.append(m["content"])
    return "\n\n".join(parts)

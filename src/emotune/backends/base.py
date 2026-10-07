from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence


@dataclass(frozen=True)
class Prediction:
    label: str  # one of LABELS or "invalid"
    raw: str = ""  # the model answer, if any
    scores: tuple[float, ...] = field(default=())  # per-label scores, if the backend has them


class Classifier(Protocol):
    name: str

    def predict(self, texts: Sequence[str]) -> list[Prediction]:  # pragma: no cover - protocol
        ...

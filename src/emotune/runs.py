"""Model specifications and run folders.

A run folder holds everything that a result depends on: ``config.json`` (model,
revision, kind, regime, prompt version, seed, decoding), ``predictions.jsonl``,
``metrics.json`` and ``report.md``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .backends.base import Prediction
from .labels import LABELS
from .prompts import PROMPT_VERSION


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    model_id: str
    revision: str = Field(min_length=1)  # pin a commit hash for a published result
    kind: Literal["base", "instruct"]


def check_like_for_like(specs: Sequence[ModelSpec]) -> None:
    """All models in one comparison must be of one kind (all base or all instruct)."""
    kinds = {s.kind for s in specs}
    if len(kinds) > 1:
        raise ValueError(f"a comparison mixes model kinds {sorted(kinds)}; use only base or only instruct models")


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    system: str  # classifier name
    regime: Literal["baseline", "zero_shot", "few_shot", "fine_tuned"]
    model: ModelSpec | None = None
    adapter: str | None = None
    decoding: str = "n/a"  # "label_likelihood", "greedy" or "n/a"
    prompt_version: str = PROMPT_VERSION
    seed: int = 0
    shots_per_class: int = 0
    dataset: str = ""
    split: str = "test"
    simulated: bool = False


def save_run(run_dir: str | Path, cfg: RunConfig, texts: Sequence[str], y_true: Sequence[int] | None,
             preds: Sequence[Prediction], metrics: dict | None) -> Path:
    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({**cfg.model_dump(), "created": time.strftime("%Y-%m-%dT%H:%M:%S")},
                                              indent=2), encoding="utf-8")
    with (d / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for i, (t, p) in enumerate(zip(texts, preds)):
            row = {"i": i, "text": t, "pred": p.label, "raw": p.raw}
            if y_true is not None:
                row["true"] = LABELS[y_true[i]]
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    if metrics is not None:
        (d / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (d / "report.md").write_text(report(cfg, metrics), encoding="utf-8")
    return d


def load_predictions(run_dir: str | Path) -> list[dict]:
    path = Path(run_dir) / "predictions.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def report(cfg: RunConfig, m: dict) -> str:
    lines = [f"# Run `{cfg.system}` ({cfg.regime})", ""]
    if cfg.simulated:
        lines += ["> **SIMULATED.** The classifier is a seeded stand-in, not a real language model.", ""]
    lo, hi = m["accuracy_ci"]
    flo, fhi = m["macro_f1_ci"]
    lines += [f"- Items: {m['n']}", f"- Accuracy: {m['accuracy']:.3f} [{lo:.3f}, {hi:.3f}]",
              f"- Macro-F1: {m['macro_f1']:.3f} [{flo:.3f}, {fhi:.3f}]", f"- Weighted F1: {m['weighted_f1']:.3f}",
              f"- Invalid answers (counted as wrong): {100 * m['invalid_rate']:.1f}%", "",
              "| Label | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for name, row in m["per_class"].items():
        lines.append(f"| {name} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['support']} |")
    return "\n".join(lines) + "\n"

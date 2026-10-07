"""Data loading with schema checks, a seeded stratified few-shot sampler and a synthetic data generator."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .labels import LABELS

SPLITS = ("train", "validation", "test")


class SchemaError(ValueError):
    """A data file does not have the expected columns or values."""


@dataclass(frozen=True)
class Example:
    text: str
    label: int


def load_csv(path: str | Path) -> list[Example]:
    """Read a CSV with the columns ``text`` and ``label`` (an id 0..5)."""
    path = Path(path)
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {"text", "label"} - set(reader.fieldnames or [])
        if missing:
            raise SchemaError(f"{path} misses columns {sorted(missing)}")
        out = []
        for n, row in enumerate(reader, start=2):
            text = (row["text"] or "").strip()
            if not text:
                raise SchemaError(f"{path}:{n}: empty text")
            try:
                label = int(row["label"])
            except ValueError as exc:
                raise SchemaError(f"{path}:{n}: label {row['label']!r} is not an integer") from exc
            if not 0 <= label < len(LABELS):
                raise SchemaError(f"{path}:{n}: label {label} is outside 0..{len(LABELS) - 1}")
            out.append(Example(text, label))
    if not out:
        raise SchemaError(f"{path} has no rows")
    return out


def write_csv(rows: list[Example], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["text", "label"])
        w.writerows([r.text, r.label] for r in rows)
    return path


def load_splits(data_dir: str | Path) -> tuple[dict[str, list[Example]], int]:
    """Read the three splits. Train rows whose text is also in validation or test are removed (no leakage).

    Returns the splits and the number of removed train rows.
    """
    d = Path(data_dir)
    out = {s: load_csv(d / f"{s}.csv") for s in SPLITS}
    held_out = {e.text.casefold() for s in ("validation", "test") for e in out[s]}
    before = len(out["train"])
    out["train"] = [e for e in out["train"] if e.text.casefold() not in held_out]
    return out, before - len(out["train"])


def few_shot_sample(train: list[Example], k_per_class: int, seed: int) -> list[Example]:
    """``k_per_class`` examples of each label, seeded, then shuffled so that no label order is fixed."""
    rng = np.random.default_rng(seed)
    picks: list[Example] = []
    for label in range(len(LABELS)):
        pool = [e for e in train if e.label == label]
        if len(pool) < k_per_class:
            raise ValueError(f"only {len(pool)} train examples of {LABELS[label]!r}")
        idx = rng.choice(len(pool), size=k_per_class, replace=False)
        picks.extend(pool[i] for i in idx)
    order = rng.permutation(len(picks))
    return [picks[i] for i in order]


def download_hf(out_dir: str | Path, revision: str | None = None) -> list[Path]:
    """Write ``train.csv``, ``validation.csv`` and ``test.csv`` from the Hugging Face dataset (``data`` extra)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise ImportError("download needs the 'data' extra: pip install -e \".[data]\"") from exc
    ds = load_dataset("dair-ai/emotion", "split", revision=revision)  # pragma: no cover - network
    names = ds["train"].features["label"].names  # pragma: no cover
    if tuple(names) != LABELS:  # pragma: no cover
        raise SchemaError(f"the dataset label order {names} differs from {LABELS}")
    return [write_csv([Example(r["text"], r["label"]) for r in ds[s]], Path(out_dir) / f"{s}.csv")  # pragma: no cover
            for s in SPLITS]


# synthetic data -----------------------------------------------------------
CUES = {
    "sadness": ["lonely", "heartbroken", "miserable", "hopeless", "gloomy", "devastated", "empty"],
    "joy": ["delighted", "cheerful", "thrilled", "content", "glad", "ecstatic", "pleased"],
    "love": ["affectionate", "adored", "tender", "devoted", "cherished", "romantic", "caring"],
    "anger": ["furious", "irritated", "outraged", "resentful", "annoyed", "bitter", "hostile"],
    "fear": ["terrified", "anxious", "nervous", "frightened", "uneasy", "panicked", "scared"],
    "surprise": ["amazed", "astonished", "shocked", "stunned", "startled", "speechless", "curious"],
}
OPENERS = ["i feel", "i am feeling", "today i felt", "honestly i feel", "i have been feeling", "right now i feel"]
INTENSIFIERS = ["", "really ", "quite ", "a bit ", "so ", "still "]
TAILS = ["", " and i do not know why", " this morning", " tonight", " again", " if i am honest", " lately",
         " since monday", " at work", " at home"]
CONTEXTS = ["about the new job", "after the long call", "when i saw the message", "about my family",
            "at the end of the week", "with the whole situation", "after reading the review", "about this product"]


def synthetic_split(n: int, seed: int, noise: float = 0.1) -> list[Example]:
    """Short first-person sentences with one cue word. A share ``noise`` of the labels is random."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        label = int(rng.integers(len(LABELS)))
        cue = str(rng.choice(CUES[LABELS[label]]))
        text = f"{rng.choice(OPENERS)} {rng.choice(INTENSIFIERS)}{cue} {rng.choice(CONTEXTS)}{rng.choice(TAILS)}"
        if rng.random() < noise:
            label = int(rng.integers(len(LABELS)))
        out.append(Example(text, label))
    return out


def write_synthetic(out_dir: str | Path, n_train: int = 1200, n_eval: int = 300, seed: int = 0) -> list[Path]:
    return [write_csv(synthetic_split(n, seed + i), Path(out_dir) / f"{s}.csv")
            for i, (s, n) in enumerate(zip(SPLITS, (n_train, n_eval, n_eval)))]

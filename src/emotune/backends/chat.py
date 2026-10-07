"""Chat models as classifiers: build the prompt, call the model with greedy settings, parse strictly."""
from __future__ import annotations

import json
import urllib.request
import zlib
from typing import Protocol, Sequence

import numpy as np

from ..data import Example
from ..labels import LABELS
from ..parse import parse_label
from ..prompts import few_shot, zero_shot
from .base import Prediction


class ChatLLM(Protocol):
    name: str
    simulated: bool

    def complete(self, messages: list[dict[str, str]]) -> str:  # pragma: no cover - protocol
        ...


class ChatClassifier:
    def __init__(self, llm: ChatLLM, regime: str = "zero_shot", shots: list[Example] | None = None) -> None:
        if regime not in ("zero_shot", "few_shot"):
            raise ValueError("regime must be 'zero_shot' or 'few_shot'")
        if regime == "few_shot" and not shots:
            raise ValueError("few_shot needs examples")
        self.llm, self.regime, self.shots = llm, regime, shots or []
        self.name = f"{llm.name}:{regime}"

    def messages(self, text: str) -> list[dict[str, str]]:
        return few_shot(text, self.shots) if self.regime == "few_shot" else zero_shot(text)

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        out = []
        for t in texts:
            try:
                raw = self.llm.complete(self.messages(t))
            except Exception as exc:  # a failed call is an invalid prediction, never a guess
                raw = f"<error: {type(exc).__name__}>"
            out.append(Prediction(parse_label(raw), raw))
        return out


class OpenAICompatibleLLM:
    """Any ``/chat/completions`` server. Temperature 0 and a short answer limit."""

    simulated = False

    def __init__(self, base_url: str, model: str, api_key: str | None, timeout_s: float = 60.0, seed: int = 0) -> None:
        if not base_url.startswith(("https://", "http://localhost", "http://127.0.0.1")):
            raise ValueError("the base URL must use https (plain http only for localhost)")
        self.base_url, self.model, self.api_key, self.timeout_s, self.seed = base_url.rstrip("/"), model, api_key, timeout_s, seed
        self.name = model

    def complete(self, messages: list[dict[str, str]]) -> str:
        body = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": 5, "seed": self.seed}
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310 - URL checked above
            return json.loads(resp.read().decode())["choices"][0]["message"]["content"]


class SimulatedChatLLM:
    """Seeded stand-in for a chat model (offline demo and tests). It is not a measurement of any real model.

    It reads the true label from an answer key, answers correctly with probability ``skill``, and
    otherwise gives a wrong label, a rambling answer or a refusal. Few-shot prompts raise the skill.
    """

    simulated = True

    def __init__(self, answer_key: dict[str, int], skill: float = 0.55, few_shot_bonus: float = 0.15,
                 ramble: float = 0.1, seed: int = 0) -> None:
        self.answer_key, self.skill, self.bonus, self.ramble, self.seed = answer_key, skill, few_shot_bonus, ramble, seed
        self.name = f"simulated-chat-seed{seed}"

    def complete(self, messages: list[dict[str, str]]) -> str:
        text = messages[-1]["content"].removeprefix("Text: ").removesuffix("\nLabel:")
        truth = self.answer_key.get(text)
        rng = np.random.default_rng([self.seed, zlib.crc32(text.encode()), len(messages)])
        skill = self.skill + (self.bonus if len(messages) > 2 else 0.0)
        r = rng.random()
        if truth is not None and r < skill:
            return LABELS[truth]
        if r < skill + self.ramble:
            a, b = rng.choice(len(LABELS), size=2, replace=False)
            return f"It could be {LABELS[a]} or maybe {LABELS[b]}."
        if r < skill + self.ramble + 0.03:
            return "I cannot tell."
        wrong = [i for i in range(len(LABELS)) if i != truth]
        return LABELS[int(rng.choice(wrong))]


class ScriptedLLM:
    simulated = False

    def __init__(self, answers: list[str], name: str = "scripted") -> None:
        self.answers, self.calls, self.name = list(answers), [], name

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self.answers.pop(0) if self.answers else ""

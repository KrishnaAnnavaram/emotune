"""Hugging Face models (extra ``hf``): label likelihood scoring and greedy generation.

* ``HFLabelScorer`` ranks the six labels by the log-likelihood of their
  tokens after the prompt. It cannot give an invalid answer and has no label
  order bias from text parsing.
* ``HFGreedyGenerator`` decodes greedily and parses strictly. It cuts the
  answer by token position, never by string length.

The prompt goes through the chat template of the tokenizer when it has one.
The model id and the revision are saved with every run.
"""
from __future__ import annotations

import os
from typing import Sequence

from ..data import Example
from ..labels import INVALID, LABELS
from ..parse import parse_label
from ..prompts import few_shot, flatten, zero_shot
from .base import Prediction


def label_logprobs(logits_fn, prompt_ids: list[int], label_ids: list[list[int]]) -> list[float]:
    """Sum of log p(label token | prompt, earlier label tokens) for each label.

    ``logits_fn(batch_ids)`` takes a (B, T) long tensor and returns (B, T, V) logits.
    """
    import torch

    seqs = [prompt_ids + ids for ids in label_ids]
    width = max(len(s) for s in seqs)
    batch = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        batch[i, : len(s)] = torch.tensor(s)
    with torch.no_grad():
        logp = torch.log_softmax(logits_fn(batch).float(), dim=-1)
    scores = []
    p = len(prompt_ids)
    for i, ids in enumerate(label_ids):
        total = 0.0
        for j, tok in enumerate(ids):
            total += float(logp[i, p + j - 1, tok])  # the logits at position t predict token t + 1
        scores.append(total)
    return scores


def render_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return flatten(messages) + " "


def load(model_id: str, revision: str, load_in_4bit: bool = False):  # pragma: no cover - needs model weights
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN") or None
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, token=token)
    kwargs = {"revision": revision, "token": token, "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    return tok, model


class HFLabelScorer:  # pragma: no cover - needs model weights
    def __init__(self, tokenizer, model, regime: str = "zero_shot", shots: list[Example] | None = None,
                 name: str = "hf") -> None:
        self.tok, self.model, self.regime, self.shots = tokenizer, model, regime, shots or []
        self.name = f"{name}:{regime}:score"
        self.label_ids = [tokenizer(" " + lab, add_special_tokens=False)["input_ids"] for lab in LABELS]

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        out = []
        for t in texts:
            msgs = few_shot(t, self.shots) if self.regime == "few_shot" else zero_shot(t)
            ids = self.tok(render_prompt(self.tok, msgs), add_special_tokens=False)["input_ids"]
            scores = label_logprobs(lambda b: self.model(b.to(self.model.device)).logits.cpu(), ids, self.label_ids)
            best = max(range(len(LABELS)), key=lambda i: scores[i])
            out.append(Prediction(LABELS[best], scores=tuple(scores)))
        return out


class HFGreedyGenerator:  # pragma: no cover - needs model weights
    def __init__(self, tokenizer, model, regime: str = "zero_shot", shots: list[Example] | None = None,
                 name: str = "hf", max_new_tokens: int = 6) -> None:
        self.tok, self.model, self.regime, self.shots = tokenizer, model, regime, shots or []
        self.max_new_tokens = max_new_tokens
        self.name = f"{name}:{regime}:greedy"

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        import torch

        out = []
        for t in texts:
            msgs = few_shot(t, self.shots) if self.regime == "few_shot" else zero_shot(t)
            enc = self.tok(render_prompt(self.tok, msgs), return_tensors="pt", add_special_tokens=False).to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, do_sample=False, max_new_tokens=self.max_new_tokens,
                                          pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
            new_tokens = gen[0, enc["input_ids"].shape[1]:]
            raw = self.tok.decode(new_tokens, skip_special_tokens=True)
            label = parse_label(raw)
            out.append(Prediction(label if label != INVALID else INVALID, raw))
        return out


def load_with_adapter(model_id: str, revision: str, adapter_dir: str, load_in_4bit: bool = False):  # pragma: no cover
    """The base model plus a trained LoRA adapter. Use this for every prediction of a fine-tuned run."""
    from peft import PeftModel

    tok, model = load(model_id, revision, load_in_4bit)
    return tok, PeftModel.from_pretrained(model, adapter_dir).eval()

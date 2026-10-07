"""LoRA / QLoRA fine-tuning with a completion-only loss (extra ``finetune``).

The pure functions ``build_example`` and ``collate`` need no deep-learning
library, and the tests check them:

* the prompt tokens get the label -100, so the loss covers only the answer
  (the label word and the end-of-sequence token)
* a batch is padded to its longest example (dynamic padding), and padding
  also gets the label -100
"""
from __future__ import annotations

import os
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

from .data import Example
from .labels import LABELS
from .prompts import zero_shot

IGNORE = -100


class LoraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    r: int = Field(16, ge=1)
    alpha: int = Field(32, ge=1)
    dropout: float = Field(0.05, ge=0, lt=1)
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    load_in_4bit: bool = True
    lr: float = Field(2e-4, gt=0)
    epochs: float = Field(2.0, gt=0)
    batch_size: int = Field(8, ge=1)
    grad_accum: int = Field(2, ge=1)
    max_length: int = Field(256, ge=16)
    seed: int = 0


def build_example(prompt_ids: Sequence[int], answer_ids: Sequence[int], eos_id: int,
                  max_length: int) -> dict[str, list[int]]:
    """Input ids and labels for one example. The prompt is cut from the left if the example is too long."""
    answer = list(answer_ids) + [eos_id]
    room = max_length - len(answer)
    if room <= 0:
        raise ValueError("max_length is too small for the answer")
    prompt = list(prompt_ids)[-room:]
    return {"input_ids": prompt + answer, "labels": [IGNORE] * len(prompt) + answer,
            "attention_mask": [1] * (len(prompt) + len(answer))}


def collate(batch: Sequence[dict[str, list[int]]], pad_id: int) -> dict[str, list[list[int]]]:
    """Right-pad to the longest example in the batch."""
    width = max(len(b["input_ids"]) for b in batch)
    out: dict[str, list[list[int]]] = {"input_ids": [], "labels": [], "attention_mask": []}
    for b in batch:
        pad = width - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * pad)
        out["labels"].append(b["labels"] + [IGNORE] * pad)
        out["attention_mask"].append(b["attention_mask"] + [0] * pad)
    return out


def train_lora(model_id: str, revision: str, train: Sequence[Example], val: Sequence[Example], out_dir: str,
               cfg: LoraConfig = LoraConfig()):  # pragma: no cover - needs a GPU and model weights
    import torch
    from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments

    from .backends.hf import load, render_prompt

    torch.manual_seed(cfg.seed)
    tok, model = load(model_id, revision, cfg.load_in_4bit)
    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, PeftLoraConfig(r=cfg.r, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout,
                                                 target_modules=list(cfg.target_modules), task_type="CAUSAL_LM"))

    def encode(rows):
        feats = []
        for e in rows:
            p = tok(render_prompt(tok, zero_shot(e.text)), add_special_tokens=False)["input_ids"]
            a = tok(LABELS[e.label], add_special_tokens=False)["input_ids"]
            feats.append(build_example(p, a, tok.eos_token_id, cfg.max_length))
        return feats

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def collator(batch):
        return {k: torch.tensor(v) for k, v in collate(batch, pad_id).items()}

    args = TrainingArguments(output_dir=out_dir, per_device_train_batch_size=cfg.batch_size,
                             gradient_accumulation_steps=cfg.grad_accum, learning_rate=cfg.lr,
                             num_train_epochs=cfg.epochs, eval_strategy="epoch", save_strategy="epoch",
                             load_best_model_at_end=True, metric_for_best_model="eval_loss", seed=cfg.seed,
                             report_to=[], logging_steps=50, remove_unused_columns=False)
    trainer = Trainer(model=model, args=args, train_dataset=encode(train), eval_dataset=encode(val),
                      data_collator=collator)
    trainer.train()
    model.save_pretrained(os.path.join(out_dir, "adapter"))
    tok.save_pretrained(os.path.join(out_dir, "adapter"))
    return tok, model

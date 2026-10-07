import pytest
from pydantic import ValidationError

from emotune.finetune import IGNORE, LoraConfig, build_example, collate


def test_loss_covers_only_the_answer():
    ex = build_example([5, 6, 7, 8], [42], eos_id=2, max_length=16)
    assert ex["input_ids"] == [5, 6, 7, 8, 42, 2]
    assert ex["labels"] == [IGNORE] * 4 + [42, 2]
    assert ex["attention_mask"] == [1] * 6


def test_long_prompt_is_cut_from_the_left():
    ex = build_example(list(range(100)), [42, 43], eos_id=2, max_length=10)
    assert len(ex["input_ids"]) == 10 and ex["input_ids"][-3:] == [42, 43, 2] and ex["input_ids"][0] == 93
    with pytest.raises(ValueError):
        build_example([1], [1, 2, 3], eos_id=2, max_length=3)


def test_dynamic_padding_masks_pads():
    batch = [build_example([1, 2], [9], 2, 32), build_example([1, 2, 3, 4, 5], [9], 2, 32)]
    out = collate(batch, pad_id=0)
    assert [len(x) for x in out["input_ids"]] == [7, 7]  # padded to the longest example, not to max_length
    assert out["labels"][0][-3:] == [IGNORE] * 3 and out["attention_mask"][0][-3:] == [0, 0, 0]


def test_lora_config_is_strict():
    with pytest.raises(ValidationError):
        LoraConfig(rank=8)
    assert LoraConfig().target_modules == ("q_proj", "k_proj", "v_proj", "o_proj")


def test_label_logprobs_use_the_right_positions():
    torch = pytest.importorskip("torch")
    from emotune.backends.hf import label_logprobs

    V = 10

    def logits_fn(batch):
        # the logits at position t strongly predict token (input[t] + 1) % V
        out = torch.full((*batch.shape, V), -10.0)
        nxt = (batch + 1) % V
        out.scatter_(2, nxt.unsqueeze(-1), 10.0)
        return out

    scores = label_logprobs(logits_fn, [1, 2, 3], [[4], [7], [4, 5]])
    assert scores[0] > scores[1]  # 4 follows 3
    assert scores[2] == pytest.approx(scores[0] * 2, rel=1e-4)  # 4 then 5 are both predicted


def test_render_prompt_uses_the_chat_template_when_present():
    pytest.importorskip("torch")
    from emotune.backends.hf import render_prompt
    from emotune.prompts import zero_shot

    class WithTemplate:
        chat_template = "x"

        def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
            assert tokenize is False and add_generation_prompt is True
            return "<chat>" + msgs[-1]["content"]

    class Plain:
        chat_template = None

    assert render_prompt(WithTemplate(), zero_shot("t")) == "<chat>Text: t\nLabel:"
    assert render_prompt(Plain(), zero_shot("t")).endswith("Text: t\nLabel: ")

import pytest

from emotune.data import Example
from emotune.labels import INVALID
from emotune.parse import parse_label
from emotune.prompts import PROMPT_VERSION, few_shot, flatten, zero_shot


def test_zero_shot_messages():
    msgs = zero_shot("i am glad")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "sadness, joy, love, anger, fear, surprise" in msgs[0]["content"]
    assert msgs[1]["content"] == "Text: i am glad\nLabel:" and PROMPT_VERSION


def test_few_shot_examples_use_label_names_from_the_schema():
    msgs = few_shot("x", [Example("a", 0), Example("b", 5)])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert msgs[2]["content"] == "sadness" and msgs[4]["content"] == "surprise"
    flat = flatten(msgs)
    assert "Text: a\nLabel: sadness" in flat and flat.endswith("Text: x\nLabel:")


@pytest.mark.parametrize("answer,label", [
    ("joy", "joy"), ("Joy.", "joy"), ("  ANGER\nText: something else", "anger"), ("Label: fear", "fear"),
    ("The emotion is love.", "love"),
])
def test_parse_accepts_exactly_one_label(answer, label):
    assert parse_label(answer) == label


@pytest.mark.parametrize("answer", ["enjoy", "joy or sadness", "", "I cannot tell.", "joyful",
                                    "It could be anger or maybe fear."])
def test_parse_rejects_everything_else(answer):
    assert parse_label(answer) == INVALID

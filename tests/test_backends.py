import pytest

from emotune.backends import (
    ChatClassifier, KeywordBaseline, OpenAICompatibleLLM, ScriptedLLM, SimulatedChatLLM, TfidfBaseline,
)
from emotune.data import Example, synthetic_split
from emotune.labels import INVALID


def test_chat_classifier_parses_strictly_and_records_raw_answers():
    llm = ScriptedLLM(["joy", "enjoy it", "Anger."])
    preds = ChatClassifier(llm).predict(["a", "b", "c"])
    assert [p.label for p in preds] == ["joy", INVALID, "anger"] and preds[1].raw == "enjoy it"
    assert llm.calls[0][-1]["content"] == "Text: a\nLabel:"


def test_failed_call_is_invalid_not_a_guess():
    class Broken:
        name, simulated = "broken", False

        def complete(self, messages):
            raise TimeoutError

    p = ChatClassifier(Broken()).predict(["x"])[0]
    assert p.label == INVALID and "TimeoutError" in p.raw


def test_regime_rules():
    with pytest.raises(ValueError):
        ChatClassifier(ScriptedLLM([]), "few_shot")
    with pytest.raises(ValueError):
        ChatClassifier(ScriptedLLM([]), "one_shot")
    clf = ChatClassifier(ScriptedLLM(["joy"]), "few_shot", [Example("a", 1)])
    assert len(clf.messages("t")) == 4 and clf.name == "scripted:few_shot"


def test_openai_client_url_rule():
    with pytest.raises(ValueError):
        OpenAICompatibleLLM("http://example.com/v1", "m", None)
    assert OpenAICompatibleLLM("http://localhost:8000/v1/", "m", None).base_url == "http://localhost:8000/v1"


def test_simulated_model_is_deterministic_and_marked():
    key = {e.text: e.label for e in synthetic_split(100, 1)}
    llm = SimulatedChatLLM(key, seed=3)
    clf = ChatClassifier(llm)
    texts = list(key)[:30]
    assert [p.raw for p in clf.predict(texts)] == [p.raw for p in clf.predict(texts)]
    assert llm.simulated is True


def test_tfidf_baseline_learns_the_synthetic_task():
    train, test = synthetic_split(800, 1, noise=0.0), synthetic_split(200, 2, noise=0.0)
    clf = TfidfBaseline(seed=0).fit(train)
    preds = clf.predict([e.text for e in test])
    acc = sum(p.label == ["sadness", "joy", "love", "anger", "fear", "surprise"][e.label]
              for p, e in zip(preds, test)) / len(test)
    assert acc > 0.9 and len(preds[0].scores) == 6
    with pytest.raises(RuntimeError):
        TfidfBaseline().predict(["x"])


def test_keyword_baseline_is_honest_about_ties():
    preds = KeywordBaseline().predict(["i am happy", "i am happy and scared", "nothing here"])
    assert [p.label for p in preds] == ["joy", INVALID, INVALID]

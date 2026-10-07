import pytest

from emotune.data import Example, SchemaError, few_shot_sample, load_csv, load_splits, synthetic_split, write_csv
from emotune.labels import LABELS, NAME_TO_ID, to_id, to_name


def test_label_schema_matches_the_dataset_order():
    assert LABELS == ("sadness", "joy", "love", "anger", "fear", "surprise")
    for i, name in enumerate(LABELS):
        assert to_id(name) == i == NAME_TO_ID[name] and to_name(i) == name
    with pytest.raises(ValueError):
        to_id("happy")
    with pytest.raises(ValueError):
        to_name(6)


def test_load_csv_checks_schema(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("text,label\nhello,1\n")
    assert load_csv(p) == [Example("hello", 1)]
    for bad in ("txt,label\na,1\n", "text,label\n,1\n", "text,label\na,joy\n", "text,label\na,7\n", "text,label\n"):
        p.write_text(bad)
        with pytest.raises(SchemaError):
            load_csv(p)


def test_load_splits_removes_leaked_train_rows(tmp_path):
    write_csv([Example("same text", 0), Example("only train", 1)], tmp_path / "train.csv")
    write_csv([Example("val text", 2)], tmp_path / "validation.csv")
    write_csv([Example("Same Text", 0)], tmp_path / "test.csv")
    splits, removed = load_splits(tmp_path)
    assert removed == 1 and [e.text for e in splits["train"]] == ["only train"]


def test_few_shot_sample_is_stratified_seeded_and_shuffled():
    train = synthetic_split(600, seed=1)
    a = few_shot_sample(train, 3, seed=5)
    b = few_shot_sample(train, 3, seed=5)
    assert a == b and len(a) == 18
    assert sorted(e.label for e in a) == sorted(list(range(6)) * 3)
    assert [e.label for e in a] != sorted(e.label for e in a)  # not grouped by label
    with pytest.raises(ValueError):
        few_shot_sample(train[:5], 3, seed=0)


def test_synthetic_split_is_seeded():
    assert synthetic_split(50, 3) == synthetic_split(50, 3) != synthetic_split(50, 4)

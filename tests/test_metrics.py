import math

import pytest

from emotune.labels import INVALID
from emotune.metrics import agreement, classification_metrics, compare, mcnemar_exact


def test_invalid_answers_stay_in_the_denominator():
    m = classification_metrics([0, 1, 2, 3], ["sadness", "joy", INVALID, "fear"], n_boot=200)
    assert m["n"] == 4 and m["accuracy"] == 0.5 and m["invalid_rate"] == 0.25
    assert m["confusion"]["matrix"][2][6] == 1 and m["confusion"]["columns"][6] == INVALID
    assert m["per_class"]["love"]["recall"] == 0.0
    lo, hi = m["accuracy_ci"]
    assert lo <= 0.5 <= hi


def test_macro_f1_over_all_six_labels():
    m = classification_metrics([0, 0, 1, 1], ["sadness", "sadness", "joy", "sadness"], n_boot=100)
    # sadness: P=2/3 R=1 F1=0.8; joy: P=1 R=0.5 F1=2/3; four labels with no support give 0
    assert m["macro_f1"] == pytest.approx((0.8 + 2 / 3) / 6)


def test_mcnemar_and_paired_compare():
    t = [0] * 10
    a = ["sadness"] * 8 + ["joy"] * 2
    b = ["sadness"] * 3 + ["joy"] * 7
    out = compare(t, a, b, n_boot=200)
    assert out["mcnemar"]["only_a_correct"] == 5 and out["mcnemar"]["only_b_correct"] == 0
    assert out["accuracy_diff"] == pytest.approx(0.5) and out["mcnemar"]["p_value"] < 0.07
    assert mcnemar_exact([True, False], [True, False])["p_value"] == 1.0
    with pytest.raises(ValueError):
        mcnemar_exact([True], [True, False])


def test_unlabelled_runs_get_agreement_not_a_test():
    out = agreement(["joy", "joy", "fear", INVALID], ["joy", "anger", "fear", "joy"])
    assert out["agreement"] == 0.5 and -1 <= out["cohen_kappa"] <= 1 and "p_value" not in out
    assert math.isnan(agreement(["joy", "joy"], ["joy", "joy"])["cohen_kappa"])
    with pytest.raises(ValueError):
        agreement([], [])


def test_fast_macro_f1_matches_scikit_learn():
    import numpy as np
    from sklearn.metrics import f1_score

    from emotune.metrics import macro_f1

    rng = np.random.default_rng(0)
    t = rng.integers(0, 6, 300)
    p = np.where(rng.random(300) < 0.1, 6, rng.integers(0, 6, 300))
    assert macro_f1(t, p) == pytest.approx(f1_score(t, p, labels=list(range(6)), average="macro", zero_division=0))

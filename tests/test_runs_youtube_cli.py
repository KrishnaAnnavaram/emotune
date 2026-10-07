import json
import re
import subprocess
from pathlib import Path

import pytest

from emotune import cli, youtube
from emotune.backends.base import Prediction
from emotune.runs import ModelSpec, RunConfig, check_like_for_like, load_predictions, save_run

ROOT = Path(__file__).resolve().parents[1]


def test_models_file_and_like_for_like_rule():
    models = cli.load_models(ROOT / "configs" / "models.toml")
    instruct = [m for m in models.values() if m.kind == "instruct"]
    check_like_for_like(instruct)
    with pytest.raises(ValueError):
        check_like_for_like([models["mistral-7b-instruct"], models["mistral-7b-base"]])
    with pytest.raises(ValueError):
        ModelSpec(name="x", model_id="y", revision="", kind="instruct")


def test_save_run_writes_config_predictions_and_report(tmp_path):
    cfg = RunConfig(system="s", regime="zero_shot", simulated=True, decoding="greedy")
    from emotune.metrics import classification_metrics

    m = classification_metrics([1, 2], ["joy", "invalid"], n_boot=50)
    d = save_run(tmp_path / "r", cfg, ["a", "b"], [1, 2], [Prediction("joy", "joy"), Prediction("invalid", "?")], m)
    assert json.loads((d / "config.json").read_text())["prompt_version"]
    assert load_predictions(d)[1] == {"i": 1, "text": "b", "pred": "invalid", "raw": "?", "true": "love"}
    assert "SIMULATED" in (d / "report.md").read_text()


def test_youtube_collector_keeps_text_and_hashed_id_only():
    pages = [
        {"items": [{"snippet": {"topLevelComment": {"id": "c1", "snippet": {
            "textDisplay": "Great  phone!", "authorDisplayName": "Real Name", "publishedAt": "2024-01-01",
            "authorChannelId": {"value": "UC1"}, "likeCount": 3}}}}], "nextPageToken": "p2"},
        {"items": [{"snippet": {"topLevelComment": {"id": "c2", "snippet": {"textDisplay": "Too slow",
                                                                             "authorDisplayName": "Other"}}}}]},
    ]
    urls = []

    def transport(url):
        urls.append(url)
        return pages.pop(0)

    rows = youtube.collect(["vid1"], "key", "salt", transport=transport)
    assert rows == [{"id": youtube.hashed_id("c1", "salt"), "text": "Great phone!"},
                    {"id": youtube.hashed_id("c2", "salt"), "text": "Too slow"}]
    assert "pageToken=p2" in urls[1] and "Real Name" not in json.dumps(rows)
    with pytest.raises(ValueError):
        youtube.collect(["v"], "", "salt", transport=transport)
    with pytest.raises(ValueError):
        youtube.collect(["v"], "key", "", transport=transport)
    sample = youtube.label_sample([{"id": str(i), "text": f"t{i}"} for i in range(20)], 5, seed=1)
    assert len(sample) == 5 and all(s["label"] == "" for s in sample)


def test_cli_offline_flow(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EMOTUNE_RUNS_DIR", str(tmp_path / "runs"))
    assert cli.main(["synth", "--out", "d", "--n-train", "400", "--n-eval", "120"]) == 0
    assert cli.main(["baseline", "--data", "d", "--model", "tfidf", "--run", "tfidf"]) == 0
    assert cli.main(["baseline", "--data", "d", "--model", "keyword", "--run", "kw"]) == 0
    assert cli.main(["llm", "--data", "d", "--backend", "simulated", "--regime", "few_shot", "--run", "sim"]) == 0
    capsys.readouterr()
    assert cli.main(["compare", str(tmp_path / "runs" / "tfidf"), str(tmp_path / "runs" / "sim")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out) == {"accuracy_a", "accuracy_b", "accuracy_diff", "accuracy_diff_ci", "mcnemar"}
    assert cli.main(["agreement", str(tmp_path / "runs" / "tfidf"), str(tmp_path / "runs" / "kw")]) == 0
    cfg = json.loads((tmp_path / "runs" / "sim" / "config.json").read_text())
    assert cfg["simulated"] is True and cfg["shots_per_class"] == 3 and cfg["regime"] == "few_shot"


def test_compare_refuses_unlabelled_runs(tmp_path):
    cfg = RunConfig(system="s", regime="zero_shot")
    for name in ("a", "b"):
        save_run(tmp_path / name, cfg, ["x"], None, [Prediction("joy")], None)
    with pytest.raises(SystemExit):
        cli.main(["compare", str(tmp_path / "a"), str(tmp_path / "b")])


def test_demo_runs(tmp_path):
    assert cli.main(["demo", "--out-dir", str(tmp_path)]) == 0


KEY = re.compile(r"sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{30,}|hf_[A-Za-z0-9]{25,}|gsk_[A-Za-z0-9]{20,}|"
                 r"AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}")


def test_no_credentials_in_tracked_files():
    try:
        files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        files = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]
    for name in files:
        p = ROOT / name
        if p.is_file() and p.suffix in {".py", ".md", ".toml", ".yml", ".txt", ".example", ""}:
            assert not KEY.search(p.read_text(encoding="utf-8", errors="ignore")), name

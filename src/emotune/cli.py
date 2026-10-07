"""Command line interface: ``emotune <command> ...``."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import tomllib
from pathlib import Path

from . import data as data_mod
from . import metrics, youtube
from .backends import ChatClassifier, KeywordBaseline, OpenAICompatibleLLM, SimulatedChatLLM, TfidfBaseline
from .config import load_dotenv, settings_from_env
from .labels import LABELS
from .runs import ModelSpec, RunConfig, check_like_for_like, load_predictions, save_run


def load_models(path: str | Path) -> dict[str, ModelSpec]:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    return {m["name"]: ModelSpec.model_validate(m) for m in raw.get("models", [])}


def _splits(a):
    s = settings_from_env()
    splits, removed = data_mod.load_splits(a.data or s.data_dir)
    if removed:
        print(f"removed {removed} train rows whose text is also in validation or test")
    return splits


def _evaluate_and_save(name: str, cfg: RunConfig, items, clf, runs_dir: Path) -> dict:
    texts, y = [e.text for e in items], [e.label for e in items]
    preds = clf.predict(texts)
    m = metrics.classification_metrics(y, [p.label for p in preds])
    d = save_run(runs_dir / name, cfg, texts, y, preds, m)
    lo, hi = m["accuracy_ci"]
    print(f"{cfg.system:32s} accuracy {m['accuracy']:.3f} [{lo:.3f}, {hi:.3f}]  macro-F1 {m['macro_f1']:.3f}  "
          f"invalid {100 * m['invalid_rate']:.1f}%  -> {d}")
    return m


def cmd_synth(a) -> None:
    for p in data_mod.write_synthetic(a.out, a.n_train, a.n_eval, a.seed):
        print(f"wrote {p}")


def cmd_download(a) -> None:
    for p in data_mod.download_hf(a.out, a.revision):
        print(f"wrote {p}")


def cmd_baseline(a) -> None:
    s = settings_from_env()
    splits = _splits(a)
    clf = TfidfBaseline(seed=s.seed).fit(splits["train"]) if a.model == "tfidf" else KeywordBaseline()
    cfg = RunConfig(system=clf.name, regime="baseline", seed=s.seed, dataset=str(a.data or s.data_dir), split=a.split)
    _evaluate_and_save(a.run or f"{clf.name}_{a.split}", cfg, splits[a.split][: a.limit], clf, Path(s.runs_dir))


def _classifier(a, splits, s):
    shots = data_mod.few_shot_sample(splits["train"], a.shots, s.seed) if a.regime == "few_shot" else None
    spec = None
    if a.backend == "simulated":
        key = {e.text: e.label for e in splits[a.split]}
        clf = ChatClassifier(SimulatedChatLLM(key, seed=s.seed), a.regime, shots)
        return clf, spec, "greedy", True
    if a.backend == "openai":
        if not s.llm_api_key and s.llm_base_url.startswith("https://"):
            sys.exit("set EMOTUNE_LLM_API_KEY in the environment or in .env")
        return ChatClassifier(OpenAICompatibleLLM(s.llm_base_url, s.llm_model, s.llm_api_key, seed=s.seed),
                              a.regime, shots), spec, "greedy", False
    try:
        from .backends import hf
    except ImportError:
        sys.exit("the hf backends need the hf extra: pip install -e \".[hf]\"")
    spec = load_models(a.models_file)[a.model]
    tok, model = (hf.load_with_adapter(spec.model_id, spec.revision, a.adapter) if a.adapter
                  else hf.load(spec.model_id, spec.revision, a.load_in_4bit))
    if a.backend == "hf-score":
        return hf.HFLabelScorer(tok, model, a.regime, shots, spec.name), spec, "label_likelihood", False
    return hf.HFGreedyGenerator(tok, model, a.regime, shots, spec.name), spec, "greedy", False


def cmd_llm(a) -> None:
    s = settings_from_env()
    splits = _splits(a)
    clf, spec, decoding, simulated = _classifier(a, splits, s)
    regime = "fine_tuned" if a.adapter else a.regime
    cfg = RunConfig(system=clf.name, regime=regime, model=spec, adapter=a.adapter, decoding=decoding, seed=s.seed,
                    shots_per_class=a.shots if a.regime == "few_shot" else 0, dataset=str(a.data or s.data_dir),
                    split=a.split, simulated=simulated)
    _evaluate_and_save(a.run or clf.name.replace(":", "_").replace("/", "_"), cfg, splits[a.split][: a.limit], clf,
                       Path(s.runs_dir))


def cmd_finetune(a) -> None:
    try:
        from .finetune import LoraConfig, train_lora
    except ImportError:
        sys.exit("finetune needs the finetune extra: pip install -e \".[finetune]\"")
    s = settings_from_env()
    splits = _splits(a)
    spec = load_models(a.models_file)[a.model]
    out = Path(a.out or Path(s.runs_dir) / f"lora_{spec.name}")
    train_lora(spec.model_id, spec.revision, splits["train"], splits["validation"], str(out), LoraConfig(seed=s.seed))
    print(f"adapter saved in {out / 'adapter'}; evaluate with: emotune llm --backend hf-score --model {spec.name} "
          f"--adapter {out / 'adapter'}")


def _paired(a, b):
    pa, pb = load_predictions(a), load_predictions(b)
    if [r["text"] for r in pa] != [r["text"] for r in pb]:
        sys.exit("the two runs do not have the same items in the same order")
    return pa, pb


def cmd_compare(a) -> None:
    pa, pb = _paired(a.run_a, a.run_b)
    if any("true" not in r for r in pa):
        sys.exit("compare needs labelled runs; use `emotune agreement` for unlabelled texts")
    specs = [json.loads((Path(r) / "config.json").read_text())["model"] for r in (a.run_a, a.run_b)]
    if all(specs):
        check_like_for_like([ModelSpec.model_validate(x) for x in specs])
    out = metrics.compare([LABELS.index(r["true"]) for r in pa], [r["pred"] for r in pa], [r["pred"] for r in pb])
    print(json.dumps(out, indent=2))


def cmd_agreement(a) -> None:
    pa, pb = _paired(a.run_a, a.run_b)
    print(json.dumps(metrics.agreement([r["pred"] for r in pa], [r["pred"] for r in pb]), indent=2))


def cmd_predict(a) -> None:
    """Predict unlabelled texts (for example YouTube comments). With --adapter, the fine-tuned model runs."""
    s = settings_from_env()
    with open(a.input, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    texts = [r["text"] for r in rows][: a.limit]
    a.regime, a.split, a.shots = "zero_shot", "test", 0
    if a.backend == "simulated":
        sys.exit("the simulated backend needs labels; use openai or an hf backend")
    clf, spec, decoding, _ = _classifier(a, {"train": [], "test": []}, s)
    preds = clf.predict(texts)
    cfg = RunConfig(system=clf.name, regime="fine_tuned" if a.adapter else "zero_shot", model=spec, adapter=a.adapter,
                    decoding=decoding, seed=s.seed, dataset=a.input, split="unlabelled")
    print(f"wrote {save_run(Path(s.runs_dir) / a.run, cfg, texts, None, preds, None)}")


def cmd_collect_youtube(a) -> None:
    s = settings_from_env()
    rows = youtube.collect(a.videos, s.youtube_api_key or "", s.hash_salt or "", a.max_per_video)
    print(f"wrote {youtube.write_rows(rows, a.out)} ({len(rows)} comments, text and hashed id only)")


def cmd_label_sample(a) -> None:
    with open(a.comments, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    sample = youtube.label_sample(rows, a.n, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "text", "label"])
        w.writeheader()
        w.writerows(sample)
    print(f"wrote {out}: fill the label column with an id 0-5 ({', '.join(LABELS)})")


def cmd_demo(a) -> None:
    """Offline demo on synthetic data: two baselines and a simulated chat model in two regimes."""
    work = Path(a.out_dir) if a.out_dir else Path(tempfile.mkdtemp(prefix="emotune_demo_"))
    data_mod.write_synthetic(work / "data", 1200, 300, a.seed)
    splits, removed = data_mod.load_splits(work / "data")
    print(f"synthetic data: train {len(splits['train'])} (removed {removed} leaked rows), test {len(splits['test'])}")
    runs = work / "runs"
    key = {e.text: e.label for e in splits["test"]}
    shots = data_mod.few_shot_sample(splits["train"], 3, a.seed)
    systems = [
        (KeywordBaseline(), "baseline", False),
        (TfidfBaseline(seed=a.seed).fit(splits["train"]), "baseline", False),
        (ChatClassifier(SimulatedChatLLM(key, seed=a.seed), "zero_shot"), "zero_shot", True),
        (ChatClassifier(SimulatedChatLLM(key, seed=a.seed), "few_shot", shots), "few_shot", True),
    ]
    for clf, regime, sim in systems:
        cfg = RunConfig(system=clf.name, regime=regime, seed=a.seed, dataset="synthetic", simulated=sim,
                        shots_per_class=3 if regime == "few_shot" else 0, decoding="greedy" if sim else "n/a")
        _evaluate_and_save(clf.name.replace(":", "_"), cfg, splits["test"], clf, runs)
    pa, pb = load_predictions(runs / "tfidf-logreg"), load_predictions(runs / "keyword")
    cmp = metrics.compare([LABELS.index(r["true"]) for r in pa], [r["pred"] for r in pa], [r["pred"] for r in pb])
    lo, hi = cmp["accuracy_diff_ci"]
    print(f"tfidf-logreg minus keyword: accuracy {cmp['accuracy_diff']:+.3f} [{lo:+.3f}, {hi:+.3f}], "
          f"McNemar p = {cmp['mcnemar']['p_value']:.2g}")
    print("synthetic data and a simulated chat model: these numbers show that the pipeline runs, not model quality")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="emotune", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def data_args(sp):
        sp.add_argument("--data", help="folder with train.csv, validation.csv, test.csv (default EMOTUNE_DATA_DIR)")
        sp.add_argument("--split", choices=["validation", "test"], default="test")
        sp.add_argument("--limit", type=int, help="use only the first N items")
        sp.add_argument("--run", help="run folder name under EMOTUNE_RUNS_DIR")

    def model_args(sp):
        sp.add_argument("--backend", choices=["simulated", "openai", "hf-score", "hf-greedy"], required=True)
        sp.add_argument("--model", help="model name from the models file (hf backends)")
        sp.add_argument("--models-file", default="configs/models.toml")
        sp.add_argument("--adapter", help="LoRA adapter folder: predictions come from the fine-tuned model")
        sp.add_argument("--load-in-4bit", action="store_true")

    s = sub.add_parser("synth", help="write a synthetic dataset")
    s.add_argument("--out", default="data/synthetic")
    s.add_argument("--n-train", type=int, default=1200)
    s.add_argument("--n-eval", type=int, default=300)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_synth)

    d = sub.add_parser("download", help="write dair-ai/emotion as CSV files (data extra)")
    d.add_argument("--out", default="data/emotion")
    d.add_argument("--revision", help="dataset commit hash to pin")
    d.set_defaults(func=cmd_download)

    b = sub.add_parser("baseline", help="evaluate a baseline classifier")
    data_args(b)
    b.add_argument("--model", choices=["tfidf", "keyword"], default="tfidf")
    b.set_defaults(func=cmd_baseline)

    m = sub.add_parser("llm", help="evaluate a language model (zero-shot, few-shot or with an adapter)")
    data_args(m)
    model_args(m)
    m.add_argument("--regime", choices=["zero_shot", "few_shot"], default="zero_shot")
    m.add_argument("--shots", type=int, default=3, help="examples per label for few_shot")
    m.set_defaults(func=cmd_llm)

    f = sub.add_parser("finetune", help="LoRA fine-tuning with a completion-only loss (finetune extra)")
    f.add_argument("--data")
    f.add_argument("--model", required=True)
    f.add_argument("--models-file", default="configs/models.toml")
    f.add_argument("--out")
    f.set_defaults(func=cmd_finetune)

    c = sub.add_parser("compare", help="paired comparison of two labelled runs (McNemar + bootstrap)")
    c.add_argument("run_a")
    c.add_argument("run_b")
    c.set_defaults(func=cmd_compare)

    g = sub.add_parser("agreement", help="agreement and Cohen's kappa of two unlabelled runs")
    g.add_argument("run_a")
    g.add_argument("run_b")
    g.set_defaults(func=cmd_agreement)

    pr = sub.add_parser("predict", help="predict an unlabelled CSV with a text column")
    model_args(pr)
    pr.add_argument("--input", required=True)
    pr.add_argument("--limit", type=int)
    pr.add_argument("--run", required=True)
    pr.set_defaults(func=cmd_predict)

    y = sub.add_parser("collect-youtube", help="collect comment text (no author data) for given video ids")
    y.add_argument("--videos", nargs="+", required=True)
    y.add_argument("--max-per-video", type=int, default=200)
    y.add_argument("--out", default="data/youtube/comments.csv")
    y.set_defaults(func=cmd_collect_youtube)

    ls = sub.add_parser("label-sample", help="seeded sample of comments for human labelling")
    ls.add_argument("--comments", required=True)
    ls.add_argument("--n", type=int, default=300)
    ls.add_argument("--seed", type=int, default=0)
    ls.add_argument("--out", default="data/youtube/to_label.csv")
    ls.set_defaults(func=cmd_label_sample)

    dm = sub.add_parser("demo", help="offline demo on synthetic data")
    dm.add_argument("--seed", type=int, default=0)
    dm.add_argument("--out-dir")
    dm.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

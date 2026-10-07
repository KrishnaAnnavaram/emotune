# Data

This repository commits no dataset, no scraped text, no model weights and no run outputs.

## Benchmark data: dair-ai/emotion

| Item | Value |
|---|---|
| Source name | `dair-ai/emotion` (English Twitter messages with six emotion labels) |
| URL | https://huggingface.co/datasets/dair-ai/emotion |
| License / terms | The dataset card limits use to educational and research purposes. Cite Saravia et al., "CARER: Contextualized Affect Representations for Emotion Recognition", EMNLP 2018 |
| Size | 16,000 train, 2,000 validation, 2,000 test rows |
| Labels | 0 sadness, 1 joy, 2 love, 3 anger, 4 fear, 5 surprise (the `ClassLabel` order, also `emotune.labels.LABELS`) |

Download it once (extra `data`):

```bash
pip install -e ".[data]"
emotune download --out data/emotion --revision <dataset commit hash>
```

## Expected files

| File | Columns |
|---|---|
| `train.csv` | `text`, `label` (integer 0..5) |
| `validation.csv` | `text`, `label` |
| `test.csv` | `text`, `label` |

The loader refuses other columns, empty texts and labels outside 0..5. It removes each train row whose
text (case-folded) is also in validation or test, and it prints the count.

## YouTube comments (optional, domain shift)

`emotune collect-youtube` uses the YouTube Data API v3 with `YOUTUBE_API_KEY`. It keeps only the
comment text and a salted SHA-256 id (`EMOTUNE_HASH_SALT`). It drops author names, channel ids, dates
and like counts. Respect the YouTube API Terms of Service. Do not publish the collected text.
`emotune label-sample` makes a seeded sample with an empty `label` column for human labelling.

## Synthetic data

`emotune synth --out data/synthetic` writes short template sentences with one cue word per label and 10%
random labels. The demo and the tests use only this data.

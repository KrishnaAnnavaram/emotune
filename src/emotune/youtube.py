"""YouTube comment collection for a domain-shift check (YouTube Data API v3, key from the environment).

Only the comment text and a salted hash of the comment id are kept. Author
names, channel ids, dates and like counts are dropped at the source.
"""
from __future__ import annotations

import csv
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np

API = "https://www.googleapis.com/youtube/v3/commentThreads"
Transport = Callable[[str], dict]


def _get(url: str) -> dict:  # pragma: no cover - network
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - fixed https host
        return json.loads(resp.read().decode())


def hashed_id(comment_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{comment_id}".encode()).hexdigest()[:16]


def collect(video_ids: list[str], api_key: str, salt: str, max_per_video: int = 200,
            transport: Transport = _get) -> list[dict]:
    if not api_key:
        raise ValueError("set YOUTUBE_API_KEY in the environment")
    if not salt:
        raise ValueError("set EMOTUNE_HASH_SALT so that comment ids cannot be reversed")
    rows, seen = [], set()
    for vid in video_ids:
        token, got = None, 0
        while got < max_per_video:
            params = {"part": "snippet", "videoId": vid, "maxResults": min(100, max_per_video - got),
                      "textFormat": "plainText", "key": api_key}
            if token:
                params["pageToken"] = token
            body = transport(f"{API}?{urllib.parse.urlencode(params)}")
            for item in body.get("items", []):
                top = item["snippet"]["topLevelComment"]
                text = " ".join(top["snippet"].get("textDisplay", "").split())
                hid = hashed_id(top["id"], salt)
                if text and hid not in seen:
                    seen.add(hid)
                    rows.append({"id": hid, "text": text})
                    got += 1
            token = body.get("nextPageToken")
            if not token:
                break
    return rows


def write_rows(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "text"])
        w.writeheader()
        w.writerows({"id": r["id"], "text": r["text"]} for r in rows)
    return path


def label_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """A seeded random sample for human labelling, with an empty ``label`` column to fill in."""
    idx = np.random.default_rng(seed).permutation(len(rows))[:n]
    return [{"id": rows[i]["id"], "text": rows[i]["text"], "label": ""} for i in sorted(idx)]

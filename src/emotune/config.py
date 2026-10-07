"""Settings from environment variables (and an optional local ``.env`` file). Secrets never live in code."""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

ENV = {
    "EMOTUNE_DATA_DIR": "data_dir",
    "EMOTUNE_RUNS_DIR": "runs_dir",
    "EMOTUNE_SEED": "seed",
    "EMOTUNE_LLM_BASE_URL": "llm_base_url",
    "EMOTUNE_LLM_MODEL": "llm_model",
    "EMOTUNE_LLM_API_KEY": "llm_api_key",
    "HF_TOKEN": "hf_token",
    "YOUTUBE_API_KEY": "youtube_api_key",
    "EMOTUNE_HASH_SALT": "hash_salt",
}


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    data_dir: Path = Path("data/emotion")
    runs_dir: Path = Path("runs")
    seed: int = 0
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str | None = Field(default=None, repr=False)
    hf_token: str | None = Field(default=None, repr=False)
    youtube_api_key: str | None = Field(default=None, repr=False)
    hash_salt: str | None = Field(default=None, repr=False)


def load_dotenv(path: str | os.PathLike = ".env") -> list[str]:
    p = Path(path)
    if not p.is_file():
        return []
    loaded = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (x.strip() for x in line.split("=", 1))
        if key in ENV and value and not os.environ.get(key):
            os.environ[key] = value.strip("\"'")
            loaded.append(key)
    return loaded


def settings_from_env() -> Settings:
    return Settings.model_validate({f: os.environ[v] for v, f in ENV.items() if os.environ.get(v)})

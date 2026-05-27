from __future__ import annotations

import os
from pathlib import Path

from config.settings import DATA_DIR, LLM_MODEL, LLM_PROVIDER

EVAL_DIR = DATA_DIR / "eval"
GOLD_QA_PATH = EVAL_DIR / "gold_qa.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
SUMMARIES_DIR = EVAL_DIR / "summaries"

for _d in [EVAL_DIR, RESULTS_DIR, SUMMARIES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

PASS_BAND_THRESHOLD = 7
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", LLM_PROVIDER)
JUDGE_MODEL = os.getenv("JUDGE_MODEL", LLM_MODEL)
JUDGE_TIMEOUT_SECONDS = 45
JUDGE_MAX_RETRIES = 2


def resolve_path(path: str | Path | None, default: Path) -> Path:
    if path is None:
        return default
    return Path(path)

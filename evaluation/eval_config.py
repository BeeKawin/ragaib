from __future__ import annotations

import os
from pathlib import Path

from config.settings import DATA_DIR

EVAL_DIR = DATA_DIR / "eval"
GOLD_QA_PATH = EVAL_DIR / "gold_qa.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
SUMMARIES_DIR = EVAL_DIR / "summaries"

for _d in [EVAL_DIR, RESULTS_DIR, SUMMARIES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

METRIC_WEIGHTS = {
    "correctness": 0.35,
    "groundedness": 0.25,
    "completeness": 0.20,
    "clarity": 0.10,
    "safety": 0.10,
}

PASS_THRESHOLD = 70.0
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-2.5-flash-lite")
JUDGE_TIMEOUT_SECONDS = 45
JUDGE_MAX_RETRIES = 2


def resolve_path(path: str | Path | None, default: Path) -> Path:
    if path is None:
        return default
    return Path(path)

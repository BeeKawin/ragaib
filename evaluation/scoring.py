from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Any

from .eval_config import PASS_BAND_THRESHOLD
from .judge import JudgeResult


@dataclass
class ScoredItem:
    item_id: str
    subject: str
    grade: str
    question: str
    reference_answer: str
    preferred_answer_type: str
    language: str
    keypoints: str
    model_answer: str
    retrieved_context: str
    scores: JudgeResult
    overall_band: int


def _mean(values: list[float]) -> float:
    return round(mean(values), 2) if values else 0.0


def build_summary(results: list[ScoredItem]) -> dict[str, Any]:
    metric_values = {
        "correctness": [r.scores.correctness for r in results],
        "groundedness": [r.scores.groundedness for r in results],
        "completeness": [r.scores.completeness for r in results],
        "clarity": [r.scores.clarity for r in results],
        "type_alignment": [r.scores.type_alignment for r in results],
    }
    bands = [r.overall_band for r in results]

    by_subject: dict[str, list[float]] = defaultdict(list)
    by_grade: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_subject[r.subject].append(r.overall_band)
        by_grade[r.grade].append(r.overall_band)

    passed = sum(1 for band in bands if band >= PASS_BAND_THRESHOLD)

    return {
        "count": len(results),
        "overall_band": _mean(bands),
        "pass_threshold": PASS_BAND_THRESHOLD,
        "pass_rate": round((passed / max(len(results), 1)) * 100.0, 2),
        "metric_averages": {k: _mean(v) for k, v in metric_values.items()},
        "by_subject": {k: _mean(v) for k, v in sorted(by_subject.items())},
        "by_grade": {k: _mean(v) for k, v in sorted(by_grade.items())},
    }

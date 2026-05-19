from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Any

from .eval_config import METRIC_WEIGHTS, PASS_THRESHOLD
from .judge import JudgeResult


@dataclass
class ScoredItem:
    item_id: str
    subject: str
    grade: str
    question: str
    reference_answer: str
    model_answer: str
    retrieved_context: str
    scores: JudgeResult
    weighted_score: float


def weighted_score(scores: JudgeResult, weights: dict[str, float] | None = None) -> float:
    w = weights or METRIC_WEIGHTS
    raw = (
        scores.correctness * w["correctness"]
        + scores.groundedness * w["groundedness"]
        + scores.completeness * w["completeness"]
        + scores.clarity * w["clarity"]
        + scores.safety * w["safety"]
    )
    return round((raw / 5.0) * 100.0, 2)


def _mean(values: list[float]) -> float:
    return round(mean(values), 2) if values else 0.0


def build_summary(results: list[ScoredItem]) -> dict[str, Any]:
    metric_values = {
        "correctness": [r.scores.correctness for r in results],
        "groundedness": [r.scores.groundedness for r in results],
        "completeness": [r.scores.completeness for r in results],
        "clarity": [r.scores.clarity for r in results],
        "safety": [r.scores.safety for r in results],
    }
    weighted = [r.weighted_score for r in results]

    by_subject: dict[str, list[float]] = defaultdict(list)
    by_grade: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_subject[r.subject].append(r.weighted_score)
        by_grade[r.grade].append(r.weighted_score)

    passed = sum(1 for s in weighted if s >= PASS_THRESHOLD)

    return {
        "count": len(results),
        "overall_weighted_score": _mean(weighted),
        "pass_threshold": PASS_THRESHOLD,
        "pass_rate": round((passed / max(len(results), 1)) * 100.0, 2),
        "metric_averages": {k: _mean(v) for k, v in metric_values.items()},
        "by_subject": {k: _mean(v) for k, v in sorted(by_subject.items())},
        "by_grade": {k: _mean(v) for k, v in sorted(by_grade.items())},
    }

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import httpx

from .eval_config import GOLD_QA_PATH, resolve_path


def _load_rows(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if path.suffix.lower() == ".csv":
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                rows.append({k: (v or "").strip() for k, v in raw.items()})
                if limit is not None and len(rows) >= limit:
                    break
    else:
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                rows.append({k: str(v or "").strip() for k, v in raw.items()})
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def _normalize_subject(value: str) -> str:
    aliases = {"mathematics": "math", "math": "math"}
    return aliases.get(value.lower(), value)


def _matches_expected(source: dict[str, Any], expected: set[str]) -> bool:
    topic = str(source.get("topic", "")).strip().lower()
    title = str(source.get("source_title", "")).strip().lower()
    section = str(source.get("section", "")).strip().lower()
    haystack = f"{topic} {title} {section}"
    for term in expected:
        if not term:
            continue
        if term in haystack:
            return True
        words = [word for word in term.split() if len(word) >= 4]
        if words and all(word in haystack for word in words):
            return True
    return False


def _expected_terms(row: dict[str, str]) -> set[str]:
    terms = {
        row.get("topic", "").strip().lower(),
        row.get("subtopic", "").strip().lower(),
        row.get("source_topic", "").strip().lower(),
    }
    return {term for term in terms if term}


def run_retrieval_eval(
    dataset_path: str | Path | None = None,
    api_base: str = "http://127.0.0.1:8000",
    top_k: int = 5,
    limit: int | None = None,
) -> dict[str, Any]:
    path = resolve_path(dataset_path, GOLD_QA_PATH)
    rows = _load_rows(path, limit=limit)
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")

    details = []
    by_grade: dict[str, list[dict[str, bool]]] = defaultdict(list)
    by_topic: dict[str, list[dict[str, bool]]] = defaultdict(list)

    with httpx.Client(base_url=api_base.rstrip("/"), timeout=60) as client:
        for row in rows:
            payload = {
                "message": row["question"],
                "subject": _normalize_subject(row.get("subject", "")),
                "grade": row.get("grade") or None,
                "top_k": top_k,
            }
            response = client.post("/chat/sources", json=payload)
            response.raise_for_status()
            sources = response.json().get("sources", [])
            expected = _expected_terms(row)

            top1_hit = bool(sources) and _matches_expected(sources[0], expected)
            hit = any(_matches_expected(source, expected) for source in sources)
            status = {
                "retrieved": bool(sources),
                "top1_hit": top1_hit,
                "hit": hit,
            }
            by_grade[row.get("grade", "")].append(status)
            by_topic[row.get("topic", "")].append(status)
            details.append(
                {
                    "id": row.get("id", ""),
                    "grade": row.get("grade", ""),
                    "expected": sorted(expected),
                    "retrieved_topics": [source.get("topic", "") for source in sources],
                    **status,
                }
            )

    def pct(count: int, total: int) -> float:
        return round((count / max(total, 1)) * 100.0, 2)

    def summarize(items: list[dict[str, bool]]) -> dict[str, float | int]:
        total = len(items)
        return {
            "count": total,
            "retrieved_rate": pct(sum(item["retrieved"] for item in items), total),
            "top1_topic_hit_rate": pct(sum(item["top1_hit"] for item in items), total),
            "topic_hit_rate": pct(sum(item["hit"] for item in items), total),
        }

    missed = [detail for detail in details if not detail["hit"]]
    topic_counts = Counter(topic for detail in details for topic in detail["retrieved_topics"])

    return {
        "dataset_path": str(path),
        "api_base": api_base,
        "top_k": top_k,
        **summarize(details),
        "by_grade": {grade: summarize(items) for grade, items in sorted(by_grade.items())},
        "by_topic": {topic: summarize(items) for topic, items in sorted(by_topic.items())},
        "most_retrieved_topics": topic_counts.most_common(10),
        "misses": missed[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval via the running API")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    summary = run_retrieval_eval(
        dataset_path=args.dataset,
        api_base=args.api_base,
        top_k=args.top_k,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

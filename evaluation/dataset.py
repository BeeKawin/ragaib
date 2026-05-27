from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .judge import normalize_answer_type


@dataclass
class EvalItem:
    id: str
    subject: Optional[str]
    grade: Optional[str]
    question: str
    reference_answer: str
    preferred_answer_type: str = "general"
    language: str = ""
    keypoints: str = ""
    notes: str = ""


def _require_str(raw: dict, key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' must be a non-empty string")
    return value.strip()


def _first_str(raw: dict, keys: list[str]) -> str:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    joined = "' or '".join(keys)
    raise ValueError(f"'{joined}' must be a non-empty string")


def _optional_str(raw: dict, key: str) -> str:
    value = raw.get(key)
    return value.strip() if isinstance(value, str) else ""


def _normalize_subject(value: object) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None

    raw = value.strip()
    aliases = {
        "mathematics": "math",
        "math": "math",
        "physics": "physics",
        "chemistry": "chemistry",
        "biology": "biology",
    }
    return aliases.get(raw.lower(), raw)


def _notes_from_csv(raw: dict) -> str:
    parts = []
    for key in [
        "topic",
        "subtopic",
        "language",
        "difficulty",
        "question_type",
        "type",
        "expected_keywords",
        "keypoints",
        "source_topic",
    ]:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(f"{key}={value.strip()}")
    return "; ".join(parts)


def _item_from_raw(raw: dict, row_label: str) -> EvalItem:
    try:
        return EvalItem(
            id=_require_str(raw, "id"),
            subject=_normalize_subject(raw.get("subject")),
            grade=(raw.get("grade") or None),
            question=_require_str(raw, "question"),
            reference_answer=_first_str(raw, ["reference_answer", "answer"]),
            preferred_answer_type=normalize_answer_type(
                raw.get("preferred_answer_type") or raw.get("type")
            ),
            language=_optional_str(raw, "language"),
            keypoints=_optional_str(raw, "keypoints"),
            notes=(raw.get("notes") or _notes_from_csv(raw)).strip(),
        )
    except ValueError as exc:
        raise ValueError(f"Invalid row at {row_label}: {exc}") from exc


def _load_jsonl(path: Path, limit: Optional[int]) -> list[EvalItem]:
    items: list[EvalItem] = []
    # utf-8-sig tolerates BOM-prefixed files (common on Windows editors/PowerShell).
    with open(path, encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc

            items.append(_item_from_raw(raw, f"line {line_no}"))
            if limit is not None and len(items) >= limit:
                break
    return items


def _load_csv(path: Path, limit: Optional[int]) -> list[EvalItem]:
    items: list[EvalItem] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row_no, raw in enumerate(reader, 2):
            items.append(_item_from_raw(raw, f"CSV row {row_no}"))
            if limit is not None and len(items) >= limit:
                break
    return items


def load_eval_dataset(path: Path, limit: Optional[int] = None) -> list[EvalItem]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".csv":
        items = _load_csv(path, limit)
    else:
        items = _load_jsonl(path, limit)

    if not items:
        raise ValueError(f"Dataset is empty: {path}")

    return items

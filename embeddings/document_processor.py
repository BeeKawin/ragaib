"""
embeddings/document_processor.py
──────────────────────────────────
Loads raw scraped JSON → cleans → splits into overlapping chunks →
returns LangChain Document objects with rich metadata.

Chunking strategy
─────────────────
• Each page section is chunked independently (topical coherence).
• Equations, key-terms, and examples get their own dedicated chunks
  so they can be retrieved as precise answers.
• All chunks carry filterable metadata (subject, grade, topic, …).
"""

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

from config.settings import CHUNK_OVERLAP, CHUNK_SIZE, RAW_DIR, SUBJECT_META


# ── Text cleaning ─────────────────────────────────────────────────────────────

_WS = re.compile(r"\s{3,}")

REQUIRED_RAW_FIELDS = (
    "subject",
    "grade",
    "topic",
    "url",
    "title",
    "sections",
    "key_terms",
    "equations",
    "examples",
    "raw_text",
)
STRING_RAW_FIELDS = ("subject", "grade", "topic", "url", "title", "raw_text")
LIST_RAW_FIELDS = ("sections", "key_terms", "equations", "examples")

BLOCKED_PAGE_PHRASES = (
    "file not found",
    "cookies are disabled",
    "no results found",
    "there isn't any ck-12 content",
)
BOILERPLATE_MARKERS = (
    "student sign up",
    "sign in with google",
    "sign in with microsoft",
    "already have an account",
    "please wait",
    "got it",
)


@dataclass
class RawValidationIssue:
    path: Path
    subject: str
    grade: str
    topic: str
    reasons: list[str]


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = _WS.sub("\n\n", text)
    return text.strip()


def validate_raw_page(raw: dict) -> list[str]:
    """Return validation issues for a raw page; an empty list means usable."""
    reasons: list[str] = []

    if not isinstance(raw, dict):
        return ["raw page must be a JSON object"]

    for field in REQUIRED_RAW_FIELDS:
        if field not in raw:
            reasons.append(f"missing required field: {field}")

    for field in STRING_RAW_FIELDS:
        value = raw.get(field)
        if not isinstance(value, str) or not value.strip():
            reasons.append(f"field must be a non-empty string: {field}")

    for field in LIST_RAW_FIELDS:
        if field in raw and not isinstance(raw.get(field), list):
            reasons.append(f"field must be a list: {field}")

    title = str(raw.get("title", ""))
    raw_text = str(raw.get("raw_text", ""))
    combined = f"{title}\n{raw_text}".lower()

    for phrase in BLOCKED_PAGE_PHRASES:
        if phrase in combined:
            reasons.append(f"blocked/error page phrase found: {phrase}")

    marker_count = sum(1 for marker in BOILERPLATE_MARKERS if marker in combined)
    if marker_count >= 4 and len(raw_text) < 1200:
        reasons.append("page appears to be mostly login/support boilerplate")

    sections = raw.get("sections") if isinstance(raw.get("sections"), list) else []
    has_section_content = any(
        isinstance(sec, dict) and len(str(sec.get("content", "")).strip()) >= 30
        for sec in sections
    )
    has_aux_content = any(
        isinstance(raw.get(field), list) and len(raw.get(field, [])) > 0
        for field in ("key_terms", "equations", "examples")
    )
    if not has_section_content and not has_aux_content and len(raw_text.strip()) < 80:
        reasons.append("no usable educational content found")

    return reasons


# ── Splitter ──────────────────────────────────────────────────────────────────

def _make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


# ── Metadata ──────────────────────────────────────────────────────────────────

def _meta(raw: dict, section: str, idx: int) -> dict:
    return {
        "subject":         raw["subject"],
        "grade":           raw["grade"],
        "topic":           raw["topic"],
        "section":         section,
        "chunk_index":     idx,
        "source_url":      raw["url"],
        "source_title":    raw["title"],
        "scraped_at":      raw.get("scraped_at", ""),
        "subject_display": SUBJECT_META.get(raw["subject"], {}).get("display", raw["subject"]),
        "has_equations":   bool(raw.get("equations")),
        "is_example":      "example" in section.lower(),
        "is_definition":   any(
            kw in section.lower()
            for kw in ["definition", "key term", "vocabulary", "glossary"]
        ),
    }


# ── Processor ─────────────────────────────────────────────────────────────────

class DocumentProcessor:
    """
    Converts raw JSON scraped pages into LangChain Document chunks.

    Usage
    ─────
        proc = DocumentProcessor()
        docs = proc.process_all()
        docs = proc.process_all(subjects=["math"], grades=["M4"])
        stats = proc.get_stats(docs)
    """

    def __init__(self):
        self.splitter = _make_splitter()
        self.last_validation_summary: dict = {}

    # ── single page ──────────────────────────────────────────────────────────

    def process_page(self, json_path: Path) -> list:
        with open(json_path, encoding="utf-8") as f:
            raw = json.load(f)

        docs  = []
        idx   = 0

        # ── main content sections ────────────────────────────────────────────
        sections = raw.get("sections") or [
            {"heading": "Content", "content": raw.get("raw_text", "")}
        ]
        for sec in sections:
            heading = sec.get("heading", "Content")
            content = clean_text(sec.get("content", ""))
            if not content or len(content) < 30:
                continue
            for chunk in self.splitter.split_text(f"## {heading}\n\n{content}"):
                docs.append(Document(page_content=chunk, metadata=_meta(raw, heading, idx)))
                idx += 1

        # ── key terms ────────────────────────────────────────────────────────
        terms = raw.get("key_terms", [])
        if terms:
            text = "Key Terms:\n" + "\n".join(f"• {t}" for t in terms)
            m = _meta(raw, "Key Terms", idx)
            m["is_definition"] = True
            docs.append(Document(page_content=text, metadata=m))
            idx += 1

        # ── equations ────────────────────────────────────────────────────────
        eqs = raw.get("equations", [])
        if eqs:
            text = "Equations:\n" + "\n".join(f"  {e}" for e in eqs[:30])
            m = _meta(raw, "Equations", idx)
            m["has_equations"] = True
            docs.append(Document(page_content=text, metadata=m))
            idx += 1

        # ── worked examples ───────────────────────────────────────────────────
        for ex in raw.get("examples", []):
            ex_clean = clean_text(ex)
            if len(ex_clean) < 30:
                continue
            for chunk in self.splitter.split_text(ex_clean):
                m = _meta(raw, "Worked Example", idx)
                m["is_example"] = True
                docs.append(Document(page_content=chunk, metadata=m))
                idx += 1

        logger.debug(f"  {json_path.name}: {len(docs)} chunks")
        return docs

    # ── batch ─────────────────────────────────────────────────────────────────

    def _json_files(self, subjects=None, grades=None) -> Generator:
        for subj_dir in sorted(RAW_DIR.iterdir()):
            if not subj_dir.is_dir():
                continue
            if subjects and subj_dir.name not in subjects:
                continue
            for grade_dir in sorted(subj_dir.iterdir()):
                if not grade_dir.is_dir():
                    continue
                if grades and grade_dir.name not in grades:
                    continue
                yield from sorted(grade_dir.glob("*.json"))

    @staticmethod
    def _path_subject_grade(path: Path) -> tuple[str, str]:
        return path.parent.parent.name, path.parent.name

    def validate_files(self, files: list[Path]) -> tuple[list[Path], list[RawValidationIssue]]:
        valid: list[Path] = []
        invalid: list[RawValidationIssue] = []

        for path in files:
            subject, grade = self._path_subject_grade(path)
            topic = path.stem
            try:
                with open(path, encoding="utf-8") as f:
                    raw = json.load(f)
                subject = str(raw.get("subject") or subject)
                grade = str(raw.get("grade") or grade)
                topic = str(raw.get("topic") or topic)
                reasons = validate_raw_page(raw)
            except Exception as exc:
                reasons = [f"failed to read/parse JSON: {exc}"]

            if reasons:
                invalid.append(
                    RawValidationIssue(
                        path=path,
                        subject=subject,
                        grade=grade,
                        topic=topic,
                        reasons=reasons,
                    )
                )
            else:
                valid.append(path)

        return valid, invalid

    @staticmethod
    def _validation_summary(
        files: list[Path],
        valid: list[Path],
        invalid: list[RawValidationIssue],
    ) -> dict:
        valid_subjects = Counter()
        valid_grades = Counter()
        for path in valid:
            subject, grade = DocumentProcessor._path_subject_grade(path)
            valid_subjects[subject] += 1
            valid_grades[grade] += 1

        invalid_subjects = Counter(issue.subject for issue in invalid)
        invalid_grades = Counter(issue.grade for issue in invalid)

        return {
            "total_files": len(files),
            "valid_files": len(valid),
            "invalid_files": len(invalid),
            "valid_by_subject": dict(sorted(valid_subjects.items())),
            "invalid_by_subject": dict(sorted(invalid_subjects.items())),
            "valid_by_grade": dict(sorted(valid_grades.items())),
            "invalid_by_grade": dict(sorted(invalid_grades.items())),
            "invalid_details": [
                {
                    "path": str(issue.path),
                    "subject": issue.subject,
                    "grade": issue.grade,
                    "topic": issue.topic,
                    "reasons": issue.reasons,
                }
                for issue in invalid
            ],
        }

    def process_all(self, subjects=None, grades=None, validate: bool = True) -> list:
        files = list(self._json_files(subjects, grades))
        if not files:
            logger.warning("No scraped JSON files found. Run the scraper first.")
            return []

        if validate:
            valid_files, invalid = self.validate_files(files)
            self.last_validation_summary = self._validation_summary(files, valid_files, invalid)
            logger.info(
                "Raw validation: "
                f"{self.last_validation_summary['valid_files']} valid | "
                f"{self.last_validation_summary['invalid_files']} invalid | "
                f"{self.last_validation_summary['total_files']} total"
            )
            for issue in invalid[:10]:
                logger.warning(
                    f"Skipping invalid raw page [{issue.subject}|{issue.grade}] "
                    f"{issue.topic}: {'; '.join(issue.reasons)}"
                )
            if len(invalid) > 10:
                logger.warning(f"...and {len(invalid) - 10} more invalid raw pages")
            files = valid_files
        else:
            self.last_validation_summary = {
                "total_files": len(files),
                "valid_files": len(files),
                "invalid_files": 0,
            }

        if not files:
            logger.warning("No valid raw JSON files found after validation.")
            return []

        logger.info(f"Processing {len(files)} pages...")
        all_docs = []
        for path in files:
            try:
                all_docs.extend(self.process_page(path))
            except Exception as exc:
                logger.error(f"Error processing {path}: {exc}")

        logger.success(f"Done. {len(all_docs)} chunks from {len(files)} pages.")
        return all_docs

    # ── stats ─────────────────────────────────────────────────────────────────

    def get_stats(self, docs: list) -> dict:
        return {
            "total_chunks": len(docs),
            "by_subject":   dict(Counter(d.metadata["subject"] for d in docs)),
            "by_grade":     dict(Counter(d.metadata["grade"]   for d in docs)),
            "avg_chunk_len": int(
                sum(len(d.page_content) for d in docs) / max(len(docs), 1)
            ),
            "validation": self.last_validation_summary,
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json as _json

    p = argparse.ArgumentParser(description="Process scraped CK-12 JSON files")
    p.add_argument("--subjects", nargs="*")
    p.add_argument("--grades",   nargs="*")
    p.add_argument("--no-validate", action="store_true")
    args = p.parse_args()

    proc  = DocumentProcessor()
    docs  = proc.process_all(
        subjects=args.subjects,
        grades=args.grades,
        validate=not args.no_validate,
    )
    stats = proc.get_stats(docs)
    print(_json.dumps(stats, indent=2))

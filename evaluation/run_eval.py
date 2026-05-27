from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from retrieval.rag_chain import get_rag_chain

from .eval_config import GOLD_QA_PATH, RESULTS_DIR, SUMMARIES_DIR, resolve_path
from .dataset import EvalItem, load_eval_dataset
from .judge import GeminiJudge
from .scoring import ScoredItem, build_summary


def _iso_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _format_context(docs: list[dict]) -> str:
    if not docs:
        return "No retrieved sources"
    lines = []
    for i, d in enumerate(docs, 1):
        lines.append(f"[{i}] {d.get('subject','')}|{d.get('grade','')} {d.get('topic','')} > {d.get('section','')}")
        url = d.get("url", "")
        if url:
            lines.append(f"URL: {url}")
        content = str(d.get("content", "")).strip()
        if content:
            lines.append("Content:")
            lines.append(content)
    return "\n".join(lines)


def _run_item(item: EvalItem, judge: GeminiJudge) -> ScoredItem:
    chain = get_rag_chain()
    model_answer = chain.ask(
        item.question,
        subject=item.subject,
        grade=item.grade,
        preferred_answer_type=item.preferred_answer_type,
        language=item.language,
        keypoints=item.keypoints,
    )
    context_docs = chain.get_context_docs(item.question, subject=item.subject, grade=item.grade)
    context = _format_context(context_docs)
    judge_result = judge.score(
        question=item.question,
        reference_answer=item.reference_answer,
        model_answer=model_answer,
        context=context,
        preferred_answer_type=item.preferred_answer_type,
    )

    return ScoredItem(
        item_id=item.id,
        subject=item.subject or "",
        grade=item.grade or "",
        question=item.question,
        reference_answer=item.reference_answer,
        preferred_answer_type=item.preferred_answer_type,
        language=item.language,
        keypoints=item.keypoints,
        model_answer=model_answer,
        retrieved_context=context,
        scores=judge_result,
        overall_band=judge_result.overall_band,
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _emit_console(summary: dict[str, Any]) -> None:
    logger.info("Eval complete")
    logger.info(f"Items: {summary['count']}")
    logger.info(f"Overall band: {summary['overall_band']}")
    logger.info(f"Pass rate: {summary['pass_rate']}% (threshold={summary['pass_threshold']})")
    metrics = summary.get("metric_averages", {})
    logger.info(
        "Metrics avg | "
        + " | ".join(
            f"{k}={v}" for k, v in metrics.items()
        )
    )


def run_evaluation(
    dataset_path: Optional[str | Path] = None,
    limit: Optional[int] = None,
    judge_model: Optional[str] = None,
) -> dict[str, Any]:
    ds_path = resolve_path(dataset_path, GOLD_QA_PATH)
    items = load_eval_dataset(ds_path, limit=limit)

    judge = GeminiJudge(model=judge_model) if judge_model else GeminiJudge()

    scored_items: list[ScoredItem] = []
    for item in items:
        logger.info(f"Evaluating {item.id} [{item.subject}|{item.grade}]")
        scored_items.append(_run_item(item, judge))

    summary = build_summary(scored_items)
    timestamp = _iso_ts()

    results_path = RESULTS_DIR / f"{timestamp}.jsonl"
    summaries_path = SUMMARIES_DIR / f"{timestamp}.json"

    jsonl_rows: list[dict[str, Any]] = []
    for s in scored_items:
        row = {
            "id": s.item_id,
            "subject": s.subject,
            "grade": s.grade,
            "question": s.question,
            "reference_answer": s.reference_answer,
            "preferred_answer_type": s.preferred_answer_type,
            "language": s.language,
            "keypoints": s.keypoints,
            "model_answer": s.model_answer,
            "retrieved_context": s.retrieved_context,
            "scores": asdict(s.scores),
            "overall_band": s.overall_band,
        }
        jsonl_rows.append(row)

    _write_jsonl(results_path, jsonl_rows)

    summary_blob = {
        "timestamp": timestamp,
        "dataset_path": str(ds_path),
        "judge_model": judge.model,
        "results_path": str(results_path),
        **summary,
    }
    with open(summaries_path, "w", encoding="utf-8") as f:
        json.dump(summary_blob, f, ensure_ascii=False, indent=2)

    _emit_console(summary_blob)
    return summary_blob


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run offline LLM-based RAG evaluation")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--judge-model", type=str, default=None)
    return p


def main() -> None:
    args = _arg_parser().parse_args()
    summary = run_evaluation(
        dataset_path=args.dataset,
        limit=args.limit,
        judge_model=args.judge_model,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

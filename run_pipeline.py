"""
run_pipeline.py
────────────────
One-shot runner: scrape → process → index → (optionally) start API.

Usage
─────
    # Full pipeline (all subjects, all grades)
    python run_pipeline.py

    # Subset
    python run_pipeline.py --subjects math physics --grades M4 M5

    # Skip scraping (already done)
    python run_pipeline.py --skip-scrape

    # Run API after indexing
    python run_pipeline.py --serve
"""

import argparse
import asyncio
import sys

from loguru import logger


def parse_args():
    p = argparse.ArgumentParser(description="EduRAG pipeline runner")
    p.add_argument("--subjects",     nargs="*", help="Subjects to process")
    p.add_argument("--grades",       nargs="*", help="Grades to process (M4 M5 M6)")
    p.add_argument("--concurrency",  type=int, default=3, help="Scraper concurrency")
    p.add_argument("--force-scrape", action="store_true", help="Re-scrape cached pages")
    p.add_argument("--skip-scrape",  action="store_true", help="Skip scraping step")
    p.add_argument("--serve",        action="store_true", help="Start API after indexing")
    p.add_argument("--port",         type=int, default=8000)
    return p.parse_args()


async def step_scrape(subjects, grades, concurrency, force):
    from scraper.openstax_scraper import run_scraper
    logger.info("━━━ Step 1: Scraping OpenStax ━━━")
    pages = await run_scraper(
        subjects=subjects,
        grades=grades,
        concurrency=concurrency,
        force=force,
    )
    logger.success(f"Scraped {len(pages)} pages.")


def step_process(subjects, grades):
    from embeddings.document_processor import DocumentProcessor
    import json

    logger.info("━━━ Step 2: Processing documents ━━━")
    proc  = DocumentProcessor()
    docs  = proc.process_all(subjects=subjects, grades=grades)
    stats = proc.get_stats(docs)
    logger.success(f"Processed: {json.dumps(stats)}")
    return docs


def step_index(docs):
    from vector_store.indexer import VectorStoreManager

    logger.info("━━━ Step 3: Building vector index ━━━")
    vsm = VectorStoreManager()
    vsm.build(docs)
    stats = vsm.stats()
    logger.success(f"Index ready: {stats}")


def step_serve(port):
    import uvicorn
    logger.info(f"━━━ Step 4: Starting API on port {port} ━━━")
    uvicorn.run("chat.api:app", host="0.0.0.0", port=port, reload=False)


async def main():
    args = parse_args()

    # ── Step 1: Scrape ────────────────────────────────────────────────────────
    if not args.skip_scrape:
        await step_scrape(
            subjects=args.subjects,
            grades=args.grades,
            concurrency=args.concurrency,
            force=args.force_scrape,
        )
    else:
        logger.info("Skipping scrape step (--skip-scrape).")

    # ── Step 2: Process ───────────────────────────────────────────────────────
    docs = step_process(subjects=args.subjects, grades=args.grades)

    if not docs:
        logger.error("No documents found. Did the scraper run successfully?")
        sys.exit(1)

    # ── Step 3: Index ─────────────────────────────────────────────────────────
    step_index(docs)

    # ── Step 4: Serve (optional) ──────────────────────────────────────────────
    if args.serve:
        step_serve(args.port)
    else:
        logger.success(
            "Pipeline complete! Start the API with:\n"
            "  uvicorn chat.api:app --reload --port 8000"
        )


if __name__ == "__main__":
    asyncio.run(main())

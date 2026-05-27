import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from config.settings import CURRICULUM, GRADE_META, SUBJECT_META
from evaluation.eval_config import SUMMARIES_DIR
from evaluation.run_eval import run_evaluation
from retrieval.rag_chain import get_rag_chain
from vector_store.indexer import get_vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting EduRAG API - loading vector store...")
    get_vector_store()
    get_rag_chain()
    logger.success("API ready.")
    yield


app = FastAPI(
    title="EduRAG - Thailand HS Science & Math",
    description="RAG-powered educational chat for M4-M6 (Math, Chemistry, Physics, Biology)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    subject: Optional[str] = None
    grade: Optional[str] = None
    top_k: int = 6


class ChatResponse(BaseModel):
    answer: str
    subject: Optional[str]
    grade: Optional[str]
    sources: list[dict]


class SourcesResponse(BaseModel):
    sources: list[dict]


class EvalRunRequest(BaseModel):
    dataset: Optional[str] = None
    limit: Optional[int] = None
    judge_model: Optional[str] = None


class EvalRunResponse(BaseModel):
    timestamp: str
    dataset_path: str
    judge_model: str
    results_path: str
    count: int
    overall_band: float
    pass_threshold: int
    pass_rate: float
    metric_averages: dict
    by_subject: dict
    by_grade: dict


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    chain = get_rag_chain()
    try:
        answer = chain.ask(req.message, subject=req.subject, grade=req.grade, top_k=req.top_k)
        sources = chain.get_sources(req.message, subject=req.subject, grade=req.grade, top_k=req.top_k)
    except Exception as exc:
        logger.exception("Chat generation failed")
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc

    return ChatResponse(answer=answer, subject=req.subject, grade=req.grade, sources=sources)


@app.post("/chat/sources", response_model=SourcesResponse)
async def chat_sources(req: ChatRequest):
    chain = get_rag_chain()
    try:
        sources = chain.get_sources(req.message, subject=req.subject, grade=req.grade, top_k=req.top_k)
    except Exception as exc:
        logger.exception("Source retrieval failed")
        raise HTTPException(status_code=500, detail=f"Source retrieval failed: {exc}") from exc
    return SourcesResponse(sources=sources)


@app.get("/health")
async def health():
    vsm = get_vector_store()
    stats = vsm.stats()
    return {"status": "ok", "vector_store": stats}


@app.get("/curriculum")
async def curriculum_list(subject: Optional[str] = None, grade: Optional[str] = None):
    entries = CURRICULUM
    if subject:
        entries = [e for e in entries if e["subject"] == subject]
    if grade:
        entries = [e for e in entries if e["grade"] == grade]

    enriched = []
    for e in entries:
        visible_entry = {k: v for k, v in e.items() if k != "ck12_url"}
        enriched.append(
            {
                **visible_entry,
                "subject_display": SUBJECT_META.get(e["subject"], {}).get("display", ""),
                "grade_display": GRADE_META.get(e["grade"], {}).get("display", ""),
            }
        )
    return {"count": len(enriched), "topics": enriched}


@app.post("/eval/run", response_model=EvalRunResponse)
async def eval_run(req: EvalRunRequest):
    try:
        summary = run_evaluation(
            dataset_path=req.dataset,
            limit=req.limit,
            judge_model=req.judge_model,
        )
    except Exception as exc:
        logger.exception("Eval run failed")
        raise HTTPException(status_code=500, detail=f"Eval run failed: {exc}") from exc
    return EvalRunResponse(**summary)


@app.get("/eval/latest")
async def eval_latest():
    summary_files = sorted(SUMMARIES_DIR.glob("*.json"))
    if not summary_files:
        raise HTTPException(status_code=404, detail="No evaluation summaries found")

    latest = summary_files[-1]
    try:
        with open(latest, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read summary: {exc}") from exc

    return {"file": str(Path(latest)), "summary": payload}


@app.websocket("/chat/stream")
async def chat_stream(ws: WebSocket):
    await ws.accept()
    chain = get_rag_chain()

    try:
        while True:
            data = await ws.receive_json()
            message = data.get("message", "")
            subject = data.get("subject")
            grade = data.get("grade")
            top_k = data.get("top_k", 6)

            if not message.strip():
                await ws.send_json({"error": "Empty message"})
                continue

            async for chunk in chain.ask_stream(message, subject=subject, grade=grade, top_k=top_k):
                await ws.send_text(chunk)

            sources = chain.get_sources(message, subject=subject, grade=grade, top_k=top_k)
            await ws.send_json({"done": True, "sources": sources})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as exc:
        logger.error(f"WebSocket error: {exc}")
        await ws.send_json({"error": str(exc)})
        await ws.close()

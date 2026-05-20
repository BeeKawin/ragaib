"""
retrieval/rag_chain.py
───────────────────────
Builds the RAG chain that:
  1. Retrieves relevant chunks from ChromaDB.
  2. Injects them into a subject-aware, grade-tuned system prompt.
  3. Streams the LLM answer back to the caller.

Supports both Anthropic (Claude) and OpenAI backends.
"""

from typing import AsyncIterator, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from config.settings import (
    GEMINI_API_KEY,
    LLM_MODEL,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    SUBJECT_META,
    GRADE_META,
)
from vector_store.indexer import get_vector_store


# ── LLM factory ───────────────────────────────────────────────────────────────

def _get_llm(streaming: bool = False):
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            api_key=GEMINI_API_KEY,
            temperature=0.3,
            max_tokens=2048,
        )
    if LLM_PROVIDER == "ollama":
        from langchain_community.chat_models.ollama import ChatOllama
        return ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=LLM_MODEL,
            temperature=0.3,
            num_ctx=OLLAMA_NUM_CTX,
            num_predict=OLLAMA_NUM_PREDICT,
        )
    raise ValueError(
        f"Unsupported LLM_PROVIDER='{LLM_PROVIDER}'. Use one of: gemini, ollama"
    )


# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """You are an expert educational tutor for Thai high-school students.
You specialise in {subject_display} ({subject_display_th}) at the {grade_display} ({grade_display_th}) level.

Curriculum context (retrieved from OpenStax):
─────────────────────────────────────────
{context}
─────────────────────────────────────────

Guidelines:
- Answer clearly and accurately using the context above.
- If the context does not cover the question, say so and give a brief general answer.
- Use step-by-step explanations for problem-solving questions.
- When relevant, show equations in plain LaTeX notation: e.g. $F = ma$.
- Keep the tone friendly and encouraging for high-school students.
- You may respond in Thai (ภาษาไทย) if the student writes in Thai.
"""

HUMAN_TEMPLATE = "{question}"


def _make_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE),
        ("human",  HUMAN_TEMPLATE),
    ])


# ── Context formatter ─────────────────────────────────────────────────────────

def _format_docs(docs: list[Document]) -> str:
    if not docs:
        return "No relevant content found in the knowledge base."
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        header = (
            f"[{i}] {meta.get('subject_display', '')} › "
            f"{meta.get('topic', '')} › {meta.get('section', '')}"
        )
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _doc_to_payload(doc: Document, include_content: bool = False) -> dict:
    meta = doc.metadata
    payload = {
        "topic": meta.get("topic", ""),
        "section": meta.get("section", ""),
        "url": meta.get("source_url", ""),
        "grade": meta.get("grade", ""),
        "subject": meta.get("subject", ""),
        "source_title": meta.get("source_title", ""),
        "chunk_index": meta.get("chunk_index", 0),
    }
    if include_content:
        payload["content"] = doc.page_content
    return payload


# ── RAG chain builder ─────────────────────────────────────────────────────────

class EduRAGChain:
    """
    Retrieval-Augmented Generation chain for the education platform.

    Usage
    ─────
        chain = EduRAGChain()

        # Standard (blocking) answer
        answer = chain.ask(
            question="What is Newton's second law?",
            subject="physics",
            grade="M4",
        )

        # Streaming answer
        async for chunk in chain.ask_stream(...):
            print(chunk, end="", flush=True)
    """

    def __init__(self):
        self.vsm    = get_vector_store()
        self.prompt = _make_prompt()
        self.parser = StrOutputParser()

    # ── subject/grade display helpers ─────────────────────────────────────────

    @staticmethod
    def _display(subject: Optional[str], grade: Optional[str]) -> dict:
        s = SUBJECT_META.get(subject or "", {})
        g = GRADE_META.get(grade   or "", {})
        return {
            "subject_display":    s.get("display",    subject or "Science"),
            "subject_display_th": s.get("display_th", ""),
            "grade_display":      g.get("display",    grade   or "High School"),
            "grade_display_th":   g.get("display_th", ""),
        }

    # ── core methods ──────────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        subject:  Optional[str] = None,
        grade:    Optional[str] = None,
        top_k:    int = 6,
    ) -> str:
        """Retrieve context and return full LLM answer (blocking)."""
        docs    = self.vsm.search(question, subject=subject, grade=grade, top_k=top_k)
        context = _format_docs(docs)
        display = self._display(subject, grade)

        llm    = _get_llm(streaming=False)
        chain  = self.prompt | llm | self.parser

        logger.info(f"RAG query [{subject}|{grade}]: {question[:80]}")
        return chain.invoke({
            "question": question,
            "context":  context,
            **display,
        })

    async def ask_stream(
        self,
        question: str,
        subject:  Optional[str] = None,
        grade:    Optional[str] = None,
        top_k:    int = 6,
    ) -> AsyncIterator[str]:
        """Streaming version — yields text chunks as they arrive."""
        docs    = self.vsm.search(question, subject=subject, grade=grade, top_k=top_k)
        context = _format_docs(docs)
        display = self._display(subject, grade)

        llm   = _get_llm(streaming=True)
        chain = self.prompt | llm | self.parser

        logger.info(f"RAG stream [{subject}|{grade}]: {question[:80]}")
        async for chunk in chain.astream({
            "question": question,
            "context":  context,
            **display,
        }):
            yield chunk

    def get_sources(
        self,
        question: str,
        subject:  Optional[str] = None,
        grade:    Optional[str] = None,
        top_k:    int = 6,
    ) -> list[dict]:
        """Return source metadata only (no LLM call) — useful for citations."""
        docs = self.vsm.search(question, subject=subject, grade=grade, top_k=top_k)
        return [_doc_to_payload(d) for d in docs]

    def get_context_docs(
        self,
        question: str,
        subject:  Optional[str] = None,
        grade:    Optional[str] = None,
        top_k:    int = 6,
    ) -> list[dict]:
        """Return retrieved chunks with text content for evaluation and debugging."""
        docs = self.vsm.search(question, subject=subject, grade=grade, top_k=top_k)
        return [_doc_to_payload(d, include_content=True) for d in docs]


# ── Singleton ─────────────────────────────────────────────────────────────────

_chain_instance: Optional[EduRAGChain] = None


def get_rag_chain() -> EduRAGChain:
    global _chain_instance
    if _chain_instance is None:
        _chain_instance = EduRAGChain()
    return _chain_instance

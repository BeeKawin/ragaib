from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from config.settings import (
    GEMINI_API_KEY,
    OLLAMA_BASE_URL,
    OLLAMA_NUM_CTX,
)

from .eval_config import JUDGE_MAX_RETRIES, JUDGE_MODEL, JUDGE_PROVIDER, JUDGE_TIMEOUT_SECONDS


@dataclass
class JudgeResult:
    correctness: int
    groundedness: int
    completeness: int
    clarity: int
    safety: int
    rationale: str


RUBRIC_KEYS = ["correctness", "groundedness", "completeness", "clarity", "safety"]


def _extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Judge output did not contain a JSON object")
    blob = text[start : end + 1]
    return json.loads(blob)


def parse_judge_response(text: str) -> JudgeResult:
    payload = _extract_json_object(text)

    for key in RUBRIC_KEYS:
        value = payload.get(key)
        if not isinstance(value, int) or value < 1 or value > 5:
            raise ValueError(f"Invalid score for '{key}': {value}")

    rationale = payload.get("rationale", "")
    if not isinstance(rationale, str):
        raise ValueError("'rationale' must be a string")

    return JudgeResult(
        correctness=payload["correctness"],
        groundedness=payload["groundedness"],
        completeness=payload["completeness"],
        clarity=payload["clarity"],
        safety=payload["safety"],
        rationale=rationale.strip(),
    )


def _build_prompt(question: str, reference_answer: str, model_answer: str, context: str) -> str:
    return f"""
You are an expert evaluator for educational RAG answers.
Score the model answer against the reference and retrieved context.

Return ONLY valid JSON with keys:
- correctness (int 1-5)
- groundedness (int 1-5)
- completeness (int 1-5)
- clarity (int 1-5)
- safety (int 1-5)
- rationale (string <= 120 words)

Scoring rubric:
- correctness: factual and conceptual accuracy versus reference.
- groundedness: whether claims are supported by retrieved context.
- completeness: coverage of key points in reference.
- clarity: understandable, coherent explanation for high-school learners.
- safety: avoids harmful/misleading/overconfident guidance.

Question:
{question}

Reference Answer:
{reference_answer}

Retrieved Context:
{context}

Model Answer:
{model_answer}
""".strip()


class GeminiJudge:
    def __init__(self, model: str = JUDGE_MODEL):
        self.model = model

    def score(self, question: str, reference_answer: str, model_answer: str, context: str) -> JudgeResult:
        if JUDGE_PROVIDER == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(
                model=self.model,
                api_key=GEMINI_API_KEY,
                temperature=0,
                max_tokens=512,
                timeout=JUDGE_TIMEOUT_SECONDS,
                max_retries=JUDGE_MAX_RETRIES,
            )
        elif JUDGE_PROVIDER == "ollama":
            from langchain_community.chat_models.ollama import ChatOllama

            llm = ChatOllama(
                base_url=OLLAMA_BASE_URL,
                model=self.model,
                temperature=0,
                num_ctx=OLLAMA_NUM_CTX,
                num_predict=512,
                format="json",
                timeout=JUDGE_TIMEOUT_SECONDS,
            )
        else:
            raise ValueError(
                f"Unsupported JUDGE_PROVIDER='{JUDGE_PROVIDER}'. Use one of: gemini, ollama"
            )

        prompt = _build_prompt(question, reference_answer, model_answer, context)
        response = llm.invoke(prompt)
        text = getattr(response, "content", "")
        if isinstance(text, list):
            text = "\n".join(str(x) for x in text)
        return parse_judge_response(str(text))

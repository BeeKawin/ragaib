from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from config.settings import (
    GEMINI_API_KEY,
    OLLAMA_API_KEY,
    OLLAMA_BASE_URL,
    OLLAMA_NUM_CTX,
    OPENROUTER_API_KEY,
    OPENROUTER_APP_NAME,
    OPENROUTER_BASE_URL,
    OPENROUTER_SITE_URL,
)

from .eval_config import JUDGE_MAX_RETRIES, JUDGE_MODEL, JUDGE_PROVIDER, JUDGE_TIMEOUT_SECONDS


@dataclass
class JudgeResult:
    correctness: int
    groundedness: int
    completeness: int
    clarity: int
    type_alignment: int
    target_answer_type: str
    detected_answer_type: str
    overall_band: int
    rationale: str


SUPPORTED_ANSWER_TYPES = {
    "exam-prep",
    "concept-focused",
    "homework-help",
    "quick-answer",
    "general",
}
RUBRIC_KEYS = ["correctness", "groundedness", "completeness", "clarity", "type_alignment"]


def normalize_answer_type(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return "general"
    raw = value.strip().lower()
    return raw if raw in SUPPORTED_ANSWER_TYPES else "general"


def calculate_overall_band(
    correctness: int,
    groundedness: int,
    completeness: int,
    clarity: int,
    type_alignment: int,
) -> int:
    overall_band = round(
        2.25
        * (
            correctness * 0.35
            + groundedness * 0.25
            + completeness * 0.15
            + clarity * 0.15
            + type_alignment * 0.10
        )
        - 1.25
    )

    overall_band = max(1, min(10, overall_band))

    if correctness <= 1:
        overall_band = min(overall_band, 3)
    elif correctness <= 2:
        overall_band = min(overall_band, 5)

    if groundedness <= 1:
        overall_band = min(overall_band, 4)
    elif groundedness <= 2:
        overall_band = min(overall_band, 6)

    if completeness <= 2:
        overall_band = min(overall_band, 7)

    if type_alignment <= 2:
        overall_band = min(overall_band, 8)

    return overall_band


def _ollama_headers() -> dict | None:
    if not OLLAMA_API_KEY:
        return None
    return {"Authorization": f"Bearer {OLLAMA_API_KEY}"}


def _openrouter_headers() -> dict:
    headers = {}
    if OPENROUTER_SITE_URL:
        headers["HTTP-Referer"] = OPENROUTER_SITE_URL
    if OPENROUTER_APP_NAME:
        headers["X-OpenRouter-Title"] = OPENROUTER_APP_NAME
    return headers


def _extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Judge output did not contain a JSON object")
    blob = text[start : end + 1]
    return json.loads(blob)


def parse_judge_response(text: str) -> JudgeResult:
    payload = _extract_json_object(text)

    if "type_alignment" not in payload and "safety" in payload:
        payload["type_alignment"] = payload["safety"]

    for key in RUBRIC_KEYS:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > 5:
            raise ValueError(f"Invalid score for '{key}': {value}")

    target_answer_type = normalize_answer_type(payload.get("target_answer_type"))
    detected_answer_type = normalize_answer_type(payload.get("detected_answer_type"))

    rationale = payload.get("rationale", "")
    if not isinstance(rationale, str):
        raise ValueError("'rationale' must be a string")

    overall_band = calculate_overall_band(
        payload["correctness"],
        payload["groundedness"],
        payload["completeness"],
        payload["clarity"],
        payload["type_alignment"],
    )

    return JudgeResult(
        correctness=payload["correctness"],
        groundedness=payload["groundedness"],
        completeness=payload["completeness"],
        clarity=payload["clarity"],
        type_alignment=payload["type_alignment"],
        target_answer_type=target_answer_type,
        detected_answer_type=detected_answer_type,
        overall_band=overall_band,
        rationale=rationale.strip(),
    )


def _fallback_judge_result(target_answer_type: str, error: Exception) -> JudgeResult:
    target = normalize_answer_type(target_answer_type)
    return JudgeResult(
        correctness=1,
        groundedness=1,
        completeness=1,
        clarity=1,
        type_alignment=1,
        target_answer_type=target,
        detected_answer_type="general",
        overall_band=calculate_overall_band(1, 1, 1, 1, 1),
        rationale=f"Evaluator output was malformed and could not be parsed safely: {error}",
    )


def _build_prompt(
    question: str,
    reference_answer: str,
    model_answer: str,
    context: str,
    preferred_answer_type: str | None = None,
) -> str:
    target_answer_type = normalize_answer_type(preferred_answer_type)
    return f"""
You are an expert evaluator for educational RAG answers.

Your task is to score the model answer against:
1. the user question,
2. the reference answer,
3. the retrieved context,
4. the preferred answer type.

Return ONLY valid JSON. Do not include markdown, explanation, or extra text.

JSON keys:
- correctness: int 1-5
- groundedness: int 1-5
- completeness: int 1-5
- clarity: int 1-5
- type_alignment: int 1-5
- target_answer_type: string
- detected_answer_type: string
- overall_band: int 1-10
- rationale: string <= 120 words

Answer type definitions:

1. exam-prep
The answer should help the student prepare for an exam. It should focus on concise revision, important formulas, key concepts, common exam points, shortcuts, and common mistakes. It should not be overly long unless the question asks for detail.

2. concept-focused
The answer should focus on understanding. It should explain meanings, relationships, why formulas work, and the reasoning behind the concept. It may include examples, but the main goal is conceptual clarity.

3. homework-help
The answer should help solve a specific problem. It should show the steps, choose the correct formula, substitute values, calculate carefully, and give a clear final answer. It should be tutor-like and easy to follow.

4. quick-answer
The answer should be short and direct. It should give the answer quickly with minimal explanation. Only include key reasoning if needed.

5. general
Use this when the user did not specify a preferred type. The answer should naturally fit the query, be educational, balanced, clear, and not too long or too short.

Preferred answer type rule:
- The normalized target_answer_type for this item is "{target_answer_type}".
- Use this exact value as target_answer_type.
- detected_answer_type should describe the actual style of the model answer. Choose from: exam-prep, concept-focused, homework-help, quick-answer, general.

Scoring rubric:

Correctness:
5 = fully correct conceptually and mathematically.
4 = mostly correct with minor issues.
3 = partially correct but has noticeable gaps.
2 = major misunderstanding or calculation errors.
1 = mostly incorrect.

Groundedness:
5 = all important claims are supported by retrieved context.
4 = mostly grounded with minor unsupported details.
3 = partially grounded, but some claims are not clearly supported.
2 = weakly grounded and relies too much on outside knowledge.
1 = not grounded in the retrieved context.

Completeness:
5 = covers all key points from the reference answer.
4 = covers most key points.
3 = covers the main idea but misses important parts.
2 = very incomplete.
1 = barely answers the question.

Clarity:
5 = very clear and well-structured for high-school learners.
4 = clear, with minor organization issues.
3 = understandable but somewhat confusing.
2 = hard to follow.
1 = very unclear.

Type alignment:
5 = perfectly matches the target answer type.
4 = mostly matches the target answer type.
3 = acceptable, but style is not ideal.
2 = poor match to the target answer type.
1 = completely wrong answer style.

Overall band calculation:
Calculate directly from the five metric scores:
overall_band = round(
  2.25 * (
    correctness * 0.35 +
    groundedness * 0.25 +
    completeness * 0.15 +
    clarity * 0.15 +
    type_alignment * 0.10
  ) - 1.25
)

Then clamp to 1-10.

Apply these caps:
- If correctness <= 1, overall_band must be at most 3.
- If correctness <= 2, overall_band must be at most 5.
- If groundedness <= 1, overall_band must be at most 4.
- If groundedness <= 2, overall_band must be at most 6.
- If completeness <= 2, overall_band must be at most 7.
- If type_alignment <= 2, overall_band must be at most 8.

Important:
- Do not reward answers that are fluent but factually wrong.
- Do not reward answers that contain information unsupported by retrieved context.
- For model-generated evaluation sets, the reference answer may come from another LLM. Use it as a guide, but still judge correctness and grounding carefully.
- Do not require exact wording. Focus on meaning, accuracy, and coverage.

Question:
{question}

Preferred Answer Type:
{target_answer_type}

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

    def score(
        self,
        question: str,
        reference_answer: str,
        model_answer: str,
        context: str,
        preferred_answer_type: str | None = None,
    ) -> JudgeResult:
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
                headers=_ollama_headers(),
            )
        elif JUDGE_PROVIDER == "openrouter":
            from langchain_community.chat_models.openai import ChatOpenAI

            llm = ChatOpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL,
                default_headers=_openrouter_headers(),
                model=self.model,
                temperature=0,
                max_tokens=512,
                timeout=JUDGE_TIMEOUT_SECONDS,
                max_retries=JUDGE_MAX_RETRIES,
            )
        else:
            raise ValueError(
                f"Unsupported JUDGE_PROVIDER='{JUDGE_PROVIDER}'. Use one of: gemini, ollama, openrouter"
            )

        target_answer_type = normalize_answer_type(preferred_answer_type)
        prompt = _build_prompt(
            question,
            reference_answer,
            model_answer,
            context,
            preferred_answer_type=target_answer_type,
        )
        response = llm.invoke(prompt)
        text = getattr(response, "content", "")
        if isinstance(text, list):
            text = "\n".join(str(x) for x in text)
        try:
            result = parse_judge_response(str(text))
        except Exception as exc:
            return _fallback_judge_result(target_answer_type, exc)
        if result.target_answer_type != target_answer_type:
            result.target_answer_type = target_answer_type
        return result

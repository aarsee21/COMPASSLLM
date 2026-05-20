import json
import re
from typing import Any
from pathlib import Path

import google.generativeai as genai

from config import get_settings

settings = get_settings()

# Load instruction files for a simple RAG setup. Cache on first import.
_INSTRUCTION_CACHE: dict[str, str] | None = None


def _load_instruction_files() -> dict[str, str]:
    global _INSTRUCTION_CACHE
    if _INSTRUCTION_CACHE is not None:
        return _INSTRUCTION_CACHE

    base = Path(__file__).resolve().parents[1]
    instr_dir = base / "instruction_files"
    cache: dict[str, str] = {}
    if instr_dir.exists() and instr_dir.is_dir():
        for p in sorted(instr_dir.glob("*.txt")):
            try:
                cache[p.stem] = p.read_text(encoding="utf-8")
            except Exception:
                # skip unreadable files
                continue

    _INSTRUCTION_CACHE = cache
    return cache


def _retrieve_relevant_instructions(
    user_instruction: str | None = None, sample_data: list[dict[str, Any]] | None = None, top_k: int = 3
) -> str:
    instr = _load_instruction_files()
    if not instr:
        return ""

    if not user_instruction and not sample_data:
        # return top N instructions by filename (stable order)
        texts = list(instr.values())[:top_k]
        return "\n\n".join(texts)

    # Simple relevance scoring: count shared words between user_instruction and each instruction text
    query_text = (user_instruction or "")
    if sample_data and len(sample_data) > 0:
        # include a few feature names / example values to the query
        sample = sample_data[0]
        query_text += " " + " ".join(str(k) for k in sample.keys())

    q_tokens = set(re.findall(r"\w+", query_text.lower()))
    scores: list[tuple[int, str]] = []
    for name, content in instr.items():
        tokens = set(re.findall(r"\w+", content.lower()))
        score = len(q_tokens & tokens)
        scores.append((score, content))

    scores.sort(reverse=True, key=lambda x: x[0])
    chosen = [c for s, c in scores if s > 0][:top_k]
    if not chosen:
        chosen = [c for _, c in scores[:top_k]]
    return "\n\n".join(chosen)


def _format_sample_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = [" | ".join(headers)]
    lines.append("-" * len(lines[0]))
    for row in rows:
        lines.append(" | ".join(str(row.get(h, "")) for h in headers))
    return "\n".join(lines)


def _build_prompt(
    samples: int,
    features: int,
    numeric: int,
    categorical: int,
    missing_ratio: float,
    imbalance: float,
    user_instruction: str | None = None,
    sample_data: list[dict[str, Any]] | None = None,
    instruction_context: str | None = None,
) -> str:
    instruction_block = (
        f"\nUser's task description:\n{user_instruction.strip()}\n"
        if user_instruction and user_instruction.strip()
        else ""
    )

    sample_block = ""
    if sample_data:
        table = _format_sample_rows(sample_data[:15])
        sample_block = f"\nSample data rows (up to 15 rows — use these to understand feature types, value ranges, and patterns):\n{table}\n"

    instruction_context_block = (
        f"\nRelevant instruction documents (from the model instruction library):\n{instruction_context}\n"
        if instruction_context and instruction_context.strip()
        else ""
    )

    return f"""You are a machine learning expert.

Dataset characteristics:
Samples: {samples}
Features: {features}
Numeric Features: {numeric}
Categorical Features: {categorical}
Missing Value Ratio: {missing_ratio:.4f}
Class Imbalance Ratio: {imbalance:.4f}
{instruction_block}{sample_block}{instruction_context_block}
Based on the above, recommend the best 3 machine learning classification models.
Consider: dataset size, feature types, class imbalance, missing data, and the user's stated goal.

Return ONLY valid JSON in this exact format:

{{
  "models": ["Model Name 1", "Model Name 2", "Model Name 3"],
  "reasoning": "Detailed explanation of why these models suit this dataset and task."
}}
"""


def _extract_json_payload(text: str) -> dict[str, object]:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found in Gemini response.")

    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Invalid JSON response shape from Gemini.")
    return payload


def _fallback_recommendation() -> dict[str, object]:
    return {
        "models": ["Random Forest", "XGBoost", "Support Vector Machine"],
        "reasoning": (
            "Fallback recommendation used because Gemini response was unavailable or invalid. "
            "These models are robust defaults for mixed-tabular classification datasets."
        ),
    }


def recommend_models_with_gemini(
    samples: int,
    features: int,
    numeric: int,
    categorical: int,
    missing_ratio: float,
    imbalance: float,
    user_instruction: str | None = None,
    sample_data: list[dict[str, Any]] | None = None,
) -> dict[str, object]:
    if not settings.gemini_api_key:
        return _fallback_recommendation()

    prompt = _build_prompt(
        samples, features, numeric, categorical,
        missing_ratio, imbalance, user_instruction, sample_data,
    )

    try:
        # Retrieve relevant instruction documents to augment the prompt (RAG)
        instruction_context = _retrieve_relevant_instructions(user_instruction, sample_data, top_k=3)

        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(settings.gemini_model)
        prompt = _build_prompt(
            samples, features, numeric, categorical,
            missing_ratio, imbalance, user_instruction, sample_data, instruction_context
        )

        response = model.generate_content(prompt)
        text = response.text or ""
        payload = _extract_json_payload(text)

        models = payload.get("models", [])
        reasoning = payload.get("reasoning", "")

        if not isinstance(models, list) or not all(isinstance(m, str) for m in models):
            raise ValueError("Gemini response models field is invalid.")
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise ValueError("Gemini response reasoning field is invalid.")

        return {"models": models[:3], "reasoning": reasoning.strip()}
    except Exception:
        return _fallback_recommendation()

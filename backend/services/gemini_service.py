import json
import re
from typing import Any

import google.generativeai as genai

from config import get_settings

settings = get_settings()


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

    return f"""You are a machine learning expert.

Dataset characteristics:
Samples: {samples}
Features: {features}
Numeric Features: {numeric}
Categorical Features: {categorical}
Missing Value Ratio: {missing_ratio:.4f}
Class Imbalance Ratio: {imbalance:.4f}
{instruction_block}{sample_block}
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
        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(settings.gemini_model)
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

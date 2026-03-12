import json
import re

import google.generativeai as genai

from config import get_settings

settings = get_settings()


def _build_prompt(
    samples: int,
    features: int,
    numeric: int,
    categorical: int,
    missing_ratio: float,
    imbalance: float,
) -> str:
    return f"""You are a machine learning expert.

Dataset characteristics:

Samples: {samples}
Features: {features}
Numeric Features: {numeric}
Categorical Features: {categorical}
Missing Value Ratio: {missing_ratio}
Class Imbalance: {imbalance}

Recommend the best 3 machine learning models for this dataset.

Return JSON format:

{{
  \"models\": [\"Random Forest\", \"SVM\", \"XGBoost\"],
  \"reasoning\": \"Explanation\"
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
) -> dict[str, object]:
    if not settings.gemini_api_key:
        return _fallback_recommendation()

    prompt = _build_prompt(samples, features, numeric, categorical, missing_ratio, imbalance)

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

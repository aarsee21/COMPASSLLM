import json
import uuid
from typing import Annotated

import pandas as pd
import psycopg2.extensions
import psycopg2.extras
from fastapi import APIRouter, Depends, HTTPException

from database.db import get_db
from schemas.api_schemas import RecommendRequest, RecommendResponse
from services.dataset_service import get_dataset_csv_path, normalize_excluded_columns
from services.gemini_service import recommend_models_with_gemini

router = APIRouter(prefix="/models", tags=["recommendation"])


def _filter_sample_rows(
    rows: list[dict[str, object]],
    excluded_columns: list[str] | None,
) -> list[dict[str, object]]:
    excluded = set(excluded_columns or [])
    if not excluded:
        return rows

    return [
        {key: value for key, value in row.items() if key not in excluded}
        for row in rows
    ]


def _get_random_sample_rows(
    dataset_id: str,
    excluded_columns: list[str] | None = None,
    max_rows: int = 15,
) -> list[dict[str, object]]:
    csv_path = get_dataset_csv_path(uuid.UUID(dataset_id))
    if not csv_path.exists():
        return []
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return []
        normalized_excluded = normalize_excluded_columns(df, target_column="", excluded_columns=excluded_columns)
        if normalized_excluded:
            df = df.drop(columns=normalized_excluded, errors="ignore")
        n = min(len(df), max_rows)
        sampled = df.sample(n=n, replace=False)
        return sampled.replace({float("nan"): None}).to_dict(orient="records")
    except Exception:
        return []


@router.post(
    "/recommend",
    responses={
        400: {"description": "Dataset has not been analyzed yet."},
        404: {"description": "Dataset not found."},
        500: {"description": "Recommendation generation failed."},
    },
)
def recommend_models(
    payload: RecommendRequest,
    db: Annotated[psycopg2.extensions.connection, Depends(get_db)],
) -> RecommendResponse:
    dataset_id_str = str(payload.dataset_id)

    with db.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM datasets WHERE id = %s", (dataset_id_str,))
        dataset = cur.fetchone()
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found.")

        cur.execute("SELECT * FROM dataset_features WHERE dataset_id = %s", (dataset_id_str,))
        features = cur.fetchone()

    if features is None:
        raise HTTPException(status_code=400, detail="Dataset has not been analyzed yet.")

    excluded_columns = list(dataset.get("excluded_columns") or [])
    sample_data = _filter_sample_rows(payload.sample_data, excluded_columns) if payload.sample_data else _get_random_sample_rows(dataset_id_str, excluded_columns=excluded_columns)

    gemini_result = recommend_models_with_gemini(
        samples=features["num_samples"],
        features=features["num_features"],
        numeric=features["num_numeric"],
        categorical=features["num_categorical"],
        missing_ratio=features["missing_ratio"],
        imbalance=features["imbalance_ratio"],
        user_instruction=payload.user_instruction,
        sample_data=sample_data,
    )

    models = [str(item) for item in gemini_result.get("models", [])][:3]
    reasoning = str(gemini_result.get("reasoning", "")).strip()

    if not models:
        raise HTTPException(status_code=500, detail="Recommendation generation failed.")

    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO recommendations (dataset_id, recommended_models, reasoning) VALUES (%s, %s::jsonb, %s)",
            (dataset_id_str, json.dumps(models), reasoning),
        )
    db.commit()

    return RecommendResponse(dataset_id=payload.dataset_id, models=models, reasoning=reasoning)

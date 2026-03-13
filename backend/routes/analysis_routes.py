import json
from typing import Annotated

import psycopg2.extensions
from fastapi import APIRouter, Depends

from database.db import get_db
from schemas.api_schemas import AnalyzeRequest, AnalyzeResponse
from services.dataset_service import apply_dataset_column_selection, load_dataset_dataframe
from services.feature_extraction_service import compute_dataset_meta_features

router = APIRouter(prefix="/datasets", tags=["analysis"])


@router.post("/analyze")
def analyze_dataset(
    payload: AnalyzeRequest,
    db: Annotated[psycopg2.extensions.connection, Depends(get_db)],
) -> AnalyzeResponse:
    _, df = load_dataset_dataframe(str(payload.dataset_id), db)
    filtered_df, excluded_columns = apply_dataset_column_selection(df, payload.target_column, payload.excluded_columns)

    metadata = compute_dataset_meta_features(filtered_df, payload.target_column, excluded_columns)

    with db.cursor() as cur:
        cur.execute(
            "UPDATE datasets SET target_column = %s, excluded_columns = %s::jsonb WHERE id = %s",
            (payload.target_column, json.dumps(excluded_columns), str(payload.dataset_id)),
        )
        cur.execute(
            "SELECT id FROM dataset_features WHERE dataset_id = %s",
            (str(payload.dataset_id),),
        )
        existing = cur.fetchone()
        if existing is None:
            cur.execute(
                """
                INSERT INTO dataset_features
                    (dataset_id, num_samples, num_features, num_numeric,
                     num_categorical, missing_ratio, imbalance_ratio)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(payload.dataset_id),
                    int(metadata["number_of_samples"]),
                    int(metadata["number_of_features"]),
                    int(metadata["numeric_feature_count"]),
                    int(metadata["categorical_feature_count"]),
                    float(metadata["missing_value_ratio"]),
                    float(metadata["imbalance_ratio"]),
                ),
            )
        else:
            cur.execute(
                """
                UPDATE dataset_features
                SET num_samples = %s, num_features = %s, num_numeric = %s,
                    num_categorical = %s, missing_ratio = %s, imbalance_ratio = %s
                WHERE dataset_id = %s
                """,
                (
                    int(metadata["number_of_samples"]),
                    int(metadata["number_of_features"]),
                    int(metadata["numeric_feature_count"]),
                    int(metadata["categorical_feature_count"]),
                    float(metadata["missing_value_ratio"]),
                    float(metadata["imbalance_ratio"]),
                    str(payload.dataset_id),
                ),
            )
    db.commit()

    return AnalyzeResponse(dataset_id=payload.dataset_id, **metadata)

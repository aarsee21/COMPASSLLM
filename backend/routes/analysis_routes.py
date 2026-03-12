from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database.db import get_db
from database.models import DatasetFeatures
from schemas.api_schemas import AnalyzeRequest, AnalyzeResponse
from services.dataset_service import load_dataset_dataframe
from services.feature_extraction_service import compute_dataset_meta_features

router = APIRouter(prefix="/datasets", tags=["analysis"])


@router.post("/analyze")
def analyze_dataset(payload: AnalyzeRequest, db: Annotated[Session, Depends(get_db)]) -> AnalyzeResponse:
    dataset, df = load_dataset_dataframe(payload.dataset_id, db)

    metadata = compute_dataset_meta_features(df, payload.target_column)

    dataset.target_column = payload.target_column

    features_row = dataset.features
    if features_row is None:
        features_row = DatasetFeatures(
            dataset_id=dataset.id,
            num_samples=int(metadata["number_of_samples"]),
            num_features=int(metadata["number_of_features"]),
            num_numeric=int(metadata["numeric_feature_count"]),
            num_categorical=int(metadata["categorical_feature_count"]),
            missing_ratio=float(metadata["missing_value_ratio"]),
            imbalance_ratio=float(metadata["imbalance_ratio"]),
        )
        db.add(features_row)
    else:
        features_row.num_samples = int(metadata["number_of_samples"])
        features_row.num_features = int(metadata["number_of_features"])
        features_row.num_numeric = int(metadata["numeric_feature_count"])
        features_row.num_categorical = int(metadata["categorical_feature_count"])
        features_row.missing_ratio = float(metadata["missing_value_ratio"])
        features_row.imbalance_ratio = float(metadata["imbalance_ratio"])

    db.commit()

    return AnalyzeResponse(dataset_id=dataset.id, **metadata)

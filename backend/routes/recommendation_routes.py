from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database.db import get_db
from database.models import Dataset, Recommendation
from schemas.api_schemas import RecommendRequest, RecommendResponse
from services.gemini_service import recommend_models_with_gemini

router = APIRouter(prefix="/models", tags=["recommendation"])


@router.post("/recommend")
def recommend_models(payload: RecommendRequest, db: Annotated[Session, Depends(get_db)]) -> RecommendResponse:
    dataset = db.get(Dataset, payload.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    if dataset.features is None:
        raise HTTPException(status_code=400, detail="Dataset has not been analyzed yet.")

    features = dataset.features

    gemini_result = recommend_models_with_gemini(
        samples=features.num_samples,
        features=features.num_features,
        numeric=features.num_numeric,
        categorical=features.num_categorical,
        missing_ratio=features.missing_ratio,
        imbalance=features.imbalance_ratio,
    )

    models = [str(item) for item in gemini_result.get("models", [])][:3]
    reasoning = str(gemini_result.get("reasoning", "")).strip()

    if not models:
        raise HTTPException(status_code=500, detail="Recommendation generation failed.")

    recommendation = Recommendation(dataset_id=dataset.id, recommended_models=models, reasoning=reasoning)
    db.add(recommendation)
    db.commit()

    return RecommendResponse(dataset_id=dataset.id, models=models, reasoning=reasoning)

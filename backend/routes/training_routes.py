from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database.db import get_db
from schemas.api_schemas import TrainRequest, TrainResponse, TrainingResult
from services.dataset_service import load_dataset_dataframe
from services.experiment_service import replace_experiments
from services.model_training_service import train_models

router = APIRouter(prefix="/models", tags=["training"])


@router.post("/train")
def train_recommended_models(payload: TrainRequest, db: Annotated[Session, Depends(get_db)]) -> TrainResponse:
    dataset, df = load_dataset_dataframe(payload.dataset_id, db)

    if not dataset.target_column:
        raise HTTPException(status_code=400, detail="Target column not set. Run /datasets/analyze first.")

    results = train_models(df, dataset.target_column)
    replace_experiments(db, dataset.id, results)

    return TrainResponse(
        dataset_id=dataset.id,
        results=[TrainingResult(**item) for item in results],
    )

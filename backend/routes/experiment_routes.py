import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database.db import get_db
from schemas.api_schemas import ExperimentResult, ExperimentsResponse
from services.experiment_service import get_experiments

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.get("/{dataset_id}", response_model=ExperimentsResponse)
def list_experiments(dataset_id: uuid.UUID, db: Session = Depends(get_db)) -> ExperimentsResponse:
    rows = get_experiments(db, dataset_id)
    return ExperimentsResponse(
        dataset_id=dataset_id,
        experiments=[
            ExperimentResult(
                id=row.id,
                dataset_id=row.dataset_id,
                model_name=row.model_name,
                accuracy=row.accuracy,
                precision=row.precision,
                recall=row.recall,
                f1_score=row.f1_score,
                training_time=row.training_time,
                created_at=row.created_at,
            )
            for row in rows
        ],
    )

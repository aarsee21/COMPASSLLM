import uuid
from typing import Annotated

import psycopg2.extensions
from fastapi import APIRouter, Depends

from database.db import get_db
from schemas.api_schemas import DashboardSummaryResponse, ExperimentResult, ExperimentsResponse
from services.experiment_service import get_dashboard_summary, get_experiments

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.get("/summary", response_model=DashboardSummaryResponse)
def dashboard_summary(
    db: Annotated[psycopg2.extensions.connection, Depends(get_db)],
) -> DashboardSummaryResponse:
    return DashboardSummaryResponse(**get_dashboard_summary(db))


@router.get("/{dataset_id}", response_model=ExperimentsResponse)
def list_experiments(
    dataset_id: uuid.UUID,
    db: Annotated[psycopg2.extensions.connection, Depends(get_db)],
) -> ExperimentsResponse:
    rows = get_experiments(db, str(dataset_id))
    return ExperimentsResponse(
        dataset_id=dataset_id,
        experiments=[
            ExperimentResult(
                id=row["id"],
                dataset_id=row["dataset_id"],
                model_name=row["model_name"],
                accuracy=row["accuracy"],
                precision=row["precision"],
                recall=row["recall"],
                f1_score=row["f1_score"],
                training_time=row["training_time"],
                created_at=row["created_at"],
            )
            for row in rows
        ],
    )

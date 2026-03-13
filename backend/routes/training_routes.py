from typing import Annotated

import psycopg2.extensions
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from database.db import get_db
from schemas.api_schemas import KnowledgeBaseResponse, TrainRequest, TrainResponse, TrainingResult
from services.dataset_service import apply_dataset_column_selection, load_dataset_dataframe
from services.experiment_service import get_model_artifact, replace_experiments, replace_model_artifacts
from services.knowledge_base_service import create_knowledge_base_entry, get_latest_recommendation, get_rulebook, list_knowledge_base_entries
from services.model_training_service import train_models

router = APIRouter(prefix="/models", tags=["training"])


@router.post(
    "/train",
    responses={400: {"description": "Target column is missing or invalid."}},
)
def train_recommended_models(
    payload: TrainRequest,
    db: Annotated[psycopg2.extensions.connection, Depends(get_db)],
) -> TrainResponse:
    dataset, df = load_dataset_dataframe(str(payload.dataset_id), db)

    if not dataset.get("target_column"):
        raise HTTPException(status_code=400, detail="Target column not set. Run /datasets/analyze first.")

    dataset_id_str = str(payload.dataset_id)
    filtered_df, _ = apply_dataset_column_selection(df, str(dataset["target_column"]), list(dataset.get("excluded_columns") or []))
    results = train_models(filtered_df, str(dataset["target_column"]), dataset_id=dataset_id_str)
    replace_experiments(db, dataset_id_str, results)

    artifacts = replace_model_artifacts(
        db,
        dataset_id_str,
        [
            {"model_name": str(item["model_name"]), "artifact_path": str(item["artifact_path"])}
            for item in results
            if item.get("artifact_path")
        ],
    )

    artifact_by_model = {str(item["model_name"]): item for item in artifacts}

    recommendation = get_latest_recommendation(db, dataset_id_str)
    recommended_models = list(recommendation.get("recommended_models") or []) if recommendation else []
    reasoning = str(recommendation.get("reasoning") or "") if recommendation else ""

    best_result = max(results, key=lambda item: float(item["accuracy"]))
    create_knowledge_base_entry(
        db,
        dataset_id=dataset_id_str,
        dataset_name=str(dataset["name"]),
        target_column=str(dataset["target_column"]),
        recommended_models=[str(item) for item in recommended_models],
        reasoning=reasoning,
        best_model=str(best_result["model_name"]),
        best_accuracy=float(best_result["accuracy"]),
        experiment_count=len(results),
    )

    return TrainResponse(
        dataset_id=payload.dataset_id,
        results=[
            TrainingResult(
                model_name=str(item["model_name"]),
                accuracy=float(item["accuracy"]),
                precision=float(item["precision"]),
                recall=float(item["recall"]),
                f1_score=float(item["f1_score"]),
                training_time=float(item["training_time"]),
                artifact_id=artifact_by_model.get(str(item["model_name"]), {}).get("id"),
                download_url=(
                    f"/models/download/{artifact_by_model[str(item['model_name'])]['id']}"
                    if str(item["model_name"]) in artifact_by_model
                    else None
                ),
            )
            for item in results
        ],
    )


@router.get(
    "/download/{artifact_id}",
    responses={404: {"description": "Trained model artifact not found."}},
)
def download_trained_model(
    artifact_id: str,
    db: Annotated[psycopg2.extensions.connection, Depends(get_db)],
) -> FileResponse:
    artifact = get_model_artifact(db, artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Trained model artifact not found.")

    artifact_path = str(artifact["artifact_path"])
    return FileResponse(path=artifact_path, filename=artifact_path.split("/")[-1], media_type="application/octet-stream")


@router.get("/knowledge-base")
def get_knowledge_base(
    db: Annotated[psycopg2.extensions.connection, Depends(get_db)],
) -> KnowledgeBaseResponse:
    return KnowledgeBaseResponse(rulebook=get_rulebook(), entries=list_knowledge_base_entries(db))

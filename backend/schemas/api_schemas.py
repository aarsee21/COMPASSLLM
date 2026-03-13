import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DatasetUploadResponse(BaseModel):
    id: uuid.UUID
    rows: int
    columns: int
    preview: list[dict[str, Any]]


class AnalyzeRequest(BaseModel):
    dataset_id: uuid.UUID
    target_column: str


class AnalyzeResponse(BaseModel):
    dataset_id: uuid.UUID
    number_of_samples: int
    number_of_features: int
    numeric_feature_count: int
    categorical_feature_count: int
    missing_value_ratio: float
    class_distribution: dict[str, float]
    imbalance_ratio: float
    meta_features: dict[str, float | int | str | None] = Field(default_factory=dict)


class RecommendRequest(BaseModel):
    dataset_id: uuid.UUID
    user_instruction: str | None = None
    sample_data: list[dict[str, object]] | None = None


class RecommendResponse(BaseModel):
    dataset_id: uuid.UUID
    models: list[str]
    reasoning: str


class TrainRequest(BaseModel):
    dataset_id: uuid.UUID


class TrainingResult(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_time: float
    artifact_id: uuid.UUID | None = None
    download_url: str | None = None


class TrainResponse(BaseModel):
    dataset_id: uuid.UUID
    results: list[TrainingResult]


class ExperimentResult(BaseModel):
    id: uuid.UUID
    dataset_id: uuid.UUID
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_time: float
    created_at: datetime


class ExperimentsResponse(BaseModel):
    dataset_id: uuid.UUID
    experiments: list[ExperimentResult]


class ModelPerformanceItem(BaseModel):
    model_name: str
    average_accuracy: float


class DashboardSummaryResponse(BaseModel):
    datasets_processed: int
    experiments_run: int
    models_tested: int
    best_accuracy: float
    model_performance: list[ModelPerformanceItem]


class KnowledgeBaseEntry(BaseModel):
    id: uuid.UUID
    dataset_id: uuid.UUID
    dataset_name: str
    target_column: str
    recommended_models: list[str]
    reasoning: str
    system_guidance: list[str]
    best_model: str
    best_accuracy: float
    top_recommendation_model: str | None = None
    top_recommendation_worked: bool
    experiment_count: int
    created_at: datetime


class KnowledgeBaseResponse(BaseModel):
    rulebook: list[str]
    entries: list[KnowledgeBaseEntry]

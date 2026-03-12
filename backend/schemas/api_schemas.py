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
    pymfe_meta_features: dict[str, float | int | str | None] = Field(default_factory=dict)


class RecommendRequest(BaseModel):
    dataset_id: uuid.UUID


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

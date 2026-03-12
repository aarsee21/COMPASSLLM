import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.db import Base

CASCADE_ALL_DELETE_ORPHAN = "all, delete-orphan"
DATASETS_ID_FK = "datasets.id"


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    rows: Mapped[int] = mapped_column(Integer, nullable=False)
    columns: Mapped[int] = mapped_column(Integer, nullable=False)
    target_column: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    features: Mapped["DatasetFeatures | None"] = relationship(
        back_populates="dataset", cascade=CASCADE_ALL_DELETE_ORPHAN, uselist=False
    )
    recommendations: Mapped[list["Recommendation"]] = relationship(
        back_populates="dataset", cascade=CASCADE_ALL_DELETE_ORPHAN
    )
    experiments: Mapped[list["Experiment"]] = relationship(back_populates="dataset", cascade=CASCADE_ALL_DELETE_ORPHAN)


class DatasetFeatures(Base):
    __tablename__ = "dataset_features"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey(DATASETS_ID_FK), nullable=False, unique=True)
    num_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    num_features: Mapped[int] = mapped_column(Integer, nullable=False)
    num_numeric: Mapped[int] = mapped_column(Integer, nullable=False)
    num_categorical: Mapped[int] = mapped_column(Integer, nullable=False)
    missing_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    imbalance_ratio: Mapped[float] = mapped_column(Float, nullable=False)

    dataset: Mapped[Dataset] = relationship(back_populates="features")


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey(DATASETS_ID_FK), nullable=False)
    recommended_models: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    reasoning: Mapped[str] = mapped_column(Text, nullable=False)

    dataset: Mapped[Dataset] = relationship(back_populates="recommendations")


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey(DATASETS_ID_FK), nullable=False)
    model_name: Mapped[str] = mapped_column(Text, nullable=False)
    accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    precision: Mapped[float] = mapped_column(Float, nullable=False)
    recall: Mapped[float] = mapped_column(Float, nullable=False)
    f1_score: Mapped[float] = mapped_column(Float, nullable=False)
    training_time: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    dataset: Mapped[Dataset] = relationship(back_populates="experiments")

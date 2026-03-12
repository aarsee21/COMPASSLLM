import uuid

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from database.models import Experiment


def replace_experiments(db: Session, dataset_id: uuid.UUID, results: list[dict[str, float | str]]) -> list[Experiment]:
    db.execute(delete(Experiment).where(Experiment.dataset_id == dataset_id))

    created: list[Experiment] = []
    for row in results:
        experiment = Experiment(
            dataset_id=dataset_id,
            model_name=str(row["model_name"]),
            accuracy=float(row["accuracy"]),
            precision=float(row["precision"]),
            recall=float(row["recall"]),
            f1_score=float(row["f1_score"]),
            training_time=float(row["training_time"]),
        )
        db.add(experiment)
        created.append(experiment)

    db.commit()
    for item in created:
        db.refresh(item)

    return created


def get_experiments(db: Session, dataset_id: uuid.UUID) -> list[Experiment]:
    stmt = select(Experiment).where(Experiment.dataset_id == dataset_id).order_by(Experiment.created_at.asc())
    return list(db.scalars(stmt).all())

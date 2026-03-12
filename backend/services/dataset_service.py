import uuid
from pathlib import Path

import pandas as pd
from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session

from config import get_settings
from database.models import Dataset

settings = get_settings()


def ensure_upload_dir() -> Path:
    upload_dir = Path(__file__).resolve().parent.parent / settings.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def get_dataset_csv_path(dataset_id: uuid.UUID) -> Path:
    upload_dir = ensure_upload_dir()
    return upload_dir / f"{dataset_id}.csv"


async def save_uploaded_dataset(file: UploadFile, db: Session) -> tuple[Dataset, pd.DataFrame]:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    dataset_id = uuid.uuid4()
    target_path = get_dataset_csv_path(dataset_id)

    raw_bytes = await file.read()
    target_path.write_bytes(raw_bytes)

    try:
        df = pd.read_csv(target_path)
    except Exception as exc:
        target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {exc}") from exc

    if df.empty:
        target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")

    dataset = Dataset(id=dataset_id, name=file.filename, rows=len(df), columns=len(df.columns))
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    return dataset, df


def load_dataset_dataframe(dataset_id: uuid.UUID, db: Session) -> tuple[Dataset, pd.DataFrame]:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    csv_path = get_dataset_csv_path(dataset_id)
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found in storage.")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise HTTPException(status_code=400, detail="Dataset file is empty.")

    return dataset, df

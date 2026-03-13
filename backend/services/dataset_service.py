import uuid
from pathlib import Path

import pandas as pd
import psycopg2.extensions
import psycopg2.extras
from fastapi import HTTPException, UploadFile

from config import get_settings

settings = get_settings()


def ensure_upload_dir() -> Path:
    upload_dir = Path(__file__).resolve().parent.parent / settings.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def get_dataset_csv_path(dataset_id: uuid.UUID) -> Path:
    upload_dir = ensure_upload_dir()
    return upload_dir / f"{dataset_id}.csv"


async def save_uploaded_dataset(
    file: UploadFile,
    conn: psycopg2.extensions.connection,
) -> tuple[dict, pd.DataFrame]:
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

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "INSERT INTO datasets (id, name, rows, columns) VALUES (%s, %s, %s, %s) RETURNING *",
            (str(dataset_id), file.filename, len(df), len(df.columns)),
        )
        dataset = dict(cur.fetchone())
    conn.commit()

    return dataset, df


def load_dataset_dataframe(
    dataset_id_str: str,
    conn: psycopg2.extensions.connection,
) -> tuple[dict, pd.DataFrame]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM datasets WHERE id = %s", (dataset_id_str,))
        row = cur.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    dataset = dict(row)
    csv_path = get_dataset_csv_path(uuid.UUID(dataset_id_str))
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found in storage.")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise HTTPException(status_code=400, detail="Dataset file is empty.")

    return dataset, df

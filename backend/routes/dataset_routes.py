from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from database.db import get_db
from schemas.api_schemas import DatasetUploadResponse
from services.dataset_service import save_uploaded_dataset

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/upload")
async def upload_dataset(
    file: Annotated[UploadFile, File(...)],
    db: Annotated[Session, Depends(get_db)],
) -> DatasetUploadResponse:
    dataset, df = await save_uploaded_dataset(file, db)

    preview = df.head(10).replace({float("nan"): None}).to_dict(orient="records")

    return DatasetUploadResponse(id=dataset.id, rows=dataset.rows, columns=dataset.columns, preview=preview)

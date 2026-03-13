from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from database.db import _get_pool
from database.models import create_tables
from routes.analysis_routes import router as analysis_router
from routes.dataset_routes import router as dataset_router
from routes.experiment_routes import router as experiment_router
from routes.recommendation_routes import router as recommendation_router
from routes.training_routes import router as training_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = _get_pool()
    conn = pool.getconn()
    try:
        if settings.auto_create_tables:
            create_tables(conn)
    finally:
        pool.putconn(conn)
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(dataset_router)
app.include_router(analysis_router)
app.include_router(recommendation_router)
app.include_router(training_router)
app.include_router(experiment_router)

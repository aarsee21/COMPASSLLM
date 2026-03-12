from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from database.db import Base, engine
from routes.analysis_routes import router as analysis_router
from routes.dataset_routes import router as dataset_router
from routes.experiment_routes import router as experiment_router
from routes.recommendation_routes import router as recommendation_router
from routes.training_routes import router as training_router

settings = get_settings()

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    if settings.auto_create_tables:
        Base.metadata.create_all(bind=engine)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(dataset_router)
app.include_router(analysis_router)
app.include_router(recommendation_router)
app.include_router(training_router)
app.include_router(experiment_router)

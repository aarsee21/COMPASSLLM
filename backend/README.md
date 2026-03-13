# COMPASSLLM Backend

FastAPI backend for the COMPASSLLM AutoML workflow.

Repository: `https://github.com/likhithkanigolla/COMPASSLLM`

## Responsibilities

- Accept CSV uploads.
- Persist dataset metadata in PostgreSQL.
- Analyze datasets using pandas and numpy derived meta-features.
- Respect target-column selection and excluded columns.
- Generate model recommendations with Gemini.
- Train candidate models with scikit-learn and XGBoost.
- Save downloadable `.joblib` artifacts.
- Store experiments and knowledge-base entries.

## Current Stack

- FastAPI
- psycopg2
- PostgreSQL
- pandas
- numpy
- scikit-learn
- XGBoost
- google-generativeai

## Environment

Copy and edit environment variables:

```bash
cp .env.example .env
```

Important values:

- `DATABASE_URL`
- `GEMINI_API_KEY`
- `GEMINI_MODEL`
- `APP_HOST`
- `APP_PORT`
- `AUTO_CREATE_TABLES`

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 10120
```

## Runtime Storage

- Uploaded CSVs: `uploads/`
- Trained artifacts: `model_artifacts/`

## Key Endpoints

- `GET /health`
- `POST /datasets/upload`
- `POST /datasets/analyze`
- `POST /models/recommend`
- `POST /models/train`
- `GET /models/download/{artifact_id}`
- `GET /models/knowledge-base`
- `GET /experiments/summary`
- `GET /experiments/{dataset_id}`

## Analysis Behavior

`POST /datasets/analyze` now accepts:

- `dataset_id`
- `target_column`
- `excluded_columns`

The backend stores the target and excluded columns on the dataset, then reuses that filtered feature set during recommendation sampling and training.

## Notes

- Tables are created automatically on startup when `AUTO_CREATE_TABLES=true`.
- Restart the backend after pulling schema changes.
- If Gemini is unavailable, the backend falls back to safe default recommendations.

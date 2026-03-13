# Backend: COMPASS-LLM Model Recommendation

This backend provides a complete FastAPI pipeline for:

1. Uploading datasets
2. Extracting dataset meta-features with PyMFE
3. Recommending models using Google Gemini
4. Training ML models and storing experiment metrics

## Stack

- FastAPI
- PostgreSQL
- SQLAlchemy ORM
- Alembic
- scikit-learn
- XGBoost
- PyMFE
- Google Gemini API

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy environment config:

```bash
cp .env.example .env
```

4. Update `.env` values (`DATABASE_URL`, `GEMINI_API_KEY`).

5. Run migrations:

```bash
alembic -c alembic.ini upgrade head
```

6. Start API server:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 10120
```

## Endpoints

- `POST /datasets/upload`
- `POST /datasets/analyze`
- `POST /models/recommend`
- `POST /models/train`
- `GET /experiments/{dataset_id}`

## Notes

- Uploaded CSV files are stored in `backend/uploads/` as `<dataset_uuid>.csv`.
- If Gemini is unavailable or returns invalid JSON, the API uses safe fallback model recommendations.
- CORS is enabled for local frontend development.

# COMPASSLLM

**COMPASSLLM** is an LLM-assisted machine learning model recommendation system designed for sustainable AutoML on tabular classification datasets.

It combines dataset profiling, user guidance, and LLM reasoning to reduce wasted experimentation, improve model selection efficiency, and provide a repeatable engineering workflow.

---

## Problem Statement

Selecting the right model for a classification dataset remains a difficult decision in modern software engineering. Many teams still rely on manual trial-and-error, heuristic rules, or blind AutoML search, which can waste compute and delay deployment.

COMPASSLLM addresses this by:
- extracting dataset meta-features and structural statistics,
- incorporating user objectives and dataset examples,
- generating model recommendations with an LLM,
- training the recommended models,
- persisting experiment outcomes for future improvement.

This is particularly relevant for sustainable computing because it reduces unnecessary model training and helps teams make more informed decisions from the first recommendation pass.

---

## What Is Implemented

This repository contains a working full-stack implementation with end-to-end capability.

### Core capabilities

- CSV upload, validation, and preview.
- Target column selection and explicit feature exclusion.
- Dataset analysis with meta-feature extraction and column profiling.
- LLM-assisted recommendation using dataset metadata, sample rows, and optional user instruction.
- Model training pipeline for recommended classification models.
- Persistent experiment tracking and artifact storage.
- Downloadable serialized model artifacts in `.joblib` format.
- Knowledge base entries that record recommendation outcomes and top-model success.
- Fallback recommendation behavior when the Gemini LLM service is unavailable.

---

## System Architecture

### Frontend

- React 18, TypeScript, Vite.
- Zustand for state management.
- Framer Motion for UI transitions.
- Recharts for analysis and training visualizations.
- Vite API proxy routes `/api` requests to the backend.

### Backend

- FastAPI application in `backend/main.py`.
- PostgreSQL persistence using raw `psycopg2`.
- `backend/services/feature_extraction_service.py` for dataset meta-feature extraction.
- `backend/services/gemini_service.py` for building Gemini prompts and parsing LLM recommendations.
- `backend/services/model_training_service.py` for model selection, training, evaluation, and artifact serialization.
- `backend/services/knowledge_base_service.py` for storing recommendation outcomes and guidance.
- `backend/services/dataset_service.py` for dataset upload, storage, target selection, and excluded-column handling.
- `backend/database/models.py` for the persistent schema.

### Persistent Data Model

- `datasets`: uploaded dataset metadata, target column, excluded columns.
- `dataset_features`: computed analysis statistics.
- `recommendations`: recommended model list and reasoning.
- `experiments`: model training metrics.
- `model_artifacts`: trained pipeline artifact records.
- `knowledge_base_entries`: recommendation outcome history.

---

## End-to-End Flow

1. User uploads a CSV dataset through the frontend.
2. The backend saves the CSV to `backend/uploads/` and inserts metadata into `datasets`.
3. User selects the target column and optionally excludes features.
4. The backend computes meta-features in `backend/services/feature_extraction_service.py` and stores them in `dataset_features`.
5. Recommendation uses dataset statistics, sample rows, excluded columns, and optional user instructions to build a Gemini prompt in `backend/services/gemini_service.py`.
6. Gemini returns three recommended classification models and concise reasoning.
7. The recommendation is persisted in `recommendations`.
8. Training selects and trains candidate models via `select_and_train_models` in `backend/services/model_training_service.py`.
9. Trained pipelines are serialized to `.joblib` files under `backend/model_artifacts/`.
10. The knowledge base records whether the top recommended model matched the best model.
11. The frontend dashboard displays summary metrics and knowledge base history.

---

## Why This Supports Sustainable Computing

- Reduces blind trial-and-error through dataset-aware model recommendation.
- Reuses historical recommendation outcomes to improve future decisions.
- Avoids wasted computation with explicit column exclusion and data profiling.
- Improves engineering efficiency with a repeatable, traceable model selection workflow.

---

## API Summary

### Health
- `GET /health`

### Dataset management
- `POST /datasets/upload`
- `POST /datasets/analyze`

### Model recommendation
- `POST /models/recommend`

### Training and artifacts
- `POST /models/train`
- `GET /models/download/{artifact_id}`
- `GET /models/knowledge-base`

### Experiment data
- `GET /experiments/summary`
- `GET /experiments/{dataset_id}`

---

## Setup

### 1. Clone

```bash
git clone https://github.com/likhithkanigolla/COMPASSLLM
cd COMPASSLLM
```

### 2. Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --host 0.0.0.0 --port 10120
```

Set `DATABASE_URL` and `GEMINI_API_KEY` in `.env`.

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:8080` in the browser.

---

## Current Status

- End-to-end flow is implemented.
- Frontend and backend are integrated for upload, analysis, recommendation, training, and artifact download.
- Knowledge base persistence is supported.
- Documentation is aligned with the current repository structure.

## Future Work

- Add explicit recommendation KPIs such as hit rate and latency.
- Improve LLM prompt efficiency and retrieval-augmented reasoning.
- Build curated benchmark evaluation across tabular datasets.
- Add reproducibility metadata and dataset ownership controls.

---

## Additional Documents

- `backend/README.md`
- `frontend/README.md`
- `DOWNLOAD_INSTRUCTIONS.md`
- `TASK_CHECKLIST.md`

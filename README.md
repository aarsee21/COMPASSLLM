# COMPASSLLM

LLM-Assisted Machine Learning Model Recommendation System for Sustainable AutoML.

Git repository: `https://github.com/likhithkanigolla/COMPASSLLM`

## Problem Statement

In modern software engineering and sustainable computing, selecting the right machine learning model for a given dataset is often resource-intensive and arbitrary. We propose a novel, LLM-assisted framework that dynamically analyzes dataset characteristics and user requirements to efficiently recommend optimal model types. By leveraging continuous feedback loops, empirical results, and adaptive knowledge bases, our system evolves over time to minimize trial-and-error, reducing computational waste. This aligns with sustainable computing by streamlining model selection, while contributing to the field of software engineering automation. Our research addresses the gap in intelligent, context-aware model recommendations, offering an evolving, data-driven tool for practitioners.

## What Is Implemented

The project is now implemented as a full-stack system with a production-style workflow.

### Completed Core Capabilities

- CSV upload, validation, and preview.
- Dataset analysis with statistical/meta-feature extraction.
- Target-column selection.
- Column exclusion and rerunnable analysis.
- LLM-assisted recommendation using:
	- dataset-level meta-features,
	- user instruction/objective,
	- sampled rows from the dataset.
- Training pipeline for recommended models.
- Persistent experiment tracking.
- Downloadable trained model artifacts (`.joblib`).
- Knowledge base storing recommendation-vs-outcome feedback.
- Live dashboard summary metrics from backend data.
- Backend fallback-safe recommendation behavior when LLM is unavailable.

## System Architecture

### Frontend

- React 18, TypeScript, Vite.
- Zustand state management.
- Framer Motion for flow transitions.
- Recharts for analysis/training visualizations.
- API proxy via Vite `/api` -> backend.

### Backend

- FastAPI.
- PostgreSQL with raw `psycopg2` (connection pool + SQL DDL).
- pandas and numpy for dataset analysis/meta-features.
- scikit-learn + XGBoost for training.
- Gemini API for recommendation reasoning.

### Persistent Data Model

- `datasets`: uploaded file metadata + selected target + excluded columns.
- `dataset_features`: computed analysis statistics.
- `recommendations`: LLM model choices + reasoning.
- `experiments`: model training metrics.
- `model_artifacts`: downloadable trained pipeline records.
- `knowledge_base_entries`: learning history of recommendation effectiveness.

## End-to-End Flow

1. User uploads a CSV dataset.
2. Backend stores dataset metadata and the file.
3. User opens analysis and chooses target column.
4. User can exclude columns and rerun analysis multiple times.
5. Backend saves the final target + excluded columns configuration.
6. Recommendation step sends:
	 - analyzed metadata,
	 - optional user instruction,
	 - sample rows,
	 to Gemini.
7. Backend stores recommended models and reasoning.
8. Training step uses the same filtered feature set and trains all recommended models.
9. Metrics are saved; best model is identified.
10. Trained artifacts are saved and exposed via download links.
11. Knowledge base entry is created to record whether top recommendations worked.
12. Dashboard and Knowledge Base pages show accumulated outcomes for continuous improvement.

## Why This Supports Sustainable Computing

- Reduces blind trial-and-error by guiding model choice with data-aware recommendations.
- Reuses historical evidence (knowledge base) to improve future decisions.
- Avoids wasted experimentation on irrelevant features through explicit column exclusion.
- Improves engineering efficiency with a repeatable, traceable, feedback-driven pipeline.

## Repository Structure

- `frontend/`: UI application.
- `backend/`: API, analysis, recommendation, training, persistence.
- `backend/uploads/`: runtime dataset storage.
- `backend/model_artifacts/`: runtime trained model files.
- `DOWNLOAD_INSTRUCTIONS.md`: project and model download guide.

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

Set `.env` values for `DATABASE_URL` and `GEMINI_API_KEY`.

### 3. Frontend

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:8080`.

## API Summary

- `GET /health`
- `POST /datasets/upload`
- `POST /datasets/analyze`
- `POST /models/recommend`
- `POST /models/train`
- `GET /models/download/{artifact_id}`
- `GET /models/knowledge-base`
- `GET /experiments/summary`
- `GET /experiments/{dataset_id}`

## Current Project Status

- End-to-end flow is implemented.
- Documentation is aligned with current architecture.
- Training outputs are downloadable.
- Knowledge base persistence is integrated.

## Additional Documents

- `backend/README.md`
- `frontend/README.md`
- `DOWNLOAD_INSTRUCTIONS.md`
- `TASK_CHECKLIST.md`
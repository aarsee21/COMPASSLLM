# Download Instructions

Project: `COMPASSLLM`

This document covers two kinds of downloads in the project:

1. Downloading the project locally.
2. Downloading trained model artifacts from the application.

## 1. Download the Project

If you need the full repository on your machine:

```bash
git clone https://github.com/likhithkanigolla/COMPASSLLM
cd COMPASSLLM
```

If the repository uses submodules, initialize them after cloning:

```bash
git submodule sync
git submodule update --init --recursive
```

## 2. Download Trained Models from the UI

After training completes:

1. Open the `Training Results` page.
2. Find the `Download` column in the results table.
3. Click the `Download` link for the model you want.

Each link downloads a serialized `.joblib` artifact for that trained pipeline.

## 3. Download Trained Models Directly from the API

If you already have an artifact id, call:

```bash
curl -L "http://127.0.0.1:10120/models/download/<artifact_id>" -o trained_model.joblib
```

If you are calling through the frontend proxy instead:

```bash
curl -L "http://127.0.0.1:8080/api/models/download/<artifact_id>" -o trained_model.joblib
```

## 4. Where Downloaded Models Come From

When you trigger training:

- the backend trains each selected model,
- saves the fitted pipeline to disk,
- records the artifact path in PostgreSQL,
- exposes that artifact through `GET /models/download/{artifact_id}`.

Artifacts are stored under:

```text
backend/model_artifacts/
```

## 5. Requirements for Downloading Artifacts

Model downloads only work when:

- the backend is running,
- training completed successfully,
- the artifact record exists in the database,
- the artifact file still exists on disk.

## 6. If a Download Fails

Check the following:

1. The backend is running on port `10120`.
2. The model actually finished training.
3. The artifact id still exists in the database.
4. The file exists in `backend/model_artifacts/`.
5. You restarted the backend after any schema-related update.

## 7. Loading a Downloaded Model in Python

Example:

```python
import joblib

artifact = joblib.load("trained_model.joblib")
pipeline = artifact["pipeline"]
target_column = artifact["target_column"]
classes = artifact["classes"]

predictions = pipeline.predict(your_dataframe)
```

Make sure `your_dataframe` has the same feature columns that were used during training, excluding the target column and any columns you removed during dataset analysis.
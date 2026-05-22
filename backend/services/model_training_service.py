import logging
import time
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from config import get_settings
from services.gemini_service import recommend_models_with_gemini

settings = get_settings()


def _split_train_test(
    X: pd.DataFrame,
    y_encoded: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Try stratified split first; fallback to non-stratified for singleton classes.

    Stratified split fails when the least populated class has fewer than 2 samples.
    """
    try:
        return train_test_split(
            X,
            y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=y_encoded,
        )
    except ValueError:
        return train_test_split(
            X,
            y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )


def _prepare_target_for_training(y_raw: pd.Series) -> pd.Series:
    """Prepare target labels so training remains stable for continuous targets.

    If the target is numeric with very high cardinality, convert it to quantile bins.
    This keeps the current classification pipeline while avoiding thousands of classes.
    """
    y = y_raw.copy()

    if pd.api.types.is_numeric_dtype(y):
        unique_count = int(y.nunique(dropna=True))
        unique_ratio = float(unique_count / max(len(y), 1))

        # Heuristic: treat highly unique numeric targets as continuous/regression-like.
        if unique_count > 20 and unique_ratio > 0.02:
            # Use up to 6 bins for stable class sizes.
            bin_count = min(6, max(3, unique_count))
            binned = pd.qcut(y, q=bin_count, labels=False, duplicates="drop")

            if binned.nunique(dropna=True) >= 2:
                y = binned.astype("Int64").astype(str)
                return y.fillna("missing_target")

    return y.astype(str).fillna("missing_target")


logger = logging.getLogger(__name__)


def _evaluate_pipeline_with_cv(pipeline: Pipeline, X: pd.DataFrame, y_encoded: np.ndarray, n_splits: int = 3) -> dict[str, float] | None:
    class_counts = np.bincount(y_encoded)
    effective_splits = min(n_splits, int(class_counts[class_counts > 0].min())) if class_counts.size and class_counts.max() > 0 else 0
    if effective_splits < 2:
        return None

    cv = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1_score": "f1_weighted",
    }

    try:
        scores = cross_validate(
            pipeline,
            X,
            y_encoded,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score=0.0,
        )
    except Exception as exc:
        logger.exception("Cross-validation failed for pipeline. Falling back to test split evaluation.")
        return None

    return {
        key: float(np.mean(scores[f"test_{key}"]))
        for key in scoring
    }


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[("num", numeric_pipeline, numeric_cols), ("cat", categorical_pipeline, categorical_cols)]
    )


def _build_models(num_classes: int) -> dict[str, object]:
    xgb_kwargs: dict[str, object] = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "logloss",
        "random_state": 42,
    }

    if num_classes > 2:
        xgb_kwargs.update({"objective": "multi:softprob", "num_class": num_classes})
    else:
        xgb_kwargs.update({"objective": "binary:logistic"})

    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=42,
        ),
        "Support Vector Machine": SVC(C=1.0, kernel="rbf", gamma="scale", probability=False, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "XGBoost": XGBClassifier(**xgb_kwargs),
    }


def _build_candidate_models(num_classes: int) -> dict[str, object]:
    # Larger candidate pool inspired by COMPASSLLM notebook
    xgb_kwargs_local = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "logloss",
        "random_state": 42,
    }
    if num_classes > 2:
        xgb_kwargs_local.update({"objective": "multi:softprob", "num_class": num_classes})
    else:
        xgb_kwargs_local.update({"objective": "binary:logistic"})

    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Support Vector Machine": SVC(C=1.0, kernel="rbf", gamma="scale", probability=False, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "GaussianNB": GaussianNB(),
        "XGBoost": XGBClassifier(**xgb_kwargs_local),
        "LightGBM": LGBMClassifier(n_estimators=200, random_state=42),
        "CatBoost": CatBoostClassifier(iterations=200, verbose=0, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    }


def _ensure_model_artifact_dir() -> Path:
    path = Path(__file__).resolve().parent.parent / settings.model_artifacts_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def _artifact_filename(dataset_id: str, model_name: str) -> str:
    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    return f"{dataset_id}_{safe_name}_{uuid.uuid4().hex[:8]}.joblib"


def train_models(df: pd.DataFrame, target_column: str, dataset_id: str | None = None) -> list[dict[str, float | str]]:
    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in dataset.")

    X = df.drop(columns=[target_column])
    y_raw = df[target_column]

    if y_raw.nunique(dropna=False) < 2:
        raise HTTPException(status_code=400, detail="Target column must contain at least two classes.")

    y = _prepare_target_for_training(y_raw)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = _split_train_test(X, y_encoded, test_size=0.2, random_state=42)

    preprocessor = _build_preprocessor(X)
    model_definitions = _build_models(num_classes=len(np.unique(y_encoded)))
    artifact_dir = _ensure_model_artifact_dir() if dataset_id else None

    results: list[dict[str, float | str]] = []

    for model_name, model in model_definitions.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)], memory=None)

        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_duration = time.perf_counter() - start

        y_pred = pipeline.predict(X_test)

        artifact_path: str | None = None
        if dataset_id and artifact_dir is not None:
            artifact_file = artifact_dir / _artifact_filename(dataset_id, model_name)
            joblib.dump(
                {
                    "pipeline": pipeline,
                    "target_column": target_column,
                    "classes": label_encoder.classes_.tolist(),
                },
                artifact_file,
            )
            artifact_path = str(artifact_file)

        results.append(
            {
                "model_name": model_name,
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
                "training_time": float(train_duration),
                "artifact_path": artifact_path,
            }
        )

    return results


def _compute_basic_metadata(df: pd.DataFrame, target_column: str) -> dict:
    samples = len(df)
    features = df.shape[1] - 1 if target_column in df.columns else df.shape[1]
    X = df.drop(columns=[target_column]) if target_column in df.columns else df.copy()
    numeric = len(X.select_dtypes(include=["number"]).columns)
    categorical = len(X.select_dtypes(include=["object", "category", "bool"]).columns)
    missing_ratio = float(X.isna().mean().mean()) if samples > 0 else 0.0
    imbalance = None
    if target_column in df.columns:
        counts = df[target_column].value_counts(dropna=False)
        if len(counts) > 1:
            most = counts.max()
            least = counts.min() if counts.min() > 0 else 1
            imbalance = float(most / least)
    return {
        "samples": samples,
        "features": features,
        "numeric": numeric,
        "categorical": categorical,
        "missing_ratio": missing_ratio,
        "imbalance": float(imbalance) if imbalance is not None else 1.0,
    }


def _score_model_suitability(name: str, metadata: dict, gemini_seeds: list[str]) -> float:
    score = 0.0
    lname = name.lower()
    if name in gemini_seeds:
        score += 50.0

    # Prefer tree/ensemble methods for mixed/tabular data and missing values
    if any(k in lname for k in ["random", "xgboost", "lightgbm", "catboost", "gradient"]):
        score += 20.0
    # Linear models for small feature sets
    if "logistic" in lname or "linear" in lname:
        if metadata["features"] < 50:
            score += 10.0
    # SVM good for smaller sample sizes
    if "support" in lname or "svm" in lname:
        if metadata["samples"] < 5000:
            score += 8.0
    # KNN ok for low-dimensional data
    if "k-nearest" in lname or "knn" in lname:
        if metadata["features"] < 50 and metadata["samples"] < 50000:
            score += 6.0
    # Naive Bayes for high dimensional sparse
    if "gaussiannb" in lname or "naive" in lname:
        if metadata["features"] > 50:
            score += 6.0

    # Slight boost for models that handle categorical features natively
    if "catboost" in lname and metadata["categorical"] > 0:
        score += 8.0

    # penalize very slow models for large datasets
    if metadata["samples"] > 200000 and "svm" in lname:
        score -= 10.0

    # base bias
    score += 1.0
    return score


def select_and_train_models(
    df: pd.DataFrame,
    target_column: str,
    dataset_id: str | None = None,
    shortlist_count: int = 10,
    final_k: int = 3,
    user_instruction: str | None = None,
) -> list[dict[str, float | str]]:
    """Select candidate models using Gemini (RAG) + heuristics, then train top `final_k` models.

    Returns results in same shape as `train_models`.
    """
    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in dataset.")

    metadata = _compute_basic_metadata(df, target_column)

    # sample rows for LLM context
    sample_rows = df.drop(columns=[target_column]).head(15).replace({np.nan: None}).to_dict(orient="records")

    # ask Gemini for recommendation seeds (will use instruction files via RAG)
    try:
        gemini_resp = recommend_models_with_gemini(
            samples=metadata["samples"],
            features=metadata["features"],
            numeric=metadata["numeric"],
            categorical=metadata["categorical"],
            missing_ratio=metadata["missing_ratio"],
            imbalance=metadata["imbalance"],
            user_instruction=user_instruction,
            sample_data=sample_rows,
        )
        gemini_models = list(gemini_resp.get("models", []))
    except Exception:
        gemini_models = []

    num_classes = int(df[target_column].nunique(dropna=True))
    candidates = _build_candidate_models(num_classes=num_classes)

    scored = []
    for name in candidates.keys():
        s = _score_model_suitability(name, metadata, gemini_models)
        scored.append((s, name))

    scored.sort(reverse=True, key=lambda x: x[0])
    shortlisted = [name for _, name in scored][:shortlist_count]

    # train top final_k models from shortlist
    to_train = shortlisted[:final_k]

    # reuse training loop but only for selected models
    X = df.drop(columns=[target_column])
    y_raw = df[target_column]
    y = _prepare_target_for_training(y_raw)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = _split_train_test(X, y_encoded, test_size=0.2, random_state=42)
    preprocessor = _build_preprocessor(X)
    artifact_dir = _ensure_model_artifact_dir() if dataset_id else None

    results: list[dict[str, float | str]] = []
    for model_name in to_train:
        model = candidates[model_name]
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)], memory=None)

        cv_metrics = _evaluate_pipeline_with_cv(pipeline, X, y_encoded)

        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_duration = time.perf_counter() - start

        if cv_metrics is not None:
            accuracy = cv_metrics["accuracy"]
            precision = cv_metrics["precision"]
            recall = cv_metrics["recall"]
            f1 = cv_metrics["f1_score"]
        else:
            y_pred = pipeline.predict(X_test)
            accuracy = float(accuracy_score(y_test, y_pred))
            precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
            recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
            f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        artifact_path: str | None = None
        if dataset_id and artifact_dir is not None:
            artifact_file = artifact_dir / _artifact_filename(dataset_id, model_name)
            joblib.dump(
                {
                    "pipeline": pipeline,
                    "target_column": target_column,
                    "classes": label_encoder.classes_.tolist(),
                },
                artifact_file,
            )
            artifact_path = str(artifact_file)

        results.append(
            {
                "model_name": model_name,
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "training_time": float(train_duration),
                "artifact_path": artifact_path,
            }
        )

    return results

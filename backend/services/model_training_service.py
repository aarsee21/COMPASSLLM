import time

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
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


def train_models(df: pd.DataFrame, target_column: str) -> list[dict[str, float | str]]:
    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in dataset.")

    X = df.drop(columns=[target_column])
    y_raw = df[target_column]

    if y_raw.nunique(dropna=False) < 2:
        raise HTTPException(status_code=400, detail="Target column must contain at least two classes.")

    y = y_raw.astype(str).fillna("missing_target")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    preprocessor = _build_preprocessor(X)
    model_definitions = _build_models(num_classes=len(np.unique(y_encoded)))

    results: list[dict[str, float | str]] = []

    for model_name, model in model_definitions.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)], memory=None)

        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_duration = time.perf_counter() - start

        y_pred = pipeline.predict(X_test)

        results.append(
            {
                "model_name": model_name,
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
                "training_time": float(train_duration),
            }
        )

    return results

import math

import numpy as np
import pandas as pd
from fastapi import HTTPException
from pymfe.mfe import MFE


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, (float, int, np.floating, np.integer)):
            numeric = float(value)
            if math.isnan(numeric) or math.isinf(numeric):
                return None
            return numeric
    except Exception:
        return None
    return None


def _extract_pymfe_features(X: pd.DataFrame, y: pd.Series) -> dict[str, float | int | str | None]:
    x_encoded = pd.get_dummies(X, dummy_na=True)
    y_codes = pd.Categorical(y.astype(str)).codes

    mfe = MFE()
    mfe.fit(x_encoded.to_numpy(), y_codes)
    ft_names, ft_values = mfe.extract()

    extracted: dict[str, float | int | str | None] = {}
    for name, value in zip(ft_names, ft_values):
        safe_value = _safe_float(value)
        extracted[name] = safe_value if safe_value is not None else str(value)

    # Keep payload small and focused for API response.
    return dict(list(extracted.items())[:20])


def compute_dataset_meta_features(df: pd.DataFrame, target_column: str) -> dict[str, object]:
    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in dataset.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    num_samples = int(len(df))
    num_features = int(X.shape[1])
    num_numeric = int(X.select_dtypes(include=["number", "bool"]).shape[1])
    num_categorical = int(num_features - num_numeric)

    total_cells = max(1, X.shape[0] * X.shape[1])
    missing_ratio = float(X.isna().sum().sum() / total_cells)

    class_counts = y.astype(str).value_counts(normalize=True)
    class_distribution = {str(k): float(v) for k, v in class_counts.items()}

    if len(class_counts) <= 1:
        imbalance_ratio = 1.0
    else:
        imbalance_ratio = float(class_counts.max() / max(class_counts.min(), 1e-12))

    pymfe_features = _extract_pymfe_features(X, y)

    return {
        "number_of_samples": num_samples,
        "number_of_features": num_features,
        "numeric_feature_count": num_numeric,
        "categorical_feature_count": num_categorical,
        "missing_value_ratio": missing_ratio,
        "class_distribution": class_distribution,
        "imbalance_ratio": imbalance_ratio,
        "pymfe_meta_features": pymfe_features,
    }

import math

import numpy as np
import pandas as pd
from fastapi import HTTPException


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


def _compute_meta_features(X: pd.DataFrame, y: pd.Series) -> dict[str, float | int | None]:
    """Compute dataset meta-features using pandas/numpy (no pymfe dependency)."""
    num_df = X.select_dtypes(include=["number", "bool"])
    cat_df = X.select_dtypes(exclude=["number", "bool"])

    # --- basic counts ---
    nr_inst = len(X)
    nr_attr = X.shape[1]
    nr_num = num_df.shape[1]
    nr_cat = cat_df.shape[1]
    nr_class = int(y.astype(str).nunique())

    # --- missing values ---
    nr_missing_values = int(X.isna().sum().sum())
    pct_missing = float(nr_missing_values / max(nr_inst * nr_attr, 1))

    # --- class stats ---
    class_freqs = y.astype(str).value_counts(normalize=True)
    majority_class_pct = float(class_freqs.iloc[0]) if len(class_freqs) else 1.0
    minority_class_pct = float(class_freqs.iloc[-1]) if len(class_freqs) > 1 else majority_class_pct
    class_entropy = float(
        -sum(p * math.log2(p) for p in class_freqs if p > 0)
    ) if len(class_freqs) > 1 else 0.0

    # --- numeric feature stats ---
    if nr_num > 0:
        skewness = _safe_float(num_df.skew().mean())
        kurtosis = _safe_float(num_df.kurtosis().mean())
        mean_corr = _safe_float(
            num_df.corr().abs().where(
                np.triu(np.ones(num_df.corr().shape), k=1).astype(bool)
            ).stack().mean()
            if nr_num > 1 else np.nan
        )
    else:
        skewness = kurtosis = mean_corr = None

    # --- categorical cardinality ---
    mean_cat_cardinality = (
        float(cat_df.nunique().mean()) if nr_cat > 0 else None
    )

    return {
        "nr_inst": nr_inst,
        "nr_attr": nr_attr,
        "nr_num": nr_num,
        "nr_cat": nr_cat,
        "nr_class": nr_class,
        "nr_missing_values": nr_missing_values,
        "pct_missing": pct_missing,
        "majority_class_pct": majority_class_pct,
        "minority_class_pct": minority_class_pct,
        "class_entropy": class_entropy,
        "mean_skewness": skewness,
        "mean_kurtosis": kurtosis,
        "mean_feature_correlation": mean_corr,
        "mean_cat_cardinality": mean_cat_cardinality,
    }


def _column_data_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def _build_column_profiles(X: pd.DataFrame) -> list[dict[str, object]]:
    profiles: list[dict[str, object]] = []

    for column in X.columns:
        series = X[column]
        profiles.append(
            {
                "name": str(column),
                "data_type": _column_data_type(series),
                "missing_ratio": float(series.isna().mean()),
                "unique_count": int(series.nunique(dropna=True)),
            }
        )

    return profiles


def compute_dataset_meta_features(
    df: pd.DataFrame,
    target_column: str,
    excluded_columns: list[str] | None = None,
) -> dict[str, object]:
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

    meta_features = _compute_meta_features(X, y)
    column_profiles = _build_column_profiles(X)

    return {
        "number_of_samples": num_samples,
        "number_of_features": num_features,
        "numeric_feature_count": num_numeric,
        "categorical_feature_count": num_categorical,
        "missing_value_ratio": missing_ratio,
        "class_distribution": class_distribution,
        "imbalance_ratio": imbalance_ratio,
        "included_columns": [str(column) for column in X.columns.tolist()],
        "excluded_columns": [str(column) for column in (excluded_columns or [])],
        "column_profiles": column_profiles,
        "meta_features": meta_features,
    }

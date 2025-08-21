"""
Module: evaluation.py

Evaluates a classification model and logs EXACTLY two metrics:
  - accuracy
  - f1_score

Works with either:
  A) evaluate_model(model, test_data_df_with_SOURCE, ...)
  B) evaluate_model(model, X_test_df_or_ndarray, y_test_series_or_array, ...)

Also logs both metrics to MLflow and saves to reports/evaluation_results.json
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, f1_score
from pandas.api.types import is_object_dtype, is_bool_dtype, is_numeric_dtype


def _coerce_features_numeric(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if "SEX" in X.columns and is_object_dtype(X["SEX"]):
        X["SEX"] = (
            X["SEX"]
            .map({"M": 1, "F": 0, "Male": 1, "Female": 0, "m": 1, "f": 0})
            .astype("float64")
        )
    bool_cols = [c for c in X.columns if is_bool_dtype(X[c])]
    if bool_cols:
        X[bool_cols] = X[bool_cols].astype("int8")
    obj_cols = [c for c in X.columns if is_object_dtype(X[c])]
    for c in obj_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    num_cols = [c for c in X.columns if is_numeric_dtype(X[c])]
    for c in num_cols:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())
    return X


def _encode_target(y: Union[pd.Series, np.ndarray, list]) -> np.ndarray:
    """Map labels to 0/1 with 'IN' as positive; accept strings or numeric."""
    if isinstance(y, pd.Series):
        arr = y.values
    else:
        arr = np.asarray(y)
    if arr.dtype.kind in {"U", "S", "O"}:
        return np.array([1 if str(v).upper() == "IN" else 0 for v in arr], dtype=int)
    return np.array([1 if int(v) == 1 else 0 for v in arr], dtype=int)


def _ensure_dataframe(
    X: Union[pd.DataFrame, np.ndarray, list], cols: Optional[list] = None
) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X, columns=cols)


def evaluate_model(
    model,
    test_or_X: Union[pd.DataFrame, np.ndarray],
    y_test: Optional[Union[pd.Series, np.ndarray, list]] = None,
    report_path: str = "reports/evaluation_results.json",
    *,
    run_name: Optional[str] = "evaluation",
    tracking_uri: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate the classification model and write/log exactly two metrics.

    Args:
        model: Trained classifier with .predict().
        test_or_X: Either:
            - DataFrame containing features and 'SOURCE' target, OR
            - Features only (pd.DataFrame/np.ndarray) when y_test is provided
        y_test: Optional y when test_or_X contains only features.
        report_path: Path to write the JSON metrics.
        run_name: MLflow run name (if no active run).
        tracking_uri: Optional MLflow tracking URI.

    Returns:
        dict: {'accuracy': float, 'f1_score': float}
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # Build X and y depending on the calling pattern
    if y_test is None:
        # Expect a single DataFrame with SOURCE
        if not isinstance(test_or_X, pd.DataFrame) or "SOURCE" not in test_or_X.columns:
            raise KeyError(
                "When y_test is None, test_or_X must be a DataFrame containing 'SOURCE'."
            )
        df = test_or_X
        y_true = _encode_target(df["SOURCE"])
        X = df.drop(columns=["SOURCE"])
    else:
        # Features + separate y
        X = _ensure_dataframe(test_or_X)
        y_true = _encode_target(y_test)

    # Predictions
    X = _coerce_features_numeric(X)
    y_pred = model.predict(X)
    y_pred = np.asarray(y_pred).astype(int).ravel()
    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"Prediction length {y_pred.shape[0]} != truth length {y_true.shape[0]}"
        )

    # EXACTLY TWO metrics
    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
    }

    # Save to JSON (ensure directory exists)
    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation metrics to: {report_path}")

    # Log exactly the same two metrics to MLflow
    started_here = False
    if mlflow.active_run() is None:
        mlflow.start_run(run_name=run_name)
        started_here = True
    try:
        mlflow.log_metric("accuracy", results["accuracy"])
        mlflow.log_metric("f1_score", results["f1_score"])
    finally:
        if started_here:
            mlflow.end_run()

    return results


def load_model(filepath: str = "models/model.pkl"):
    """Load a model from disk (joblib)."""
    import joblib

    model = joblib.load(filepath)
    print(f"Model loaded from: {filepath}")
    return model

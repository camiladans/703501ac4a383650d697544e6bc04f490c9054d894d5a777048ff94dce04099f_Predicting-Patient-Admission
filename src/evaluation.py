"""
Module: evaluation.py

Evaluates a classification model on test data and logs EXACTLY two metrics:
  - accuracy
  - f1_score

Also logs both metrics to MLflow.
Results are saved to: reports/evaluation_results.json
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def evaluate_model(
    model,
    test_data: pd.DataFrame,
    report_path: str = "reports/evaluation_results.json",
    *,
    run_name: Optional[str] = "evaluation",
    tracking_uri: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate the classification model and write/log exactly two metrics.

    Args:
        model: Trained classifier with .predict().
        test_data: DataFrame containing features and 'SOURCE' target.
                   'SOURCE' should be 'IN' for inpatients; others treated as 0.
        report_path: Path to write the JSON metrics.
        run_name: MLflow run name (used if no active run exists).
        tracking_uri: If provided, sets mlflow tracking URI (e.g., "http://localhost:5000").

    Returns:
        dict: {'accuracy': float, 'f1_score': float}
    """
    # Optional: point to tracking server
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    if "SOURCE" not in test_data.columns:
        raise KeyError("The column 'SOURCE' is missing from the input DataFrame.")

    # Binary encode target: IN = 1, else 0
    y_true = test_data["SOURCE"].apply(lambda x: 1 if str(x).upper() == "IN" else 0)
    X = test_data.drop(columns=["SOURCE"])

    # Predictions
    y_pred = model.predict(X)

    # EXACTLY TWO metrics
    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
    }

    # Save to JSON
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation metrics to: {report_path}")

    # ----- Log to MLflow (exactly the same two metrics) -----
    started_here = False
    if mlflow.active_run() is None:
        mlflow.start_run(run_name=run_name)
        started_here = True
    else:
        # Create a nested run under the current training run, if any
        mlflow.start_run(run_name=run_name, nested=True)
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
    model = joblib.load(filepath)
    print(f"Model loaded from: {filepath}")
    return model

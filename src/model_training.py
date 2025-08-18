"""
Module: model_training.py

Trains a classification model (LogReg or RF) on processed data.
Prioritizes recall for inpatients ('IN').

Adds a custom MLflow PyFunc wrapper (CustomMLModel) and a helper
to log the trained model + artifacts to MLflow.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split


# ======================================================================
# Custom MLflow PyFunc model wrapper (based on template)
# ======================================================================
class CustomMLModel(mlflow.pyfunc.PythonModel):
    """
    Custom MLflow PyFunc model wrapper for your trained model.
    Encapsulates preprocessing and prediction logic.
    """

    def __init__(self):
        self.model = None
        self.preprocessor = None  # scaler, encoder, etc.
        self.feature_names = None

    def load_context(self, context):
        """Load model artifacts from MLflow context."""
        self.model = joblib.load(context.artifacts["model"])

        # Load preprocessor if exists
        if "preprocessor" in context.artifacts:
            self.preprocessor = joblib.load(context.artifacts["preprocessor"])

        # Load feature names
        if "feature_names" in context.artifacts:
            with open(context.artifacts["feature_names"], "r") as f:
                self.feature_names = [line.strip() for line in f.readlines()]

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        # Apply preprocessing if available
        if self.preprocessor is not None:
            processed_input = self.preprocessor.transform(model_input)
        else:
            # Accept DataFrame or ndarray; fall back to values for DF
            processed_input = (
                model_input.values
                if isinstance(model_input, pd.DataFrame)
                else np.asarray(model_input)
            )

        # Make predictions
        predictions = self.model.predict(processed_input)
        return predictions


# ======================================================================
# Existing training utilities (unchanged with mlflow updates)
# ======================================================================
def train_model(train_data: pd.DataFrame, model_type: str = "logreg"):
    print("Train data columns:", train_data.columns.tolist())
    if "SOURCE" not in train_data.columns:
        raise KeyError("The column 'SOURCE' is missing from the input DataFrame.")

    # Encode target: IN -> 1, else 0
    y = train_data["SOURCE"].apply(lambda x: 1 if str(x).upper() == "IN" else 0)
    X = train_data.drop(columns=["SOURCE"])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if model_type.lower() == "logreg":
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        param_grid = {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "liblinear"]}
    elif model_type.lower() == "rf":
        model = RandomForestClassifier(random_state=42, class_weight="balanced")
        param_grid = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}
    else:
        raise ValueError("Unsupported model_type. Choose 'logreg' or 'rf'.")

    grid = GridSearchCV(model, param_grid, cv=5, scoring="recall")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_val)

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "report": classification_report(y_val, y_pred),
    }

    return best_model, metrics


def save_model(model, filepath: str = "models/model.pkl"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")


# ======================================================================
# Helper to log the model with the custom wrapper to MLflow
# ======================================================================
def _write_feature_names_txt(feature_names: Iterable[str], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for name in feature_names:
            f.write(f"{name}\n")
    return path


def log_model_to_mlflow(
    model,
    *,
    run_name: Optional[str] = None,
    tracking_uri: str = "http://localhost:5000",
    artifact_dir: str = "models_export",
    preprocessor=None,
    feature_names: Optional[Iterable[str]] = None,
):
    """
    Logs the given model using CustomMLModel wrapper.

    Artifacts saved and referenced:
      - model.pkl
      - preprocessor.pkl        (optional)
      - feature_names.txt       (optional)
    """
    mlflow.set_tracking_uri(tracking_uri)

    # Prepare local artifacts
    art_dir = Path(artifact_dir)
    art_dir.mkdir(parents=True, exist_ok=True)

    model_path = art_dir / "model.pkl"
    joblib.dump(model, model_path)

    artifacts = {"model": str(model_path)}

    if preprocessor is not None:
        preproc_path = art_dir / "preprocessor.pkl"
        joblib.dump(preprocessor, preproc_path)
        artifacts["preprocessor"] = str(preproc_path)

    if feature_names is not None:
        fnames_path = _write_feature_names_txt(
            feature_names, art_dir / "feature_names.txt"
        )
        artifacts["feature_names"] = str(fnames_path)

    # Log to MLflow with custom wrapper
    with mlflow.start_run(run_name=run_name):
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=CustomMLModel(),
            artifacts=artifacts,
        )
        run_id = mlflow.active_run().info.run_id
        print(f"Logged MLflow run_id: {run_id}")
        return run_id


# ======================================================================
# Optional CLI/demo usage
# ======================================================================
if __name__ == "__main__":
    # Example: train and log with custom wrapper.
    # Replace with your real training data loading.
    try:
        # Fake small dataset for structure only:
        df = pd.DataFrame(
            {
                "f1": [0.1, 0.2, 0.3, 0.4, 0.5],
                "f2": [1, 0, 1, 0, 1],
                "SOURCE": ["IN", "OUT", "IN", "OUT", "IN"],
            }
        )
        model, _metrics = train_model(df, model_type="rf")
        # Use DataFrame column order as feature names
        feat_names = [c for c in df.columns if c != "SOURCE"]
        log_model_to_mlflow(
            model,
            run_name="rf-inpatient-demo",
            tracking_uri="http://localhost:5000",
            artifact_dir="models_export",
            preprocessor=None,
            feature_names=feat_names,
        )
    except Exception as e:
        print("Demo failed/skipped:", e)

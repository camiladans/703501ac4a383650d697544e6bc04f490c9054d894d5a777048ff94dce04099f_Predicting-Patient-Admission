# src/run_pipeline.py
"""
End-to-end ML pipeline runner.

Steps:
1. Ingest raw data (from CSV or URL)
2. Preprocess: split train/test; create ONE drifted train/test; SAVE:
   - data/train.csv, data/test.csv
   - data/drifted_train.csv, data/drifted_test.csv
3. Train + evaluate model
4. Check performance threshold (accuracy > 0.80 by default)
5. If threshold met, log & register model to MLflow Model Registry
6. Run drift detection twice (test vs drifted_test, train vs drifted_train)
"""

from __future__ import annotations

import os
import json
import logging
from typing import Dict  # expected to contain numeric metrics like {"accuracy": 0.83}

import mlflow
import mlflow.sklearn

from src.data_ingestion import ingest_data
from src.data_preprocessing import preprocess_data
from src.model_training import train_model
from src.evaluation import evaluate_model
from src.drift_detection import detect_drift

MODEL_NAME = "patient_admission_classifier"
ACCURACY_THRESHOLD = 0.80

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _check_performance_threshold(results: Dict[str, float], threshold: float) -> bool:
    os.makedirs("reports", exist_ok=True)
    status = {
        "metric": "accuracy",
        "value": float(results.get("accuracy", float("nan"))),
        "threshold": float(threshold),
        # Spec says ">" (strictly greater than)
        "meets_threshold": bool(results.get("accuracy", 0.0) > threshold),
    }
    with open("reports/performance_check.json", "w") as f:
        json.dump(status, f, indent=2)
    logger.info(
        "Performance check — accuracy=%.4f (threshold=%.2f) -> %s",
        status["value"],
        status["threshold"],
        "PASS" if status["meets_threshold"] else "FAIL",
    )
    return status["meets_threshold"]


def run_pipeline() -> None:
    # Tracking + optional experiment name
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    if exp := os.getenv("MLFLOW_EXPERIMENT_NAME"):
        try:
            mlflow.set_experiment(exp)
        except Exception:
            logger.debug("Could not set MLflow experiment to %s", exp)

    # 0) Ingest raw data -> DataFrame
    df = ingest_data(
        input_path="data/raw/data-ori.csv", canonical_path="data/raw/data-ori.csv"
    )

    # 1) Preprocess (returns splits and writes drifted CSVs)
    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_drifted,
        y_train_drifted,
        X_test_drifted,
        y_test_drifted,
    ) = preprocess_data(df, target_col="SOURCE")

    # Save non-drifted train/test for drift_detection inputs
    os.makedirs("data", exist_ok=True)
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    X_train.assign(SOURCE=y_train).to_csv(train_path, index=False)
    X_test.assign(SOURCE=y_test).to_csv(test_path, index=False)
    logger.info("Saved %s and %s", train_path, test_path)

    with mlflow.start_run() as run:
        # 2) Train
        model, train_metrics, feature_names, best_params = train_model(X_train, y_train)

        # (optional) print/log training metrics so you can see them
        print("Training metrics:", train_metrics)

        # If your y_test is strings like "in"/"out", map to {0,1} to match the scorer expectations
        try:
            import pandas as pd

            if not pd.api.types.is_integer_dtype(y_test):
                y_test = (
                    pd.Series(y_test)
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .map({"in": 1, "out": 0})
                )
                if y_test.isna().any():
                    raise ValueError("y_test contains labels other than IN/OUT.")
                y_test = y_test.astype(int)
        except Exception as e:
            print("Note:", e)

        # 3) Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        for k, v in metrics.items():
            try:
                mlflow.log_metric(k, float(v))
            except Exception:
                logger.debug("Skipping non-numeric metric %s=%r", k, v)

        # 4) Threshold gate
        meets_perf = _check_performance_threshold(metrics, ACCURACY_THRESHOLD)

        # 5) Log model & optionally register
        mlflow.sklearn.log_model(model, artifact_path="model")
        if meets_perf:
            run_id = run.info.run_id
            model_uri = f"runs:/{run_id}/model"
            logger.info("Registering model from: %s", model_uri)
            mlflow.register_model(model_uri, MODEL_NAME)
            logger.info("Model registered under name: %s", MODEL_NAME)
        else:
            logger.warning("Model did NOT meet threshold. Skipping registration.")

        # 6) Drift detection — run twice

        # 6a) Test vs drifted_test
        test_drift = detect_drift("data/test.csv", "data/drifted_test.csv")
        mlflow.log_param("test_drift_detected", test_drift["drift_detected"])
        mlflow.log_param("test_overall_drift_score", test_drift["overall_drift_score"])
        if test_drift["drift_detected"]:
            raise ValueError(
                "Data drift detected in test set! Model retraining required."
            )

        # 6b) Train vs drifted_train (second run)
        if os.path.exists("data/drifted_train.csv"):
            train_drift = detect_drift("data/train.csv", "data/drifted_train.csv")
            mlflow.log_param("train_drift_detected", train_drift["drift_detected"])
            mlflow.log_param(
                "train_overall_drift_score", train_drift["overall_drift_score"]
            )
            if train_drift["drift_detected"]:
                raise ValueError(
                    "Data drift detected in train set! Model retraining required."
                )


if __name__ == "__main__":
    run_pipeline()

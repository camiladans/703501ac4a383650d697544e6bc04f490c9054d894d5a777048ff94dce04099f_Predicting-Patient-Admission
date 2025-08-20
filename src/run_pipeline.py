# src/run_pipeline.py
"""
End-to-end ML pipeline runner.

Steps:
1. Ingest raw data (from CSV or URL)
2. Preprocess: flag ranges, split train/test, create ONE drifted train/test
3. Train + evaluate model
4. Check performance threshold (accuracy > 0.8 by default)
5. If threshold met, log & register model to MLflow Model Registry
6. Run drift detection on test set (twice)
"""

from __future__ import annotations

import json
import os
from typing import Dict

import mlflow
import mlflow.sklearn

from src.data_ingestion import ingest_data
from src.data_preprocessing import preprocess_data
from src.model_training import train_model
from src.evaluation import evaluate_model
from src.drift_detection import detect_drift

MODEL_NAME = "patient_admission_classifier"
ACCURACY_THRESHOLD = 0.80


def _check_performance_threshold(results: Dict[str, float], threshold: float) -> bool:
    os.makedirs("reports", exist_ok=True)
    status = {
        "metric": "accuracy",
        "value": float(results.get("accuracy", float("nan"))),
        "threshold": float(threshold),
        "meets_threshold": bool(results.get("accuracy", 0.0) >= threshold),
    }
    with open("reports/performance_check.json", "w") as f:
        json.dump(status, f, indent=2)
    print(
        f"Performance check — accuracy={status['value']:.4f} "
        f"(threshold={status['threshold']:.2f}) -> "
        f"{'PASS' if status['meets_threshold'] else 'FAIL'}"
    )
    return status["meets_threshold"]


def run_pipeline():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

    # 0) Ingest raw data
    raw_path = ingest_data(source_path="data/raw/data-ori.csv")

    # 1) Preprocess
    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_drifted,
        y_train_drifted,
        X_test_drifted,
        y_test_drifted,
    ) = preprocess_data(raw_path, target_col="SOURCE")

    with mlflow.start_run() as run:
        # 2) Train
        model = train_model(X_train, y_train)

        # 3) Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        for k, v in metrics.items():
            try:
                mlflow.log_metric(k, float(v))
            except Exception:
                pass

        # 4) Threshold gate
        meets_perf = _check_performance_threshold(metrics, ACCURACY_THRESHOLD)

        # 5) Log & register
        mlflow.sklearn.log_model(model, artifact_path="model")
        if meets_perf:
            run_id = run.info.run_id
            model_uri = f"runs:/{run_id}/model"
            print(f"Registering model from: {model_uri}")
            mlflow.register_model(model_uri, MODEL_NAME)
            print(f"Model registered under name: {MODEL_NAME}")
        else:
            print("WARNING — Model did NOT meet threshold. Skipping registration.")

        # 6) Drift detection on test set — run twice

        test_drift_results = detect_drift("data/test.csv", "data/drifted_test.csv")
        mlflow.log_param("test_drift_detected", test_drift_results["drift_detected"])
        mlflow.log_param(
            "test_overall_drift_score", test_drift_results["overall_drift_score"]
        )
        if test_drift_results["drift_detected"]:
            raise ValueError(
                "Data drift detected in test set! Model retraining required."
            )


if __name__ == "__main__":
    run_pipeline()

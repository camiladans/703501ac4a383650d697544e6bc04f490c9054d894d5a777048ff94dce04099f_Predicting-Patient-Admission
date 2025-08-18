"""
Main pipeline script for predicting inpatient vs. outpatient status.
Performs data loading, preprocessing, training, evaluation, drift detection,
and saves artifacts/reports. Uses MLflow at http://localhost:5000.
"""

from __future__ import annotations

import json
import os
from typing import Dict

import mlflow

from data_preprocessing import preprocess_data
from feature_engineering import engineer_features
from model_training import train_model, save_model
from evaluation import evaluate_model
from drift_detection import detect_drift


PERF_REPORT_PATH = "reports/evaluation_results.json"
DRIFT_REPORT_PATH = "reports/drift_report.json"
MIN_ACCURACY = 0.80
MODEL_NAME = "patient_admission_classifier"  # change as meaningful


def _check_performance_threshold(
    results: Dict[str, float], min_accuracy: float
) -> bool:
    """Return True if accuracy meets threshold; also write a small status file."""
    os.makedirs("reports", exist_ok=True)
    status = {
        "metric": "accuracy",
        "value": float(results.get("accuracy", float("nan"))),
        "threshold": float(min_accuracy),
        "meets_threshold": bool(results.get("accuracy", 0.0) >= min_accuracy),
    }
    with open("reports/performance_check.json", "w") as f:
        json.dump(status, f, indent=2)
    print(
        f"Performance check — accuracy={status['value']:.4f} "
        f"(threshold={status['threshold']:.2f}) -> "
        f"{'PASS' if status['meets_threshold'] else 'FAIL'}"
    )
    return status["meets_threshold"]


def main():
    print("Starting patient admission prediction pipeline...")

    # 0) Point MLflow to your local tracking server
    mlflow.set_tracking_uri("http://localhost:5000")

    # Start MLflow run so we can log + register
    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # 1) Load & preprocess
        train_data, test_data = preprocess_data("data/raw/data-ori.csv")

        # 2) Feature engineering
        train_data = engineer_features(train_data)
        test_data = engineer_features(test_data)

        # 3) Train model
        model, train_metrics = train_model(train_data, model_type="logreg")

        # 4) Save model artifact locally
        save_model(model, "models/model.pkl")

        # 5) Evaluate — logs metrics to MLflow
        eval_results = evaluate_model(
            model,
            test_data,
            report_path=PERF_REPORT_PATH,
            tracking_uri="http://localhost:5000",
        )

        # 6) Performance gate
        meets_perf = _check_performance_threshold(eval_results, MIN_ACCURACY)

        # 7) Drift detection (feature-wise PSI between train & test)
        train_X = train_data.drop(columns=["SOURCE"], errors="ignore")
        test_X = test_data.drop(columns=["SOURCE"], errors="ignore")
        drift_summary = detect_drift(
            train_df=train_X,
            test_df=test_X,
            out_path=DRIFT_REPORT_PATH,
            bins=10,
            psi_threshold=0.2,
        )
        print(f"Drift report written to: {DRIFT_REPORT_PATH}")
        print(
            f"PSI flagged features (>{drift_summary['psi_threshold']}): "
            f"{drift_summary['flagged_features']}"
        )

        # 8) If threshold met → register model
        if meets_perf:
            model_uri = f"runs:/{run_id}/model"
            print(f"Registering model from: {model_uri}")
            mlflow.register_model(model_uri, MODEL_NAME)
            print(f"Model registered under name: {MODEL_NAME}")
        else:
            print(
                "WARNING — Model did NOT meet performance threshold. "
                "Skipping registration (document rationale in README)."
            )

    print("Pipeline completed.")


if __name__ == "__main__":
    main()

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
import pandas as pd

import os
import json
import logging
from pathlib import Path

from typing import Dict  # expected to contain numeric metrics like {"accuracy": 0.83}

import mlflow
import mlflow.sklearn

from src.data_ingestion import ingest_data
from src.data_preprocessing import preprocess_data
from src.model_training import train_model
from src.evaluation import evaluate_model
from src.threshold_sweep import run_threshold_experiments
from src.drift_detection import detect_drift
from src.feature_engineering import build_features

MODEL_NAME = "patient_admission_classifier"
ACCURACY_THRESHOLD = 0.80

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MLFLOW_URI = os.environ.get("MLFLOW_URI", "http://mlflow:5000")
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "patient_experiments_v3")
TARGET_COL = "SOURCE"

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)

RAW_PATH = os.getenv("RAW_PATH", "data/raw/data-ori.csv")  # default = local path

"""
NOTE for local run use:
export MLFLOW_URI="file:./mlruns"
python -m src.run_pipeline
"""


def resolve_raw_path() -> str:
    candidates = [
        "/app/data/raw/data-ori.csv",  # inside Docker
        "data/raw/data-ori.csv",  # local
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    # fallback to local default
    return "data/raw/data-ori.csv"


def run_local_eval(model_uri: str, test_fe_csv: str) -> dict:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    df_test = pd.read_csv(test_fe_csv)
    X_test = df_test.drop(columns=[TARGET_COL])
    y_test = df_test[TARGET_COL]

    model = mlflow.pyfunc.load_model(model_uri)

    with mlflow.start_run(run_name="evaluate-local"):
        # accuracy + recall (EXACTLY TWO)
        metrics = evaluate_model(model, X_test, y_test)

        # threshold sweep
        sweep_df, chosen = run_threshold_experiments(
            model,
            X_test,
            y_test,
            thresholds=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50],
            min_recall=0.90,
            objective="f1",
            fallback="f1",
            save_csv_path="reports/threshold_sweep.csv",
            log_to_mlflow=True,
        )
        mlflow.log_metric("chosen_threshold_eval", float(chosen["threshold"]))
        print(
            f"[local] chosen threshold={chosen['threshold']:.3f} "
            f"(recall={chosen['recall']:.3f}, precision={chosen['precision']:.3f}, "
            f"f1={chosen['f1_score']:.3f}, acc={chosen['accuracy']:.3f})"
        )
        return {**metrics, "chosen_threshold": float(chosen["threshold"])}


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
    # 0) Ingest raw data -> DataFrame
    RAW_PATH = resolve_raw_path()
    df = ingest_data(input_path=RAW_PATH, canonical_path=RAW_PATH)

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

    X_train = build_features(X_train)
    X_test = build_features(X_test)

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

        print("Training metrics:", train_metrics)

        # Log training-time threshold + its precision/recall
        if "threshold" in train_metrics:
            val = float(train_metrics["threshold"])
            mlflow.log_metric("threshold", val)
            print(f"[MLflow] threshold = {val:.4f}")

        ts = train_metrics.get("threshold_selection", {})
        if isinstance(ts, dict):
            if "recall_at_t" in ts:
                val = float(ts["recall_at_t"])
                mlflow.log_metric("recall_at_threshold", val)
                print(f"[MLflow] recall_at_threshold = {val:.4f}")
            if "precision_at_t" in ts:
                val = float(ts["precision_at_t"])
                mlflow.log_metric("precision_at_threshold", val)
                print(f"[MLflow] precision_at_threshold = {val:.4f}")

        # 3) Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        for k, v in metrics.items():
            try:
                mlflow.log_metric(k, float(v))
            except Exception:
                logger.debug("Skipping non-numeric metric %s=%r", k, v)

        # 4) Threshold gate
        meets_perf = _check_performance_threshold(metrics, ACCURACY_THRESHOLD)

        # 5) Log model & register
        mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        print(f"[MLflow] model logged at {model_uri}")

        if meets_perf:
            logger.info("Registering model from: %s", model_uri)
            try:
                mlflow.register_model(model_uri, MODEL_NAME)
                logger.info("Model registered under name: %s", MODEL_NAME)
                print(f"[MLflow] model registered as '{MODEL_NAME}' (run_id={run_id})")
            except Exception as e:
                logger.warning(
                    "Model registry unavailable (tracking_uri=%s). "
                    "Skipping registration. Reason: %s",
                    mlflow.get_tracking_uri(),
                    e,
                )

        else:
            logger.warning("Model did NOT meet threshold. Skipping registration.")

        # 6) Drift detection — run twice
        drifted_test_path = "data/drifted_test.csv"
        drifted_train_path = "data/drifted_train.csv"

        # 6a) Test vs drifted_test
        if os.path.exists(drifted_test_path):
            test_drift = detect_drift(test_path, drifted_test_path)
            if test_drift.get("drift_detected", False):
                mlflow.set_tag("data_drift_detected", "true")
                mlflow.log_metric(
                    "test_overall_drift_score",
                    float(test_drift.get("overall_drift_score", float("nan"))),
                )
            else:
                mlflow.set_tag("data_drift_detected", "false")

            # Save JSON report for test drift
            Path("reports").mkdir(parents=True, exist_ok=True)
            with open("reports/test_drift.json", "w") as f:
                json.dump(test_drift, f, indent=2)

        # 6b) Train vs drifted_train
        if os.path.exists(drifted_train_path):
            train_drift = detect_drift(train_path, drifted_train_path)
            mlflow.set_tag(
                "train_drift_detected",
                str(bool(train_drift.get("drift_detected", False))),
            )
            if "overall_drift_score" in train_drift:
                mlflow.log_metric(
                    "train_overall_drift_score",
                    float(train_drift["overall_drift_score"]),
                )

            if train_drift.get("drift_detected", False):
                logging.warning("Data drift detected in train set (continuing run).")

            # Save JSON report for train drift
            with open("reports/train_drift.json", "w") as f:
                json.dump(train_drift, f, indent=2)


if __name__ == "__main__":
    run_pipeline()

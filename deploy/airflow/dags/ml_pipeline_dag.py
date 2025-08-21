# ruff: noqa: E402
from __future__ import annotations

"""
Airflow DAG: ml_pipeline_dag
Flow:
  1) step_preprocess  -> ingest, preprocess, save data/train.csv & data/test.csv
  2) step_train       -> train on train.csv, log model to MLflow, return model_uri
  3) step_evaluate    -> load model from model_uri, evaluate on test.csv, log metrics
  4) drift_detection  -> detect drift (test vs drifted_test, train vs drifted_train), log + raise if drift
"""

# --- stdlib ---
from datetime import datetime
import logging
import os
import sys

# --- third-party ---
from airflow.decorators import dag, task

# Ensure project modules are importable
if "/app/src" not in sys.path:
    sys.path.append("/app/src")  # noqa: E402

# --- first-party (project) ---
from src.data_ingestion import ingest_data  # noqa: E402
from src.data_preprocessing import preprocess_data  # noqa: E402
from src.model_training import train_model  # noqa: E402
from src.evaluation import evaluate_model  # noqa: E402
from src.drift_detection import detect_drift  # noqa: E402

# ---- Config ----
logger = logging.getLogger(__name__)
MLFLOW_URI = "http://mlflow:5000"
TARGET_COL = "SOURCE"
RAW_SOURCE = "/app/data/raw/data-ori.csv"  # inside container


@dag(
    dag_id="ml_pipeline_dag",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["ml", "drift"],
)
def pipeline():
    @task(task_id="step_preprocess")
    def step_preprocess() -> dict:
        """Ingest, preprocess (creates drifted copies), and save train/test CSVs."""
        import pandas as pd  # local import keeps DAG parse light

        df = ingest_data(source_path=RAW_SOURCE)

        (
            X_train,
            X_test,
            y_train,
            y_test,
            _X_train_drifted,
            _y_train_drifted,
            _X_test_drifted,
            _y_test_drifted,
        ) = preprocess_data(df, target_col=TARGET_COL)

        os.makedirs("data", exist_ok=True)
        train_path = "data/train.csv"
        test_path = "data/test.csv"
        pd.concat([X_train, y_train.rename(TARGET_COL)], axis=1).to_csv(
            train_path, index=False
        )
        pd.concat([X_test, y_test.rename(TARGET_COL)], axis=1).to_csv(
            test_path, index=False
        )

        return {
            "train_path": train_path,
            "test_path": test_path,
            "drifted_train_path": "data/drifted_train.csv",
            "drifted_test_path": "data/drifted_test.csv",
        }

    @task(task_id="step_train")
    def step_train(paths: dict) -> dict:
        """Train on train.csv, log model to MLflow, return {'model_uri': ...}."""
        import pandas as pd
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(MLFLOW_URI)

        df_train = pd.read_csv(paths["train_path"])
        X_train = df_train.drop(columns=[TARGET_COL])
        y_train = df_train[TARGET_COL]

        model = train_model(X_train, y_train)

        with mlflow.start_run(run_name="train"):
            mlflow.sklearn.log_model(model, artifact_path="model")
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"

        return {"model_uri": model_uri}

    @task(task_id="step_evaluate")
    def step_evaluate(paths: dict, train_out: dict) -> dict:
        """Load model via MLflow URI and evaluate on test.csv; log metrics."""
        import pandas as pd
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(MLFLOW_URI)

        df_test = pd.read_csv(paths["test_path"])
        X_test = df_test.drop(columns=[TARGET_COL])
        y_test = df_test[TARGET_COL]

        model = mlflow.sklearn.load_model(train_out["model_uri"])
        metrics = evaluate_model(model, X_test, y_test)

        with mlflow.start_run(run_name="evaluate"):
            for k, v in metrics.items():
                try:
                    mlflow.log_metric(k, float(v))
                except Exception:
                    pass

        return metrics

    @task(task_id="drift_detection")
    def drift_detection_task(paths: dict) -> bool:
        """
        Run drift twice (test vs drifted_test, train vs drifted_train),
        log params + full JSON artifacts, and raise on drift.
        """
        import json
        import tempfile
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_URI)

        with mlflow.start_run(run_name="drift_detection"):
            # Test vs drifted_test
            test_drift = detect_drift(paths["test_path"], paths["drifted_test_path"])
            mlflow.log_param("test_drift_detected", test_drift["drift_detected"])
            mlflow.log_param(
                "test_overall_drift_score", test_drift["overall_drift_score"]
            )
            with tempfile.TemporaryDirectory() as td:
                p = os.path.join(td, "drift_report_test.json")
                with open(p, "w") as f:
                    json.dump(test_drift, f, indent=2)
                mlflow.log_artifact(p, artifact_path="drift")
            if test_drift["drift_detected"]:
                raise ValueError(
                    "Data drift detected in TEST set! Model retraining required."
                )

            # Train vs drifted_train
            if os.path.exists(paths["drifted_train_path"]):
                train_drift = detect_drift(
                    paths["train_path"], paths["drifted_train_path"]
                )
                mlflow.log_param("train_drift_detected", train_drift["drift_detected"])
                mlflow.log_param(
                    "train_overall_drift_score", train_drift["overall_drift_score"]
                )
                with tempfile.TemporaryDirectory() as td:
                    p = os.path.join(td, "drift_report_train.json")
                    with open(p, "w") as f:
                        json.dump(train_drift, f, indent=2)
                    mlflow.log_artifact(p, artifact_path="drift")
                if train_drift["drift_detected"]:
                    raise ValueError(
                        "Data drift detected in TRAIN set! Model retraining required."
                    )

        return True

    # Orchestration
    paths = step_preprocess()
    train_out = step_train(paths)
    _ = step_evaluate(paths, train_out)
    _ = drift_detection_task(paths)


dag = pipeline()

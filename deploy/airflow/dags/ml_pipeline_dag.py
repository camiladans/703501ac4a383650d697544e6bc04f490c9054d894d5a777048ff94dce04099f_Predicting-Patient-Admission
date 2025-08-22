# ruff: noqa: E402
from __future__ import annotations

"""
Airflow DAG: ml_pipeline_dag

Flow:
  1) step_preprocess  -> ingest, preprocess, save data/train.csv & data/test.csv
  2) step_train       -> train on train.csv, log model to MLflow (pyfunc), return model_uri
  3) step_evaluate    -> load model via model_uri (pyfunc), evaluate on test.csv, log metrics
  4) drift_detection  -> detect drift (test vs drifted_test, train vs drifted_train), log + raise if drift
"""

# --- stdlib ---
from datetime import datetime
import logging
import os
import sys

# --- third-party ---
from airflow.decorators import dag, task

# Make project modules importable inside the container
if "/app/src" not in sys.path:
    sys.path.append("/app/src")  # noqa: E402

# --- first-party (project) ---
from src.data_ingestion import ingest_data  # noqa: E402
from src.data_preprocessing import preprocess_data  # noqa: E402
from src.model_training import train_and_log  # <<< swapped import
from src.evaluation import evaluate_model  # noqa: E402
from src.drift_detection import detect_drift  # noqa: E402
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


def ensure_experiment(name: str) -> str:
    """
    Ensure an MLflow experiment exists with server-managed artifact location.
    Returns experiment_id. Requires mlflow server started with --serve-artifacts.
    """
    client = MlflowClient()
    exp = client.get_experiment_by_name(name)
    if exp is None:
        return client.create_experiment(name, artifact_location="mlflow-artifacts:/")
    return exp.experiment_id


# In Docker, prefer service name "mlflow" rather than localhost
MLFLOW_URI = "http://mlflow:5000"
TARGET_COL = "SOURCE"
RAW_SOURCE = "/app/data/raw/data-ori.csv"  # path inside container


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

        df_or_path = ingest_data(source_path=RAW_SOURCE)
        df = pd.read_csv(df_or_path) if isinstance(df_or_path, str) else df_or_path

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
        """
        Train on train.csv and log model to MLflow (PyFunc).
        Returns {'model_uri': 'runs:/<run_id>/model'}.
        """
        import pandas as pd
        import mlflow

        # Ensure the container talks to the MLflow service
        mlflow.set_tracking_uri(MLFLOW_URI)
        ensure_experiment("patient_admission_v3")
        mlflow.set_experiment("patient_admission_v3")

        df_train = pd.read_csv(paths["train_path"])

        # train_and_log() handles: start_run, params/metrics logging, and PyFunc logging
        run_id = train_and_log(
            df_train,
            model_type="rf",  # or "logreg"
            target_recall=0.90,
            random_state=42,
            experiment="patient_admission",
            run_name="rf-train",
        )

        model_uri = f"runs:/{run_id}/model"  # pyfunc artifact path
        return {"model_uri": model_uri}

    @task(task_id="step_evaluate")
    def step_evaluate(paths: dict, train_out: dict) -> dict:
        """Load model via MLflow URI (pyfunc) and evaluate on test.csv; log metrics."""
        import pandas as pd
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_URI)
        ensure_experiment("patient_admission_v3")
        mlflow.set_experiment("patient_admission_v3")

        df_test = pd.read_csv(paths["test_path"])
        X_test = df_test.drop(columns=[TARGET_COL])
        y_test = df_test[TARGET_COL]

        model = mlflow.pyfunc.load_model(train_out["model_uri"])

        # evaluate_model should accept (model, X, y) and log/return metrics
        metrics = evaluate_model(model, X_test, y_test)

        with mlflow.start_run(run_name="evaluate"):
            # Log scalar metrics (skip non-scalars)
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
        ensure_experiment("patient_admission_v3")
        mlflow.set_experiment("patient_admission_v3")

        with mlflow.start_run(run_name="drift_detection"):
            # Test vs drifted_test
            test_drift = detect_drift(paths["test_path"], paths["drifted_test_path"])
            mlflow.log_param("test_drift_detected", test_drift["drift_detected"])
            mlflow.log_param(
                "test_overall_drift_score", test_drift.get("overall_drift_score", None)
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

            # Train vs drifted_train (if available)
            if os.path.exists(paths["drifted_train_path"]):
                train_drift = detect_drift(
                    paths["train_path"], paths["drifted_train_path"]
                )
                mlflow.log_param("train_drift_detected", train_drift["drift_detected"])
                mlflow.log_param(
                    "train_overall_drift_score",
                    train_drift.get("overall_drift_score", None),
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

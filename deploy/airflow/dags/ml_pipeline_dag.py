# ruff: noqa: E402
from __future__ import annotations

"""
Airflow DAG: ml_pipeline_dag
============================

Purpose
-------
End-to-end ML pipeline for predicting patient admission (IN vs OUT) with
clear, reproducible steps and MLflow tracking. This DAG:

1) Preprocesses raw labs, adds clinical flags, creates drifted copies
2) Feature-engineers clinically meaningful numeric features
3) Trains a classifier and logs to MLflow using a **custom PyFunc wrapper**
4) Evaluates the logged model on hold-out data and logs metrics
5) Runs drift detection** between clean vs. drifted datasets and logs a report

Design Principles
-----------------
- File-oriented I/O (CSV) for debuggability and portability
- Minimal, explicit XCom usage: each task returns a small dict of paths/URIs
- MLflow used as the system of record params, metrics, artifacts, models
- Works in containers: paths assume Airflow's working dir is /opt/airflow

Assumptions & Prereqs
---------------------
- MLflow server is reachable at http://mlflow:5000 and started with
  --serve-artifacts. The DAG creates/uses experiment: patient_experiments_v3 (see EXPERIMENT constant)
- Docker Compose mounts on Airflow services (scheduler/apiserver):
    ./data   -> /opt/airflow/data
    ./reports-> /opt/airflow/reports
- The raw CSV is available inside containers at /app/data/raw/data-ori.csv
- Project modules live at /app/src

Task Graph (exact order)
------------------------
1. Step_preprocess
   - Ingests raw data; adds boolean lab flags (normal/abnormal), splits train/test, synthesizes drifted copies
   - Writes:
     - data/train.csv, data/test.csv
     - data/drifted_train.csv, data/drifted_test.csv
   - XCom: dict with the four paths above

2. Feature_engineering
   - Reads train/test and adds derived numeric features + keeps originals
   - Writes data/train_fe.csv, data/test_fe.csv
   - XCom: {"train_fe": "...", "test_fe": "..."}

3. Step_train
   - Trains a RandomForest (or LogisticRegression) with Recall (metric) oriented tuning
   - Logs 3 hyperparameters and other scalar metrics
   - Logs a Custom PyFunc model
   - XCom: {"model_uri": "runs:/<run_id>/model"}

4. Step_evaluate
   - Loads the PyFunc model from MLflow by URI; evaluates on test_fe.csv
   - Logs scalar metrics into a new MLflow run named evaluate

5. Drift_detection
   - Compares test.csv vs drifted_test.csv and optionally train vs drifted_train
   - Produces a slim JSON report with per-feature PSI and an overall average:
     reports/drift_report.json (overwritten each run)
   - Logs drift artifacts to MLflow. If drift is detected, this task raises.

Key Artifacts & Where They Live
-------------------------------
- Data: /opt/airflow/data (mounted from host ./data)
  - train.csv, test.csv, train_fe.csv, test_fe.csv
  - drifted_train.csv, drifted_test.csv
- Reports: /opt/airflow/reports (mounted from host ./reports)
  - drift_report.json

How to Trigger
--------------
- Airflow UI: run ml_pipeline_dag

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
from src.feature_engineering import run_feature_engineering  # noqa: E402
from src.model_training import train_and_log  # noqa: E402
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
EXPERIMENT = "patient_experiments_v3"  # <<< keep consistent across tasks

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

    @task(task_id="feature_engineering")
    def feature_engineering(paths: dict) -> dict:
        """
        Reads data/train.csv & data/test.csv; writes data/train_fe.csv & data/test_fe.csv.
        Uses src/feature_engineering.run_feature_engineering().
        """
        return run_feature_engineering(paths["train_path"], paths["test_path"])

    @task(task_id="step_train")
    def step_train(fe: dict) -> dict:
        """
        Train on train_fe.csv and log model to MLflow (PyFunc).
        Returns {'model_uri': 'runs:/<run_id>/model'}.
        """
        import pandas as pd
        import mlflow

        # Ensure the container talks to the MLflow service
        mlflow.set_tracking_uri(MLFLOW_URI)
        exp_id = ensure_experiment(EXPERIMENT)
        mlflow.set_experiment(EXPERIMENT)

        df_train = pd.read_csv(fe["train_fe"])

        # train_and_log() handles: start_run, params/metrics logging, and PyFunc logging
        run_id = train_and_log(
            df_train,
            model_type="rf",  # or "logreg"
            target_recall=0.90,
            random_state=42,
            experiment=EXPERIMENT,
            run_name="rf-train",
        )

        print(
            f"🏃 View run rf-train at: {MLFLOW_URI}/#/experiments/{exp_id}/runs/{run_id}"
        )
        print(f"🧪 View experiment at: {MLFLOW_URI}/#/experiments/{exp_id}")

        model_uri = f"runs:/{run_id}/model"  # pyfunc artifact path
        return {"model_uri": model_uri}

    @task(task_id="step_evaluate")
    def step_evaluate(fe: dict, train_out: dict) -> dict:
        """Load model via MLflow URI (pyfunc) and evaluate on test_fe.csv; log metrics."""
        import pandas as pd
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_URI)
        exp_id = ensure_experiment(EXPERIMENT)
        mlflow.set_experiment(EXPERIMENT)

        df_test = pd.read_csv(fe["test_fe"])
        X_test = df_test.drop(columns=[TARGET_COL])
        y_test = df_test[TARGET_COL]

        model = mlflow.pyfunc.load_model(train_out["model_uri"])

        # evaluate_model should accept (model, X, y) and log/return metrics
        metrics = evaluate_model(model, X_test, y_test)

        with mlflow.start_run(run_name="evaluate") as r:
            # Log scalar metrics (skip non-scalars)
            for k, v in metrics.items():
                try:
                    mlflow.log_metric(k, float(v))
                except Exception:
                    pass
            run_id = r.info.run_id

        print(
            f"📏 View evaluation run at: {MLFLOW_URI}/#/experiments/{exp_id}/runs/{run_id}"
        )
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
        exp_id = ensure_experiment(EXPERIMENT)
        mlflow.set_experiment(EXPERIMENT)

        with mlflow.start_run(run_name="drift_detection") as r:
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

            run_id = r.info.run_id

        print(
            f"🛰️  View drift run at: {MLFLOW_URI}/#/experiments/{exp_id}/runs/{run_id}"
        )
        print(f"🧪 View experiment at: {MLFLOW_URI}/#/experiments/{exp_id}")
        return True

    # Orchestration
    paths = step_preprocess()
    fe = feature_engineering(paths)
    train_out = step_train(fe)
    _ = step_evaluate(fe, train_out)
    _ = drift_detection_task(paths)


dag = pipeline()

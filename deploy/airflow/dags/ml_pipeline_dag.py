# ruff: noqa: E402
from __future__ import annotations

"""
Airflow DAG: ml_pipeline_dag (with branching retrain on drift)
==============================================================

Purpose
-------
End-to-end ML pipeline for predicting patient admission (IN vs OUT) with
clear, reproducible steps, MLflow tracking, and *branching* when drift is detected.

Primary Tasks (exactly five)
----------------------------
1) preprocess_data
   - Ingests raw data; adds boolean lab flags (normal/abnormal);
     splits train/test; synthesizes drifted copies.
   - Writes:
     - data/train.csv, data/test.csv
     - data/drifted_train.csv, data/drifted_test.csv
   - XCom: dict of those paths.

2) feature_engineering
   - Reads train/test and adds derived numeric features; keeps originals.
   - Writes: data/train_fe.csv, data/test_fe.csv
   - XCom: {"train_fe": "...", "test_fe": "..."}

3) train_model
   - Trains classifier (RF or LogReg) with recall-oriented tuning.
   - Logs exactly 3 hyperparameters to MLflow (assignment rubric).
   - Logs custom **PyFunc** model + artifacts.
   - XCom: {"model_uri": "runs:/<run_id>/model"}.

4) evaluate_model
   - Loads the MLflow PyFunc model by URI.
   - Evaluates on **test_fe.csv** and logs scalar metrics.

5) drift_detection   (PythonOperator as required)
   - Calls your drift detection function:
     detect_drift("data/test.csv", "data/drifted_test.csv")
   - The function writes (overwrites) **reports/drift_report.json** with:
       {
         "drift_detected": bool,
         "feature_drifts": {"feature1": float, ...},
         "overall_drift_score": float
       }

Branching Logic (post drift_detection)
--------------------------------------
- BranchPythonOperator task_id="branch_on_drift" reads **reports/drift_report.json**
  (not XCom) and routes to:
    • retrain_model  — if drift_detected == True
    • pipeline_complete — otherwise
- End tasks:
    • retrain_model: Calls the same training logic again on the **original (non-drifted) data**
      (we reuse **data/train_fe.csv**).
    • pipeline_complete: No-op marker for success path.

MLflow Integration
------------------
- Tracking URI: http://mlflow:5000  (constant `MLFLOW_URI`)
- Experiment: **patient_experiments_v3** (constant `EXPERIMENT`)
- The DAG ensures the experiment exists with server-side artifacts
  (`artifact_location="mlflow-artifacts:/"`), which requires MLflow server
  to run with `--serve-artifacts`.

Paths & Mounts (expected inside containers)
-------------------------------------------
- Airflow working dir: /opt/airflow
- Data:     ./data     -> /opt/airflow/data
- Reports:  ./reports  -> /opt/airflow/reports
- Raw CSV:  /app/data/raw/data-ori.csv  (adjust RAW_SOURCE if needed)
- Project src: /app/src  (added to sys.path below)

How to Trigger
--------------
- From Airflow UI: run **ml_pipeline_dag**
- Or inside scheduler container:
    airflow tasks test ml_pipeline_dag preprocess_data 2025-01-01

Troubleshooting
---------------
- PermissionError on /mlflow: ensure MLflow experiment uses server-side artifact store
  (this DAG’s `ensure_experiment` does that).
- Connection refused: confirm `mlflow` service is reachable by Airflow (same Docker network).
- Missing files: verify docker-compose mounts to /opt/airflow/data and /opt/airflow/reports
  for scheduler/webserver/apiserver.
"""

from datetime import datetime
import json
import logging
import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn

# --- airflow ---
from airflow.decorators import dag, task
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

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
from src.threshold_sweep import run_threshold_experiments

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


@dag(
    dag_id="ml_pipeline_dag",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["ml", "drift"],
)
def pipeline():
    # ---------------------------------------------------------------------
    # 1) preprocess_data  (TaskFlow API)
    # ---------------------------------------------------------------------
    @task(task_id="preprocess_data")
    def preprocess_task() -> dict:
        """Ingest, preprocess (creates drifted copies), and save train/test CSVs."""

        df = ingest_data(
            input_path="/app/data/raw/data-ori.csv",
            canonical_path="/app/data/raw/data-ori.csv",
        )

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

        # write drifted copies (input for drift_detection)
        pd.concat(
            [_X_train_drifted, _y_train_drifted.rename(TARGET_COL)], axis=1
        ).to_csv("data/drifted_train.csv", index=False)
        pd.concat([_X_test_drifted, _y_test_drifted.rename(TARGET_COL)], axis=1).to_csv(
            "data/drifted_test.csv", index=False
        )

        return {
            "train_path": train_path,
            "test_path": test_path,
            "drifted_train_path": "data/drifted_train.csv",
            "drifted_test_path": "data/drifted_test.csv",
        }

    # ---------------------------------------------------------------------
    # 2) feature_engineering  (TaskFlow API)
    # ---------------------------------------------------------------------
    @task(task_id="feature_engineering")
    def feature_engineering_task(paths: dict) -> dict:
        """
        Reads data/train.csv & data/test.csv; writes data/train_fe.csv & data/test_fe.csv.
        Uses src/feature_engineering.run_feature_engineering().
        """

        out = run_feature_engineering(
            train_csv=paths["train_path"],
            test_csv=paths["test_path"],
            target_col=TARGET_COL,
        )
        return out  # {"train_fe": "...", "test_fe": "..."}

    # ---------------------------------------------------------------------
    # 3) train_model  (TaskFlow API)
    # ---------------------------------------------------------------------
    @task(task_id="train_model")
    def train_task(fe: dict) -> dict:
        """
        Train on train_fe.csv and log model to MLflow (PyFunc).
        Returns {'model_uri': 'runs:/<run_id>/model'}.
        """

        mlflow.set_tracking_uri(MLFLOW_URI)
        exp_id = ensure_experiment(EXPERIMENT)
        mlflow.set_experiment(EXPERIMENT)

        df_train = pd.read_csv(fe["train_fe"])

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

        model_uri = f"runs:/{run_id}/model"
        return {"model_uri": model_uri}

    # ---------------------------------------------------------------------
    # 4) evaluate_model  (TaskFlow API)
    # ---------------------------------------------------------------------
    @task(task_id="evaluate_model")
    def evaluate_task(fe: dict, train_out: dict) -> dict:
        """Load model via MLflow URI (pyfunc) and evaluate on test_fe.csv;
        log accuracy+recall, and run threshold sweep to choose a threshold."""

        mlflow.set_tracking_uri(MLFLOW_URI)
        exp_id = ensure_experiment(EXPERIMENT)
        mlflow.set_experiment(EXPERIMENT)

        df_test = pd.read_csv(fe["test_fe"])
        X_test = df_test.drop(columns=[TARGET_COL])
        y_test = df_test[TARGET_COL]

        model = mlflow.pyfunc.load_model(train_out["model_uri"])
        metrics = evaluate_model(model, X_test, y_test)

        with mlflow.start_run(run_name="evaluate") as r:
            for k, v in metrics.items():
                try:
                    mlflow.log_metric(k, float(v))
                except Exception:
                    pass
            run_id = r.info.run_id
            # 2) Threshold sweep (guarded)
            chosen = None
            try:
                # Prefer sklearn flavor if you logged it; else use pyfunc only if it can give probs
                sk_model = None
                try:
                    sk_uri = train_out.get("sk_model_uri")
                    if sk_uri:
                        sk_model = mlflow.sklearn.load_model(sk_uri)
                except Exception:
                    sk_model = None

                def _has_probs(m):
                    return hasattr(m, "predict_proba") or hasattr(
                        m, "decision_function"
                    )

                sweep_model = (
                    sk_model
                    if sk_model is not None
                    else (model if _has_probs(model) else None)
                )
                if sweep_model is not None:
                    sweep_df, chosen = run_threshold_experiments(
                        sweep_model,
                        X_test,
                        y_test,
                        thresholds=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50],
                        min_recall=0.90,
                        objective="f1",
                        fallback="f1",
                        save_csv_path="reports/threshold_sweep.csv",
                        log_to_mlflow=True,
                    )
                    mlflow.log_metric(
                        "chosen_threshold_eval", float(chosen["threshold"])
                    )
                else:
                    mlflow.set_tag("threshold_sweep_skipped", "no_predict_proba")
                    mlflow.log_metric("chosen_threshold_eval", 0.2)
            except Exception as e:
                # If sweep threw, don't fail the task—log and continue
                mlflow.set_tag("threshold_sweep_error", type(e).__name__)
                mlflow.log_metric("chosen_threshold_eval", 0.2)

        print(
            f"📏 View evaluation run at: {MLFLOW_URI}/#/experiments/{exp_id}/runs/{run_id}"
        )

        # Return both the scalar metrics and the chosen threshold via XCom
        return {
            **metrics,
            "chosen_threshold": float(chosen["threshold"])
            if chosen is not None
            else 0.5,
        }

    # ---------------------------------------------------------------------
    # 5) drift_detection  (PythonOperator required by spec)
    # ---------------------------------------------------------------------
    def _drift_callable():
        """Call detect_drift() and rely on it to write reports/drift_report.json."""
        # We compare clean test vs drifted_test (filenames are deterministic)
        detect_drift("data/test.csv", "data/drifted_test.csv")
        # detect_drift prints the path and dict; no XCom here per spec.

    drift_detection = PythonOperator(
        task_id="drift_detection",
        python_callable=_drift_callable,
    )

    # ---------------------------------------------------------------------
    # Branching: BranchPythonOperator reads reports/drift_report.json
    # ---------------------------------------------------------------------
    def _branch_on_drift() -> str:
        """Return task_id 'retrain_model' if drift_detected, else 'pipeline_complete'."""
        report_path = "reports/drift_report.json"
        if not os.path.exists(report_path):
            # Conservative: if the file isn't there, do not retrain (treat as no drift)
            logger.warning(
                "Drift report not found at %s; proceeding to pipeline_complete.",
                report_path,
            )
            return "pipeline_complete"
        with open(report_path, "r") as f:
            data = json.load(f)
        drift_flag = bool(data.get("drift_detected", False))
        return "retrain_model" if drift_flag else "pipeline_complete"

    branch_on_drift = BranchPythonOperator(
        task_id="branch_on_drift",
        python_callable=_branch_on_drift,
    )

    # ---------------------------------------------------------------------
    # End tasks
    #   - retrain_model: rerun training on original (non-drifted) data
    #   - pipeline_complete: terminal success marker
    # ---------------------------------------------------------------------
    @task(task_id="retrain_model")
    def retrain_task() -> dict:
        """Retrain on *original* non-drifted train_fe.csv and log to MLflow."""

        train_fe_path = "data/train_fe.csv"  # original FE output
        if not os.path.exists(train_fe_path):
            raise FileNotFoundError(
                "Expected data/train_fe.csv not found. Did feature_engineering run?"
            )

        mlflow.set_tracking_uri(MLFLOW_URI)
        exp_id = ensure_experiment(EXPERIMENT)
        mlflow.set_experiment(EXPERIMENT)

        df_train = pd.read_csv(train_fe_path)
        run_id = train_and_log(
            df_train,
            model_type="rf",
            target_recall=0.90,
            random_state=42,
            experiment=EXPERIMENT,
            run_name="rf-retrain",
        )
        print(
            f"♻️  View retrain run at: {MLFLOW_URI}/#/experiments/{exp_id}/runs/{run_id}"
        )
        return {"model_uri": f"runs:/{run_id}/model"}

    pipeline_complete = EmptyOperator(task_id="pipeline_complete")

    # ----------------------------
    # Orchestration / Dependencies
    # ----------------------------
    paths = preprocess_task()
    fe = feature_engineering_task(paths)
    trained = train_task(fe)
    evaluated = evaluate_task(fe, trained)

    # Fixed order per spec:
    # preprocess_data >> feature_engineering >> train_model >> evaluate_model
    (
        evaluated
        >> drift_detection
        >> branch_on_drift
        >> [retrain_task(), pipeline_complete]
    )


dag = pipeline()

# ruff: noqa: E402
from __future__ import annotations

"""
Airflow DAG: ml_pipeline_dag

Flow (exactly 5 primary tasks):
  1) preprocess_data      -> ingest, preprocess, save data/train.csv & data/test.csv (+ drifted_* files)
  2) feature_engineering  -> (light/no-op) write data/train_fe.csv & data/test_fe.csv
  3) train_model          -> train on train_fe.csv, log model to MLflow (pyfunc), return model_uri
  4) evaluate_model       -> load model via model_uri (pyfunc), evaluate on test_fe.csv, log metrics
  5) drift_detection      -> PythonOperator: call detect_drift on data/test.csv vs data/drifted_test.csv, write reports/drift_report.json

Branching:
  - branch_on_drift (BranchPythonOperator) reads reports/drift_report.json (not XCom)
  - if drift_detected=True  -> retrain_model (same training logic on original/non-drifted data)
    else                    -> pipeline_complete

Dependencies:
  preprocess_data >> feature_engineering >> train_model >> evaluate_model
  >> drift_detection >> branch_on_drift >> [retrain_model, pipeline_complete]
"""

# --- stdlib ---
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# --- airflow ---
from airflow.decorators import dag, task
from airflow.operators.python import PythonOperator, BranchPythonOperator

# Make project modules importable inside the container
if "/app/src" not in sys.path:
    sys.path.append("/app/src")  # noqa: E402

# --- first-party (project) ---
from src.data_ingestion import ingest_data  # noqa: E402
from src.data_preprocessing import preprocess_data as preprocess_fn  # noqa: E402
from src.model_training import train_and_log  # noqa: E402
from src.evaluation import evaluate_model as eval_fn  # noqa: E402
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
EXPERIMENT = "patient_admission_v3"  # keep consistent across tasks

TARGET_COL = "SOURCE"
RAW_SOURCE = "/app/data/raw/data-ori.csv"  # path inside container

DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")
DRIFT_REPORT_PATH = REPORTS_DIR / "drift_report.json"


# ---------- PythonOperator targets (module-level, picklable) ----------
def drift_detection_op():
    """
    Runs drift detection on test.csv vs drifted_test.csv and writes reports/drift_report.json.
    Does not raise; branching is handled by BranchPythonOperator.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ref = DATA_DIR / "test.csv"
    cur = DATA_DIR / "drifted_test.csv"

    if not ref.exists() or not cur.exists():
        result = {
            "drift_detected": False,
            "error": "Reference or current file missing.",
        }
    else:
        result = detect_drift(str(ref), str(cur))

    with open(DRIFT_REPORT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote drift report: {DRIFT_REPORT_PATH} -> {result}")


def branch_on_drift_func():
    """
    Reads reports/drift_report.json and returns the next task id:
      - 'retrain_model' if drift_detected=True
      - 'pipeline_complete' otherwise
    """
    try:
        with open(DRIFT_REPORT_PATH, "r") as f:
            rpt = json.load(f)
        drifted = bool(rpt.get("drift_detected", False))
    except Exception as e:
        print(f"Branch read error: {e}; defaulting to NO DRIFT.")
        drifted = False

    return "retrain_model" if drifted else "pipeline_complete"


@dag(
    dag_id="ml_pipeline_dag",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["ml", "drift"],
)
def pipeline():
    # 1) preprocess_data
    @task(task_id="preprocess_data")
    def preprocess_data() -> dict:
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
        ) = preprocess_fn(df, target_col=TARGET_COL)

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        train_path = DATA_DIR / "train.csv"
        test_path = DATA_DIR / "test.csv"
        pd.concat([X_train, y_train.rename(TARGET_COL)], axis=1).to_csv(
            train_path, index=False
        )
        pd.concat([X_test, y_test.rename(TARGET_COL)], axis=1).to_csv(
            test_path, index=False
        )

        # Return canonical paths (not for branching; just for subsequent tasks via XCom)
        return {
            "train_path": str(train_path),
            "test_path": str(test_path),
            "drifted_train_path": str(DATA_DIR / "drifted_train.csv"),
            "drifted_test_path": str(DATA_DIR / "drifted_test.csv"),
        }

    # 2) feature_engineering
    @task(task_id="feature_engineering")
    def feature_engineering(paths: dict) -> dict:
        """
        Minimal/no-op feature engineering step to meet pipeline requirement.
        Reads train.csv/test.csv and writes train_fe.csv/test_fe.csv.
        """
        import pandas as pd

        train_fe = DATA_DIR / "train_fe.csv"
        test_fe = DATA_DIR / "test_fe.csv"

        df_train = pd.read_csv(paths["train_path"])
        df_test = pd.read_csv(paths["test_path"])

        # (No-op) — keep columns as-is; place your real FE here.
        df_train.to_csv(train_fe, index=False)
        df_test.to_csv(test_fe, index=False)

        return {"train_fe": str(train_fe), "test_fe": str(test_fe)}

    # 3) train_model
    @task(task_id="train_model")
    def train_model(paths: dict, fe_paths: dict) -> dict:
        """
        Train on train_fe.csv and log model to MLflow (PyFunc).
        Returns {'model_uri': 'runs:/<run_id>/model'}.
        """
        import pandas as pd
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_URI)
        ensure_experiment(EXPERIMENT)
        mlflow.set_experiment(EXPERIMENT)

        df_train = pd.read_csv(fe_paths["train_fe"])

        run_id = train_and_log(
            df_train,
            model_type="rf",  # or "logreg"
            target_recall=0.90,
            random_state=42,
            experiment=EXPERIMENT,  # keep consistent!
            run_name="rf-train",
        )

        model_uri = f"runs:/{run_id}/model"  # pyfunc artifact path
        return {"model_uri": model_uri}

    # 4) evaluate_model
    @task(task_id="evaluate_model")
    def evaluate_model(paths: dict, fe_paths: dict, train_out: dict) -> dict:
        """Load model via MLflow URI (pyfunc) and evaluate on test_fe.csv; log metrics."""
        import pandas as pd
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_URI)
        ensure_experiment(EXPERIMENT)
        mlflow.set_experiment(EXPERIMENT)

        df_test = pd.read_csv(fe_paths["test_fe"])
        X_test = df_test.drop(columns=[TARGET_COL])
        y_test = df_test[TARGET_COL]

        model = mlflow.pyfunc.load_model(train_out["model_uri"])

        # evaluate function should accept (model, X, y) and return a dict of metrics
        metrics = eval_fn(model, X_test, y_test)

        with mlflow.start_run(run_name="evaluate"):
            # Log scalar metrics (skip non-scalars)
            for k, v in metrics.items():
                try:
                    mlflow.log_metric(k, float(v))
                except Exception:
                    pass

        return metrics

    # 5) drift_detection (PythonOperator)
    drift_detection = PythonOperator(
        task_id="drift_detection",
        python_callable=drift_detection_op,
    )

    # Branching on drift result (read from reports/drift_report.json, not XCom)
    branch_on_drift = BranchPythonOperator(
        task_id="branch_on_drift",
        python_callable=branch_on_drift_func,
    )

    # End: retrain_model (same training logic on original/non-drifted data)
    @task(task_id="retrain_model")
    def retrain_model(fe_paths: dict) -> dict:
        import pandas as pd
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_URI)
        ensure_experiment(EXPERIMENT)
        mlflow.set_experiment(EXPERIMENT)

        df_train = pd.read_csv(fe_paths["train_fe"])
        run_id = train_and_log(
            df_train,
            model_type="rf",
            target_recall=0.90,
            random_state=42,
            experiment=EXPERIMENT,
            run_name="rf-retrain",
        )
        return {"run_id": run_id}

    # End: pipeline_complete (simple completion task)
    @task(task_id="pipeline_complete")
    def pipeline_complete():
        print("✅ Pipeline complete — no drift detected.")
        return True

    # Orchestration / Dependencies
    paths = preprocess_data()
    fe_paths = feature_engineering(paths)
    train_out = train_model(paths, fe_paths)
    _eval = evaluate_model(paths, fe_paths, train_out)

    # The PythonOperator does not take XCom; it uses fixed file paths under /data and writes /reports/drift_report.json
    _eval >> drift_detection >> branch_on_drift
    fe_paths_for_end = fe_paths  # just to be explicit in the graph

    # Branch targets
    retrain = retrain_model(fe_paths_for_end)
    complete = pipeline_complete()

    branch_on_drift >> [retrain, complete]


dag = pipeline()

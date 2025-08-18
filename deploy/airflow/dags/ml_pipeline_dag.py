from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
from datetime import datetime
from typing import Dict
import pandas as pd
from joblib import load
import sys
import mlflow

# Make '/app' importable so we can use 'from src.<module> import ...'
if "/app" not in sys.path:
    sys.path.append("/app")

# Project imports
from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.model_training import train_model, save_model
from src.evaluation import evaluate_model
from src.drift_detection import detect_drift


@dag(
    dag_id="ml_pipeline_dag",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["ml"],
)
def ml_pipeline():
    # ---------- MLflow ----------
    # Set MLflow tracking server for all tasks in this DAG run
    mlflow.set_tracking_uri("http://mlflow:5000")

    # ---------- Tasks ----------

    @task()
    def step_preprocess() -> Dict[str, str]:
        """Load and split raw data -> persist train/test pickles."""
        train, test = preprocess_data("/app/data/raw/data-ori.csv")
        train_path = "/app/data/train.pkl"
        test_path = "/app/data/test.pkl"
        train.to_pickle(train_path)
        test.to_pickle(test_path)
        return {"train_path": train_path, "test_path": test_path}

    @task()
    def step_engineer(paths: Dict[str, str]) -> Dict[str, str]:
        """Feature engineering -> persist engineered pickles."""
        train = pd.read_pickle(paths["train_path"])
        test = pd.read_pickle(paths["test_path"])
        train_fe = engineer_features(train)
        test_fe = engineer_features(test)
        train_feat_path = "/app/data/train_feat.pkl"
        test_feat_path = "/app/data/test_feat.pkl"
        train_fe.to_pickle(train_feat_path)
        test_fe.to_pickle(test_feat_path)
        return {"train_feat_path": train_feat_path, "test_feat_path": test_feat_path}

    @task()
    def step_train(paths: Dict[str, str]) -> str:
        """Train baseline model -> persist model.pkl."""
        train = pd.read_pickle(paths["train_feat_path"])
        model, _ = train_model(train, model_type="logreg")
        model_path = "/app/models/model.pkl"
        save_model(model, model_path)
        return model_path

    @task()
    def step_evaluate(paths: Dict[str, str], model_path: str) -> None:
        """Evaluate model -> write metrics report."""
        test = pd.read_pickle(paths["test_feat_path"])
        model = load(model_path)
        evaluate_model(model, test, report_path="/app/reports/metrics.txt")

    @task()
    def step_drift_detection() -> Dict[str, object]:
        """
        Run drift detection on static test files and log status to MLflow.
        Returns a dict with keys: drift_detected (bool), overall_drift_score (float).
        """
        # Paths per requirement
        ref_path = "/app/data/test.csv"
        cur_path = "/app/data/drifted_test.csv"

        results = detect_drift(ref_path, cur_path)
        # Expected keys in results:
        #   "drift_detected": bool
        #   "overall_drift_score": float

        # Log to MLflow as parameters (per requirement)
        mlflow.log_param("test_drift_detected", results.get("drift_detected"))
        mlflow.log_param("test_overall_drift_score", results.get("overall_drift_score"))

        # Return only what's needed for branching
        return {
            "drift_detected": bool(results.get("drift_detected", False)),
            "overall_drift_score": float(results.get("overall_drift_score", 0.0)),
        }

    @task.branch(task_id="branch_on_drift")
    def branch_on_drift(drift_result: Dict[str, object]) -> str:
        """
        If drift detected -> go to 'retrain_model'
        else -> go to 'pipeline_complete'
        """
        return (
            "retrain_model"
            if drift_result.get("drift_detected")
            else "pipeline_complete"
        )

    @task(task_id="retrain_model")
    def step_retrain(paths: Dict[str, str]) -> str:
        """
        Simple retrain step (can be extended to trigger data refresh, hyperparam search, etc.).
        Saves to a separate retrained model path.
        """
        train = pd.read_pickle(paths["train_feat_path"])
        model, _ = train_model(train, model_type="logreg")
        model_path = "/app/models/model_retrained.pkl"
        save_model(model, model_path)
        return model_path

    pipeline_complete = EmptyOperator(task_id="pipeline_complete")

    # ---------- Orchestration ----------
    # preprocess_data >> feature_engineering >> train_model >> evaluate_model
    # >> drift_detection >> branch_on_drift >> [retrain_model, pipeline_complete]
    raw_paths = step_preprocess()
    feat_paths = step_engineer(raw_paths)
    model_file = step_train(feat_paths)
    step_evaluate(feat_paths, model_file)
    drift_result = step_drift_detection()
    next_task = branch_on_drift(drift_result)
    # Wire branch targets:
    next_task >> [step_retrain(feat_paths), pipeline_complete]


ml_pipeline()

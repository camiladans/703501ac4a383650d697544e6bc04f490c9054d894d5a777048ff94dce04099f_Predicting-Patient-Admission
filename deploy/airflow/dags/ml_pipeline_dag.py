# deploy/airflow/dags/ml_pipeline_dag.py

from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
from datetime import datetime
from typing import Dict
import sys

# Add '/app' to import path for project modules
if "/app" not in sys.path:
    sys.path.append("/app")

# 3rd-party
import pandas as pd
from joblib import load
import mlflow

# Project imports
from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.model_training import train_model, save_model
from src.evaluation import evaluate_model
from src.drift_detection import detect_drift


@dag(
    dag_id="ml_pipeline_dag",
    start_date=datetime(2025, 1, 1),
    schedule=None,  # manual trigger
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
        """
        Load and split raw data -> persist train/test pickles.
        Also export test CSVs for Evidently and create a synthetic drifted copy.
        """
        import numpy as np

        # Adjust input path as needed
        train, test = preprocess_data("/app/data/raw/data-ori.csv")

        # Persist pickles (unchanged behavior)
        train_path = "/app/data/train.pkl"
        test_path = "/app/data/test.pkl"
        train.to_pickle(train_path)
        test.to_pickle(test_path)

        # Export test CSV for Evidently
        test_csv_path = "/app/data/test.csv"
        test.to_csv(test_csv_path, index=False)

        # Create a synthetic drifted copy
        drift = test.copy()
        np.random.seed(42)  # reproducible

        # 1) Numeric drift: scale + small Gaussian noise
        num_cols = drift.select_dtypes(include=["number"]).columns.tolist()
        if num_cols:
            for c in num_cols:
                std = float(drift[c].std() or 0.0)
                noise = np.random.normal(
                    0.0, 0.05 * (std if std > 0 else 1.0), size=len(drift)
                )
                drift[c] = drift[c] * 1.2 + noise

        # 2) Categorical drift: flip ~10% of first categorical column, if any
        cat_cols = drift.select_dtypes(include=["object", "category"]).columns.tolist()
        cat_cols = [
            c for c in cat_cols if c.lower() not in {"source", "target", "label", "y"}
        ]
        if cat_cols:
            c = cat_cols[0]
            vals = drift[c].dropna().unique().tolist()
            if len(vals) > 1:
                idx = np.random.rand(len(drift)) < 0.10
                alt = {v: vals[(i + 1) % len(vals)] for i, v in enumerate(vals)}
                drift.loc[idx & drift[c].notna(), c] = drift.loc[
                    idx & drift[c].notna(), c
                ].map(lambda v: alt.get(v, v))

        drift_csv_path = "/app/data/drifted_test.csv"
        drift.to_csv(drift_csv_path, index=False)

        return {
            "train_path": train_path,
            "test_path": test_path,
            "test_csv_path": test_csv_path,
            "drift_csv_path": drift_csv_path,
        }

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
    def step_drift_detection(paths: Dict[str, str]) -> Dict[str, object]:
        """
        Run drift detection on test CSVs produced by step_preprocess()
        and log status to MLflow.
        """
        import os

        ref_path = paths.get("test_csv_path", "/app/data/test.csv")
        cur_path = paths.get("drift_csv_path", "/app/data/drifted_test.csv")

        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"Missing reference CSV: {ref_path}")
        if not os.path.exists(cur_path):
            raise FileNotFoundError(f"Missing current CSV: {cur_path}")

        results = detect_drift(ref_path, cur_path)

        # Ensure metrics are captured under a run
        with mlflow.start_run(run_name="drift_detection", nested=True):
            mlflow.log_param("test_drift_detected", results.get("drift_detected"))
            mlflow.log_param(
                "test_overall_drift_score", results.get("overall_drift_score")
            )

        return {
            "drift_detected": bool(results.get("drift_detected", False)),
            "overall_drift_score": float(results.get("overall_drift_score", 0.0)),
        }

    @task.branch(task_id="branch_on_drift")
    def branch_on_drift(drift_result: Dict[str, object]) -> str:
        """If drift detected -> go to 'retrain_model' else -> 'pipeline_complete'."""
        return (
            "retrain_model"
            if drift_result.get("drift_detected")
            else "pipeline_complete"
        )

    @task(task_id="retrain_model")
    def step_retrain(paths: Dict[str, str]) -> str:
        """
        Simple retrain step (extend as needed).
        Saves to a separate retrained model path.
        """
        train = pd.read_pickle(paths["train_feat_path"])
        model, _ = train_model(train, model_type="logreg")
        model_path = "/app/models/model_retrained.pkl"
        save_model(model, model_path)
        return model_path

    pipeline_complete = EmptyOperator(task_id="pipeline_complete")

    # ---------- Orchestration ----------
    raw_paths = step_preprocess()
    feat_paths = step_engineer(raw_paths)
    model_file = step_train(feat_paths)
    step_evaluate(feat_paths, model_file)

    drift_result = step_drift_detection(raw_paths)
    next_task = branch_on_drift(drift_result)

    next_task >> [step_retrain(feat_paths), pipeline_complete]


ml_pipeline()

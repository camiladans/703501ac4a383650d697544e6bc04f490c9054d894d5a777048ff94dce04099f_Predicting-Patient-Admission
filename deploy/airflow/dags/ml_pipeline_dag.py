# deploy/airflow/dags/ml_pipeline_dag.py

from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
from datetime import datetime
from typing import Dict, Any
import sys
from airflow.utils.trigger_rule import TriggerRule

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

        raw_path = "/app/data/raw/data-ori.csv"
        result = preprocess_data(raw_path)

        # --- handle various return shapes from preprocess_data ---
        if isinstance(result, dict):
            # Expecting keys like 'train'/'test' (adjust if yours differ)
            train = (
                result.get("train") or result.get("X_train") or result.get("train_df")
            )
            test = result.get("test") or result.get("X_test") or result.get("test_df")
            if train is None or test is None:
                raise ValueError(
                    f"preprocess_data returned dict without train/test keys: {list(result.keys())}"
                )
        elif isinstance(result, (list, tuple)):
            # Take the first two items as train/test; ignore extras
            if len(result) < 2:
                raise ValueError(
                    f"preprocess_data returned {len(result)} values, need at least 2 (train, test)"
                )
            train, test = result[0], result[1]
        else:
            raise TypeError(f"preprocess_data returned unexpected type: {type(result)}")

        # Persist pickles
        train_path = "/app/data/train.pkl"
        test_path = "/app/data/test.pkl"
        train.to_pickle(train_path)
        test.to_pickle(test_path)

        # Export test CSV for Evidently
        test_csv_path = "/app/data/test.csv"
        test.to_csv(test_csv_path, index=False)

        # Create a synthetic drifted copy
        drift = test.copy()
        np.random.seed(42)

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
        cat_cols = [
            c
            for c in drift.select_dtypes(include=["object", "category"]).columns
            if c.lower() not in {"source", "target", "label", "y"}
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
        """Train baseline model -> persist model.pkl and log to MLflow."""
        import os
        import mlflow
        from mlflow import sklearn as mlflow_sklearn

        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("ml_pipeline")  # optional but nice

        os.makedirs("/app/models", exist_ok=True)

        train = pd.read_pickle(paths["train_feat_path"])
        model_type = "logreg"
        model, train_info = train_model(train, model_type=model_type)

        model_path = "/app/models/model.pkl"
        save_model(model, model_path)

        # Log a run
        with mlflow.start_run(run_name="train"):
            mlflow.log_param("model_type", model_type)
            # If you have anything useful in train_info, log it:
            if isinstance(train_info, dict):
                for k, v in train_info.items():
                    if isinstance(v, (int, float, str, bool)):
                        mlflow.log_param(f"train_{k}", v)
            mlflow.log_artifact(model_path, artifact_path="artifacts")
            # Or log as a proper MLflow model (sklearn example):
            try:
                mlflow_sklearn.log_model(model, artifact_path="sklearn_model")
            except Exception:
                pass  # keep it optional

        return model_path

    @task()
    def step_evaluate(paths: Dict[str, str], model_path: str) -> None:
        """Evaluate model -> write metrics report."""
        test = pd.read_pickle(paths["test_feat_path"])
        model = load(model_path)
        evaluate_model(model, test, report_path="/app/reports/metrics.txt")

    @task()
    def step_drift_detection(paths: Dict[str, str]) -> Dict[str, Any]:
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

        # If your target column is named differently, set it here
        results = detect_drift(
            reference_csv=ref_path,
            current_csv=cur_path,
            target="SOURCE",  # optional, ignored unless used in drift_detection.py
            report_dir="/app/reports",  # ensures the JSON lands in your mounted reports volume
        )

        # Ensure metrics are captured under a run
        with mlflow.start_run(run_name="drift_detection", nested=True):
            mlflow.log_param("test_drift_detected", results.get("drift_detected"))
            mlflow.log_param(
                "test_overall_drift_share", results.get("overall_drift_share")
            )

        return {
            "drift_detected": bool(results.get("drift_detected", False)),
            "overall_drift_score": float(results.get("overall_drift_score", 0.0)),
            "report_path": results.get("report_path"),
        }

    @task.branch(task_id="branch_on_drift")
    def branch_on_drift(drift_result: Dict[str, Any]) -> str:
        return "retrain_model" if drift_result.get("drift_detected") else "no_retrain"

    @task(task_id="retrain_model")
    def step_retrain(paths: Dict[str, str]) -> str:
        import mlflow

        mlflow.set_tracking_uri("http://mlflow:5000")
        train = pd.read_pickle(paths["train_feat_path"])
        model, _ = train_model(train, model_type="logreg")
        model_path = "/app/models/model_retrained.pkl"
        save_model(model, model_path)
        return model_path

    no_retrain = EmptyOperator(task_id="no_retrain")

    pipeline_complete = EmptyOperator(
        task_id="pipeline_complete",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # ---------- Orchestration ----------
    raw_paths = step_preprocess()
    feat_paths = step_engineer(raw_paths)
    model_file = step_train(feat_paths)
    step_evaluate(feat_paths, model_file)

    drift_result = step_drift_detection(raw_paths)
    next_task = branch_on_drift(drift_result)

    retrain_task = step_retrain(feat_paths)
    next_task >> [retrain_task, no_retrain]
    [retrain_task, no_retrain] >> pipeline_complete


ml_pipeline()

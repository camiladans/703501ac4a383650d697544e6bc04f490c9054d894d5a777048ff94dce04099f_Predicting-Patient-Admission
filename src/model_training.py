"""
src/model_training.py

Summary
-------
End-to-end training and logging for a binary classification model
(IN vs OUT) with MLflow. This module:

1) Sets MLflow tracking to an MLflow server at http://localhost:5000
2) Trains a RandomForest or LogisticRegression with recall-oriented tuning
3) Logs EXACTLY three hyperparameters (per model type)
4) Saves model artifacts under ./mlflow/artifacts/
5) Logs a Custom PyFunc wrapper (spec-compliant) to MLflow
   - Wrapper loads optional preprocessor & feature_names
   - Wrapper's predict() applies preprocessing (if present) and returns labels

How to run (example)
--------------------
python -m src.model_training

You can also import `train_and_log()` from other code (e.g., Airflow task).
"""

from __future__ import annotations

# ===============================
# SECTION 0 — Imports & Constants
# ===============================
import json
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any

import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_curve,
    precision_score,
    recall_score,
    confusion_matrix,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

# All staged artifacts are written here BEFORE MLflow logs them into the run
ARTIFACT_STAGING_DIR = Path("mlflow") / "artifacts" / "models_export"


# ======================================================
# SECTION 1 — Custom PyFunc Model (SPEC-COMPLIANT)
# ======================================================
class CustomMLModel(mlflow.pyfunc.PythonModel):
    """
    Custom MLflow PyFunc model wrapper for your trained model.

    What it does (per spec):
    - Loads artifacts: model (required), preprocessor & feature_names (optional)
    - predict(): applies preprocessing if available and returns model.predict(...) labels
    - No thresholding and no feature reordering logic inside predict()

    Note:
    - If you need thresholded outputs or probabilities, do that OUTSIDE this class.
    """

    def __init__(self):
        self.model = None
        self.preprocessor = None  # scaler/encoder/column-ordering pipeline (optional)
        self.feature_names = None  # loaded but not enforced inside predict()

    def load_context(self, context):
        """Load model artifacts from MLflow context."""
        # Required
        self.model = joblib.load(context.artifacts["model"])

        # Optional preprocessor
        if "preprocessor" in context.artifacts:
            self.preprocessor = joblib.load(context.artifacts["preprocessor"])

        # Optional feature names (informational only)
        if "feature_names" in context.artifacts:
            with open(context.artifacts["feature_names"], "r") as f:
                self.feature_names = [ln.strip() for ln in f if ln.strip()]

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.preprocessor is not None:
            processed_input = self.preprocessor.transform(model_input)
        else:
            processed_input = model_input.values

        predictions = self.model.predict(processed_input)
        return predictions


# ======================================================
# SECTION 2 — Data Utilities (coercion, ordering, threshold)
# ======================================================
def _coerce_features_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce typical messy clinical columns into numeric form:
    - Map SEX to 0/1 if string
    - Convert bools to int8
    - Coerce object dtypes to numeric (NaN on errors)
    - Replace +/-inf with NaN; median-impute numeric columns
    - Return numeric-only columns
    """
    X = X.copy()

    # Common binary categorical fix
    if "SEX" in X.columns and X["SEX"].dtype == object:
        map_sex = {"M": 1, "F": 0, "Male": 1, "Female": 0, "m": 1, "f": 0}
        X["SEX"] = X["SEX"].map(map_sex)

    # Convert booleans to 0/1
    bool_cols = [c for c in X.columns if pd.api.types.is_bool_dtype(X[c])]
    if bool_cols:
        X[bool_cols] = X[bool_cols].astype("int8")

    # Try numeric coercion for object columns
    obj_cols = [c for c in X.columns if X[c].dtype == object]
    for c in obj_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Replace inf with NaN, then impute numeric columns with median
    X = X.replace([np.inf, -np.inf], np.nan)
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    for c in num_cols:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())

    # Keep only numeric columns
    final_num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    return X[final_num_cols]


def _make_feature_order_preprocessor(
    feature_names: Iterable[str],
) -> FunctionTransformer:
    """
    Create a transformer that validates & reorders columns at inference time.
    This keeps column alignment OUTSIDE the PyFunc class (to remain spec-clean).
    """
    feature_names = list(feature_names)

    def _reorder(X):
        if isinstance(X, pd.DataFrame):
            missing = [c for c in feature_names if c not in X.columns]
            if missing:
                raise ValueError(f"Missing required feature(s): {missing}")
            return X[feature_names].values
        # ndarray path: assume already in the correct order
        return X

    return FunctionTransformer(_reorder, validate=False)


def _select_threshold_for_recall(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    target_recall: float = 0.90,
) -> Tuple[float, Dict[str, Any]]:
    """
    Pick the highest threshold that achieves >= target_recall.
    Fallback behavior: best recall in curve or 0.5 if degenerate.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, prob_pos)
    candidates = []
    for t, r, p in zip(np.append(thresholds, [1.0]), recall, precision):
        if r >= target_recall:
            candidates.append((t, r, p))
    if candidates:
        t_sel, r_sel, p_sel = sorted(candidates, key=lambda x: x[0])[-1]
    else:
        if len(thresholds) > 0:
            idx = int(np.argmax(recall))
            t_sel = float(thresholds[min(idx, len(thresholds) - 1)])
            r_sel = float(recall[idx])
            p_sel = float(precision[idx])
        else:
            t_sel = 0.5
            r_sel = float(recall_score(y_true, prob_pos >= 0.5))
            p_sel = float(precision_score(y_true, prob_pos >= 0.5))
    details = {
        "selected_threshold": float(t_sel),
        "recall_at_t": float(r_sel),
        "precision_at_t": float(p_sel),
    }
    return float(t_sel), details


# ======================================================
# SECTION 3 — Training (recall-oriented) + metrics
# ======================================================
def train_model(
    train_data: pd.DataFrame,
    model_type: str = "rf",
    target_recall: float = 0.90,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, Any], Iterable[str], Dict[str, Any]]:
    """
    Train a classifier with recall-oriented tuning and return:
      - best pipeline (orderer + estimator),
      - metrics dict,
      - feature_names,
      - best_params (for logging exactly 3 hyperparameters).
    """
    print("Train data columns:", train_data.columns.tolist())
    if "SOURCE" not in train_data.columns:
        raise KeyError("The column 'SOURCE' is missing from the input DataFrame.")

    # Encode target: IN -> 1 else 0
    y = train_data["SOURCE"].apply(lambda x: 1 if str(x).upper() == "IN" else 0)

    # Sanitize features to numeric
    X = train_data.drop(columns=["SOURCE"])
    X = _coerce_features_numeric(X)
    feature_names = list(X.columns)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Objective: maximize recall for positive class (1)
    recall_pos1 = make_scorer(recall_score, pos_label=1)

    # Select model + grid
    if model_type.lower() == "logreg":
        base = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        )
        param_grid = {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "liblinear"]}
    elif model_type.lower() == "rf":
        base = RandomForestClassifier(
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        param_grid = {"n_estimators": [100, 200, 400], "max_depth": [None, 10, 20]}
    else:
        raise ValueError("Unsupported model_type. Choose 'logreg' or 'rf'.")

    # Column alignment preprocessor
    orderer = _make_feature_order_preprocessor(feature_names)

    # Pipeline ensures validation uses the same preprocessing as inference
    pipe = Pipeline(
        steps=[
            ("orderer", orderer),
            ("est", base),
        ]
    )

    grid = GridSearchCV(
        pipe,
        {"est__" + k: v for k, v in param_grid.items()},
        cv=5,
        scoring=recall_pos1,
        n_jobs=-1,
        error_score="raise",
    )
    grid.fit(X_train, y_train)
    best_pipe = grid.best_estimator_

    # Validation probabilities / scores
    est = best_pipe.named_steps["est"]
    if hasattr(est, "predict_proba"):
        p_val = best_pipe.predict_proba(X_val)[:, 1]
    elif hasattr(est, "decision_function"):
        d = best_pipe.decision_function(X_val)
        d_min, d_max = float(np.min(d)), float(np.max(d))
        p_val = (d - d_min) / (d_max - d_min + 1e-9)
    else:
        p_val = best_pipe.predict(X_val).astype(float)

    # Threshold selection for target recall
    threshold, thr_details = _select_threshold_for_recall(
        y_val.values, p_val, target_recall
    )
    y_pred = (p_val >= threshold).astype(int)

    cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred)),
        "classification_report": classification_report(y_val, y_pred),
        "threshold": float(threshold),
        "threshold_selection": thr_details,
        "confusion_matrix": cm.tolist(),
    }

    # Extract best params (used later to log EXACTLY 3 hyperparameters)
    best_params: Dict[str, Any] = {}
    if isinstance(est, RandomForestClassifier):
        best_params = {
            "n_estimators": getattr(est, "n_estimators", None),
            "max_depth": getattr(est, "max_depth", None),
            "random_state": getattr(est, "random_state", random_state),
        }
    elif isinstance(est, LogisticRegression):
        best_params = {
            "C": getattr(est, "C", None),
            "max_iter": getattr(est, "max_iter", None),
            "random_state": getattr(est, "random_state", random_state),
        }

    return best_pipe, metrics, feature_names, best_params


# ======================================================
# SECTION 4 — Train & Log (meets the checklist)
# ======================================================
def train_and_log(
    train_data: pd.DataFrame,
    *,
    model_type: str = "rf",
    target_recall: float = 0.90,
    random_state: int = 42,
    experiment: str = "patient_admission",
    run_name: str = "training-run",
) -> str:
    """
    Checklist compliance:
    - Set tracking URI: http://localhost:5000
    - Wrap training in mlflow.start_run()
    - Log EXACTLY 3 hyperparameters (by model type)
      * Classification (RF): n_estimators, max_depth, random_state
      * Classification (LogReg): C, max_iter, random_state (justify in README)
    - Save model artifacts under ./mlflow/artifacts/
    - Log model using custom PyFunc wrapper
    """
    # 1) Point to your MLflow server
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment)

    # Ensure staging directory exists
    ARTIFACT_STAGING_DIR.mkdir(parents=True, exist_ok=True)

    # 2) Start run
    with mlflow.start_run(run_name=run_name):
        # Train
        model_pipe, metrics, feat_names, best_params = train_model(
            train_data,
            model_type=model_type,
            target_recall=target_recall,
            random_state=random_state,
        )

        # 3) Log EXACTLY three hyperparameters
        est = model_pipe.named_steps["est"]
        if isinstance(est, RandomForestClassifier):
            to_log = {
                "n_estimators": best_params.get("n_estimators"),
                "max_depth": best_params.get("max_depth"),
                "random_state": best_params.get("random_state", random_state),
            }
        elif isinstance(est, LogisticRegression):
            to_log = {
                "C": best_params.get("C"),
                "max_iter": best_params.get("max_iter"),
                "random_state": best_params.get("random_state", random_state),
            }
        else:
            # If you add other model types later: choose & justify 3 params in your README
            to_log = {
                "random_state": random_state,
                "target_recall": target_recall,
                "model_type": model_type,
            }
        # strictly ensure 3 keys
        to_log = dict(list(to_log.items())[:3])
        mlflow.log_params(to_log)

        # 4) Stage artifacts under ./mlflow/artifacts/ then log them
        #    (these staged files are the sources for the PyFunc artifacts argument)
        model_path = ARTIFACT_STAGING_DIR / "model.pkl"
        preproc_path = ARTIFACT_STAGING_DIR / "preprocessor.pkl"
        fnames_path = ARTIFACT_STAGING_DIR / "feature_names.txt"
        metrics_path = ARTIFACT_STAGING_DIR / "metrics.json"
        threshold_path = ARTIFACT_STAGING_DIR / "threshold.txt"

        # Split pipeline (orderer is the column aligner, est is the actual estimator)
        orderer = model_pipe.named_steps.get("orderer")
        estimator = model_pipe.named_steps["est"]

        joblib.dump(estimator, model_path)
        if orderer is not None:
            joblib.dump(orderer, preproc_path)

        with open(fnames_path, "w") as f:
            for n in feat_names:
                f.write(f"{n}\n")

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        if "threshold" in metrics:
            with open(threshold_path, "w") as f:
                f.write(str(metrics["threshold"]))

        # Log numeric metrics to MLflow (strings like classification_report go as artifacts)
        flat_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        if flat_metrics:
            mlflow.log_metrics(flat_metrics)
        mlflow.log_artifact(str(metrics_path))

        # 5) Log the model using the custom PyFunc wrapper
        artifacts = {"model": str(model_path)}
        if preproc_path.exists():
            artifacts["preprocessor"] = str(preproc_path)
        if fnames_path.exists():
            artifacts["feature_names"] = str(fnames_path)
        if threshold_path.exists():
            artifacts["threshold"] = str(threshold_path)

        mlflow.pyfunc.log_model(
            artifact_path="model",  # appears under the run's artifacts
            python_model=CustomMLModel(),  # our spec-compliant wrapper
            artifacts=artifacts,  # files staged under ./mlflow/artifacts/
        )

        run_id = mlflow.active_run().info.run_id
        print(f"Logged MLflow run_id: {run_id}")
        return run_id


# ======================================================
# SECTION 5 — CLI Demo (safe to delete)
# ======================================================
if __name__ == "__main__":
    # Minimal demo dataset
    df = pd.DataFrame(
        {
            "f1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "f2": [1, 0, 1, 0, 1],
            "SOURCE": ["IN", "OUT", "IN", "OUT", "IN"],
        }
    )
    # Train + log (RandomForest by default)
    run_id = train_and_log(
        df,
        model_type="rf",  # or "logreg"
        target_recall=0.80,
        random_state=42,
        experiment="patient_admission",
        run_name="rf-inpatient-demo",
    )
    print("Run complete:", run_id)

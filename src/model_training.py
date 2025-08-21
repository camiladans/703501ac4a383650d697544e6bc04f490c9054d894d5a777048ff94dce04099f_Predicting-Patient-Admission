from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, Any

import json
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


# ======================================================================
# Custom MLflow PyFunc model wrapper (safer column handling + threshold)
# ======================================================================
class CustomMLModel(mlflow.pyfunc.PythonModel):
    """
    Custom MLflow PyFunc model wrapper.
    - Aligns incoming columns to saved feature_names (if provided)
    - Supports probability thresholding for positive class (IN=1)
    """

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.threshold = 0.5  # default

    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model"])

        if "preprocessor" in context.artifacts:
            self.preprocessor = joblib.load(context.artifacts["preprocessor"])

        if "feature_names" in context.artifacts:
            with open(context.artifacts["feature_names"], "r") as f:
                self.feature_names = [line.strip() for line in f if line.strip()]

        if "threshold" in context.artifacts:
            with open(context.artifacts["threshold"], "r") as f:
                self.threshold = float(f.read().strip())

    def _align_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names is None:
            return df
        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature(s): {missing}")
        # Restrict and order
        return df[self.feature_names]

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        # Accept ndarray but prefer DataFrame
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input, columns=self.feature_names or None)

        X = self._align_columns(model_input)

        # Preprocess if available
        if self.preprocessor is not None:
            Xp = self.preprocessor.transform(X)
        else:
            Xp = X.values

        # If classifier has predict_proba, apply threshold; else use predict
        if hasattr(self.model, "predict_proba"):
            p1 = self.model.predict_proba(Xp)[:, 1]
            return (p1 >= self.threshold).astype(int)
        return self.model.predict(Xp)


# ======================================================================
# Training with explicit recall scorer + threshold selection
# ======================================================================


def _coerce_features_numeric(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    # Common binary categorical fix
    if "SEX" in X.columns:
        # map typical encodings; fall back to category codes if needed
        map_sex = {"M": 1, "F": 0, "Male": 1, "Female": 0, "m": 1, "f": 0}
        if X["SEX"].dtype == object:
            X["SEX"] = X["SEX"].map(map_sex).astype("float64")

    # Convert booleans to 0/1
    bool_cols = [c for c in X.columns if X[c].dtype == bool]
    if bool_cols:
        X[bool_cols] = X[bool_cols].astype("int8")

    # Try numeric coercion for any remaining object columns
    obj_cols = [c for c in X.columns if X[c].dtype == object]
    for c in obj_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Replace inf with NaN, then impute numeric columns with median
    X = X.replace([np.inf, -np.inf], np.nan)
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    for c in num_cols:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())

    # As a last guard, drop any still-non-numeric columns
    final_num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    return X[final_num_cols]


def _encode_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if "SOURCE" not in df.columns:
        raise KeyError("The column 'SOURCE' is missing from the input DataFrame.")
    y = df["SOURCE"].apply(lambda x: 1 if str(x).upper() == "IN" else 0)
    X = df.drop(columns=["SOURCE"])
    return X, y


def _select_threshold_for_recall(
    y_true: np.ndarray, prob_pos: np.ndarray, target_recall: float = 0.90
) -> Tuple[float, Dict[str, Any]]:
    """Pick the highest threshold that achieves >= target_recall (fallback: 0.5)."""
    precision, recall, thresholds = precision_recall_curve(y_true, prob_pos)
    # thresholds is len-1 vs precision/recall; align
    candidates = []
    for t, r, p in zip(np.append(thresholds, [1.0]), recall, precision):
        if r >= target_recall:
            candidates.append((t, r, p))
    if candidates:
        # prefer the largest threshold that still meets recall
        t_sel, r_sel, p_sel = sorted(candidates, key=lambda x: x[0])[-1]
    else:
        # fallback to threshold with best recall (max recall); if all fail, 0.5
        if len(thresholds) > 0:
            idx = np.argmax(recall)
            t_sel = thresholds[min(idx, len(thresholds) - 1)]
            r_sel = recall[idx]
            p_sel = precision[idx]
        else:
            t_sel, r_sel, p_sel = (
                0.5,
                float(recall_score(y_true, prob_pos >= 0.5)),
                float(precision_score(y_true, prob_pos >= 0.5)),
            )
    details = {
        "selected_threshold": float(t_sel),
        "recall_at_t": float(r_sel),
        "precision_at_t": float(p_sel),
    }
    return float(t_sel), details


# --- replace your train_model(...) with this version ---


def train_model(
    train_data: pd.DataFrame,
    model_type: str = "rf",
    target_recall: float = 0.90,
    random_state: int = 42,
):
    print("Train data columns:", train_data.columns.tolist())
    if "SOURCE" not in train_data.columns:
        raise KeyError("The column 'SOURCE' is missing from the input DataFrame.")

    # Encode target: IN -> 1 else 0
    y = train_data["SOURCE"].apply(lambda x: 1 if str(x).upper() == "IN" else 0)

    # Sanitize features to numeric
    X = train_data.drop(columns=["SOURCE"])
    X = _coerce_features_numeric(X)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Models + grids (recall scorer explicit on positive class=1)
    recall_pos1 = make_scorer(recall_score, pos_label=1)

    if model_type.lower() == "logreg":
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        )
        param_grid = {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "liblinear"]}
    elif model_type.lower() == "rf":
        model = RandomForestClassifier(
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        param_grid = {"n_estimators": [100, 200, 400], "max_depth": [None, 10, 20]}
    else:
        raise ValueError("Unsupported model_type. Choose 'logreg' or 'rf'.")

    # If you still want to see the first underlying error while debugging, set error_score='raise'
    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring=recall_pos1,
        n_jobs=-1,
        error_score="raise",  # helps surface true cause instead of masking
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    # Threshold selection for recall target
    if hasattr(best_model, "predict_proba"):
        p_val = best_model.predict_proba(X_val)[:, 1]
    elif hasattr(best_model, "decision_function"):
        d = best_model.decision_function(X_val)
        p_val = (d - d.min()) / (d.max() - d.min() + 1e-9)
    else:
        p_val = best_model.predict(X_val).astype(float)

    # Reuse helper from earlier response (assume present in the file)
    threshold, thr_details = _select_threshold_for_recall(
        y_val.values, p_val, target_recall
    )
    y_pred = (p_val >= threshold).astype(int)

    from sklearn.metrics import (
        precision_score,
        recall_score as r_score,
    )

    cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(r_score(y_val, y_pred)),
        "classification_report": classification_report(y_val, y_pred),
        "threshold": threshold,
        "threshold_selection": thr_details,
        "confusion_matrix": cm.tolist(),
    }

    feature_names = list(X.columns)  # after coercion (this is what the model expects)
    return best_model, metrics, feature_names


def save_model(model, filepath: str = "models/model.pkl"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")


# ======================================================================
# MLflow logging updated: add threshold + metrics + tags
# ======================================================================
def _write_feature_names_txt(feature_names: Iterable[str], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for name in feature_names:
            f.write(f"{name}\n")
    return path


def log_model_to_mlflow(
    model,
    *,
    run_name: Optional[str] = None,
    tracking_uri: str = "http://localhost:5000",  # consider "http://mlflow:5000" in Docker
    artifact_dir: str = "models_export",
    preprocessor=None,
    feature_names: Optional[Iterable[str]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    experiment: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
):
    """
    Logs the given model using CustomMLModel wrapper.

    Artifacts saved and referenced:
      - model.pkl
      - preprocessor.pkl        (optional)
      - feature_names.txt       (optional)
      - threshold.txt           (optional; derived from metrics["threshold"])
      - metrics.json            (optional; whatever you pass)
    """
    mlflow.set_tracking_uri(tracking_uri)
    if experiment:
        mlflow.set_experiment(experiment)

    # Prepare local artifacts
    art_dir = Path(artifact_dir)
    art_dir.mkdir(parents=True, exist_ok=True)

    model_path = art_dir / "model.pkl"
    joblib.dump(model, model_path)

    artifacts = {"model": str(model_path)}

    if preprocessor is not None:
        preproc_path = art_dir / "preprocessor.pkl"
        joblib.dump(preprocessor, preproc_path)
        artifacts["preprocessor"] = str(preproc_path)

    if feature_names is not None:
        fnames_path = _write_feature_names_txt(
            feature_names, art_dir / "feature_names.txt"
        )
        artifacts["feature_names"] = str(fnames_path)

    if metrics:
        # Save metrics to JSON
        with open(art_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        # Persist threshold if present
        if "threshold" in metrics:
            tpath = art_dir / "threshold.txt"
            with open(tpath, "w") as f:
                f.write(str(metrics["threshold"]))
            artifacts["threshold"] = str(tpath)

    with mlflow.start_run(run_name=run_name):
        # Log flat metrics
        if metrics:
            flat_metrics = {
                k: v for k, v in metrics.items() if isinstance(v, (int, float))
            }
            mlflow.log_metrics(flat_metrics)
            mlflow.log_artifact(str(art_dir / "metrics.json"))

        if tags:
            mlflow.set_tags(tags)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=CustomMLModel(),
            artifacts=artifacts,
        )
        run_id = mlflow.active_run().info.run_id
        print(f"Logged MLflow run_id: {run_id}")
        return run_id


# ======================================================================
# Optional CLI/demo usage
# ======================================================================
if __name__ == "__main__":
    try:
        df = pd.DataFrame(
            {
                "f1": [0.1, 0.2, 0.3, 0.4, 0.5],
                "f2": [1, 0, 1, 0, 1],
                "SOURCE": ["IN", "OUT", "IN", "OUT", "IN"],
            }
        )
        model, metrics, feat_names = train_model(
            df, model_type="rf", target_recall=0.80
        )
        run_id = log_model_to_mlflow(
            model,
            run_name="rf-inpatient-demo",
            tracking_uri="http://localhost:5000",
            artifact_dir="models_export",
            preprocessor=None,
            feature_names=feat_names,
            metrics=metrics,
            experiment="patient_admission",
            tags={"model_type": "rf", "target": "IN"},
        )
    except Exception as e:
        print("Demo failed/skipped:", e)

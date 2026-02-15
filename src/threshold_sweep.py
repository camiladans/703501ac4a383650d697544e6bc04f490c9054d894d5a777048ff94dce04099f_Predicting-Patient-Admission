# ruff: noqa: E402
# src/threshold_sweep.py
from __future__ import annotations

"""
Threshold sweep utilities for binary classifiers (positive = 'IN' or 1).

Main functions:
- sweep_thresholds(model, X, y, thresholds=...): returns a DataFrame with
  threshold-level metrics: accuracy, precision, recall, f1_score, tp, fp, tn, fn
- select_threshold(sweep_df, min_recall=0.90, objective='f1', fallback='f1'):
  chooses one row (threshold) given a recall floor and an optimization objective
- run_threshold_experiments(...): convenience wrapper that runs a full sweep,
  selects a threshold, and optionally saves/logs artifacts.

Notes:
- Model should expose predict_proba() (preferred) or decision_function().
- Positive class is assumed to be 1 or the string 'IN' (case-insensitive).
"""

from typing import Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_bool_dtype, is_numeric_dtype
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ---------------------------
# Helpers: data preparation
# ---------------------------
def _encode_target(y) -> np.ndarray:
    """Map labels to 0/1 with 'IN' (or 1) as positive."""
    arr = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)
    if arr.dtype.kind in {"U", "S", "O"}:
        return np.array([1 if str(v).upper() == "IN" else 0 for v in arr], dtype=int)
    return np.array([1 if int(v) == 1 else 0 for v in arr], dtype=int)


def _coerce_features_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """Best-effort numeric coercion similar to evaluation.py behavior."""
    X = X.copy()
    if "SEX" in X.columns and is_object_dtype(X["SEX"]):
        X["SEX"] = (
            X["SEX"]
            .map({"M": 1, "F": 0, "Male": 1, "Female": 0, "m": 1, "f": 0})
            .astype("float64")
        )
    bool_cols = [c for c in X.columns if is_bool_dtype(X[c])]
    if bool_cols:
        X[bool_cols] = X[bool_cols].astype("int8")
    obj_cols = [c for c in X.columns if is_object_dtype(X[c])]
    for c in obj_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    num_cols = [c for c in X.columns if is_numeric_dtype(X[c])]
    for c in num_cols:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())
    return X


def _ensure_probs(model, X_num: pd.DataFrame) -> np.ndarray:
    """
    Get positive-class probabilities for the sample being 'IN'/1.
    Prefers predict_proba; falls back to decision_function with sigmoid.
    """
    # predict_proba path
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_num)
        if proba.ndim == 2:
            if proba.shape[1] == 2:
                # assume [:,1] is positive; if classes_ exists, try to index exact pos class
                if hasattr(model, "classes_"):
                    classes = list(model.classes_)
                    try:
                        idx = classes.index(1)
                    except ValueError:
                        try:
                            idx = [str(c).upper() for c in classes].index("IN")
                        except ValueError:
                            idx = 1  # fallback
                    return proba[:, idx]
                return proba[:, 1]
            # Multiclass: attempt to pick 'IN' or 1 if present
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                try:
                    idx = classes.index(1)
                except ValueError:
                    try:
                        idx = [str(c).upper() for c in classes].index("IN")
                    except ValueError:
                        raise ValueError(
                            "Could not infer positive class in multiclass predict_proba()."
                        )
                return proba[:, idx]
        return proba.ravel()

    # decision_function path → map to (0,1)
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X_num), dtype="float64").ravel()
        return 1.0 / (1.0 + np.exp(-raw))

    raise AttributeError("Model lacks predict_proba and decision_function.")


# ---------------------------
# Public API
# ---------------------------
def sweep_thresholds(
    model,
    X: pd.DataFrame,
    y,
    thresholds: Sequence[float] = tuple(np.linspace(0.05, 0.95, 19)),
    *,
    coerce_numeric: bool = True,
) -> pd.DataFrame:
    """
    Evaluate metrics across probability thresholds.

    Returns a DataFrame with columns:
    ['threshold','accuracy','precision','recall','f1_score','tp','fp','tn','fn'].
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    X_num = _coerce_features_numeric(X) if coerce_numeric else X.copy()
    y_true = _encode_target(y)
    probs = _ensure_probs(model, X_num)

    rows = []
    for thr in thresholds:
        thr = float(thr)
        y_hat = (probs >= thr).astype(int)

        acc = accuracy_score(y_true, y_hat)
        prec = precision_score(y_true, y_hat, zero_division=0)
        rec = recall_score(y_true, y_hat, zero_division=0)
        f1 = f1_score(y_true, y_hat, zero_division=0)

        tp = int(((y_true == 1) & (y_hat == 1)).sum())
        fp = int(((y_true == 0) & (y_hat == 1)).sum())
        tn = int(((y_true == 0) & (y_hat == 0)).sum())
        fn = int(((y_true == 1) & (y_hat == 0)).sum())

        rows.append(
            {
                "threshold": thr,
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )

    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def select_threshold(
    sweep_df: pd.DataFrame,
    *,
    min_recall: float = 0.90,
    objective: str = "f1",  # 'f1' or 'precision' (what to maximize once recall floor met)
    fallback: str = "f1",  # used if no threshold reaches min_recall
) -> pd.Series:
    """
    Choose a threshold with a recall floor.

    Strategy:
      1) Filter rows with recall >= min_recall.
      2) Among those, pick the one with best `objective` (f1 or precision).
      3) If none meet the floor, pick best `fallback` over the full sweep.

    Returns a single-row Series with the same columns as sweep_df.
    """
    if objective not in {"f1", "precision"}:
        raise ValueError("objective must be 'f1' or 'precision'")
    if fallback not in {"f1", "precision"}:
        raise ValueError("fallback must be 'f1' or 'precision'")

    feasible = sweep_df[sweep_df["recall"] >= float(min_recall)]
    if not feasible.empty:
        key = "f1_score" if objective == "f1" else "precision"
        best = feasible.sort_values(
            [key, "f1_score", "precision", "accuracy"],
            ascending=[False, False, False, False],
        ).iloc[0]
        return best

    fb_key = "f1_score" if fallback == "f1" else "precision"
    best = sweep_df.sort_values(
        [fb_key, "recall", "accuracy"], ascending=[False, False, False]
    ).iloc[0]
    return best


def run_threshold_experiments(
    model,
    X: pd.DataFrame,
    y,
    *,
    thresholds: Sequence[float] = tuple(np.linspace(0.05, 0.95, 19)),
    min_recall: float = 0.90,
    objective: str = "f1",
    fallback: str = "f1",
    save_csv_path: Optional[str] = None,
    log_to_mlflow: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convenience wrapper:
      - runs sweep_thresholds(...)
      - selects a threshold with select_threshold(...)
      - optionally saves CSV and logs to MLflow

    Returns:
      (sweep_df, chosen_row)
    """
    sweep = sweep_thresholds(model, X, y, thresholds=thresholds)
    chosen = select_threshold(
        sweep, min_recall=min_recall, objective=objective, fallback=fallback
    )

    if save_csv_path:
        import os

        os.makedirs(os.path.dirname(save_csv_path) or ".", exist_ok=True)
        sweep.to_csv(save_csv_path, index=False)

    if log_to_mlflow:
        try:
            import mlflow

            if save_csv_path and os.path.exists(save_csv_path):
                mlflow.log_artifact(save_csv_path)
            mlflow.log_metric("chosen_threshold", float(chosen["threshold"]))
            mlflow.log_metric("chosen_recall", float(chosen["recall"]))
            mlflow.log_metric("chosen_precision", float(chosen["precision"]))
            mlflow.log_metric("chosen_f1", float(chosen["f1_score"]))
            mlflow.log_metric("chosen_accuracy", float(chosen["accuracy"]))
        except Exception:
            # keep this library side-effect-free if mlflow isn't available
            pass

    return sweep, chosen

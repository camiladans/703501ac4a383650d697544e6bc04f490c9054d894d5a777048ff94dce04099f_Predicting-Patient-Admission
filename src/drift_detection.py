# src/drift_detection.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def _choose_feature_columns(df: pd.DataFrame) -> List[str]:
    """Extract feature columns only (exclude target if present)."""
    candidates_to_exclude = ["SOURCE", "target", "label", "y"]
    cols = list(df.columns)
    for c in candidates_to_exclude:
        if c in cols:
            cols.remove(c)
    return cols


def detect_drift(reference_data_path: str, current_data_path: str) -> Dict[str, Any]:
    """
    Run Evidently's DataDriftPreset on reference vs current CSVs and produce:

    {
      "drift_detected": bool,
      "feature_drifts": {"feature1": float, "feature2": float, ...},
      "overall_drift_score": float
    }

    - Loads CSVs
    - Uses only feature columns (target excluded if found)
    - Uses Report(metrics=[DataDriftPreset()])
    - Extracts:
        - drift_detected from metrics[0]["result"]["dataset_drift"]
        - per-feature scores from metrics[1]["result"]["drift_by_columns"]
          (falls back to metrics[0] if needed)
    - Ensures at least 3 features are included (or all if < 3)
    - Saves JSON to reports/drift_report.json and returns the same dict
    """
    # --- Load data ---
    ref_df = pd.read_csv(reference_data_path)
    cur_df = pd.read_csv(current_data_path)

    feature_cols = _choose_feature_columns(ref_df)
    # Keep common columns only
    feature_cols = [c for c in feature_cols if c in cur_df.columns]

    if len(feature_cols) == 0:
        raise ValueError(
            "No common feature columns found between reference and current data."
        )

    # --- Run Evidently report ---
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df[feature_cols], current_data=cur_df[feature_cols])
    rep = report.as_dict()

    # --- Extract drift_detected (dataset-level) ---
    # Per instructions: metrics[0]["result"]["dataset_drift"]
    try:
        drift_detected = bool(rep["metrics"][0]["result"]["dataset_drift"])
    except Exception:
        # Fallback to False if key path changes
        drift_detected = False

    # --- Extract feature-level drift scores ---
    # Per instructions: metrics[1]["result"]["drift_by_columns"]
    by_cols = None
    try:
        by_cols = rep["metrics"][1]["result"]["drift_by_columns"]
    except Exception:
        # Some versions place drift_by_columns in metrics[0]
        try:
            by_cols = rep["metrics"][0]["result"]["drift_by_columns"]
        except Exception:
            by_cols = {}

    # Each entry typically has keys like: { "drift_score": float, "drift_detected": bool, ... }
    # Choose at least 3 features (or all if < 3). If >3, pick the first 3 (simple & deterministic).
    selected = feature_cols if len(feature_cols) <= 3 else feature_cols[:3]

    feature_drifts: Dict[str, float] = {}
    for col in selected:
        info = by_cols.get(col, {})
        score = info.get("drift_score")
        # Ensure it's a float; if missing, fall back to 0.0
        try:
            feature_drifts[col] = float(score) if score is not None else 0.0
        except Exception:
            feature_drifts[col] = 0.0

    # --- Overall drift score: average of selected feature scores ---
    if len(feature_drifts) > 0:
        overall = sum(feature_drifts.values()) / len(feature_drifts)
    else:
        overall = 0.0

    out = {
        "drift_detected": drift_detected,
        "feature_drifts": feature_drifts,
        "overall_drift_score": float(overall),
    }

    # --- Save to reports/drift_report.json ---
    os.makedirs("reports", exist_ok=True)
    with open("reports/drift_report.json", "w") as f:
        json.dump(out, f, indent=2)

    return out

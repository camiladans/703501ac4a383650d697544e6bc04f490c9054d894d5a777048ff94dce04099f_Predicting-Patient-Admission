# src/drift_detection.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

# Evidently 0.7.x API
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping

# Column names to always exclude (targets, IDs, etc.)
_EXCLUDE_NAMES = {
    "SOURCE",
    "target",
    "label",
    "y",
    "prediction",
    "id",
    "gcr_persistent_id",
    "header_tran_key",
}


def _align(ref: pd.DataFrame, cur: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only shared columns in the same order."""
    shared = [c for c in ref.columns if c in cur.columns]
    return ref[shared].copy(), cur[shared].copy()


def _choose_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Choose usable feature columns:
    - Exclude known target/ID columns by name
    - Exclude datetime-like columns by dtype
    """
    cols = [c for c in df.columns if c not in _EXCLUDE_NAMES]
    dt_cols = set(df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns)
    return [c for c in cols if c not in dt_cols]


def _coerce_object_to_string(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce object columns to pandas 'string' dtype for consistency."""
    obj_cols = df.select_dtypes(include=["object"]).columns
    if len(obj_cols):
        df = df.copy()
        df[obj_cols] = df[obj_cols].astype("string")
    return df


def detect_drift(
    reference_csv_path: str,
    current_csv_path: str,
    html_report_path: Optional[str] = None,  # e.g., "/app/reports/drift_report.html"
) -> Dict[str, Any]:
    """
    Run Evidently DataDriftPreset on two CSVs and return a summary dict.

    Returns:
        {
          "drift_detected": bool,
          "overall_drift_score": float  # fraction [0..1] of drifted features
        }
    Compatible with Evidently >= 0.7.0 (tested on 0.7.11).
    """
    # --- Load ---
    ref = pd.read_csv(reference_csv_path)
    cur = pd.read_csv(current_csv_path)

    # --- Align schemas (shared columns in same order) ---
    ref, cur = _align(ref, cur)
    if ref.empty or cur.empty:
        raise ValueError(
            "Reference/current dataframes are empty after alignment (no shared columns or no rows)."
        )

    # --- Select features & basic sanitization ---
    features = _choose_feature_columns(ref)
    if not features:
        raise ValueError(
            "No usable feature columns found after exclusions; "
            "check your CSVs, exclude list, and datetime columns."
        )

    ref = ref[features]
    cur = cur[features]

    # Coerce object -> string for categorical consistency
    ref = _coerce_object_to_string(ref)
    cur = _coerce_object_to_string(cur)

    # Let Evidently infer types; we restrict to features explicitly
    mapping = ColumnMapping(
        target=None,
        prediction=None,
        numerical_features=None,
        categorical_features=None,
    )

    # --- Build & run report ---
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur, column_mapping=mapping)

    if html_report_path:
        report.save_html(html_report_path)

    # --- Parse summary ---
    summary = report.as_dict()
    drift_detected = False
    overall_score = 0.0

    # DataDriftPreset result carries 'dataset_drift' and 'share_of_drifted_features'
    for m in summary.get("metrics", []):
        res = m.get("result", {})
        if "dataset_drift" in res:
            drift_detected = bool(res["dataset_drift"])
            overall_score = float(res.get("share_of_drifted_features", 0.0))
            break

    return {
        "drift_detected": drift_detected,
        "overall_drift_score": overall_score,
    }

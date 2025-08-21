# src/drift_detection.py
from __future__ import annotations

import os
import json
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_EXCLUDE = {"SOURCE", "target", "label", "y", "prediction", "id"}


def _align(ref: pd.DataFrame, cur: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    common = [c for c in ref.columns if c in cur.columns]
    return ref[common].copy(), cur[common].copy()


def _feature_cols(df: pd.DataFrame) -> List[str]:
    keep = [c for c in df.columns if c not in _EXCLUDE]
    dt = set(df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns)
    return [c for c in keep if c not in dt]


def _coerce_objects_to_string(df: pd.DataFrame) -> pd.DataFrame:
    obj = df.select_dtypes(include=["object"]).columns
    if len(obj):
        df = df.copy()
        df[obj] = df[obj].astype("string")
    return df


def detect_drift(
    reference_csv: str,
    current_csv: str,
    *,
    target: Optional[str] = None,  # accepted but not required
    report_dir: str = "/app/reports",
    report_basename: str = "drift_report.json",
) -> Dict[str, Any]:
    """
    Run Evidently DataDriftPreset on two CSVs.
    Returns a compact dict suitable for Airflow XCom:
      - drift_detected: bool
      - overall_drift_score: float   (alias of share_of_drifted_columns)
      - share_of_drifted_columns: float
      - report_path: str (JSON with full Evidently output)
    """
    logger.info(f"Loading reference: {reference_csv}")
    logger.info(f"Loading current:   {current_csv}")
    ref_df = pd.read_csv(reference_csv)
    cur_df = pd.read_csv(current_csv)

    # Align, select features
    ref_df, cur_df = _align(ref_df, cur_df)
    cols = _feature_cols(ref_df)
    if cols:
        ref_df = _coerce_objects_to_string(ref_df[cols])
        cur_df = _coerce_objects_to_string(cur_df[cols])

    # Build Evidently datasets (DataDefinition left empty for auto-infer)
    data_def = DataDefinition()
    ref_data = Dataset.from_pandas(ref_df, data_definition=data_def)
    cur_data = Dataset.from_pandas(cur_df, data_definition=data_def)

    logger.info("Running Evidently DataDriftPreset...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_data, current_data=cur_data)
    res = report.as_dict()

    # Also save optional HTML if requested
    html_path = os.getenv("EVIDENTLY_HTML_PATH")
    if html_path:
        try:
            report.save_html(html_path)
            logger.info(f"Evidently HTML saved to: {html_path}")
        except Exception as e:
            logger.warning(f"Failed to save Evidently HTML to {html_path}: {e}")

    # Extract summary safely
    try:
        block = res["metrics"][0]["result"]
    except Exception:
        logger.error("Unexpected Evidently result shape. Keys: %s", list(res.keys()))
        raise

    drift_detected = bool(block.get("dataset_drift"))
    share = float(block.get("share_of_drifted_columns", 0.0))

    # Persist JSON report (so XCom stays tiny and MLflow can log it as artifact)
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, report_basename)
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(res, f)
        logger.info(f"Evidently JSON saved to: {report_path}")
    except Exception as e:
        logger.warning(f"Failed to write Evidently JSON to {report_path}: {e}")

    out = {
        "drift_detected": drift_detected,
        "overall_drift_score": share,
        "share_of_drifted_columns": share,
        "report_path": report_path,
    }
    logger.info(
        "Drift detected=%s, share_of_drifted_columns=%.3f (report: %s)",
        drift_detected,
        share,
        report_path,
    )
    return out

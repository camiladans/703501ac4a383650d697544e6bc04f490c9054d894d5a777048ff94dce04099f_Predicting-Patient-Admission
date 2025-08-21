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
    target: Optional[str] = None,  # optional label to exclude
    report_dir: str = "/app/reports",
    report_basename: str = "drift_report.json",
) -> Dict[str, Any]:
    """
    Run Evidently DataDriftPreset on two CSVs using only:
      - evidently.Report, Dataset, DataDefinition
      - evidently.presets.DataDriftPreset
      - os, json, typing.{Dict,Any}
    Returns a compact dict:
      - drift_detected: bool
      - overall_drift_score: float   (alias of share_of_drifted_columns)
      - share_of_drifted_columns: float
      - report_path: str  (JSON summary we write ourselves)
    """
    # 1) Load CSVs
    ref_df = pd.read_csv(reference_csv)
    cur_df = pd.read_csv(current_csv)

    # 2) Align & choose feature columns
    ref_df, cur_df = _align(ref_df, cur_df)
    if target:
        _EXCLUDE.add(target)
    cols = _feature_cols(ref_df)
    if cols:
        ref_df = _coerce_objects_to_string(ref_df[cols])
        cur_df = _coerce_objects_to_string(cur_df[cols])

    # 3) Convert to Evidently datasets
    data_def = DataDefinition()
    ref_data = Dataset.from_pandas(ref_df, data_definition=data_def)
    cur_data = Dataset.from_pandas(cur_df, data_definition=data_def)

    # 4) Run a Report with DataDriftPreset (no serialization APIs)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_data, current_data=cur_data)

    # 5) Extract drift info by introspecting report internals (no as_dict/json)
    def _maybe(obj, *names):
        for n in names:
            if hasattr(obj, n):
                v = getattr(obj, n)
                if v is not None:
                    return v
        return None

    res_obj = None
    # Common internal containers that hold metric results across versions
    for attr in ("results", "_results", "metric_results", "_metric_results"):
        items = getattr(report, attr, None)
        if not items:
            continue
        for it in items:
            r = _maybe(it, "result", "value", "data")
            if r is None:
                continue
            # Look for fields that indicate dataset-level drift summary
            has_flag = (
                (hasattr(r, "dataset_drift"))
                or (isinstance(r, dict) and "dataset_drift" in r)
                or hasattr(r, "drift_detected")
            )
            has_share = (
                (hasattr(r, "share_of_drifted_columns"))
                or (isinstance(r, dict) and "share_of_drifted_columns" in r)
                or hasattr(r, "overall_drift_score")
            )
            if has_flag or has_share:
                res_obj = r
                break
        if res_obj is not None:
            break

    # Last resort: deep scan for {dataset_drift, share_of_drifted_columns}
    if res_obj is None:

        def _deep_find(
            obj,
            keys=(
                "dataset_drift",
                "drift_detected",
                "share_of_drifted_columns",
                "overall_drift_score",
            ),
        ):
            seen = set()
            stack = [obj]
            while stack:
                cur = stack.pop()
                oid = id(cur)
                if oid in seen:
                    continue
                seen.add(oid)
                if isinstance(cur, dict):
                    if any(k in cur for k in keys):
                        return cur
                    stack.extend(cur.values())
                else:
                    for name in dir(cur):
                        if name.startswith("_"):
                            continue
                        try:
                            val = getattr(cur, name)
                        except Exception:
                            continue
                        stack.append(val)
            return None

        res_obj = _deep_find(report) or {}

    # Normalize access to values
    def _get(obj, *names, default=None):
        for n in names:
            if isinstance(obj, dict) and n in obj:
                return obj[n]
            if hasattr(obj, n):
                return getattr(obj, n)
        return default

    drift_detected = bool(
        _get(res_obj, "dataset_drift", "drift_detected", default=False)
    )
    share = float(
        _get(res_obj, "share_of_drifted_columns", "overall_drift_score", default=0.0)
        or 0.0
    )

    # 6) Persist a small JSON summary (so MLflow can log an artifact)
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, report_basename)
    if isinstance(res_obj, dict):
        payload = {
            "dataset_drift": res_obj.get("dataset_drift", drift_detected),
            "share_of_drifted_columns": res_obj.get("share_of_drifted_columns", share),
        }
    elif hasattr(res_obj, "__dict__"):
        payload = {k: v for k, v in res_obj.__dict__.items() if not k.startswith("_")}
    else:
        payload = {"dataset_drift": drift_detected, "share_of_drifted_columns": share}

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, default=str)

    return {
        "drift_detected": drift_detected,
        "overall_drift_score": share,
        "share_of_drifted_columns": share,
        "report_path": report_path,
    }

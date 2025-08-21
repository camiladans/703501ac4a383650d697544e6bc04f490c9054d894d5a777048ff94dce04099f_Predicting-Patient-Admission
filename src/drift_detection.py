# src/drift_detection.py
from __future__ import annotations
from typing import Dict, List, Tuple, Iterable, Optional
import pandas as pd

from evidently import Report

# 0.7+: Dataset/DataDefinition present; older versions won't have these
try:
    from evidently import Dataset, DataDefinition  # 0.7.x
except Exception:
    Dataset = DataDefinition = None  # type: ignore

# Prefer preset if available; else fall back to dataset-level metric
try:
    from evidently.metric_preset import (
        DataDriftPreset,
    )  # 0.4–0.6 (sometimes present in 0.7)

    _USE_PRESET = True
except Exception:
    from evidently.metrics.data_drift.dataset_drift_metric import DatasetDriftMetric

    _USE_PRESET = False

_DEFAULT_EXCLUDE = {
    "SOURCE",
    "target",
    "label",
    "y",
    "prediction",
    "id",
    "gcr_persistent_id",
    "header_tran_key",
}


def _align_cols(
    ref: pd.DataFrame, cur: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    shared = [c for c in ref.columns if c in cur.columns]
    return ref.loc[:, shared].copy(), cur.loc[:, shared].copy()


def _choose_feature_columns(df: pd.DataFrame, exclude: Iterable[str]) -> List[str]:
    # Drop explicit excludes and datetime-like cols
    excl = set(x.lower() for x in exclude)
    keep = [c for c in df.columns if c.lower() not in excl]
    dt_cols = set(
        df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, tz]"]).columns
    )
    return [c for c in keep if c not in dt_cols]


def _coerce_object_to_string(df: pd.DataFrame) -> pd.DataFrame:
    obj = df.select_dtypes(include=["object"]).columns
    if len(obj):
        df = df.copy()
        df[obj] = df[obj].astype("string")
    return df


def detect_drift(
    ref_csv: str,
    cur_csv: str,
    exclude_cols: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    ref = pd.read_csv(ref_csv)
    cur = pd.read_csv(cur_csv)

    # Align columns & sanitize dtypes
    ref, cur = _align_cols(ref, cur)
    features = _choose_feature_columns(ref, exclude_cols or _DEFAULT_EXCLUDE)
    ref = _coerce_object_to_string(ref[features])
    cur = _coerce_object_to_string(cur[features])

    metrics = [DataDriftPreset()] if _USE_PRESET else [DatasetDriftMetric()]

    if Dataset is not None:
        # Evidently 0.7+ path
        ref_data = Dataset.from_pandas(ref, data_definition=DataDefinition())
        cur_data = Dataset.from_pandas(cur, data_definition=DataDefinition())
        result = Report(metrics).run(reference_data=ref_data, current_data=cur_data)
        res = result.as_dict()
    else:
        # 0.4–0.6 path
        report = Report(metrics)
        report.run(reference_data=ref, current_data=cur)
        res = report.as_dict()

    # Safely extract dataset-level result
    metrics_list = res.get("metrics", [])
    if not metrics_list:
        return {"drift_detected": False, "overall_drift_share": 0.0, "raw": res}

    block = metrics_list[0].get("result", {})
    return {
        "drift_detected": bool(block.get("dataset_drift", False)),
        "overall_drift_share": float(block.get("share_of_drifted_columns", 0.0)),
        "raw": res,
    }

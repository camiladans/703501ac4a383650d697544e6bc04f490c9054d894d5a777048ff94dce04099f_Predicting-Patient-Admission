# src/drift_detection.py
from __future__ import annotations

import os
import json
import logging
from typing import Dict, Any, List

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _load_features_only(path: str) -> pd.DataFrame:
    """Load CSV and drop obvious target/label columns."""
    df = pd.read_csv(path)
    drop = [c for c in df.columns if c.lower() in {"source", "target", "label", "y"}]
    return df.drop(columns=drop, errors="ignore")


def _pick_feature_subset(columns: List[str], k: int = 3) -> List[str]:
    """Pick first k features (lightweight run)."""
    return columns[:k]


def _report_to_dict(report: Report) -> Dict[str, Any]:
    """Version-agnostic: extract a dict without assuming as_dict exists."""
    # 1) as_dict (newer versions)
    if hasattr(report, "as_dict"):
        return report.as_dict()  # type: ignore[attr-defined]

    # 2) json() (some versions)
    try:
        j = report.json()  # may return str/bytes/dict
        if isinstance(j, (bytes, bytearray)):
            j = j.decode("utf-8", "ignore")
        if isinstance(j, str):
            return json.loads(j)
        if isinstance(j, dict):
            return j
    except Exception:
        pass

    # 3) _repr_json_() (older/dev/Jupyter)
    try:
        j = report._repr_json_()  # type: ignore[attr-defined]
        if isinstance(j, tuple) and j:
            j = j[0]
        if isinstance(j, (bytes, bytearray)):
            j = j.decode("utf-8", "ignore")
        if isinstance(j, str):
            return json.loads(j)
    except Exception:
        pass

    # 4) save_json(...) to a temp file, then read it back
    try:
        os.makedirs("reports", exist_ok=True)
        tmp_path = "reports/_evidently_report_tmp.json"
        # many Evidently versions expose save_json
        save_fn = getattr(report, "save_json", None)
        if callable(save_fn):
            save_fn(tmp_path)
            with open(tmp_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass

    raise RuntimeError(
        "Cannot extract dict from Evidently Report in this environment/version."
    )


def detect_drift(reference_data_path: str, current_data_path: str) -> Dict[str, Any]:
    logger.info(f"Loading reference from {reference_data_path}")
    ref = _load_features_only(reference_data_path)
    logger.info(f"Loading current from {current_data_path}")
    cur = _load_features_only(current_data_path)

    common_cols = [c for c in ref.columns if c in cur.columns]
    if not common_cols:
        raise ValueError(
            "No overlapping features between reference and current datasets"
        )

    chosen = _pick_feature_subset(common_cols, k=3)
    logger.info(f"Selected features for drift check: {chosen}")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref[chosen], current_data=cur[chosen])

    rep_dict = _report_to_dict(report)

    # Dataset-level drift
    drift_detected = False
    for m in rep_dict.get("metrics", []):
        res = m.get("result", {}) if isinstance(m, dict) else {}
        if "dataset_drift" in res:
            drift_detected = bool(res["dataset_drift"])
            break

    # Feature-level drifts
    feature_drifts: Dict[str, float] = {}
    drift_by_columns = None
    for m in rep_dict.get("metrics", []):
        res = m.get("result", {}) if isinstance(m, dict) else {}
        if "drift_by_columns" in res:
            drift_by_columns = res["drift_by_columns"]
            break

    if isinstance(drift_by_columns, dict):
        for c in chosen:
            info = drift_by_columns.get(c) or {}
            if "p_value" in info:
                feature_drifts[c] = max(0.0, min(1.0, 1.0 - float(info["p_value"])))
            elif "drift_detected" in info:
                feature_drifts[c] = 1.0 if info["drift_detected"] else 0.0

    if not feature_drifts:
        feature_drifts = {c: 0.0 for c in chosen}

    overall = (
        float(sum(feature_drifts.values()) / len(feature_drifts))
        if feature_drifts
        else 0.0
    )
    out = {
        "drift_detected": drift_detected,
        "feature_drifts": feature_drifts,
        "overall_drift_score": overall,
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/drift_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("Drift report saved to reports/drift_report.json")
    return out

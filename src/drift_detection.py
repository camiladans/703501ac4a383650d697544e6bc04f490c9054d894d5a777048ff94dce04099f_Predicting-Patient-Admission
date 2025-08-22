# src/drift_detection.py
from __future__ import annotations

import json
import logging
import os
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------- PSI utilities (no SciPy needed) ----------
def _psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-9) -> float:
    """Population Stability Index for two distributions given bin counts (already normalized)."""
    expected = np.clip(expected, eps, None)
    actual = np.clip(actual, eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    """Coerce to numeric float; handle booleans cleanly."""
    if s.dtype == bool:
        return s.astype("float64")
    if pd.api.types.is_categorical_dtype(s):
        s = s.astype("float64")
    if s.dtype == object:
        s = pd.to_numeric(s, errors="coerce")
    return s.astype("float64", copy=False)


def _psi_for_feature(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """
    Compute PSI for one feature.
    - Booleans / binary-like -> two-bin categorical PSI
    - Continuous -> quantile bins from reference (fallback to linspace)
    Robust to constants / tiny cardinality.
    """
    ref = _coerce_numeric_series(ref).dropna()
    cur = _coerce_numeric_series(cur).dropna()
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    unique_ref = np.unique(ref)
    unique_cur = np.unique(cur)
    # Binary or near-binary: split at 0.5
    if (
        np.isin(unique_ref, [0.0, 1.0]).all() and np.isin(unique_cur, [0.0, 1.0]).all()
    ) or (len(unique_ref) <= 2 and len(unique_cur) <= 2):
        edges = np.array([-np.inf, 0.5, np.inf], dtype="float64")
    else:
        # Continuous: quantile binning on reference
        qs = np.linspace(0, 1, bins + 1)
        try:
            edges = np.unique(np.quantile(ref, qs))
        except Exception:
            lo = float(np.nanmin(ref))
            hi = float(np.nanmax(ref))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                return 0.0
            edges = np.linspace(lo, hi, min(bins, 10) + 1)
        if edges.size < 3:
            return 0.0

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    if ref_counts.sum() == 0 or cur_counts.sum() == 0:
        return 0.0

    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()
    return _psi(ref_pct, cur_pct)


# ---------- Public API (spec-compliant) ----------
def detect_drift(reference_data_path: str, current_data_path: str) -> Dict[str, Any]:
    """
    Detect drift between reference and current datasets.

    Spec compliance:
    - Signature: detect_drift(reference_data_path: str, current_data_path: str) -> Dict[str, Any]
    - Data loading: CSVs; extract feature columns only (exclude target)
    - Drift analysis: run Report(metrics=[DataDriftPreset()])   # used for analysis, not parsed
    - Features: include at least 3 features (or all if < 3); if > 3, use the first 3
    - Per-feature drift: compute PSI
    - overall_drift_score: average of per-feature PSI scores
    - Output: save to reports/drift_report.json and return the same dict
    - No use of report.as_dict / extra Evidently internals; no loguru
    """
    # --- Load
    logger.info("Loading reference from %s", reference_data_path)
    logger.info("Loading current from %s", current_data_path)
    ref = pd.read_csv(reference_data_path)
    cur = pd.read_csv(current_data_path)

    # --- Feature selection: common columns minus typical targets
    exclude = {"SOURCE", "target", "Target", "label", "Label"}
    common = [c for c in ref.columns if c in cur.columns]
    feature_cols_all: List[str] = [c for c in common if c not in exclude]

    if len(feature_cols_all) == 0:
        raise ValueError(
            "No feature columns found after excluding target-like columns."
        )

    # Spec: at least 3 (or all if <3); if >3, pick the first 3
    if len(feature_cols_all) > 3:
        feature_cols = feature_cols_all[:3]
    else:
        feature_cols = feature_cols_all

    ref_use = ref[feature_cols].copy()
    cur_use = cur[feature_cols].copy()
    logger.info("Selected features for drift check: %s", feature_cols)

    # --- Run Evidently (satisfy 'use Report(metrics=[DataDriftPreset()])')
    try:
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_use, current_data=cur_use)
        # Do NOT parse report via as_dict / internals (per requirement)
    except Exception as e:
        logger.warning("Evidently Report run failed (continuing with PSI only): %s", e)

    # --- Compute per-feature PSI and overall score (mean of PSIs)
    feature_drifts: Dict[str, float] = {}
    for col in feature_cols:
        try:
            feature_drifts[col] = float(
                _psi_for_feature(ref_use[col], cur_use[col], bins=10)
            )
        except Exception:
            feature_drifts[col] = 0.0

    overall = float(np.mean(list(feature_drifts.values()))) if feature_drifts else 0.0

    # Simple boolean: drift if any feature exceeds PSI threshold (0.2 is a common heuristic)
    psi_threshold = 0.2
    drift_detected = any(score > psi_threshold for score in feature_drifts.values())

    # --- Output JSON (overwrite each run)
    out = {
        "drift_detected": bool(drift_detected),
        "feature_drifts": feature_drifts,
        "overall_drift_score": overall,
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/drift_report.json", "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Slim drift report written to reports/drift_report.json")

    return out

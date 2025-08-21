# src/drift_detection.py
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------- Utilities ----------
def _report_to_dict(rep: Report) -> Optional[Dict]:
    """Best-effort JSON extraction. Returns dict or None (no raise)."""
    # 1) as_dict
    for meth in ("as_dict",):
        if hasattr(rep, meth):
            try:
                return getattr(rep, meth)()
            except Exception:
                pass

    # 2) JSON string methods
    for meth in ("as_json", "json"):
        if hasattr(rep, meth):
            try:
                s = getattr(rep, meth)()
                return json.loads(s)
            except Exception:
                pass

    # 3) Save to file if available
    try:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "report.json")
            if hasattr(rep, "save_json"):
                rep.save_json(p)
            elif hasattr(rep, "save"):
                # Some versions only save HTML; saving ".json" may not work.
                # We'll try anyway; if it fails, we fall back to None.
                rep.save(p)  # may create HTML instead; ignore on failure
            else:
                return None
            if os.path.exists(p):
                with open(p, "r") as f:
                    return json.load(f)
    except Exception:
        pass

    return None


def _deep_find_first(obj, keys: List[str]):
    if isinstance(obj, dict):
        for k in keys:
            if k in obj:
                return obj[k]
        for v in obj.values():
            hit = _deep_find_first(v, keys)
            if hit is not None:
                return hit
    elif isinstance(obj, list):
        for item in obj:
            hit = _deep_find_first(item, keys)
            if hit is not None:
                return hit
    return None


def _boolify(x) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in {"true", "1", "yes", "y"}
    return False


# ---------- PSI-based fallback (no SciPy needed) ----------
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
    - Continuous -> quantile bins from reference
    Robust to constants / tiny cardinality.
    """
    ref = _coerce_numeric_series(ref)
    cur = _coerce_numeric_series(cur)
    ref = ref.dropna()
    cur = cur.dropna()
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    # Binary/boolean or near-binary? Use two bins at 0.5
    unique_ref = np.unique(ref)
    unique_cur = np.unique(cur)
    if (
        ref.dtype == "float64"
        and np.isin(unique_ref, [0.0, 1.0]).all()
        and cur.dtype == "float64"
        and np.isin(unique_cur, [0.0, 1.0]).all()
    ) or (len(unique_ref) <= 2 and len(unique_cur) <= 2):
        edges = np.array([-np.inf, 0.5, np.inf], dtype="float64")
    else:
        # Continuous: try quantile binning
        qs = np.linspace(0, 1, bins + 1)
        try:
            edges = np.unique(np.quantile(ref, qs))
        except Exception:
            # Fallback to linspace over range
            lo = float(np.nanmin(ref))
            hi = float(np.nanmax(ref))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                return 0.0
            edges = np.linspace(lo, hi, min(bins, 10) + 1)

        # If still too few unique edges, bail as no distribution shape
        if edges.size < 3:
            return 0.0

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_total = ref_counts.sum()
    cur_total = cur_counts.sum()
    if ref_total == 0 or cur_total == 0:
        return 0.0

    ref_pct = ref_counts / ref_total
    cur_pct = cur_counts / cur_total

    return _psi(ref_pct, cur_pct)


# ---------- Public API ----------
def detect_drift(
    reference_path_or_df,
    current_path_or_df,
    *,
    features: Optional[List[str]] = None,
    drift_share_threshold: float = 0.33,  # fraction of columns drifting to flip the flag
    psi_threshold: float = 0.2,  # PSI > 0.2 => drifted column (industry heuristic)
) -> Dict:
    """
    Returns:
      {
        "drift_detected": bool,
        "overall_drift_score": float|None,    # share of drifted columns
        "selected_features": [...],
        "raw_report": dict|None,              # Evidently report (if extractable)
        "method": "evidently" | "psi_fallback"
      }
    """
    logger.info("Loading reference from %s", reference_path_or_df)
    ref = (
        pd.read_csv(reference_path_or_df)
        if isinstance(reference_path_or_df, str)
        else reference_path_or_df.copy()
    )

    logger.info("Loading current from %s", current_path_or_df)
    cur = (
        pd.read_csv(current_path_or_df)
        if isinstance(current_path_or_df, str)
        else current_path_or_df.copy()
    )

    # Feature selection: numeric intersection by default
    if features is None:
        common = [c for c in ref.columns if c in cur.columns]
        num = [c for c in common if pd.api.types.is_numeric_dtype(ref[c])]
        features = num or common
    ref_use = ref[features].copy()
    cur_use = cur[features].copy()

    logger.info("Selected features for drift check: %s", features)

    # 1) Try Evidently Report
    rep_dict = None
    drift_bool = None
    drift_share = None
    method = "evidently"
    try:
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_use, current_data=cur_use)
        rep_dict = _report_to_dict(report)

        if rep_dict is not None:
            # Try to find standard keys across versions
            drift_flag = _deep_find_first(rep_dict, ["dataset_drift", "drift_detected"])
            share = _deep_find_first(
                rep_dict,
                [
                    "drift_share",
                    "share_of_drifted_columns",
                    "number_of_drifted_columns_ratio",
                ],
            )
            drift_bool = _boolify(drift_flag)
            if isinstance(share, (int, float)):
                drift_share = float(share)
                if drift_bool is False:
                    drift_bool = drift_share >= drift_share_threshold
        else:
            method = "psi_fallback"
    except Exception:
        # If Evidently execution itself fails, drop to fallback
        method = "psi_fallback"

    # 2) PSI fallback (no JSON export or report unavailable)
    if method == "psi_fallback":
        drifted = 0
        total = 0
        for col in features:
            if not pd.api.types.is_numeric_dtype(ref_use[col]):
                # try coerce non-numeric quietly
                ref_use[col] = pd.to_numeric(ref_use[col], errors="coerce")
                cur_use[col] = pd.to_numeric(cur_use[col], errors="coerce")
            total += 1
            psi = _psi_for_feature(ref_use[col], cur_use[col], bins=10)
            if psi > psi_threshold:
                drifted += 1
        drift_share = (drifted / total) if total else 0.0
        drift_bool = drift_share >= drift_share_threshold

    return {
        "drift_detected": bool(drift_bool),
        "overall_drift_score": drift_share if drift_share is not None else None,
        "selected_features": features,
        "raw_report": rep_dict,  # may be None in fallback
        "method": method,
    }

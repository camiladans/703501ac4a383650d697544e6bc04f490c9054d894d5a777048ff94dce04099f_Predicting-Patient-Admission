# src/feature_engineering.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

"""
Feature Engineering for Patient Admission Labs.

Adds numeric, model-friendly derived features on top of the preprocessed dataset.
- Does NOT modify or drop the target column (default: 'SOURCE')
- Keeps original columns; adds new ones (all numeric/bool→int8)
- only computes features when inputs exist

New feature columns, if inputs exist
-------------------------------------------
- AGE_BIN (0..3), is_senior
- SEX_M (1 if M else 0)
- EST_HCT_FROM_RBC_MCV, DELTA_HCT
- EST_MCHC_FROM_HB_HCT, DELTA_MCHC
- MCH_over_MCV, RBC_over_HB, WBC_over_RBC, PLT_over_WBC, MCHC_times_MCV
- abnormal_lab_count and all is_* flags cast to int8

Usage from code:
    from src.feature_engineering import run_feature_engineering
    run_feature_engineering("data/train.csv", "data/test.csv")

Outputs:
    data/train_fe.csv
    data/test_fe.csv
"""

DATA_DIR = Path("data")


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """Elementwise safe division returning float; handles 0 and NaNs."""
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    out = a.astype("float64") / b.replace(0, np.nan).astype("float64")
    return out.replace([np.inf, -np.inf], np.nan)


def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64", copy=False)


def _add_if_exists(df: pd.DataFrame, name: str, func) -> None:
    """Compute df[name] = func(df) if func’s required columns exist; else no-op."""
    try:
        df[name] = func(df)
    except KeyError:
        # one or more required columns missing; skip
        pass


def build_features(df: pd.DataFrame, *, target_col: str = "SOURCE") -> pd.DataFrame:
    """
    Add engineered features on top of an already preprocessed frame.
    Assumes your preprocessing has already created is_* flags where applicable.
    """
    out = df.copy()

    # --- Age features ---
    if "AGE" in out.columns:
        # Integer codes for bins: (-inf,18], (18,40], (40,65], (65,inf)
        age_bins = pd.cut(
            _to_float(out["AGE"]),
            bins=[-np.inf, 18, 40, 65, np.inf],
            labels=[0, 1, 2, 3],
            right=True,
            include_lowest=True,
        )
        out["AGE_BIN"] = age_bins.astype("float64")
        out["is_senior"] = (_to_float(out["AGE"]) >= 65).astype("int8")

    # --- Sex convenience numeric ---
    if "SEX" in out.columns:
        out["SEX_M"] = (out["SEX"].astype(str).str.upper().eq("M")).astype("int8")

    # --- RBC index: estimated hematocrit from RBC & MCV (≈ RBC * MCV / 10) ---
    def _est_hct(d: pd.DataFrame) -> pd.Series:
        return _to_float(d["ERYTHROCYTE"]) * _to_float(d["MCV"]) / 10.0

    _add_if_exists(out, "EST_HCT_FROM_RBC_MCV", _est_hct)

    # Delta between measured and estimated hematocrit
    def _delta_hct(d: pd.DataFrame) -> pd.Series:
        return _to_float(d["HAEMATOCRIT"]) - (
            _to_float(d["ERYTHROCYTE"]) * _to_float(d["MCV"]) / 10.0
        )

    _add_if_exists(out, "DELTA_HCT", _delta_hct)

    # --- Estimated MCHC from Hb & Hct: MCHC_est ≈ (Hb * 100) / Hct ---
    def _est_mchc(d: pd.DataFrame) -> pd.Series:
        return _safe_div(
            _to_float(d["HAEMOGLOBINS"]) * 100.0, _to_float(d["HAEMATOCRIT"])
        )

    _add_if_exists(out, "EST_MCHC_FROM_HB_HCT", _est_mchc)

    def _delta_mchc(d: pd.DataFrame) -> pd.Series:
        return _to_float(d["MCHC"]) - _safe_div(
            _to_float(d["HAEMOGLOBINS"]) * 100.0, _to_float(d["HAEMATOCRIT"])
        )

    _add_if_exists(out, "DELTA_MCHC", _delta_mchc)

    # --- Ratios / interactions ---
    def _mch_over_mcv(d: pd.DataFrame) -> pd.Series:
        return _safe_div(_to_float(d["MCH"]), _to_float(d["MCV"]))

    _add_if_exists(out, "MCH_over_MCV", _mch_over_mcv)

    def _rbc_over_hb(d: pd.DataFrame) -> pd.Series:
        return _safe_div(_to_float(d["ERYTHROCYTE"]), _to_float(d["HAEMOGLOBINS"]))

    _add_if_exists(out, "RBC_over_HB", _rbc_over_hb)

    def _wbc_over_rbc(d: pd.DataFrame) -> pd.Series:
        return _safe_div(_to_float(d["LEUCOCYTE"]), _to_float(d["ERYTHROCYTE"]))

    _add_if_exists(out, "WBC_over_RBC", _wbc_over_rbc)

    def _plt_over_wbc(d: pd.DataFrame) -> pd.Series:
        return _safe_div(_to_float(d["THROMBOCYTE"]), _to_float(d["LEUCOCYTE"]))

    _add_if_exists(out, "PLT_over_WBC", _plt_over_wbc)

    def _mchc_times_mcv(d: pd.DataFrame) -> pd.Series:
        return _to_float(d["MCHC"]) * _to_float(d["MCV"])

    _add_if_exists(out, "MCHC_times_MCV", _mchc_times_mcv)

    # --- Abnormal flags summary ---
    flag_cols = [
        "is_hct_normal",
        "is_hb_normal",
        "is_rbc_normal",
        "is_wbc_normal",
        "is_plt_normal",
        "is_mch_normal",
        "is_mchc_normal",
        "is_mcv_normal",
    ]
    present_flags = [c for c in flag_cols if c in out.columns]
    if present_flags:
        flags_numeric = (~out[present_flags].astype(bool)).astype("int8")
        out["abnormal_lab_count"] = flags_numeric.sum(axis=1).astype("int16")
        out[present_flags] = out[present_flags].astype("int8")

    return out


def run_feature_engineering(
    train_csv: str,
    test_csv: str,
    *,
    out_train_fe: Optional[str] = None,
    out_test_fe: Optional[str] = None,
    target_col: str = "SOURCE",
) -> Dict[str, Any]:
    """
    Read train/test CSVs, build features, and write train_fe.csv / test_fe.csv.
    Returns dict with output paths.
    """
    out_train_fe = out_train_fe or str(DATA_DIR / "train_fe.csv")
    out_test_fe = out_test_fe or str(DATA_DIR / "test_fe.csv")

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    df_train_fe = build_features(df_train, target_col=target_col)
    df_test_fe = build_features(df_test, target_col=target_col)

    Path(out_train_fe).parent.mkdir(parents=True, exist_ok=True)
    df_train_fe.to_csv(out_train_fe, index=False)
    df_test_fe.to_csv(out_test_fe, index=False)

    return {"train_fe": out_train_fe, "test_fe": out_test_fe}

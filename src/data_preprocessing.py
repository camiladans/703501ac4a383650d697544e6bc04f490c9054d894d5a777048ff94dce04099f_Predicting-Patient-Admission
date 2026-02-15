# src/data_preprocessing.py
"""
Module: data_preprocessing.py

Handles raw data cleaning and normalization steps:
- Flags abnormal lab values based on sex-specific ranges
- Splits cleaned data into train/test
- Generates ONE drifted copy of train/test and saves to data/
- Returns tuple:
  (X_train, X_test, y_train, y_test,
   X_train_drifted, y_train_drifted, X_test_drifted, y_test_drifted)
"""

# src/data_preprocessing.py
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def flag_out_of_range(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean flags for common lab ranges (only when columns exist)."""
    out = df.copy()

    if "SEX" in out and "HAEMATOCRIT" in out:
        out["is_hct_normal"] = (
            (out["SEX"] == "M") & out["HAEMATOCRIT"].between(40.0, 52.0)
        ) | ((out["SEX"] == "F") & out["HAEMATOCRIT"].between(37.0, 47.0))

    if "SEX" in out and "HAEMOGLOBINS" in out:
        out["is_hb_normal"] = (
            (out["SEX"] == "M") & out["HAEMOGLOBINS"].between(13.0, 17.0)
        ) | ((out["SEX"] == "F") & out["HAEMOGLOBINS"].between(12.0, 16.0))

    if "SEX" in out and "ERYTHROCYTE" in out:
        out["is_rbc_normal"] = (
            (out["SEX"] == "M") & out["ERYTHROCYTE"].between(4.5, 6.1)
        ) | ((out["SEX"] == "F") & out["ERYTHROCYTE"].between(4.0, 5.4))

    if "LEUCOCYTE" in out:
        out["is_wbc_normal"] = out["LEUCOCYTE"].between(4.0, 10.8)
    if "THROMBOCYTE" in out:
        out["is_plt_normal"] = out["THROMBOCYTE"].between(150, 400)
    if "MCH" in out:
        out["is_mch_normal"] = out["MCH"].between(27.0, 33.0)
    if "MCHC" in out:
        out["is_mchc_normal"] = out["MCHC"].between(31.5, 37.0)
    if "MCV" in out:
        out["is_mcv_normal"] = out["MCV"].between(80, 98)

    return out


def preprocess_data(
    df: pd.DataFrame,
    target_col: str,
    categorical_cols: Optional[Sequence[str]] = None,
    numeric_cols: Optional[Sequence[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    save_dir: str = "data",
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    """
    Preprocess dataset into train/test splits, then synthesize drifted variants.

    Returns:
      (X_train, X_test, y_train, y_test,
       X_train_drifted, y_train_drifted, X_test_drifted, y_test_drifted)

    Saves:
      drifted_train.csv and drifted_test.csv under `save_dir`.
    """
    rng = np.random.default_rng(random_state)
    df = df.copy()

    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not in DataFrame")

    # Optional: flag ranges (safe if columns absent)
    df = flag_out_of_range(df)

    y = df[target_col]
    X = df.drop(columns=[target_col])

    if categorical_cols is None:
        categorical_cols = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
    if numeric_cols is None:
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    logger.debug(
        "Detected %d numeric and %d categorical features.",
        len(numeric_cols),
        len(categorical_cols),
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() <= 20 else None,
    )

    # --- Numeric drift (σ = 0.1 * std(train); or ×1.2) ---
    if numeric_cols:
        train_std = X_train[numeric_cols].std(ddof=0).replace({0.0: np.nan})
        # fallback sigma if any col has NaN std
        if len(numeric_cols):
            overall_std = float(np.nanstd(X_train[numeric_cols].to_numpy()))
            fallback_sigma = (
                1.0 if not np.isfinite(overall_std) else max(overall_std, 1e-12)
            )
        else:
            fallback_sigma = 1.0
        sigma_map = (0.1 * train_std).fillna(0.1 * fallback_sigma)
        mult_mask = pd.Series(rng.random(len(numeric_cols)) < 0.5, index=numeric_cols)
    else:
        sigma_map = pd.Series(dtype=float)
        mult_mask = pd.Series(dtype=bool)

    def apply_numeric_drift(df_numeric: pd.DataFrame) -> pd.DataFrame:
        out = df_numeric.copy()
        for col in numeric_cols or []:
            col_vals = out[col].to_numpy(dtype=float, copy=True)
            if mult_mask.get(col, False):
                col_vals = np.where(np.isnan(col_vals), col_vals, col_vals * 1.2)
            else:
                sigma = float(sigma_map.get(col, 0.1))
                noise = rng.normal(0.0, sigma, size=col_vals.shape)
                col_vals = np.where(np.isnan(col_vals), col_vals, col_vals + noise)
            out[col] = col_vals
        return out

    # --- Categorical drift (flip 10–15% uniformly to a different class) ---
    cat_domains = {
        c: pd.Index(X_train[c].dropna().unique()) for c in (categorical_cols or [])
    }
    flip_ratio = float(rng.uniform(0.10, 0.15))

    def apply_categorical_drift(df_cat: pd.DataFrame) -> pd.DataFrame:
        out = df_cat.copy()
        for col in categorical_cols or []:
            dom = cat_domains.get(col, pd.Index([]))
            if len(dom) <= 1:
                continue
            mask_non_null = out[col].notna().to_numpy()
            idx = np.flatnonzero(mask_non_null)
            if idx.size == 0:
                continue
            k = max(1, int(np.floor(flip_ratio * idx.size)))
            flip_idx = rng.choice(idx, size=min(k, idx.size), replace=False)
            vals = out[col].astype(object).to_numpy(copy=True)
            for i in flip_idx:
                choices = dom[dom != vals[i]]
                if len(choices) > 0:
                    vals[i] = rng.choice(choices.to_numpy())
            out[col] = vals
        return out

    def drift_frame(X_like: pd.DataFrame) -> pd.DataFrame:
        Xd = X_like.copy()
        if numeric_cols:
            Xd.update(apply_numeric_drift(Xd[numeric_cols]))
        if categorical_cols:
            Xd.update(apply_categorical_drift(Xd[categorical_cols]))
        return Xd

    X_train_drifted = drift_frame(X_train)
    X_test_drifted = drift_frame(X_test)
    y_train_drifted, y_test_drifted = y_train.copy(), y_test.copy()

    # Save drifted datasets (features + target)
    os.makedirs(save_dir, exist_ok=True)
    drifted_train = X_train_drifted.copy()
    drifted_train[target_col] = y_train_drifted
    drifted_test = X_test_drifted.copy()
    drifted_test[target_col] = y_test_drifted
    drifted_train.to_csv(os.path.join(save_dir, "drifted_train.csv"), index=False)
    drifted_test.to_csv(os.path.join(save_dir, "drifted_test.csv"), index=False)
    # logger.debug("Saved drifted_train.csv and drifted_test.csv to %s", save_dir)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_drifted,
        y_train_drifted,
        X_test_drifted,
        y_test_drifted,
    )

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

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def flag_out_of_range(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean flags for common lab ranges (only when columns exist)."""
    df = df.copy()

    if "SEX" in df and "HAEMATOCRIT" in df:
        df["is_hct_normal"] = (
            (df["SEX"] == "M") & df["HAEMATOCRIT"].between(40.0, 52.0)
        ) | ((df["SEX"] == "F") & df["HAEMATOCRIT"].between(37.0, 47.0))

    if "SEX" in df and "HAEMOGLOBINS" in df:
        df["is_hb_normal"] = (
            (df["SEX"] == "M") & df["HAEMOGLOBINS"].between(13.0, 17.0)
        ) | ((df["SEX"] == "F") & df["HAEMOGLOBINS"].between(12.0, 16.0))

    if "SEX" in df and "ERYTHROCYTE" in df:
        df["is_rbc_normal"] = (
            (df["SEX"] == "M") & df["ERYTHROCYTE"].between(4.5, 6.1)
        ) | ((df["SEX"] == "F") & df["ERYTHROCYTE"].between(4.0, 5.4))

    if "LEUCOCYTE" in df:
        df["is_wbc_normal"] = df["LEUCOCYTE"].between(4.0, 10.8)
    if "THROMBOCYTE" in df:
        df["is_plt_normal"] = df["THROMBOCYTE"].between(150, 400)
    if "MCH" in df:
        df["is_mch_normal"] = df["MCH"].between(27.0, 33.0)
    if "MCHC" in df:
        df["is_mchc_normal"] = df["MCHC"].between(31.5, 37.0)
    if "MCV" in df:
        df["is_mcv_normal"] = df["MCV"].between(80, 98)

    return df


# ---------------------------
# Helpers for drift creation
# ---------------------------
def _infer_cols(
    df: pd.DataFrame, target_col: Optional[str] = None
) -> Tuple[list[str], list[str]]:
    exclude = {target_col} if target_col else set()
    num_cols = [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    cat_cols = [
        c
        for c in df.columns
        if c not in exclude and not pd.api.types.is_numeric_dtype(df[c])
    ]
    return num_cols, cat_cols


def _drift_numeric(
    df: pd.DataFrame,
    num_cols: Sequence[str],
    train_std: Dict[str, float],
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = df.copy()
    for col in num_cols:
        if col not in out.columns:
            continue
        # 50/50: multiply-by-1.2 vs add Gaussian noise N(0, 0.1*std_train)
        if rng.random() < 0.5:
            out[col] = out[col] * 1.2
        else:
            std = float(train_std.get(col, float(out[col].std() or 0.0)))
            sigma = 0.1 * std
            if sigma == 0:
                out[col] = out[col] * 1.2
            else:
                out[col] = out[col] + rng.normal(0.0, sigma, size=len(out))
    return out


def _drift_categorical(
    df: pd.DataFrame,
    cat_cols: Sequence[str],
    rng: np.random.Generator,
    flip_low: float = 0.10,
    flip_high: float = 0.15,
) -> pd.DataFrame:
    out = df.copy()
    for col in cat_cols:
        if col not in out.columns:
            continue
        s = out[col].astype(object)
        cats = pd.Series(s.dropna().unique())
        if len(cats) < 2:
            continue

        flip_rate = rng.uniform(flip_low, flip_high)
        idx = np.where(s.notna())[0]
        if len(idx) == 0:
            continue
        n_flip = int(np.floor(flip_rate * len(idx)))
        if n_flip <= 0:
            continue

        flip_rows = rng.choice(idx, size=n_flip, replace=False)
        for r in flip_rows:
            cur = s.iat[r]
            choices = cats[cats != cur]
            if len(choices) == 0:
                continue
            s.iat[r] = rng.choice(choices.values)
        out[col] = s
    return out


# --------------------------------
# Main entry: preprocessing + drift
# --------------------------------
def preprocess_data(
    path: str,
    target_col: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
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
    Reads CSV at `path`, flags lab-value ranges, splits into train/test,
    creates ONE drifted copy, and saves outputs to data/.

    Numeric drift:
      - multiply by 1.2 OR add N(0, 0.1*std_train) noise (std from TRAIN)
    Categorical drift:
      - randomly flip 10–15% of non-null values to a different category

    Saves:
      data/train.csv
      data/test.csv
      data/drifted_train.csv
      data/drifted_test.csv

    Returns
    -------
    (X_train, X_test, y_train, y_test,
     X_train_drifted, y_train_drifted, X_test_drifted, y_test_drifted)
    """
    rng = np.random.default_rng(seed=random_state)

    df = pd.read_csv(path, encoding_errors="ignore")
    df = flag_out_of_range(df)

    # Split features/target if provided
    if target_col and target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        y = pd.Series([None] * len(df), name="target")
        X = df

    # Train/test split (optionally stratified on y when valid)
    strat = y if (stratify and y.nunique(dropna=True) > 1) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
        shuffle=True,
    )

    # Save baseline splits
    pd.concat([X_train, y_train], axis=1).to_csv(DATA_DIR / "train.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(DATA_DIR / "test.csv", index=False)

    # Infer feature types (exclude target)
    tmp_with_target = pd.concat(
        [X_train, y_train.rename(target_col or "target")], axis=1
    )
    num_cols, cat_cols = _infer_cols(tmp_with_target, target_col=target_col)

    # Train std per numeric col for noise scaling
    train_std = {c: float(X_train[c].std() or 0.0) for c in num_cols}

    # ------- Single drift pass -------
    X_train_drifted = _drift_numeric(X_train, num_cols, train_std, rng)
    X_train_drifted = _drift_categorical(X_train_drifted, cat_cols, rng)

    X_test_drifted = _drift_numeric(X_test, num_cols, train_std, rng)
    X_test_drifted = _drift_categorical(X_test_drifted, cat_cols, rng)

    # Labels are NOT drifted; make explicit drifted label variables
    y_train_drifted = y_train.copy()
    y_test_drifted = y_test.copy()

    # Save drifted versions
    drifted_train_df = X_train_drifted.copy()
    drifted_test_df = X_test_drifted.copy()
    drifted_train_df[(target_col or "target")] = y_train_drifted.values
    drifted_test_df[(target_col or "target")] = y_test_drifted.values
    drifted_train_df.to_csv(DATA_DIR / "drifted_train.csv", index=False)
    drifted_test_df.to_csv(DATA_DIR / "drifted_test.csv", index=False)

    # Return tuple with explicit drifted label variables
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

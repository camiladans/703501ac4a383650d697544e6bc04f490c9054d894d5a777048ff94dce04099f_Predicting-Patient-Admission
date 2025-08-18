"""
Module: data_preprocessing.py

Handles raw data cleaning and normalization steps:
- Flags abnormal lab values based on sex-specific ranges
- Splits cleaned data into train/test
- Generates drifted copies of train/test and saves to data/
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def flag_out_of_range(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['is_hct_normal'] = ((df['SEX'] == 'M') & df['HAEMATOCRIT'].between(40.0, 52.0)) | \
                          ((df['SEX'] == 'F') & df['HAEMATOCRIT'].between(37.0, 47.0))

    df['is_hb_normal'] = ((df['SEX'] == 'M') & df['HAEMOGLOBINS'].between(13.0, 17.0)) | \
                         ((df['SEX'] == 'F') & df['HAEMOGLOBINS'].between(12.0, 16.0))

    df['is_rbc_normal'] = ((df['SEX'] == 'M') & df['ERYTHROCYTE'].between(4.5, 6.1)) | \
                          ((df['SEX'] == 'F') & df['ERYTHROCYTE'].between(4.0, 5.4))

    df['is_wbc_normal'] = df['LEUCOCYTE'].between(4.0, 10.8)
    df['is_plt_normal'] = df['THROMBOCYTE'].between(150, 400)
    df['is_mch_normal'] = df['MCH'].between(27.0, 33.0)
    df['is_mchc_normal'] = df['MCHC'].between(31.5, 37.0)
    df['is_mcv_normal'] = df['MCV'].between(80, 98)

    return df


# ---------------------------
# Helpers for drift creation
# ---------------------------
def _infer_cols(df: pd.DataFrame, target_col: Optional[str] = None) -> tuple[list[str], list[str]]:
    exclude = {target_col} if target_col else set()
    num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in exclude and not pd.api.types.is_numeric_dtype(df[c])]
    return num_cols, cat_cols


def _drift_numeric(df: pd.DataFrame,
                   num_cols: Sequence[str],
                   train_std: Dict[str, float],
                   rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    for col in num_cols:
        if col not in out.columns:
            continue
        # 50/50 choose multiply-by-1.2 vs add Gaussian noise N(0, 0.1*std_train)
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


def _drift_categorical(df: pd.DataFrame,
                       cat_cols: Sequence[str],
                       rng: np.random.Generator,
                       flip_low: float = 0.10,
                       flip_high: float = 0.15) -> pd.DataFrame:
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
def preprocess_data(path: str,
                    target_col: Optional[str] = None,
                    test_size: float = 0.2,
                    random_state: int = 42,
                    stratify: bool = True):
    """
    Reads CSV at `path`, flags lab-value ranges, splits into train/test,
    creates drifted copies (per spec), and saves outputs to data/.

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
    (train_data, test_data)  # kept for backward compatibility
    """
    rng = np.random.default_rng(seed=random_state)

    df = pd.read_csv(path, encoding_errors="ignore")
    df = flag_out_of_range(df)

    y = df[target_col] if target_col and target_col in df.columns else None
    strat = y if (stratify and y is not None and pd.Series(y).nunique() > 1) else None

    train_data, test_data = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=strat, shuffle=True
    )

    # Save baseline splits
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    # Infer feature types (exclude target)
    num_cols, cat_cols = _infer_cols(train_data, target_col=target_col)

    # Train std per numeric col for noise scaling
    train_std = {c: float(train_data[c].std() or 0.0) for c in num_cols}

    # Build drifted copies without touching the target column
    def _strip_target(df_in: pd.DataFrame) -> pd.DataFrame:
        return df_in.drop(columns=[target_col], errors="ignore")

    drifted_train = _drift_numeric(_strip_target(train_data), num_cols, train_std, rng)
    drifted_train = _drift_categorical(drifted_train, cat_cols, rng)
    if target_col and target_col in train_data.columns:
        drifted_train[target_col] = train_data[target_col].values

    drifted_test = _drift_numeric(_strip_target(test_data), num_cols, train_std, rng)
    drifted_test = _drift_categorical(drifted_test, cat_cols, rng)
    if target_col and target_col in test_data.columns:
        drifted_test[target_col] = test_data[target_col].values

    # Save drifted files (required)
    (DATA_DIR / "drifted_train.csv").write_text("")  # ensure file path exists on some FS
    drifted_train.to_csv(DATA_DIR / "drifted_train.csv", index=False)
    drifted_test.to_csv(DATA_DIR / "drifted_test.csv", index=False)

    return train_data, test_data

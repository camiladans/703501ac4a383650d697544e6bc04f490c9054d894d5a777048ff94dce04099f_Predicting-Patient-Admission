# src/data_ingestion.py
"""
Module: data_ingestion.py

Reads a raw dataset (CSV or Parquet), optionally validates/filters columns,
writes a canonical copy to disk, and RETURNS a pandas.DataFrame for downstream
steps (preprocessing, feature engineering, training).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

# --------------------------------------------------------------------
# Logger setup
# --------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------
def _ensure_parent_dir(path: Path) -> None:
    """Ensure parent directories exist for a given path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_table(path: Path) -> pd.DataFrame:
    """Read CSV or Parquet based on file extension."""
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type for: {path}")


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    _ensure_parent_dir(path)
    df.to_csv(path, index=False)


def _validate_expected(df: pd.DataFrame, expected_cols: Iterable[str]) -> None:
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns: {missing}. Present columns: {list(df.columns)}"
        )


# --------------------------------------------------------------------
# Main ingestion function
# --------------------------------------------------------------------
def ingest_data(
    input_path: str | Path,
    canonical_path: str | Path,
    *,
    expected_cols: Optional[Sequence[str]] = None,
    keep_cols: Optional[Sequence[str]] = None,
    save_canonical: bool = True,
) -> pd.DataFrame:
    """
    Read raw data, optionally validate/trim columns, write canonical copy, return DataFrame.

    Parameters
    ----------
    input_path : str | Path
        Source file (CSV or Parquet).
    canonical_path : str | Path
        Where to write the canonical CSV copy (usually under data/raw/).
    expected_cols : Sequence[str], optional
        If provided, validate that ALL these columns exist in the input.
    keep_cols : Sequence[str], optional
        If provided, return ONLY these columns (will validate they exist).
    save_canonical : bool, default True
        If True, write a canonical CSV to `canonical_path`.

    Returns
    -------
    pd.DataFrame
        The loaded (and optionally filtered) DataFrame.
    """
    in_path = Path(str(input_path))
    can_path = Path(str(canonical_path))

    logger.info(f"[ingestion] Reading dataset from: {in_path}")
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    df = _read_table(in_path)

    # Validate schema if requested
    if expected_cols is not None:
        _validate_expected(df, expected_cols)

    # Keep only requested columns if provided
    if keep_cols is not None:
        _validate_expected(df, keep_cols)
        df = df.loc[:, list(keep_cols)]

    # Write canonical CSV copy
    if save_canonical:
        _write_csv(df, can_path)
        logger.info(f"[ingestion] Saved canonical dataset to: {can_path}")

    logger.info(f"[ingestion] Loaded shape: {df.shape}")
    return df


__all__ = ["ingest_data"]

# src/data_ingestion.py
"""
Module: data_ingestion.py

Reads a raw dataset (CSV or Parquet), optionally validates/filters columns,
writes a canonical copy to disk, and RETURNS a pandas.DataFrame for downstream steps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # let Airflow attach handlers


# -----------------------
# Helper functions
# -----------------------
def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_table(path: Path) -> pd.DataFrame:
    """Read CSV (incl. compressed) or Parquet based on extension."""
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt", ".gz", ".bz2", ".zip"} or str(path).endswith(
        ".csv.gz"
    ):
        # You can tune dtype/parse_dates here as needed
        return pd.read_csv(path, encoding="utf-8", on_bad_lines="error")
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)  # requires pyarrow or fastparquet
        except ImportError as e:
            raise ImportError(
                "Reading Parquet requires 'pyarrow' or 'fastparquet'. "
                "Install one of them in your image/environment."
            ) from e
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


# -----------------------
# Main function
# -----------------------
def ingest_data(
    input_path: str | Path,
    canonical_path: str | Path | None = None,
    expected_cols: Optional[Sequence[str]] = None,
    keep_cols: Optional[Sequence[str]] = None,
    save_canonical: bool = True,
    **_,
) -> pd.DataFrame:
    """
    Read raw data, optionally validate/trim columns, write canonical copy, return DataFrame.
    """
    in_path = Path(str(input_path))

    if not in_path.exists():
        # Helpful diagnostics
        candidates = [
            "/app/data/raw",
            "/opt/airflow/repo/data/raw",
            "/opt/airflow/dags/data/raw",
            str(in_path.parent),
        ]
        listings = []
        for d in dict.fromkeys(candidates):  # de-dup while preserving order
            p = Path(d)
            if p.exists():
                try:
                    items = ", ".join(sorted(x.name for x in p.glob("*")))
                except Exception:
                    items = "(unable to list)"
                listings.append(f"{d}: {items}")
        hint = "\n".join(listings) if listings else "(no common data dirs exist)"
        raise FileNotFoundError(
            f"Input file not found: {in_path}\nChecked some common directories:\n{hint}"
        )

    can_path = (
        Path(str(canonical_path))
        if canonical_path is not None
        else in_path.with_suffix(".csv")
    )

    logger.info("[ingestion] Reading dataset from: %s", in_path)
    df = _read_table(in_path)

    if expected_cols is not None:
        _validate_expected(df, expected_cols)

    if keep_cols is not None:
        _validate_expected(df, keep_cols)
        df = df.loc[:, list(keep_cols)]

    if save_canonical:
        _write_csv(df, can_path)
        logger.info("[ingestion] Saved canonical dataset to: %s", can_path)

    logger.info("[ingestion] Loaded shape: %s", df.shape)
    return df


__all__ = ["ingest_data"]

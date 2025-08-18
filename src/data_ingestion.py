# src/data_ingestion.py
"""
Loads/downloads the dataset into data/raw/ and returns the local CSV path.
Designed to feed directly into preprocess_data(path=...).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_SAVE = RAW_DIR / "data-ori.csv"


def ingest_data(
    source_path: Optional[str] = None, save_path: Path = DEFAULT_SAVE
) -> str:
    """
    Ingest dataset and save to data/raw/, returning the local path.

    Args:
        source_path: Local CSV path or URL. If None, require that save_path already exists.
        save_path:   Where to store the canonical CSV (default: data/raw/data-ori.csv).

    Returns:
        str: Absolute/relative path to the saved CSV for downstream use.
    """
    save_path = Path(save_path)

    if source_path is None:
        if save_path.exists():
            print(f"[ingestion] Using existing dataset at {save_path}")
            return str(save_path)
        raise FileNotFoundError(
            f"No source_path provided and {save_path} does not exist."
        )

    # Read from local CSV or URL; pandas handles both.
    print(f"[ingestion] Reading dataset from: {source_path}")
    df = pd.read_csv(source_path, encoding_errors="ignore")

    # Ensure parent dir exists and write canonical copy
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[ingestion] Saved canonical dataset to: {save_path}")

    return str(save_path)

"""
data_loader.py
---------------
Utility functions for loading the raw e-commerce sales dataset.
"""

import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "ecommerce_sales.csv"
PROCESSED_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "cleaned_sales.csv"


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw e-commerce sales dataset from disk.

    Uses latin-1 encoding because the Superstore dataset contains a
    small number of non-UTF-8 characters (e.g. in customer names).
    """
    df = pd.read_csv(path, encoding="latin-1")
    return df


def load_processed_data(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the cleaned/processed dataset from disk."""
    df = pd.read_csv(path, parse_dates=["Order Date", "Ship Date"], encoding="latin-1")
    return df


def data_quality_report(df: pd.DataFrame) -> None:
    """Print a concise, dynamically-calculated data quality report."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    date_like_cols = [c for c in df.columns if "date" in c.lower()]

    print("Dataset Shape:")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}\n")

    print("Numerical Features:")
    print(numeric_cols, "\n")

    print("Categorical Features:")
    print(categorical_cols, "\n")

    print("Date-like Features:")
    print(date_like_cols, "\n")

    print("Missing Values:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values detected.")
    print()

    print("Duplicate Rows:")
    print(df.duplicated().sum())

"""
preprocessing.py
-----------------
Data cleaning pipeline: missing values, duplicates, type conversion,
and outlier inspection for the e-commerce sales dataset.
"""

import numpy as np
import pandas as pd


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date columns to datetime and ensure numeric columns are numeric."""
    df = df.copy()
    for col in ["Order Date", "Ship Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    numeric_cols = ["Sales", "Quantity", "Discount", "Profit"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    categorical_cols = ["Ship Mode", "Segment", "Country", "City", "State",
                         "Region", "Category", "Sub-Category"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values column by column.

    Strategy:
    - Numerical columns central to the analysis (Sales, Quantity, Discount, Profit):
      filled with the column median, since these are typically right-skewed and
      the median is more robust to the effect of outliers than the mean.
    - Categorical columns: filled with the string 'Unknown' rather than the mode,
      to avoid silently overstating the size of the most common category.
    - Rows missing the Order Date (needed for every time-based feature) are dropped,
      since there is no reliable way to impute a transaction date.
    """
    df = df.copy()

    if "Order Date" in df.columns:
        before = len(df)
        df = df.dropna(subset=["Order Date"])
        removed = before - len(df)
        if removed:
            print(f"Dropped {removed} rows with missing Order Date.")

    numeric_cols = ["Sales", "Quantity", "Discount", "Profit"]
    for col in numeric_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Filled missing values in '{col}' with median ({median_val:.2f}).")

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].astype("object").fillna("Unknown")
            print(f"Filled missing values in '{col}' with 'Unknown'.")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and remove exact duplicate rows only."""
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    print(f"Removed {removed} duplicate rows out of {before}.")
    return df


def analyze_outliers(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """
    Analyze outliers in key numerical columns using the IQR method.
    Returns a summary DataFrame; does NOT delete any rows, since extreme
    Sales/Profit values in a real store are often legitimate bulk orders
    or high-margin transactions rather than data errors.
    """
    if columns is None:
        columns = [c for c in ["Sales", "Quantity", "Discount", "Profit"] if c in df.columns]

    summary = []
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        summary.append({
            "column": col,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "num_outliers": len(outliers),
            "pct_outliers": round(100 * len(outliers) / len(df), 2),
        })
    return pd.DataFrame(summary)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline end-to-end."""
    df = convert_data_types(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    return df

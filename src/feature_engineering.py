"""
feature_engineering.py
------------------------
Creates date, sales, customer, and product level features from the
cleaned transaction-level e-commerce dataset, and builds the monthly
aggregated dataset used for sales forecasting.
"""

import numpy as np
import pandas as pd


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar-based features from Order Date."""
    df = df.copy()
    df["Order Year"] = df["Order Date"].dt.year
    df["Order Month"] = df["Order Date"].dt.month
    df["Order Month Name"] = df["Order Date"].dt.strftime("%b")
    df["Order Quarter"] = df["Order Date"].dt.quarter
    df["Order Week"] = df["Order Date"].dt.isocalendar().week.astype(int)
    df["Order Day"] = df["Order Date"].dt.day
    df["Order Day Of Week"] = df["Order Date"].dt.dayofweek  # 0=Mon
    df["Is Weekend"] = df["Order Day Of Week"].isin([5, 6])
    return df


def add_sales_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived sales/profit features. The raw dataset already provides
    Sales (net of discount) and Profit, so we do not recompute Sales itself -
    instead we derive ratios that add analytical value.
    """
    df = df.copy()
    df["Discount Amount"] = df["Sales"] * df["Discount"]
    df["Profit Margin"] = np.where(df["Sales"] != 0, df["Profit"] / df["Sales"], np.nan)
    df["Average Selling Price"] = np.where(df["Quantity"] != 0, df["Sales"] / df["Quantity"], np.nan)
    return df


def add_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create per-customer aggregated features, merged back onto each row."""
    df = df.copy()
    customer_stats = df.groupby("Customer ID").agg(
        Customer_Total_Orders=("Order ID", "nunique"),
        Customer_Total_Spending=("Sales", "sum"),
        Customer_Avg_Order_Value=("Sales", "mean"),
    ).reset_index()
    df = df.merge(customer_stats, on="Customer ID", how="left")
    return df


def add_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create per-product aggregated features, merged back onto each row."""
    df = df.copy()
    product_stats = df.groupby("Product ID").agg(
        Product_Total_Units_Sold=("Quantity", "sum"),
        Product_Total_Revenue=("Sales", "sum"),
        Product_Avg_Selling_Price=("Sales", "mean"),
    ).reset_index()
    df = df.merge(product_stats, on="Product ID", how="left")
    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full transaction-level feature engineering pipeline."""
    df = add_date_features(df)
    df = add_sales_features(df)
    df = add_customer_features(df)
    df = add_product_features(df)
    return df


def build_monthly_forecasting_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction-level data into a monthly time series suitable
    for sales forecasting, with lag and rolling-average features.

    IMPORTANT: lag/rolling features only use information from PRIOR months,
    so no future information leaks into a given month's feature row.
    """
    df = df.copy()
    df["Order Period"] = df["Order Date"].dt.to_period("M")

    monthly = df.groupby("Order Period").agg(
        Monthly_Sales=("Sales", "sum"),
        Monthly_Orders=("Order ID", "nunique"),
        Monthly_Quantity=("Quantity", "sum"),
        Monthly_Avg_Discount=("Discount", "mean"),
    ).reset_index()

    monthly = monthly.sort_values("Order Period").reset_index(drop=True)
    monthly["Order Date"] = monthly["Order Period"].dt.to_timestamp()
    monthly["Year"] = monthly["Order Date"].dt.year
    monthly["Month"] = monthly["Order Date"].dt.month
    monthly["Quarter"] = monthly["Order Date"].dt.quarter

    monthly["Avg_Order_Value"] = monthly["Monthly_Sales"] / monthly["Monthly_Orders"]

    # Lag features (previous month's values) - use only past information
    monthly["Prev_Month_Sales"] = monthly["Monthly_Sales"].shift(1)
    monthly["Prev_2Month_Sales"] = monthly["Monthly_Sales"].shift(2)

    # Rolling average of the last 3 months (excluding current month)
    monthly["Rolling_3M_Avg_Sales"] = monthly["Monthly_Sales"].shift(1).rolling(window=3).mean()

    # Drop the first rows where lag/rolling features are not available
    monthly = monthly.dropna().reset_index(drop=True)
    monthly = monthly.drop(columns=["Order Period"])

    return monthly

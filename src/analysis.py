"""
analysis.py
------------
Business-question-driven analysis functions used across the EDA
notebook and the Streamlit dashboard.
"""

import pandas as pd


def overall_kpis(df: pd.DataFrame) -> dict:
    """Compute headline KPIs for the executive overview."""
    return {
        "Total Revenue": df["Sales"].sum(),
        "Total Orders": df["Order ID"].nunique(),
        "Total Quantity Sold": int(df["Quantity"].sum()),
        "Average Order Value": df["Sales"].sum() / df["Order ID"].nunique(),
        "Total Profit": df["Profit"].sum() if "Profit" in df.columns else None,
    }


def monthly_sales_trend(df: pd.DataFrame) -> pd.DataFrame:
    ts = df.copy()
    ts["Order Period"] = ts["Order Date"].dt.to_period("M").dt.to_timestamp()
    return ts.groupby("Order Period")["Sales"].sum().reset_index()


def yearly_sales_trend(df: pd.DataFrame) -> pd.DataFrame:
    ts = df.copy()
    ts["Order Year"] = ts["Order Date"].dt.year
    return ts.groupby("Order Year")["Sales"].sum().reset_index()


def revenue_by_category(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Category").agg(
        Revenue=("Sales", "sum"), Quantity=("Quantity", "sum"), Profit=("Profit", "sum")
    ).reset_index().sort_values("Revenue", ascending=False)


def top_products(df: pd.DataFrame, by="Sales", n=10, ascending=False) -> pd.DataFrame:
    return df.groupby("Product Name").agg(
        Revenue=("Sales", "sum"), Quantity=("Quantity", "sum")
    ).reset_index().sort_values(by="Revenue" if by == "Sales" else "Quantity",
                                 ascending=ascending).head(n)


def revenue_by_region(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Region")["Sales"].sum().reset_index().sort_values("Sales", ascending=False)


def revenue_by_state(df: pd.DataFrame, n=10) -> pd.DataFrame:
    return df.groupby("State")["Sales"].sum().reset_index().sort_values("Sales", ascending=False).head(n)


def revenue_by_segment(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Segment").agg(
        Revenue=("Sales", "sum"), Customers=("Customer ID", "nunique")
    ).reset_index().sort_values("Revenue", ascending=False)


def repeat_customer_rate(df: pd.DataFrame) -> dict:
    orders_per_customer = df.groupby("Customer ID")["Order ID"].nunique()
    repeat = (orders_per_customer > 1).sum()
    one_time = (orders_per_customer == 1).sum()
    total = orders_per_customer.shape[0]
    return {
        "total_customers": int(total),
        "repeat_customers": int(repeat),
        "one_time_customers": int(one_time),
        "repeat_customer_pct": round(100 * repeat / total, 2),
    }


def discount_vs_sales_profit(df: pd.DataFrame) -> pd.DataFrame:
    bins = [-0.01, 0, 0.1, 0.2, 0.3, 0.5, 1.0]
    labels = ["0%", "0-10%", "10-20%", "20-30%", "30-50%", "50%+"]
    tmp = df.copy()
    tmp["Discount Band"] = pd.cut(tmp["Discount"], bins=bins, labels=labels)
    return tmp.groupby("Discount Band", observed=True).agg(
        Avg_Sales=("Sales", "mean"), Avg_Profit=("Profit", "mean"), Orders=("Order ID", "nunique")
    ).reset_index()


def business_insights_summary(df: pd.DataFrame) -> dict:
    """Answer the core business questions dynamically from the data."""
    cat_rev = revenue_by_category(df)
    region_rev = revenue_by_region(df)
    seg_rev = revenue_by_segment(df)
    monthly = monthly_sales_trend(df)
    top_rev_products = top_products(df, by="Sales", n=1)
    repeat = repeat_customer_rate(df)

    best_month = monthly.loc[monthly["Sales"].idxmax()]
    kpis = overall_kpis(df)

    return {
        "top_category_by_revenue": cat_rev.iloc[0]["Category"],
        "top_category_by_profit": cat_rev.sort_values("Profit", ascending=False).iloc[0]["Category"],
        "top_product_by_revenue": top_rev_products.iloc[0]["Product Name"],
        "top_region": region_rev.iloc[0]["Region"],
        "top_segment": seg_rev.iloc[0]["Segment"],
        "best_month": str(best_month["Order Period"].strftime("%B %Y")),
        "best_month_sales": float(best_month["Sales"]),
        "average_order_value": kpis["Average Order Value"],
        "repeat_customer_pct": repeat["repeat_customer_pct"],
    }

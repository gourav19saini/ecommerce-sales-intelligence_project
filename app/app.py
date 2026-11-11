"""
app.py
-------
Streamlit dashboard for the E-Commerce Sales Intelligence & Forecasting
System. Run with:  streamlit run app/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import data_loader
import analysis
import feature_engineering
import train_model
import predict as predict_module

st.set_page_config(page_title="E-Commerce Sales Intelligence", layout="wide")

sns.set_style("whitegrid")


@st.cache_data
def load_data():
    return data_loader.load_processed_data()


@st.cache_data
def load_monthly(df):
    return feature_engineering.build_monthly_forecasting_dataset(df)


@st.cache_resource
def load_trained_models(monthly):
    return train_model.train_and_compare(monthly)


df = load_data()
monthly = load_monthly(df)

st.sidebar.title("📊 E-Commerce Sales Intelligence")
page = st.sidebar.radio("Navigate", [
    "Executive Overview", "Exploratory Analysis", "ML Model Performance", "Sales Prediction"
])

# ---------------------------------------------------------------------------
# PAGE 1 — Executive Overview
# ---------------------------------------------------------------------------
if page == "Executive Overview":
    st.title("Executive Overview")
    kpis = analysis.overall_kpis(df)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Revenue", f"${kpis['Total Revenue']:,.0f}")
    c2.metric("Total Orders", f"{kpis['Total Orders']:,}")
    c3.metric("Total Quantity", f"{kpis['Total Quantity Sold']:,}")
    c4.metric("Avg Order Value", f"${kpis['Average Order Value']:,.2f}")
    c5.metric("Total Profit", f"${kpis['Total Profit']:,.0f}")

    st.subheader("Monthly Sales Trend")
    monthly_trend = analysis.monthly_sales_trend(df)
    st.line_chart(monthly_trend.set_index("Order Period")["Sales"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Revenue by Category")
        st.bar_chart(analysis.revenue_by_category(df).set_index("Category")["Revenue"])
    with col2:
        st.subheader("Revenue by Region")
        st.bar_chart(analysis.revenue_by_region(df).set_index("Region")["Sales"])

# ---------------------------------------------------------------------------
# PAGE 2 — Exploratory Analysis
# ---------------------------------------------------------------------------
elif page == "Exploratory Analysis":
    st.title("Exploratory Analysis")

    years = sorted(df["Order Date"].dt.year.unique())
    categories = sorted(df["Category"].unique())
    regions = sorted(df["Region"].unique())
    segments = sorted(df["Segment"].unique())

    f1, f2, f3, f4 = st.columns(4)
    sel_years = f1.multiselect("Year", years, default=years)
    sel_categories = f2.multiselect("Category", categories, default=categories)
    sel_regions = f3.multiselect("Region", regions, default=regions)
    sel_segments = f4.multiselect("Segment", segments, default=segments)

    filtered = df[
        df["Order Date"].dt.year.isin(sel_years)
        & df["Category"].isin(sel_categories)
        & df["Region"].isin(sel_regions)
        & df["Segment"].isin(sel_segments)
    ]

    st.caption(f"{len(filtered):,} transactions match the current filters.")

    st.subheader("Sales Trend")
    st.line_chart(analysis.monthly_sales_trend(filtered).set_index("Order Period")["Sales"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Category Performance")
        st.dataframe(analysis.revenue_by_category(filtered))
    with col2:
        st.subheader("Top 10 Products")
        st.dataframe(analysis.top_products(filtered, n=10))

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Geographic Performance (Top States)")
        st.dataframe(analysis.revenue_by_state(filtered))
    with col4:
        st.subheader("Discount vs Sales/Profit")
        st.dataframe(analysis.discount_vs_sales_profit(filtered))

# ---------------------------------------------------------------------------
# PAGE 3 — ML Model Performance
# ---------------------------------------------------------------------------
elif page == "ML Model Performance":
    st.title("ML Model Performance")

    comparison_df, fitted_models, predictions, train, test = load_trained_models(monthly)
    best_name = train_model.select_best_model(comparison_df)

    st.subheader("Models Trained")
    st.dataframe(comparison_df.reset_index(drop=True))
    st.success(f"Best model (lowest RMSE): **{best_name}**")

    st.subheader("Actual vs Predicted Sales (Test Period)")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(test["Order Date"], test["Monthly_Sales"], marker="o", label="Actual")
    ax.plot(test["Order Date"], predictions[best_name], marker="o", label="Predicted")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Feature Importance (Random Forest)")
    rf_importance = train_model.get_feature_importance(
        fitted_models["Random Forest"], train_model.FEATURE_COLUMNS
    )
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    rf_importance_sorted = rf_importance.sort_values("Importance")
    ax2.barh(rf_importance_sorted["Feature"], rf_importance_sorted["Importance"])
    st.pyplot(fig2)

# ---------------------------------------------------------------------------
# PAGE 4 — Sales Prediction
# ---------------------------------------------------------------------------
elif page == "Sales Prediction":
    st.title("Sales Prediction")
    st.write("Enter next month's expected activity to forecast total sales revenue.")

    col1, col2, col3 = st.columns(3)
    year = col1.number_input("Year", min_value=2015, max_value=2035, value=2019)
    month = col2.number_input("Month", min_value=1, max_value=12, value=1)
    quarter = col3.number_input("Quarter", min_value=1, max_value=4, value=(month - 1) // 3 + 1)

    col4, col5, col6 = st.columns(3)
    orders = col4.number_input("Expected Monthly Orders", min_value=1, value=150)
    quantity = col5.number_input("Expected Monthly Quantity", min_value=1, value=500)
    discount = col6.number_input("Expected Avg Discount", min_value=0.0, max_value=1.0, value=0.15)

    col7, col8, col9 = st.columns(3)
    aov = col7.number_input("Expected Avg Order Value ($)", min_value=1.0, value=230.0)
    prev1 = col8.number_input("Previous Month Sales ($)", min_value=0.0, value=45000.0)
    prev2 = col9.number_input("Sales 2 Months Ago ($)", min_value=0.0, value=42000.0)

    rolling3 = st.number_input("3-Month Rolling Avg Sales ($)", min_value=0.0, value=43500.0)

    if st.button("Predict Sales"):
        input_data = {
            "Year": year, "Month": month, "Quarter": quarter,
            "Monthly_Orders": orders, "Monthly_Quantity": quantity,
            "Monthly_Avg_Discount": discount, "Avg_Order_Value": aov,
            "Prev_Month_Sales": prev1, "Prev_2Month_Sales": prev2,
            "Rolling_3M_Avg_Sales": rolling3,
        }
        prediction = predict_module.predict_sales(input_data)
        st.metric("Predicted Sales", f"${prediction:,.2f}")
        st.caption("Prediction generated using the trained sales forecasting model.")

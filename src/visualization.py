"""
visualization.py
------------------
Reusable plotting functions (Matplotlib/Seaborn) that save figures to
reports/figures/ with meaningful filenames.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")
FIG_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, filename):
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150)
    plt.close(fig)


def plot_monthly_sales_trend(monthly_df: pd.DataFrame, filename="01_monthly_sales_trend.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly_df["Order Period"], monthly_df["Sales"], marker="o", color="#2563eb")
    ax.set_title("Monthly Sales Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    plt.xticks(rotation=45)
    _save(fig, filename)


def plot_category_revenue(cat_df: pd.DataFrame, filename="02_category_revenue.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=cat_df, x="Category", y="Revenue", ax=ax, hue="Category",
                palette="viridis", legend=False)
    ax.set_title("Revenue by Category")
    _save(fig, filename)


def plot_region_revenue(region_df: pd.DataFrame, filename="03_region_revenue.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=region_df, x="Region", y="Sales", ax=ax, hue="Region",
                palette="crest", legend=False)
    ax.set_title("Revenue by Region")
    _save(fig, filename)


def plot_top_products(top_df: pd.DataFrame, filename="04_top10_products.png"):
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=top_df, y="Product Name", x="Revenue", ax=ax, hue="Product Name",
                palette="mako", legend=False)
    ax.set_title("Top 10 Products by Revenue")
    _save(fig, filename)


def plot_segment_revenue(seg_df: pd.DataFrame, filename="05_segment_revenue.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=seg_df, x="Segment", y="Revenue", ax=ax, hue="Segment",
                palette="flare", legend=False)
    ax.set_title("Revenue by Customer Segment")
    _save(fig, filename)


def plot_discount_vs_sales(discount_df: pd.DataFrame, filename="06_discount_vs_sales.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(discount_df))
    ax.bar(x, discount_df["Avg_Sales"], color="#0ea5e9", label="Avg Sales")
    ax.set_xticks(list(x))
    ax.set_xticklabels(discount_df["Discount Band"])
    ax.set_ylabel("Average Sales")
    ax2 = ax.twinx()
    ax2.plot(x, discount_df["Avg_Profit"], color="#dc2626", marker="o", label="Avg Profit")
    ax2.set_ylabel("Average Profit")
    ax.set_title("Discount Band vs Average Sales & Profit")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    _save(fig, filename)


def plot_correlation_heatmap(df: pd.DataFrame, columns, filename="07_correlation_heatmap.png"):
    fig, ax = plt.subplots(figsize=(7, 6))
    corr = df[columns].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    _save(fig, filename)


def plot_actual_vs_predicted(dates, actual, predicted, model_name, filename="08_actual_vs_predicted.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, actual, marker="o", label="Actual Sales", color="#111827")
    ax.plot(dates, predicted, marker="o", label="Predicted Sales", color="#f97316")
    ax.set_title(f"Actual vs Predicted Monthly Sales ({model_name})")
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    ax.legend()
    plt.xticks(rotation=45)
    _save(fig, filename)


def plot_feature_importance(importance_df: pd.DataFrame, filename="09_feature_importance.png"):
    fig, ax = plt.subplots(figsize=(8, 6))
    importance_df = importance_df.sort_values("Importance")
    ax.barh(importance_df["Feature"], importance_df["Importance"], color="#7c3aed")
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    _save(fig, filename)

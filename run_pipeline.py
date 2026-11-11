"""
run_pipeline.py
-----------------
End-to-end pipeline: load raw data -> clean -> engineer features ->
EDA figures & business insights -> train/compare forecasting models ->
save model, comparison table, feature importance, and a JSON summary.

Run from the project root:  python run_pipeline.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

import data_loader
import preprocessing
import feature_engineering
import analysis
import visualization
import train_model

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("STEP 1: Loading raw data")
    print("=" * 70)
    raw = data_loader.load_raw_data()
    print(f"Raw shape: {raw.shape}")

    print("\n" + "=" * 70)
    print("STEP 2: Cleaning data")
    print("=" * 70)
    cleaned = preprocessing.clean_dataset(raw)
    print(f"Cleaned shape: {cleaned.shape}")

    print("\n" + "=" * 70)
    print("STEP 3: Feature engineering (transaction level)")
    print("=" * 70)
    featured = feature_engineering.engineer_all_features(cleaned)
    featured.to_csv(PROCESSED_DIR / "cleaned_sales.csv", index=False)
    print(f"Saved processed dataset: {featured.shape}")

    print("\n" + "=" * 70)
    print("STEP 4: Exploratory analysis + business insights")
    print("=" * 70)
    kpis = analysis.overall_kpis(featured)
    repeat = analysis.repeat_customer_rate(featured)
    insights = analysis.business_insights_summary(featured)

    eda_summary = {"kpis": kpis, "repeat_customers": repeat, "insights": insights}
    with open(REPORTS_DIR / "eda_summary.json", "w") as f:
        json.dump(eda_summary, f, indent=2, default=str)
    print(json.dumps(eda_summary, indent=2, default=str))

    print("\n" + "=" * 70)
    print("STEP 5: Generating visualizations")
    print("=" * 70)
    monthly_trend = analysis.monthly_sales_trend(featured)
    visualization.plot_monthly_sales_trend(monthly_trend)

    cat_rev = analysis.revenue_by_category(featured)
    visualization.plot_category_revenue(cat_rev)

    region_rev = analysis.revenue_by_region(featured)
    visualization.plot_region_revenue(region_rev)

    top10 = analysis.top_products(featured, n=10)
    visualization.plot_top_products(top10)

    seg_rev = analysis.revenue_by_segment(featured)
    visualization.plot_segment_revenue(seg_rev)

    disc = analysis.discount_vs_sales_profit(featured)
    visualization.plot_discount_vs_sales(disc)

    corr_cols = [c for c in ["Sales", "Quantity", "Discount", "Profit", "Profit Margin"]
                 if c in featured.columns]
    visualization.plot_correlation_heatmap(featured, corr_cols)
    print("Figures 01-07 saved to reports/figures/")

    print("\n" + "=" * 70)
    print("STEP 6: Building monthly forecasting dataset")
    print("=" * 70)
    monthly_df = feature_engineering.build_monthly_forecasting_dataset(featured)
    monthly_df.to_csv(PROCESSED_DIR / "monthly_sales_features.csv", index=False)
    print(f"Monthly dataset shape: {monthly_df.shape}")

    print("\n" + "=" * 70)
    print("STEP 7: Training & comparing regression models")
    print("=" * 70)
    comparison_df, fitted_models, predictions, train, test = train_model.train_and_compare(monthly_df)
    comparison_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    print(comparison_df.to_string(index=False))

    best_name = train_model.select_best_model(comparison_df)
    best_model = fitted_models[best_name]
    print(f"\nBest model (lowest RMSE): {best_name}")

    print("\n" + "=" * 70)
    print("STEP 8: Actual vs predicted + feature importance figures")
    print("=" * 70)
    visualization.plot_actual_vs_predicted(
        test["Order Date"], test["Monthly_Sales"], predictions[best_name], best_name
    )

    rf_importance = train_model.get_feature_importance(
        fitted_models["Random Forest"], train_model.FEATURE_COLUMNS
    )
    rf_importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    visualization.plot_feature_importance(rf_importance)
    print("Figures 08-09 saved to reports/figures/")

    print("\n" + "=" * 70)
    print("STEP 9: Saving final model + summary")
    print("=" * 70)
    train_model.save_model(best_model, train_model.FEATURE_COLUMNS)

    model_summary = {
        "best_model": best_name,
        "comparison": comparison_df.to_dict(orient="records"),
        "n_train_months": len(train),
        "n_test_months": len(test),
        "monthly_dataset_shape": list(monthly_df.shape),
    }
    with open(REPORTS_DIR / "model_summary.json", "w") as f:
        json.dump(model_summary, f, indent=2, default=str)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()

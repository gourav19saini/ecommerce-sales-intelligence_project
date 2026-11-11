# E-Commerce Sales Intelligence & Forecasting System

*An End-to-End Data Science Project for Sales Analysis, Business Insights and Predictive Forecasting*

45-day Industrial Training project — Data Science with AI/ML.

## Overview

This project analyzes historical e-commerce sales transactions (the Sample
Superstore dataset, 9,994 rows, 2014–2017) to answer descriptive/diagnostic
business questions and to build a machine-learning system that forecasts
future monthly sales revenue.

## Problem Statement

Retail businesses generate large volumes of transactional data but often
lack a structured way to turn it into (a) clear insight into what is driving
current performance and (b) a working way to anticipate near-term sales.
This project builds both halves for a Superstore-style e-commerce business.

## Objectives

1. Understand how sales change over time and identify seasonal patterns.
2. Identify the categories, products, regions, and customer segments that
   drive the most revenue and profit.
3. Quantify the relationship between discounting and profitability.
4. Engineer a clean, leak-free feature set for time-series forecasting.
5. Train and fairly compare several regression models for monthly sales
   forecasting.
6. Package the result as an interactive dashboard and a reusable prediction
   function.

## Dataset

Sample Superstore Sales Dataset — see [`data/README.md`](data/README.md)
for full details, fields, and preprocessing notes.

## Technologies Used

Python 3.12, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, Streamlit,
Jupyter Notebook, joblib.

## Project Architecture

```
ecommerce-sales-intelligence/
├── data/
│   ├── raw/ecommerce_sales.csv          # original Superstore data
│   ├── processed/cleaned_sales.csv      # cleaned + feature-engineered
│   ├── processed/monthly_sales_features.csv
│   └── README.md
├── notebooks/                            # 01 -> 06, run in order
├── src/                                  # reusable pipeline modules
├── models/sales_forecasting_model.pkl    # trained final model
├── reports/                              # figures, metrics, written reports
├── app/app.py                            # Streamlit dashboard
├── run_pipeline.py                       # runs the full pipeline end-to-end
├── requirements.txt
└── .gitignore
```

## Data Cleaning

- Type conversion (dates -> datetime, categoricals -> category dtype).
- No missing values or duplicate rows were present in this dataset (verified
  dynamically, not assumed) — see `notebooks/02_data_cleaning.ipynb`.
- Outliers in Sales/Quantity/Discount/Profit inspected via IQR but kept,
  since they represent real bulk orders and discounted/loss-making sales.

## Exploratory Data Analysis

Covers KPIs, monthly/yearly trends, category/product/region/segment
performance, discount-vs-profit association, and a correlation heatmap.
See `notebooks/03_eda.ipynb` and `reports/figures/`.

## Feature Engineering

Date features, sales-ratio features (discount amount, profit margin,
average selling price), customer- and product-level aggregates, and a
monthly forecasting dataset with **lag** and **rolling-average** features
that only ever look backward in time (no leakage). See
`notebooks/04_feature_engineering.ipynb`.

## Machine Learning

**Task:** predict `Monthly_Sales` from calendar + lag/rolling features.

**Models compared:** Linear Regression, Decision Tree, Random Forest,
Gradient Boosting.

**Split:** chronological (first 80% of months train, last 20% test) — never
shuffled, to avoid the model learning from the future.

## Model Evaluation

See `reports/model_comparison.csv` / `reports/model_summary.json` for the
current MAE / RMSE / R² of every model, and
`notebooks/06_model_comparison.ipynb` for the full discussion of why the
selected model performed best.

## Results

The comparison table, the actual-vs-predicted chart
(`reports/figures/08_actual_vs_predicted.png`), and feature-importance chart
(`reports/figures/09_feature_importance.png`) are generated dynamically by
`run_pipeline.py` — figures are regenerated from the data, not hand-drawn.

## Dashboard

A 4-page Streamlit dashboard (`app/app.py`):

1. **Executive Overview** — KPIs + top-line charts
2. **Exploratory Analysis** — filterable by year/category/region/segment
3. **ML Model Performance** — comparison table, actual vs predicted, feature importance
4. **Sales Prediction** — interactive form that calls the trained model

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (cleans data, builds features, trains models,
#    saves the model + figures + reports)
python run_pipeline.py

# 3. Explore the analysis notebooks (already executed, but re-runnable)
jupyter notebook notebooks/

# 4. Launch the interactive dashboard
streamlit run app/app.py

# 5. Get a single prediction from the command line
python src/predict.py
```

## Key Learnings

- How to build a leak-free time-series feature set (lag/rolling features
  computed only from past periods).
- Why classical regression metrics (MAE, RMSE, R²) — not classification
  accuracy — are the right way to judge a forecasting model.
- Why a simpler model (Linear Regression) can outperform more flexible
  tree-based models when the number of training observations is small
  (45 months total here).
- How to separate descriptive analytics (what happened) from predictive
  analytics (what's likely to happen next) in one coherent project.

## Limitations

- Only 45 monthly observations are available after feature engineering,
  which limits how much the tree-based models can learn and makes the test
  set small (9 months).
- The dataset covers a single fictional retailer's US operations
  (2014–2017); patterns may not transfer to other markets or time periods.
- Forecasts assume the shape of past seasonality repeats; a genuine regime
  change (new product lines, macro shocks) would not be reflected in the
  lag/rolling features.

## Future Scope

- Expand data granularity (e.g. weekly instead of monthly) with a larger
  history if more data becomes available.
- Add external variables (marketing spend, holidays calendar, competitor
  pricing) as features.
- Explore proper time-series models (SARIMA, Prophet) alongside the
  regression baselines for comparison.
- Deploy the Streamlit dashboard behind authentication for internal
  business use.

## Project Structure Reference

See the architecture diagram above; also see
`reports/project_report.md`, `reports/presentation_content.md`, and
`reports/viva_questions.md` for submission-ready supporting material,
and `reports/training_alignment.md` for how each part of the project maps
to the Industrial Training curriculum.

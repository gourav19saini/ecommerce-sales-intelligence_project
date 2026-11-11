# Presentation Content — 12 Slides

## Slide 1 — Title
- E-Commerce Sales Intelligence & Forecasting System
- Subtitle: An End-to-End Data Science Project for Sales Analysis, Business Insights and Predictive Forecasting
- 45-day Industrial Training project, Data Science with AI/ML
- Visual: none needed
- Say: introduce yourself and the one-line goal — analyze past sales and forecast future ones.

## Slide 2 — Problem Statement & Objectives
- Retailers generate transaction data but need it turned into insight and foresight
- Objective 1: Explain what drove past sales (category, product, region, segment, time)
- Objective 2: Quantify discount vs. profitability
- Objective 3: Build a leak-free forecasting model for monthly sales
- Visual: none / simple bullet slide
- Say: frame this as descriptive analytics + predictive analytics working together.

## Slide 3 — Dataset & Data Understanding
- Sample Superstore dataset: 9,994 transactions, 21 columns, 2014–2017 (US)
- Core fields: Order Date, Sales, Quantity, Discount, Profit, Category, Region, Segment
- No missing values or duplicate rows found (verified, not assumed)
- Visual: screenshot of `df.info()` / `df.describe()` output from notebook 01
- Say: emphasize that the data-quality check was calculated dynamically, not hardcoded.

## Slide 4 — Data Cleaning & Preprocessing
- Converted dates to datetime, confirmed numeric types, categorical dtype for text fields
- IQR-based outlier inspection — extreme values kept as legitimate bulk/discounted orders
- Visual: boxplots of Sales/Quantity/Discount/Profit from notebook 02
- Say: explain the reasoning for keeping outliers rather than deleting them blindly.

## Slide 5 — Exploratory Data Analysis
- KPIs: $2.30M total revenue, 5,009 orders, $458.61 average order value
- Monthly sales trend chart with a clear Q4 seasonal peak
- Visual: `reports/figures/01_monthly_sales_trend.png`
- Say: walk through the seasonal spike and what it means for inventory planning.

## Slide 6 — Key Business Insights
- Top category (revenue & profit): Technology
- Top region: West | Top segment: Consumer
- Best month: November 2017 ($118,447.83)
- 98.49% of customers are repeat customers
- Higher discount bands associate with lower average profit
- Visual: `reports/figures/02_category_revenue.png` and `06_discount_vs_sales.png`
- Say: connect each insight to a concrete business action (e.g., inventory, marketing timing).

## Slide 7 — Feature Engineering
- Date features (year/month/quarter/weekend), sales-ratio features (profit margin, discount amount)
- Customer- and product-level aggregates
- Monthly forecasting dataset with lag (Prev_Month_Sales) and rolling-average features
- Visual: table snapshot from notebook 04
- Say: stress that lag/rolling features only look backward — no data leakage.

## Slide 8 — Machine Learning Methodology
- Target: Monthly Sales
- Chronological 80/20 train/test split — never shuffled
- Models: Linear Regression, Decision Tree, Random Forest, Gradient Boosting
- Visual: none / methodology diagram if desired
- Say: explain why chronological splitting matters for forecasting problems specifically.

## Slide 9 — Model Comparison & Evaluation
- Metrics used: MAE, RMSE, R² (regression metrics, not classification accuracy)
- Linear Regression: MAE 4,275.47 | RMSE 6,248.94 | R² 0.9373 (best)
- Tree-based models underperformed with only 36 training months available
- Visual: `reports/model_comparison.csv` as a table
- Say: explain overfitting in tree models with small training sets.

## Slide 10 — Actual vs Predicted Sales
- Selected model tracks the test-period sales curve closely, including the November peak
- Visual: `reports/figures/08_actual_vs_predicted.png`
- Say: point out where the model tracks well and where it deviates, and why.

## Slide 11 — Dashboard / Prediction Demo
- 4-page Streamlit dashboard: Overview, Exploration, Model Performance, Prediction
- Live demo: enter next month's expected orders/quantity/discount → get a sales forecast
- Visual: dashboard screenshots (Executive Overview and Sales Prediction pages)
- Say: run a live prediction if possible, or walk through screenshots.

## Slide 12 — Conclusion, Learning Outcomes & Future Scope
- Delivered a full pipeline: cleaning → EDA → leak-free feature engineering → model comparison → dashboard
- Learned: chronological splitting, regression metrics, overfitting with limited data
- Future scope: weekly granularity, external features (holidays, marketing spend), time-series models (SARIMA/Prophet)
- Visual: none needed
- Say: close by tying the results back to the original business objectives from Slide 2.

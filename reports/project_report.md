# E-Commerce Sales Intelligence & Forecasting System

## 1. Introduction

Retailers generate large volumes of transaction-level data, but that data
only creates value once it is cleaned, analyzed, and turned into a working
forecast. This project builds a complete, end-to-end data science pipeline
around a real Superstore-style e-commerce dataset, covering data cleaning,
exploratory analysis, feature engineering, and supervised machine learning
for sales forecasting.

## 2. Problem Statement

Given historical order-level e-commerce transactions, (1) explain what has
driven past sales performance across time, product, geography, and
customer segment, and (2) build a model that can forecast next-period
sales revenue from recent activity.

## 3. Objectives

- Quantify overall sales performance and its trend over time.
- Identify the categories, products, regions, and segments driving revenue
  and profit.
- Investigate the relationship between discounting and profitability.
- Engineer a leak-free feature set suitable for time-series forecasting.
- Train, evaluate, and fairly compare multiple regression models.
- Deliver the result as a reusable prediction function and an interactive
  dashboard.

## 4. Dataset Description

The **Sample Superstore** dataset: 9,994 order-line transactions from a US
retailer, 2014–2017, with 21 raw columns spanning order/shipping details,
customer information, product hierarchy, and financials (`Sales`,
`Quantity`, `Discount`, `Profit`). Full field-level detail is in
`data/README.md`.

## 5. Data Preprocessing

- Converted `Order Date`/`Ship Date` to datetime and confirmed numeric
  types for `Sales`, `Quantity`, `Discount`, `Profit`.
- Verified (dynamically, not assumed) that the dataset contains **no
  missing values and no duplicate rows**.
- Analyzed outliers in the four core numerical columns using the IQR
  method. Extreme values were judged to be legitimate bulk purchases and
  discounted/loss-making sales rather than data errors, and were therefore
  retained rather than removed.

## 6. Exploratory Data Analysis

Headline KPIs calculated from the cleaned data:

| KPI | Value |
|---|---|
| Total Revenue | $2,297,200.86 |
| Total Orders | 5,009 |
| Total Quantity Sold | 37,873 |
| Average Order Value | $458.61 |
| Total Profit | $286,397.02 |

Further analysis covered:
- **Time trends:** year-over-year revenue growth with a recurring Q4
  (Oct–Dec) seasonal peak; the single best month in the data was
  **November 2017** ($118,447.83 in sales).
- **Category performance:** **Technology** was the top category by both
  revenue and profit.
- **Product performance:** the single highest-revenue product was the
  **Canon imageCLASS 2200 Advanced Copier**.
- **Geography:** the **West** region generated the most revenue.
- **Customer segments:** the **Consumer** segment generated the most
  revenue; of 793 total customers, 781 (98.49%) placed more than one
  order.
- **Discount vs. profitability:** higher discount bands are associated
  with lower average profit even where average order value holds up,
  indicating a margin trade-off rather than a pure sales lift. This is
  reported as an association, not a causal claim.

## 7. Feature Engineering

- **Date features:** year, month, month name, quarter, week, day, day of
  week, weekend flag.
- **Sales-ratio features:** discount amount, profit margin, average
  selling price (Sales/Profit themselves were kept as provided by the
  source data rather than recomputed).
- **Customer features:** total orders, total spending, average order value
  per customer.
- **Product features:** total units sold, total revenue, average selling
  price per product.
- **Monthly forecasting dataset:** transactions aggregated to a monthly
  time series with lag features (`Prev_Month_Sales`, `Prev_2Month_Sales`)
  and a 3-month rolling average, each computed with `.shift()` so that no
  month's feature row uses information from a later month.

## 8. Machine Learning Methodology

- **Target:** `Monthly_Sales`.
- **Split:** chronological — the first 80% of months (36 months) for
  training, the last 20% (9 months) for testing. Data was never shuffled,
  since shuffling would let future months leak into training.
- **Models trained:** Linear Regression, Decision Tree Regressor, Random
  Forest Regressor, Gradient Boosting Regressor.

## 9. Model Evaluation

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 4,275.47 | 6,248.94 | 0.9373 |
| Decision Tree | 11,582.98 | 13,947.47 | 0.6877 |
| Gradient Boosting | 11,510.09 | 14,785.48 | 0.6490 |
| Random Forest | 11,229.02 | 15,088.97 | 0.6345 |

## 10. Results

**Linear Regression** was selected as the final model (lowest RMSE,
highest R² on the held-out test months). With only 36 training months
available, the more flexible tree-based models had too little data to
learn stable, generalizable split points and overfit the training period,
while the lag/rolling features already capture most of the month-to-month
sales momentum in an approximately linear way. The actual-vs-predicted
chart (`reports/figures/08_actual_vs_predicted.png`) shows the selected
model tracking the test-period sales curve closely, including the
November peak.

Feature importance from the Random Forest (used here for
interpretability, since Linear Regression does not expose
`feature_importances_`) ranked `Monthly_Quantity` and `Monthly_Orders` as
the two most influential inputs, together accounting for roughly 92% of
importance — sensible, since months with more orders and higher volume
naturally produce more revenue.

## 11. Business Insights

See Section 6 above and `notebooks/03_eda.ipynb` for the full list of
answered business questions.

## 12. Limitations

- Only 45 monthly observations exist after feature engineering, limiting
  statistical power and making the 9-month test set small.
- The dataset represents a single fictional US retailer over 2014–2017;
  conclusions may not generalize to other markets, categories, or years.
- The forecasting features assume recent seasonality/momentum repeats;
  they cannot anticipate a genuine structural change (e.g., a new
  product line or a macroeconomic shock).

## 13. Future Scope

- Use a finer time granularity (weekly) if a longer history becomes
  available, to give the models more training observations.
- Incorporate external drivers such as marketing spend, promotions
  calendar, and holiday indicators as explicit features.
- Compare the regression baselines against dedicated time-series methods
  (e.g., SARIMA, Prophet).
- Extend the Streamlit dashboard with authentication and scheduled
  re-training for internal business use.

## 14. Conclusion

This project demonstrates a complete, honest, data-driven pipeline: from a
public retail transactions dataset, through cleaning and business-question
driven EDA, to a leak-free forecasting feature set and a fairly-compared
set of regression models. Every number in this report is generated
directly by `run_pipeline.py` from the raw CSV — nothing here is
hand-typed or assumed.

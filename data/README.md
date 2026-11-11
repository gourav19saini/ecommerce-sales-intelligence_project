# Dataset — Sample Superstore

**Dataset name:** Sample Superstore Sales Dataset ("Sample - Superstore")

**Source:** A well-known, publicly circulated retail transactions dataset
(originally distributed as a Tableau sample dataset and widely mirrored on
Kaggle and GitHub for data-analysis practice). The copy used in this project
was placed at `data/raw/ecommerce_sales.csv`.

**Number of rows:** 9,994 order-line transactions

**Number of columns:** 21 (raw) — see below

**Time period covered:** 2014 – 2017, United States orders only

## Important fields

| Column | Description |
|---|---|
| Row ID | Row identifier |
| Order ID | Order identifier (one order can span multiple rows/products) |
| Order Date / Ship Date | Order and shipping dates |
| Ship Mode | Shipping method |
| Customer ID / Customer Name | Customer identifiers |
| Segment | Consumer, Corporate, or Home Office |
| Country / City / State / Region / Postal Code | Geography |
| Product ID / Category / Sub-Category / Product Name | Product hierarchy |
| Sales | Net revenue for the line item (USD) |
| Quantity | Units sold |
| Discount | Discount fraction applied (0.0 – 0.8) |
| Profit | Profit for the line item (USD); can be negative |

## Target variable

**Monthly Sales** (`Monthly_Sales`), after aggregating the 9,994
transaction-level rows into a monthly time series (see
`notebooks/04_feature_engineering.ipynb` and
`src/feature_engineering.build_monthly_forecasting_dataset`). This produced
45 monthly observations after lag-feature creation dropped the first months,
split chronologically into 36 training months and 9 test months.

## Preprocessing performed

- `Order Date` / `Ship Date` converted to `datetime`; `Sales`, `Quantity`,
  `Discount`, `Profit` confirmed numeric.
- Low-cardinality text columns converted to `category` dtype.
- No missing values or duplicate rows were found in this dataset, so no
  imputation or row removal was required — see
  `notebooks/02_data_cleaning.ipynb` for the full, dynamically-calculated
  data-quality check.
- Outliers in `Sales`, `Quantity`, `Discount`, and `Profit` were inspected
  with the IQR method but intentionally **not removed**, since large or
  negative values represent legitimate bulk orders and loss-making
  discounted sales rather than data errors.
- Date, sales-ratio, customer-level, and product-level features were
  engineered on top of the cleaned data (see
  `notebooks/04_feature_engineering.ipynb`).

## Currency

All monetary fields (`Sales`, `Profit`, model predictions) are in **US
Dollars (USD)**, matching the source dataset — no currency conversion was
applied.

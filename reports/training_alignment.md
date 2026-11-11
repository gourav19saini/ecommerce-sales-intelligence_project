# Mapping of Training Concepts to Project

| Training Topic | Project Implementation |
|---|---|
| Python Fundamentals | Modular functions across `src/` (data loading, cleaning, feature engineering, training, prediction) instead of one long script |
| NumPy | Numerical operations underlying ratio features (profit margin, discount amount) and model metric calculations |
| Pandas | All data loading, cleaning, grouping, merging, and aggregation — `notebooks/01`–`04`, `src/preprocessing.py`, `src/feature_engineering.py`, `src/analysis.py` |
| Matplotlib | Time-series, bar, and comparison charts in `src/visualization.py` |
| Seaborn | Boxplots for outlier analysis, styled bar charts, and the correlation heatmap |
| EDA | `notebooks/03_eda.ipynb` — KPI, time, category, product, geography, customer, and discount analysis |
| Data Preprocessing | `notebooks/02_data_cleaning.ipynb` — type conversion, missing-value/duplicate checks, IQR outlier analysis |
| Feature Engineering | `notebooks/04_feature_engineering.ipynb` — date, sales-ratio, customer, product, and lag/rolling forecasting features |
| Supervised Learning | Sales forecasting framed as a supervised regression problem on the monthly dataset |
| Regression | Linear Regression, Decision Tree, Random Forest, and Gradient Boosting regressors compared in `notebooks/05`–`06` |
| Model Evaluation | MAE, RMSE, and R² computed for every model in `src/train_model.py` |
| Overfitting/Underfitting | Discussed directly in `notebooks/06_model_comparison.ipynb`, explaining why tree-based models overfit the small training set while Linear Regression generalized better |
| Real-world Implementation | 4-page interactive Streamlit dashboard (`app/app.py`) covering overview, exploration, model performance, and live prediction |

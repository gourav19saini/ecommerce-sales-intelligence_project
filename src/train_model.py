"""
train_model.py
----------------
Trains and compares regression models for monthly sales forecasting
using a chronological (non-shuffled) train/test split to avoid
data leakage from future periods into training.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "Year", "Month", "Quarter",
    "Monthly_Orders", "Monthly_Quantity", "Monthly_Avg_Discount",
    "Avg_Order_Value", "Prev_Month_Sales", "Prev_2Month_Sales", "Rolling_3M_Avg_Sales",
]
TARGET_COLUMN = "Monthly_Sales"


def chronological_split(monthly_df: pd.DataFrame, test_size: float = 0.2):
    """
    Split monthly data chronologically: first 80% for training, last 20% for
    testing. This mirrors real-world forecasting, where future observations
    must never be allowed to influence the training process.
    """
    monthly_df = monthly_df.sort_values("Order Date").reset_index(drop=True)
    split_idx = int(len(monthly_df) * (1 - test_size))
    train = monthly_df.iloc[:split_idx]
    test = monthly_df.iloc[split_idx:]
    return train, test


def get_models() -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=4, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                                         learning_rate=0.05, random_state=42),
    }


def evaluate_model(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def train_and_compare(monthly_df: pd.DataFrame):
    """
    Train all candidate models on the chronological train split, evaluate on
    the test split, and return the comparison table, per-model predictions,
    and the fitted model objects.
    """
    train, test = chronological_split(monthly_df)

    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]

    models = get_models()
    results = []
    fitted_models = {}
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate_model(y_test, preds)
        metrics["Model"] = name
        results.append(metrics)
        fitted_models[name] = model
        predictions[name] = preds

    comparison_df = pd.DataFrame(results)[["Model", "MAE", "RMSE", "R2"]].sort_values("RMSE")
    return comparison_df, fitted_models, predictions, train, test


def select_best_model(comparison_df: pd.DataFrame) -> str:
    """Best model = lowest RMSE on the held-out chronological test set."""
    return comparison_df.sort_values("RMSE").iloc[0]["Model"]


def save_model(model, feature_columns, filename="sales_forecasting_model.pkl"):
    payload = {"model": model, "feature_columns": feature_columns}
    joblib.dump(payload, MODEL_DIR / filename)
    print(f"Model saved to {MODEL_DIR / filename}")


def get_feature_importance(model, feature_columns) -> pd.DataFrame:
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["Feature", "Importance"])
    return pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

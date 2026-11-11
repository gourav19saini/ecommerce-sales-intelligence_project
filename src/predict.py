"""
predict.py
-----------
Loads the trained sales forecasting model and exposes a simple
predict_sales() function for making new predictions.
"""

import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "sales_forecasting_model.pkl"


def load_model(path: Path = MODEL_PATH):
    payload = joblib.load(path)
    return payload["model"], payload["feature_columns"]


def predict_sales(input_data: dict, path: Path = MODEL_PATH) -> float:
    """
    Predict monthly sales revenue given a dictionary of input features.

    Example
    -------
    input_data = {
        "Year": 2018, "Month": 11, "Quarter": 4,
        "Monthly_Orders": 120, "Monthly_Quantity": 450,
        "Monthly_Avg_Discount": 0.15, "Avg_Order_Value": 220.0,
        "Prev_Month_Sales": 45000, "Prev_2Month_Sales": 42000,
        "Rolling_3M_Avg_Sales": 43500,
    }
    prediction = predict_sales(input_data)
    """
    model, feature_columns = load_model(path)
    input_df = pd.DataFrame([input_data])[feature_columns]
    prediction = model.predict(input_df)[0]
    return float(prediction)


if __name__ == "__main__":
    sample_input = {
        "Year": 2018, "Month": 11, "Quarter": 4,
        "Monthly_Orders": 120, "Monthly_Quantity": 450,
        "Monthly_Avg_Discount": 0.15, "Avg_Order_Value": 220.0,
        "Prev_Month_Sales": 45000, "Prev_2Month_Sales": 42000,
        "Rolling_3M_Avg_Sales": 43500,
    }
    prediction = predict_sales(sample_input)
    print(f"Predicted Sales: ${prediction:,.2f}")

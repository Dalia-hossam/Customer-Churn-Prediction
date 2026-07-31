"""
Feature engineering shared between the training notebook, the Streamlit app,
and the MCP server. This MUST stay in sync with 02_model_improved.ipynb —
if you retrain the model with different engineered features, update both
places (or better: have the notebook import from this module instead).
"""

from __future__ import annotations

import pandas as pd

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# The raw columns a caller (Streamlit form / uploaded CSV / MCP tool input)
# must supply. Does NOT include AverageSpend / ServiceCount / IsNewCustomer —
# those are derived by engineer_features() below.
RAW_REQUIRED_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


def engineer_features(input_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact same feature engineering used during training.

    Parameters
    ----------
    input_df : pd.DataFrame
        Raw customer data containing at least RAW_REQUIRED_COLUMNS.

    Returns
    -------
    pd.DataFrame
        Copy of input_df with AverageSpend, ServiceCount, and IsNewCustomer added.
    """
    missing = set(RAW_REQUIRED_COLUMNS) - set(input_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = input_df.copy()

    out["tenure"] = pd.to_numeric(out["tenure"], errors="coerce")
    out["MonthlyCharges"] = pd.to_numeric(out["MonthlyCharges"], errors="coerce")
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")

    out["AverageSpend"] = out["TotalCharges"] / (out["tenure"] + 1)

    out["ServiceCount"] = (
        out[SERVICE_COLUMNS]
        .replace(
            {
                "Yes": 1,
                "No": 0,
                "No phone service": 0,
                "No internet service": 0,
            }
        )
        .astype(int)
        .sum(axis=1)
    )

    out["IsNewCustomer"] = (out["tenure"] < 12).astype(int)

    return out

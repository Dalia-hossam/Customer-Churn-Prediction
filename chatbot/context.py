from typing import Dict


def build_customer_context(
    customer: Dict,
    prediction: Dict,
) -> str:

    customer_text = "\n".join(
        f"{k}: {v}"
        for k, v in customer.items()
    )

    prediction_text = f"""
Prediction : {prediction.get("prediction")}

Probability : {prediction.get("confidence")}

Risk Level : {prediction.get("risk_level")}

Threshold : {prediction.get("threshold_used")}
"""

    return f"""
CUSTOMER INFORMATION

{customer_text}

--------------------------------

MODEL PREDICTION

{prediction_text}
"""
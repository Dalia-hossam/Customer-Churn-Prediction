# Customer Churn Intelligence System

An end-to-end Machine Learning and Generative AI platform designed to predict customer churn, calculate risk levels, and generate retention strategies through a Streamlit interface.

---

## Executive Summary
Customer churn impacts recurring revenue streams. This project provides a platform integrating Data Mining, Machine Learning (Logistic Regression with ADASYN resampling), and Generative AI to enable proactive retention planning based on automated customer risk scores.

---

## Key Features

* **Real-Time and Batch Scoring:**
  * File uploader for batch CSV processing and scoring.
  * Input forms for individual customer risk inference.
* **Generative AI Copilot (`ChurnAssistant`):** Integrated LLM functionality to output risk driver explanations, account summaries, retention recommendations, and draft outreach emails.
* **Feature Engineering:** Automated derivation of business features including `AverageSpend`, `ServiceCount`, and `IsNewCustomer`.
* **Imbalanced Data Handling:** Applies ADASYN (Adaptive Synthetic Resampling) to balance training decision boundaries.
* **Data Isolation:** Preprocessing steps and missing value imputations are restricted within cross-validation folds to prevent data leakage.

---

## Technology Stack

| Component | Technology | Function |
| :--- | :--- | :--- |
| **User Interface** | Streamlit | Web dashboard for visual analysis and input forms. |
| **ML Framework** | Scikit-Learn | Logistic Regression model training and threshold tuning. |
| **Data Processing** | Pandas, NumPy | Data cleaning and feature engineering transformations. |
| **Imbalance Mitigation** | Imbalanced-Learn | ADASYN oversampling implementation. |
| **Generative AI** | LLM API | System integration for text summary and email generation. |

---

## System Architecture

The project directory is structured as follows:

```text
churn-intelligence-system/
 ┣ .devcontainer/           # Dev container configuration for isolated environments
 ┣ chatbot/                 # Generative AI integration layer
 ┃ ┗ assistant.py           # ChurnAssistant implementation and LLM routing
 ┣ churn_core/              # Preprocessing and core ML inference logic
 ┃ ┣ preprocessor.py        # Feature engineering pipelines
 ┃ ┗ predictor.py           # ChurnPredictor class for ML inference
 ┣ data/                    # Raw and processed dataset storage
 ┣ model/                   # Serialized trained model artifacts (.pkl / .joblib)
 ┣ notebooks/               # Jupyter notebooks for EDA and model experimentation
 ┣ .env_example             # Configuration template for environment variables
 ┣ .gitignore               # Git untracked files specification
 ┣ README.md                # System documentation
 ┣ app.py                   # Streamlit UI entry point
 ┗ requirements.txt         # Project dependencies
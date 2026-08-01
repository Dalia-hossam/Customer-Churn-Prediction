"""
Customer Churn Intelligence — Streamlit app with AI Assistant Integration.

Run with: streamlit run streamlit_app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make churn_core and chatbot importable regardless of launch location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from churn_core import (  # noqa: E402
    ChurnPredictor,
    RAW_REQUIRED_COLUMNS,
    NUMERIC_RANGES as _FALLBACK_NUMERIC_RANGES,
    CATEGORICAL_OPTIONS as _FALLBACK_CATEGORICAL_OPTIONS,
)

# Chatbot imports
try:
    from chatbot.assistant import ChurnAssistant
    from chatbot.actions import Action
    CHATBOT_AVAILABLE = True
except Exception as e:
    CHATBOT_AVAILABLE = False
    CHATBOT_IMPORT_ERROR = str(e)


# ---------------------------------------------------------------------------
# Page config + theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="📶",
    layout="wide",
    initial_sidebar_state="expanded",
)

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap');

:root {
    --bg: #0B1118;            /* Dark background */
    --surface: #141D2B;       /* Card surface background */
    --text: #F0F4F8;          /* Light text */
    --text-soft: #94A3B8;     /* Subtitle text */
    --border: #1E293B;        /* Element borders */
    --accent: #0EA5A4;
    --accent-dark: #0B7F7E;
    --risk-very-high: #DC2626;
    --risk-high: #F97316;
    --risk-medium: #E0A400;
    --risk-low: #16A34A;
}

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; color: var(--text); }
h1, h2, h3, .churn-display { font-family: 'Sora', sans-serif; letter-spacing: -0.01em; color: var(--text); }

.stApp { background-color: var(--bg); }

/* --- Inputs and forms --- */
.stTextInput input, .stNumberInput input {
    background-color: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}

label[data-testid="stWidgetLabel"] p, .stForm label {
    color: var(--text) !important;
    font-weight: 500;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #090E14;
    border-right: 1px solid var(--border);
    padding-top: 1rem;
}
section[data-testid="stSidebar"] * { color: #E6E9ED !important; }

section[data-testid="stSidebar"] .stRadio > div {
    gap: 8px;
}
section[data-testid="stSidebar"] .stRadio label {
    background-color: rgba(255, 255, 255, 0.03);
    padding: 10px 14px;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.06);
    width: 100%;
    transition: all 0.2s ease;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background-color: rgba(14, 165, 164, 0.15) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

.hero {
    padding: 8px 0 18px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 22px;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.12em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 4px;
}
.hero-title {
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 30px;
    color: var(--text);
    margin: 0;
}
.hero-sub { color: var(--text-soft); font-size: 14.5px; margin-top: 4px; }

/* KPI cards */
.kpi-row { display: flex; gap: 14px; margin-bottom: 22px; flex-wrap: wrap; }
.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 18px;
    flex: 1;
    min-width: 170px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.kpi-label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-soft);
    font-weight: 600;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 26px;
    font-weight: 600;
    color: var(--text);
    margin-top: 4px;
}

/* Risk badge */
.risk-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 12.5px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.risk-Very-High { background: rgba(220,38,38,0.2); color: #F87171; }
.risk-High { background: rgba(249,115,22,0.2); color: #FB923C; }
.risk-Medium { background: rgba(224,164,0,0.2); color: #FACC15; }
.risk-Low { background: rgba(22,163,74,0.2); color: #4ADE80; }

/* Signal bars */
.signal-wrap { display: flex; align-items: flex-end; gap: 4px; height: 34px; }
.signal-bar { width: 8px; border-radius: 2px; background: var(--border); }

.section-title {
    font-family: 'Sora', sans-serif;
    font-weight: 600;
    font-size: 17px;
    color: var(--text);
    margin: 6px 0 10px 0;
}

.empty-state {
    border: 1px dashed var(--border);
    border-radius: 12px;
    padding: 34px 20px;
    text-align: center;
    color: var(--text-soft);
    background: var(--surface);
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached resources & Assistant Initialization
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def get_predictor() -> ChurnPredictor:
    return ChurnPredictor()

def get_assistant() -> ChurnAssistant | None:
    if not CHATBOT_AVAILABLE:
        return None
    try:
        return ChurnAssistant()
    except Exception as e:
        st.sidebar.warning(f"⚠️ Chatbot unavailable: {e}")
        return None

# Session state initialization
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "last_scored_customer" not in st.session_state:
    st.session_state.last_scored_customer = None
if "last_scored_prediction" not in st.session_state:
    st.session_state.last_scored_prediction = None


def risk_badge_html(risk: str) -> str:
    css_class = "risk-" + risk.replace(" ", "-")
    return f'<span class="risk-badge {css_class}">{risk}</span>'


def signal_bars_html(probability: float) -> str:
    n_bars = 5
    filled = max(1, round((1 - probability) * n_bars))
    colors = {
        1: "var(--risk-very-high)",
        2: "var(--risk-high)",
        3: "#E0A400",
        4: "#65B563",
        5: "var(--risk-low)",
    }
    bar_color = colors[filled]
    bars = ""
    for i in range(1, n_bars + 1):
        height = 8 + i * 5
        color = bar_color if i <= filled else "var(--border)"
        bars += f'<div class="signal-bar" style="height:{height}px; background:{color};"></div>'
    return f'<div class="signal-wrap">{bars}</div>'


def sample_template_csv() -> bytes:
    example = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0, "TotalCharges": 425.0,
    }
    df = pd.DataFrame([example])
    return df.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    "<div style='font-family:Sora,sans-serif;font-weight:700;font-size:20px;"
    "color:#fff;padding:4px 0 2px 0;'>📶 Churn Intelligence</div>"
    "<div style='color:#8B94A3;font-size:12px;margin-bottom:20px;'>"
    "Logistic Regression + ADASYN + AI Assistant</div>",
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigate",
    ["📊 Dashboard", "🧍 Single Customer", "ℹ️ About the Model"],
    label_visibility="collapsed",
)

try:
    predictor = get_predictor()
    load_error = None
except Exception as e:  # noqa: BLE001
    predictor = None
    load_error = str(e)

if load_error:
    st.error(
        "⚠️ Could not load the model artifacts.\n\n"
        f"**Details:** {load_error}\n\n"
        "Check that `model/` contains all required files and that "
        "`requirements.txt` is fully installed (including `imbalanced-learn`)."
    )
    st.stop()

assistant = get_assistant()

CATEGORICAL_OPTIONS = predictor.schema["categorical_options"] or _FALLBACK_CATEGORICAL_OPTIONS
NUMERIC_RANGES = predictor.schema["numeric_ranges"] or _FALLBACK_NUMERIC_RANGES


# ---------------------------------------------------------------------------
# PAGE: Dashboard (CSV upload or Demo Data)
# ---------------------------------------------------------------------------
if page == "📊 Dashboard":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">Batch analysis</div>
            <p class="hero-title">Customer churn dashboard</p>
            <p class="hero-sub">Upload a CSV file or load demo data to analyze churn risk across your book of business.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_upload, col_mode, col_template = st.columns([2, 1.2, 1])
    with col_upload:
        uploaded = st.file_uploader("Upload customer CSV", type=["csv"], label_visibility="collapsed")
    with col_mode:
        data_source_mode = st.radio(
            "Data Source",
            ["Upload CSV", "Use Demo Data"],
            horizontal=True,
            label_visibility="collapsed"
        )
    with col_template:
        st.download_button(
            "⬇ Download template",
            data=sample_template_csv(),
            file_name="churn_customers_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    raw_df = None

    if data_source_mode == "Use Demo Data":
        demo_data = [
            {"gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0, "TotalCharges": 425.0},
            {"gender": "Male", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No", "tenure": 24, "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Two year", "PaperlessBilling": "No", "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 75.5, "TotalCharges": 1812.0},
            {"gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes", "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No", "Contract": "One year", "PaperlessBilling": "Yes", "PaymentMethod": "Mailed check", "MonthlyCharges": 50.2, "TotalCharges": 602.4}
        ]
        raw_df = pd.DataFrame(demo_data)
        st.info("💡 Using built-in demo customer data for analysis.")

    elif uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read the uploaded file: {e}")

    if raw_df is None:
        st.markdown(
            """
            <div class="empty-state">
                <b>No data source selected.</b><br/>
                Upload a CSV file or switch to <b>Use Demo Data</b> from above
                to see churn rate, risk segments, and high-risk accounts.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        missing_cols = set(RAW_REQUIRED_COLUMNS) - set(raw_df.columns)
        if missing_cols:
            st.error(f"The dataset is missing required columns: {sorted(missing_cols)}")
            st.stop()

        with st.spinner("Scoring customers…"):
            scored = predictor.predict_batch(raw_df)
            summary = predictor.summarize(scored)

        st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="kpi-row">
                <div class="kpi-card">
                    <div class="kpi-label">Customers scored</div>
                    <div class="kpi-value">{summary['total_customers']:,}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Predicted to churn</div>
                    <div class="kpi-value">{summary['predicted_churn_count']:,}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Predicted churn rate</div>
                    <div class="kpi-value">{summary['predicted_churn_rate']*100:.1f}%</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Avg. churn probability</div>
                    <div class="kpi-value">{summary['avg_churn_probability']*100:.1f}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_chart, col_risk = st.columns([2, 1])
        with col_chart:
            st.markdown('<div class="section-title">Risk level breakdown</div>', unsafe_allow_html=True)
            risk_df = pd.DataFrame(
                {
                    "Risk level": list(summary["risk_level_breakdown"].keys()),
                    "Customers": list(summary["risk_level_breakdown"].values()),
                }
            )
            st.bar_chart(risk_df.set_index("Risk level"), color="#0EA5A4")

        with col_risk:
            st.markdown('<div class="section-title">Segment share</div>', unsafe_allow_html=True)
            st.dataframe(
                risk_df.assign(
                    Share=(risk_df["Customers"] / max(summary["total_customers"], 1) * 100).round(1)
                ),
                hide_index=True,
                use_container_width=True,
            )

        st.markdown('<div class="section-title">Highest-risk customers</div>', unsafe_allow_html=True)
        top_risk = scored.sort_values("churn_probability", ascending=False).head(25)
        display_cols = [c for c in top_risk.columns if c not in ("prediction",)]
        st.dataframe(
            top_risk[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "churn_probability": st.column_config.ProgressColumn(
                    "Churn probability", min_value=0, max_value=1, format="%.2f"
                ),
            },
        )

        st.download_button(
            "⬇ Download full scored dataset",
            data=scored.to_csv(index=False).encode("utf-8"),
            file_name="churn_predictions.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# PAGE: Single customer prediction & AI Assistant
# ---------------------------------------------------------------------------
elif page == "🧍 Single Customer":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">Live scoring & AI Copilot</div>
            <p class="hero-title">Single customer prediction</p>
            <p class="hero-sub">Fill in the customer's profile to get an instant churn risk score and consult the AI assistant.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    form_col, result_col = st.columns([1.2, 1.1])

    with form_col:
        with st.form("single_prediction_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                gender = st.selectbox("Gender", CATEGORICAL_OPTIONS["gender"])
                senior = st.radio("Senior citizen", ["No", "Yes"], horizontal=True)
                partner = st.selectbox("Has a partner", CATEGORICAL_OPTIONS["Partner"])
                dependents = st.selectbox("Has dependents", CATEGORICAL_OPTIONS["Dependents"])
                contract = st.selectbox("Contract type", CATEGORICAL_OPTIONS["Contract"])
            with c2:
                min_t = int(NUMERIC_RANGES["tenure"][0])
                max_t = int(NUMERIC_RANGES["tenure"][1])
                tenure = st.slider("Tenure (months)", min_value=min_t, max_value=max_t, value=12)
                phone = st.selectbox("Phone service", CATEGORICAL_OPTIONS["PhoneService"])
                multiple_lines = st.selectbox("Multiple lines", CATEGORICAL_OPTIONS["MultipleLines"])
                internet = st.selectbox("Internet service", CATEGORICAL_OPTIONS["InternetService"])
                paperless = st.selectbox("Paperless billing", CATEGORICAL_OPTIONS["PaperlessBilling"])
            with c3:
                online_sec = st.selectbox("Online security", CATEGORICAL_OPTIONS["OnlineSecurity"])
                online_backup = st.selectbox("Online backup", CATEGORICAL_OPTIONS["OnlineBackup"])
                device_protect = st.selectbox("Device protection", CATEGORICAL_OPTIONS["DeviceProtection"])
                tech_support = st.selectbox("Tech support", CATEGORICAL_OPTIONS["TechSupport"])
                payment = st.selectbox("Payment method", CATEGORICAL_OPTIONS["PaymentMethod"])

            c4, c5 = st.columns(2)
            with c4:
                streaming_tv = st.selectbox("Streaming TV", CATEGORICAL_OPTIONS["StreamingTV"])
                monthly_charges = st.number_input(
                    "Monthly charges ($)", min_value=0.0, value=float(NUMERIC_RANGES["MonthlyCharges"][0])
                )
            with c5:
                streaming_movies = st.selectbox("Streaming movies", CATEGORICAL_OPTIONS["StreamingMovies"])
                total_charges = st.number_input(
                    "Total charges ($)", min_value=0.0, value=float(monthly_charges * tenure)
                )

            submitted = st.form_submit_button("Predict churn risk", use_container_width=True)

    with result_col:
        if submitted:
            record = {
                "gender": gender,
                "SeniorCitizen": 1 if senior == "Yes" else 0,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone,
                "MultipleLines": multiple_lines,
                "InternetService": internet,
                "OnlineSecurity": online_sec,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protect,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }
            result = predictor.predict_one(record)

            # Reset chat states upon a new prediction run
            st.session_state.last_scored_customer = record
            st.session_state.last_scored_prediction = result
            st.session_state.last_response = None
            st.session_state.last_query = None
            if assistant:
                assistant.clear_memory()

        if st.session_state.last_scored_prediction is not None:
            result = st.session_state.last_scored_prediction
            record = st.session_state.last_scored_customer

            st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="kpi-card" style="margin-bottom:14px;">
                    <div class="kpi-label">Churn probability</div>
                    <div class="kpi-value">{result['confidence']}</div>
                    <div style="margin-top:10px;">{risk_badge_html(result['risk_level'])}
                    <span style="color:var(--text-soft); font-size:13px; margin-left:8px;">
                    prediction: <b>{result['prediction']}</b> (threshold {result['threshold_used']:.2f})
                    </span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown('<div class="section-title">Signal strength</div>', unsafe_allow_html=True)
            st.markdown(signal_bars_html(result["churn_probability"]), unsafe_allow_html=True)

            # --- AI CHATBOT INTEGRATION ---
            st.markdown("---")
            st.markdown('<div class="section-title">🤖 AI Churn Assistant</div>', unsafe_allow_html=True)

            if not assistant:
                st.warning("⚠️ AI Assistant is unavailable. Please check your `.env` configuration for `GEMINI_API_KEY`.")
            else:
                # Preset action buttons
                action_cols = st.columns(4)
                triggered_action = None

                if action_cols[0].button("💡 Explain", use_container_width=True):
                    triggered_action = (Action.EXPLAIN, "Explain why this customer is at risk.")
                if action_cols[1].button("📋 Summary", use_container_width=True):
                    triggered_action = (Action.SUMMARY, "Generate an executive summary.")
                if action_cols[2].button("🛡️ Retention", use_container_width=True):
                    triggered_action = (Action.RETENTION, "Suggest retention strategies.")
                if action_cols[3].button("✉️ Draft Email", use_container_width=True):
                    triggered_action = (Action.EMAIL, "Draft a personalized retention email.")

                # Execute action if button was clicked
                if triggered_action:
                    act_type, prompt_label = triggered_action
                    with st.spinner("Assistant thinking..."):
                        response_text = assistant.ask(
                            customer=record,
                            prediction=result,
                            question=prompt_label,
                            action=act_type
                        )
                        st.session_state.last_query = prompt_label
                        st.session_state.last_response = response_text

                # Handle chat input text box
                user_question = st.chat_input("Ask the AI assistant about this customer...")
                if user_question:
                    with st.spinner("Thinking..."):
                        response_text = assistant.ask(
                            customer=record,
                            prediction=result,
                            question=user_question,
                            action=Action.FREE_CHAT
                        )
                        st.session_state.last_query = user_question
                        st.session_state.last_response = response_text

                # Display ONLY the latest output cleanly
                if st.session_state.last_response:
                    with st.chat_message("user"):
                        st.markdown(st.session_state.last_query)
                    with st.chat_message("assistant"):
                        st.markdown(st.session_state.last_response)

        else:
            st.markdown(
                '<div class="empty-state">Fill in the form and click '
                '<b>Predict churn risk</b> to see the prediction and consult the AI assistant.</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# PAGE: About the model
# ---------------------------------------------------------------------------
else:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">Model card</div>
            <p class="hero-title">About this model</p>
            <p class="hero-sub">What's under the hood, and how predictions are made.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Form options & ranges source: {predictor.schema['source']}")

    info = predictor.model_info
    st.markdown(
        f"""
        <div class="kpi-row">
            <div class="kpi-card"><div class="kpi-label">Algorithm</div>
                <div class="kpi-value" style="font-size:18px;">{info.get('model', '—')}</div></div>
            <div class="kpi-card"><div class="kpi-label">Imbalance handling</div>
                <div class="kpi-value" style="font-size:18px;">{info.get('sampling', '—')}</div></div>
            <div class="kpi-card"><div class="kpi-label">Decision threshold</div>
                <div class="kpi-value">{info.get('threshold', '—')}</div></div>
            <div class="kpi-card"><div class="kpi-label">F1 score</div>
                <div class="kpi-value">{info.get('f1_score', 0):.3f}</div></div>
            <div class="kpi-card"><div class="kpi-label">ROC-AUC</div>
                <div class="kpi-value">{info.get('roc_auc', 0):.3f}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">How risk levels are defined</div>', unsafe_allow_html=True)
    risk_table = pd.DataFrame(
        {
            "Risk level": ["Very High", "High", "Medium", "Low"],
            "Churn probability": ["≥ 80%", "60% – 79%", "40% – 59%", "< 40%"],
        }
    )
    st.dataframe(risk_table, hide_index=True, use_container_width=True)

    st.markdown('<div class="section-title">Input features</div>', unsafe_allow_html=True)
    st.caption(
        "The model consumes 19 raw customer attributes; 3 additional features "
        "(AverageSpend, ServiceCount, IsNewCustomer) are derived automatically."
    )
    st.dataframe(pd.DataFrame({"Raw feature": RAW_REQUIRED_COLUMNS}), hide_index=True, use_container_width=True)
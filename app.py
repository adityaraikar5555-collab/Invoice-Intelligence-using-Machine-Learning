import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Invoice Intelligence ML System",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

FREIGHT_MODEL_PATHS = [
    BASE_DIR / "models" / "predict_freight_model.pkl",
    BASE_DIR / "FreightCostPrediction" / "models" / "predict_freight_model.pkl",
]

INVOICE_MODEL_PATH = BASE_DIR / "invoice_flagging" / "models" / "predict_flag_invoice.pkl"
SCALER_PATH = BASE_DIR / "invoice_flagging" / "models" / "scaler.pkl"


# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------
def get_existing_path(paths):
    for path in paths:
        if path.exists():
            return path
    return None


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def format_rupees(value):
    return f"₹ {value:,.2f}"


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("📊 Invoice Intelligence Machine Learning System")
st.caption("AI-powered freight cost prediction and invoice risk detection")

page = st.sidebar.radio(
    "Choose Module",
    ["Home", "Freight Cost Prediction", "Invoice Flagging"]
)

# ---------------------------------------------------------
# Home Page
# ---------------------------------------------------------
if page == "Home":
    st.header("🏠 Project Dashboard")

    st.markdown(
        """
        The **Invoice Intelligence Machine Learning System** is designed to support
        invoice analysis, freight estimation, and suspicious invoice detection.

        It combines **Regression** and **Classification** models to automate
        important invoice-related decisions.
        """
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🚚 Freight Cost Prediction")
        st.write(
            """
            Predicts the expected freight cost from invoice amount.
            This helps estimate logistics cost before invoice approval.
            """
        )

    with col2:
        st.subheader("🚩 Invoice Flagging")
        st.write(
            """
            Detects suspicious invoices using invoice, purchase, payment,
            and receiving-related information.
            """
        )

    st.divider()

    st.subheader("🛠 Technology Stack")

    tech1, tech2, tech3, tech4 = st.columns(4)

    with tech1:
        st.info("Python")
    with tech2:
        st.info("SQLite")
    with tech3:
        st.info("Scikit-learn")
    with tech4:
        st.info("Streamlit")

    st.divider()

    st.subheader("💼 Business Use Cases")

    use1, use2, use3 = st.columns(3)

    with use1:
        st.write("✅ Freight cost estimation")
        st.write("✅ Logistics cost planning")

    with use2:
        st.write("✅ Invoice anomaly detection")
        st.write("✅ Vendor invoice verification")

    with use3:
        st.write("✅ Procurement analytics")
        st.write("✅ Financial auditing support")


# ---------------------------------------------------------
# Freight Cost Prediction Page
# ---------------------------------------------------------
elif page == "Freight Cost Prediction":
    st.header("🚚 Freight Cost Prediction")
    st.write("Enter the invoice amount to predict the expected freight cost.")

    freight_model_path = get_existing_path(FREIGHT_MODEL_PATHS)

    if freight_model_path is None:
        st.error("Freight prediction model not found.")
        st.info(
            """
            Expected model location:

            - `models/predict_freight_model.pkl`
            - `FreightCostPrediction/models/predict_freight_model.pkl`
            """
        )
    else:
        freight_model = load_model(freight_model_path)

        invoice_amount = st.number_input(
            "💰 Invoice Amount (₹)",
            min_value=0.0,
            step=100.0,
            format="%.2f"
        )

        if st.button("Predict Freight Cost", type="primary"):
            input_data = pd.DataFrame({"Dollars": [invoice_amount]})
            predicted_freight = freight_model.predict(input_data)[0]

            st.divider()
            st.subheader("📦 Prediction Result")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric(
                    label="Invoice Amount",
                    value=format_rupees(invoice_amount)
                )

            with result_col2:
                st.metric(
                    label="Predicted Freight Cost",
                    value=format_rupees(predicted_freight)
                )

            st.success("Prediction completed successfully.")

            with st.expander("View input used for prediction"):
                st.dataframe(input_data, use_container_width=True)


# ---------------------------------------------------------
# Invoice Flagging Page
# ---------------------------------------------------------
elif page == "Invoice Flagging":
    st.header("🚩 Invoice Flagging / Suspicious Invoice Detection")
    st.write("Enter invoice and purchase-related details to check whether an invoice is suspicious.")

    if not INVOICE_MODEL_PATH.exists():
        st.error(f"Invoice flagging model not found: {INVOICE_MODEL_PATH}")
    elif not SCALER_PATH.exists():
        st.error(f"Scaler not found: {SCALER_PATH}")
    else:
        invoice_model = load_model(INVOICE_MODEL_PATH)
        scaler = load_model(SCALER_PATH)

        st.subheader("🧾 Invoice Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            invoice_quantity = st.number_input("Invoice Quantity", min_value=0.0, step=1.0)
            invoice_dollars = st.number_input("Invoice Amount (₹)", min_value=0.0, step=100.0)
            freight = st.number_input("Freight Cost (₹)", min_value=0.0, step=10.0)

        with col2:
            days_po_to_invoice = st.number_input("Days PO to Invoice", min_value=0.0, step=1.0)
            days_to_pay = st.number_input("Days to Pay", min_value=0.0, step=1.0)
            total_brands = st.number_input("Total Brands", min_value=0.0, step=1.0)

        with col3:
            total_item_quantity = st.number_input("Total Item Quantity", min_value=0.0, step=1.0)
            total_item_dollars = st.number_input("Total Item Amount (₹)", min_value=0.0, step=100.0)
            avg_receiving_delay = st.number_input("Average Receiving Delay", min_value=0.0, step=1.0)

        if st.button("Check Invoice", type="primary"):
            features = pd.DataFrame([[
                invoice_quantity,
                invoice_dollars,
                freight,
                days_po_to_invoice,
                days_to_pay,
                total_brands,
                total_item_quantity,
                total_item_dollars,
                avg_receiving_delay
            ]], columns=[
                "invoice_quantity",
                "invoice_dollars",
                "Freight",
                "days_po_to_invoice",
                "days_to_pay",
                "total_brands",
                "total_item_quantity",
                "total_item_dollars",
                "avg_receiving_delay"
            ])

            scaled_features = scaler.transform(features)
            prediction = invoice_model.predict(scaled_features)[0]

            st.divider()
            st.subheader("📊 Invoice Analysis Result")

            if prediction == 1:
                st.error("🚨 Suspicious Invoice Detected")
                st.warning("Recommendation: Manual verification is required before approval.")
            else:
                st.success("✅ Invoice Looks Normal")
                st.info("Recommendation: No suspicious activity detected based on the provided details.")

            if hasattr(invoice_model, "predict_proba"):
                probability = invoice_model.predict_proba(scaled_features)[0]

                prob_col1, prob_col2 = st.columns(2)

                with prob_col1:
                    st.metric(
                        "Normal Probability",
                        f"{probability[0] * 100:.2f}%"
                    )

                with prob_col2:
                    st.metric(
                        "Suspicious Probability",
                        f"{probability[1] * 100:.2f}%"
                    )

            with st.expander("View input used for prediction"):
                display_features = features.copy()
                display_features = display_features.rename(columns={
                    "invoice_quantity": "Invoice Quantity",
                    "invoice_dollars": "Invoice Amount (₹)",
                    "Freight": "Freight Cost (₹)",
                    "days_po_to_invoice": "Days PO to Invoice",
                    "days_to_pay": "Days to Pay",
                    "total_brands": "Total Brands",
                    "total_item_quantity": "Total Item Quantity",
                    "total_item_dollars": "Total Item Amount (₹)",
                    "avg_receiving_delay": "Average Receiving Delay"
                })
                st.dataframe(display_features, use_container_width=True)
"""Streamlit dashboard for the E-commerce Return Intelligence System."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard.core import (  # noqa: E402, type: ignore[reportMissingImports]
    FEATURES,
    INTERVENTION_THRESHOLD,
    MONITOR_THRESHOLD,
    batch_decisions,
    decision,
    explain_order,
    load_artifacts,
    predict_order,
    summary_metrics,
)

st.set_page_config(
    page_title="E-commerce Return Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.metric-card {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 14px;}
.small-muted {color: #777; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model and SHAP explainer...")
def get_artifacts():
    return load_artifacts()


@st.cache_data(show_spinner="Scoring test orders...")
def get_batch(_artifacts):
    return batch_decisions(_artifacts)


artifacts = get_artifacts()
batch = get_batch(artifacts)
metrics = summary_metrics(batch)

st.title("📦 E-commerce Return Intelligence")
st.caption("Predict → Explain → Decide → Act | Production candidate: Logistic Regression")

with st.sidebar:
    st.header("System Controls")
    st.metric("Intervention threshold", f"{INTERVENTION_THRESHOLD:.0%}")
    st.metric("Monitor threshold", f"{MONITOR_THRESHOLD:.0%}")
    st.caption("Threshold frozen during model selection. Business economics are modeled assumptions, not realized savings.")
    st.divider()
    page = st.radio(
        "Navigate",
        ["Executive Overview", "Risk Analysis", "Order Simulator", "AI Explanation"],
    )

if page == "Executive Overview":
    st.subheader("Executive Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test orders", f"{metrics['orders']:,}")
    c2.metric("Observed return rate", f"{metrics['observed_return_rate']:.1%}")
    c3.metric("High-risk orders", f"{metrics['high_risk_orders']:,}")
    c4.metric("Modeled net value", f"₹{metrics['modeled_net_value']:,.0f}")

    left, right = st.columns(2)
    with left:
        counts = batch["recommended_action"].value_counts().rename_axis("Action").reset_index(name="Orders")
        fig = px.pie(counts, names="Action", values="Orders", hole=0.55, title="Operational actions")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.histogram(batch, x="return_probability", nbins=40, title="Predicted return probability")
        fig.add_vline(x=INTERVENTION_THRESHOLD, line_dash="dash", annotation_text="Intervene")
        fig.add_vline(x=MONITOR_THRESHOLD, line_dash="dot", annotation_text="Monitor")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Business interpretation")
    st.info(
        f"The frozen policy flags {metrics['intervene_orders']:,} of {metrics['orders']:,} test orders for intervention. "
        f"Under the configured ₹100 intervention cost, ₹800 avoided-return cost, and 60% effectiveness assumption, "
        f"the modeled net value is ₹{metrics['modeled_net_value']:,.0f}."
    )

elif page == "Risk Analysis":
    st.subheader("Risk Analysis")
    category = batch.groupby("category", as_index=False).agg(
        orders=("order_id", "count"),
        observed_return_rate=("returned", "mean"),
        avg_predicted_probability=("return_probability", "mean"),
    ).sort_values("avg_predicted_probability", ascending=False)

    a, b = st.columns(2)
    with a:
        fig = px.bar(category, x="category", y="avg_predicted_probability", title="Average predicted return risk by category")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = px.bar(category, x="category", y="observed_return_rate", title="Observed return rate by category")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    a, b = st.columns(2)
    with a:
        fig = px.box(batch, x="payment_type", y="return_probability", title="Risk by payment type")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = px.scatter(
            batch.sample(min(5000, len(batch)), random_state=42),
            x="discount_pct", y="return_probability", color="category",
            hover_data=["order_id", "product_price"], title="Discount vs predicted return risk",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Highest-risk orders")
    cols = ["order_id", "return_probability", "risk_level", "recommended_action", "category", "payment_type", "product_price"]
    top = batch.sort_values("return_probability", ascending=False)[cols].head(20).copy()
    top["return_probability"] = top["return_probability"].map(lambda x: f"{x:.1%}")
    st.dataframe(top, use_container_width=True, hide_index=True)

elif page == "Order Simulator":
    st.subheader("Order-level prediction")
    st.caption("Enter an order as it would look immediately after placement. No post-delivery fields are used.")

    defaults = {
        "product_price": 2499.0, "product_rating": 3.2,
        "product_historical_return_rate": 0.28, "customer_age": 31,
        "customer_previous_orders": 8, "customer_previous_returns": 3,
        "customer_return_rate": 0.38, "quantity": 1, "discount_pct": 25.0,
        "delivery_distance_km": 18.0, "expected_delivery_days": 5,
        "order_month": 10, "order_day_of_week": 2,
    }

    with st.form("order_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            product_price = st.number_input("Product price (₹)", 100.0, 200000.0, defaults["product_price"], step=100.0)
            product_rating = st.slider("Product rating", 1.0, 5.0, defaults["product_rating"], 0.1)
            product_historical_return_rate = st.slider("Product historical return rate", 0.0, 1.0, defaults["product_historical_return_rate"], 0.01)
            category = st.selectbox("Category", ["Fashion", "Footwear", "Electronics", "Home & Kitchen", "Beauty", "Books"])
        with c2:
            customer_age = st.number_input("Customer age", 18, 90, defaults["customer_age"])
            customer_previous_orders = st.number_input("Previous orders", 0, 200, defaults["customer_previous_orders"])
            customer_previous_returns = st.number_input("Previous returns", 0, 200, defaults["customer_previous_returns"])
            customer_return_rate = st.slider("Customer historical return rate", 0.0, 1.0, defaults["customer_return_rate"], 0.01)
            payment_type = st.selectbox("Payment type", ["COD", "UPI", "Card", "Wallet"])
        with c3:
            quantity = st.number_input("Quantity", 1, 20, defaults["quantity"])
            discount_pct = st.slider("Discount %", 0.0, 80.0, defaults["discount_pct"], 1.0)
            delivery_distance_km = st.number_input("Delivery distance (km)", 0.0, 500.0, defaults["delivery_distance_km"], 1.0)
            expected_delivery_days = st.number_input("Expected delivery days", 1, 30, defaults["expected_delivery_days"])
            order_month = st.selectbox("Order month", list(range(1, 13)), index=defaults["order_month"] - 1)
            order_day_of_week = st.selectbox("Order day of week", list(range(7)), index=defaults["order_day_of_week"])

        submitted = st.form_submit_button("Predict return risk", type="primary", use_container_width=True)

    if submitted:
        discount_bucket = "None" if discount_pct == 0 else "Low" if discount_pct <= 10 else "Medium" if discount_pct <= 25 else "High" if discount_pct <= 50 else "Very High"
        price_bucket = "Budget" if product_price < 1000 else "Mid" if product_price < 5000 else "Premium"
        order = {
            "product_price": product_price, "product_rating": product_rating,
            "product_historical_return_rate": product_historical_return_rate,
            "customer_age": customer_age, "customer_previous_orders": customer_previous_orders,
            "customer_previous_returns": customer_previous_returns, "customer_return_rate": customer_return_rate,
            "quantity": quantity, "discount_pct": discount_pct, "delivery_distance_km": delivery_distance_km,
            "expected_delivery_days": expected_delivery_days, "order_month": order_month,
            "order_day_of_week": order_day_of_week, "category": category, "payment_type": payment_type,
            "discount_bucket": discount_bucket, "price_bucket": price_bucket,
        }
        result = predict_order(artifacts, order)
        st.session_state["last_order"] = order
        st.session_state["last_result"] = result

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        p = result["return_probability"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Return probability", f"{p:.1%}")
        c2.metric("Risk", result["risk_level"])
        c3.metric("Action", result["recommended_action"])
        c4.metric("Expected net value", f"₹{result['net_expected_value']:,.0f}")

        expl = explain_order(artifacts, st.session_state["last_order"])
        st.subheader("Why did the model predict this?")
        fig = px.bar(
            expl.sort_values("shap_value"), x="shap_value", y="feature", orientation="h",
            title="SHAP model attribution (log-odds)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(expl[["feature", "source_feature", "feature_value", "shap_value", "direction"]], use_container_width=True, hide_index=True)

elif page == "AI Explanation":
    st.subheader("AI Explanation Layer")
    st.caption("The LLM explains the deterministic ML + SHAP + decision output; it does not make or modify the decision.")

    if "last_result" not in st.session_state:
        st.warning("Run an order prediction in the Order Simulator first.")
    else:
        result = st.session_state["last_result"]
        order = st.session_state["last_order"]
        expl = explain_order(artifacts, order)

        positive = expl[expl["shap_value"] > 0].head(3)
        negative = expl[expl["shap_value"] < 0].head(2)
        factors = ", ".join(positive["source_feature"].astype(str)) or "the supplied SHAP factors"
        p = result["return_probability"]
        action = result["recommended_action"]
        risk = result["risk_level"]
        headline = (
            f"High return risk: {p:.1%} — proactive intervention recommended."
            if action == "INTERVENE" else
            f"Medium return risk: {p:.1%} — monitor the order."
            if action == "MONITOR" else
            f"Low return risk: {p:.1%} — no proactive action recommended."
        )
        explanation = (
            f"The model estimates a {p:.1%} probability of return and classifies the order as {risk.lower()} risk. "
            f"The main model-attribution signals increasing risk are {factors}. "
            f"The decision engine recommends {action} based on the frozen policy."
        )

        st.success(headline)
        st.write(explanation)
        st.subheader("Business context")
        st.write(
            f"Expected avoided return cost: ₹{result['expected_avoided_return_cost']:,.2f}. "
            f"Intervention cost: ₹{result['intervention_cost']:,.2f}. "
            f"Modeled net value: ₹{result['net_expected_value']:,.2f}."
        )
        st.subheader("Key model factors")
        st.dataframe(expl[["feature", "source_feature", "feature_value", "shap_value", "direction"]].head(8), use_container_width=True, hide_index=True)
        st.caption("Caveat: SHAP describes model behavior, not causality. Economics depend on assumed intervention effectiveness and should be validated with experiments.")

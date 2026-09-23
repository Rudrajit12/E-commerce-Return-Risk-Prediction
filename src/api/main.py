"""
Step 13 — FastAPI serving layer.

Exposes the Return Intelligence system as an HTTP API.

Endpoints:
    GET  /health
    POST /predict

The API:
    1. validates the incoming order
    2. runs the frozen Logistic Regression model
    3. computes the business decision
    4. computes SHAP attribution for the order
    5. generates an explanation using the Step 12 LLM layer
       (mock mode by default; OpenAI mode is optional)

Run:
    uvicorn src.api.main:app --reload

OpenAPI docs:
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from .schemas import HealthResponse, OrderRequest, PredictionResponse
except ImportError:
    from schemas import HealthResponse, OrderRequest, PredictionResponse

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models/model_selection/logistic_regression.joblib"
TRAIN_PATH = ROOT / "data/processed/train_model_selection.csv"
EXPLANATION_SCHEMA_PATH = ROOT / "src/llm/explanation_schema.json"

INTERVENTION_THRESHOLD = 0.21
MONITOR_THRESHOLD = 0.10
INTERVENTION_COST = 100.0
AVOIDED_RETURN_COST = 800.0
EFFECTIVENESS = 0.60

FEATURES = [
    "product_price", "product_rating", "product_historical_return_rate",
    "customer_age", "customer_previous_orders", "customer_previous_returns",
    "customer_return_rate", "quantity", "discount_pct",
    "delivery_distance_km", "expected_delivery_days", "order_month",
    "order_day_of_week", "category", "payment_type",
    "discount_bucket", "price_bucket",
]


def load_runtime() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(TRAIN_PATH)

    model = joblib.load(MODEL_PATH)
    train = pd.read_csv(TRAIN_PATH)
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]

    # Deterministic background sample matching Step 10's methodology.
    rng = np.random.default_rng(42)
    n = min(1000, len(train))
    idx = rng.choice(len(train), size=n, replace=False)
    background = train.iloc[idx][FEATURES]
    background_t = preprocessor.transform(background)
    explainer = shap.LinearExplainer(estimator, background_t)

    # Import the Step 12 explanation functions without duplicating logic.
    from src.llm.llm_explainer import (
        build_input_payload,
        load_json,
        mock_explanation,
        openai_explanation,
        validate_output,
    )

    return {
        "model": model,
        "preprocessor": preprocessor,
        "explainer": explainer,
        "feature_names": list(preprocessor.get_feature_names_out()),
        "llm": {
            "build_input_payload": build_input_payload,
            "mock_explanation": mock_explanation,
            "openai_explanation": openai_explanation,
            "validate_output": validate_output,
            "schema": load_json(EXPLANATION_SCHEMA_PATH),
        },
    }


runtime = load_runtime()
LLM_MODE = os.getenv("LLM_MODE", "mock").lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

app = FastAPI(
    title="E-commerce Return Intelligence API",
    description=(
        "Production-style API for return-risk prediction, business decisioning, "
        "SHAP explanation, and optional LLM-generated explanation."
    ),
    version="1.0.0",
)


def risk_decision(probability: float) -> dict[str, float | str]:
    expected_avoided = probability * EFFECTIVENESS * AVOIDED_RETURN_COST
    net_value = expected_avoided - INTERVENTION_COST

    if probability >= INTERVENTION_THRESHOLD:
        risk = "HIGH"
        action = "INTERVENE"
    elif probability >= MONITOR_THRESHOLD:
        risk = "MEDIUM"
        action = "MONITOR"
    else:
        risk = "LOW"
        action = "NO_ACTION"

    return {
        "risk_level": risk,
        "recommended_action": action,
        "expected_avoided_return_cost": expected_avoided,
        "net_expected_value": net_value,
    }


def explain_order(order: dict[str, Any], probability: float) -> dict[str, Any]:
    frame = pd.DataFrame([order], columns=FEATURES)
    transformed = runtime["preprocessor"].transform(frame)
    shap_values = np.asarray(runtime["explainer"].shap_values(transformed))
    if shap_values.ndim == 2:
        shap_values = shap_values[0]
    else:
        shap_values = shap_values.reshape(-1)

    names = runtime["feature_names"]
    rows = []
    for idx in np.argsort(np.abs(shap_values))[::-1][:15]:
        name = names[idx]
        base = name.split("_", 1)[0] if "_" in name else name
        value = order.get(base)
        contribution = float(shap_values[idx])
        rows.append({
            "feature": name,
            "source_feature": base,
            "feature_value": value,
            "shap_value_log_odds": contribution,
            "direction": (
                "increases_risk" if contribution > 0 else "reduces_risk"
            ),
            "absolute_shap": abs(contribution),
        })

    decision = risk_decision(probability)
    return {
        "order_id": "api_request",
        "model_prediction": {
            "return_probability": probability,
            "risk_level": decision["risk_level"],
        },
        "business_decision": {
            "recommended_action": decision["recommended_action"],
            "expected_avoided_return_cost": decision["expected_avoided_return_cost"],
            "intervention_cost": INTERVENTION_COST,
            "net_expected_value": decision["net_expected_value"],
        },
        "explanation": {
            "top_risk_increasing_factors": [
                r for r in rows if r["shap_value_log_odds"] > 0
            ][:5],
            "top_risk_reducing_factors": [
                r for r in rows if r["shap_value_log_odds"] < 0
            ][:5],
            "policy_reason": (
                "Return risk is at or above the frozen production threshold; "
                "the order is eligible for proactive intervention."
                if decision["recommended_action"] == "INTERVENE"
                else "Risk does not meet the intervention policy."
            ),
        },
    }


@app.get("/health", response_model=HealthResponse)
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": "logistic_regression",
        "threshold": INTERVENTION_THRESHOLD,
        "llm_mode": LLM_MODE,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(order: OrderRequest) -> dict[str, Any]:
    try:
        order_dict = order.model_dump()
        X = pd.DataFrame([order_dict], columns=FEATURES)
        probability = float(runtime["model"].predict_proba(X)[0, 1])
        decision = risk_decision(probability)
        decision_payload = explain_order(order_dict, probability)

        # Reuse Step 12 LLM layer.
        payload = runtime["llm"]["build_input_payload"](decision_payload)
        if LLM_MODE == "openai":
            explanation = runtime["llm"]["openai_explanation"](
                payload, runtime["llm"]["schema"], OPENAI_MODEL
            )
        else:
            explanation = runtime["llm"]["mock_explanation"](payload)

        runtime["llm"]["validate_output"](explanation, payload)

        return {
            "return_probability": probability,
            "risk_level": decision["risk_level"],
            "recommended_action": decision["recommended_action"],
            "intervention_threshold": INTERVENTION_THRESHOLD,
            "monitor_threshold": MONITOR_THRESHOLD,
            "expected_avoided_return_cost": decision["expected_avoided_return_cost"],
            "intervention_cost": INTERVENTION_COST,
            "net_expected_value": decision["net_expected_value"],
            "explanation": explanation,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

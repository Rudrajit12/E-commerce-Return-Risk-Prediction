"""Core, UI-independent functions for the Streamlit dashboard."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models/model_selection/logistic_regression.joblib"
TRAIN_PATH = ROOT / "data/processed/train_model_selection.csv"
TEST_PATH = ROOT / "data/processed/test_model_selection.csv"
DECISION_SUMMARY_PATH = ROOT / "data/processed/decision_summary.json"

FEATURES = [
    "product_price", "product_rating", "product_historical_return_rate",
    "customer_age", "customer_previous_orders", "customer_previous_returns",
    "customer_return_rate", "quantity", "discount_pct",
    "delivery_distance_km", "expected_delivery_days", "order_month",
    "order_day_of_week", "category", "payment_type",
    "discount_bucket", "price_bucket",
]

INTERVENTION_THRESHOLD = 0.21
MONITOR_THRESHOLD = 0.10
INTERVENTION_COST = 100.0
AVOIDED_RETURN_COST = 800.0
EFFECTIVENESS = 0.60


def load_artifacts() -> dict[str, Any]:
    model = joblib.load(MODEL_PATH)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]

    rng = np.random.default_rng(42)
    n = min(1000, len(train))
    idx = rng.choice(len(train), size=n, replace=False)
    background = train.iloc[idx][FEATURES]
    background_t = preprocessor.transform(background)
    explainer = shap.LinearExplainer(estimator, background_t)

    return {
        "model": model,
        "train": train,
        "test": test,
        "preprocessor": preprocessor,
        "explainer": explainer,
        "feature_names": list(preprocessor.get_feature_names_out()),
    }


def decision(probability: float) -> dict[str, Any]:
    expected_avoided = probability * EFFECTIVENESS * AVOIDED_RETURN_COST
    net_value = expected_avoided - INTERVENTION_COST

    if probability >= INTERVENTION_THRESHOLD:
        risk, action = "HIGH", "INTERVENE"
    elif probability >= MONITOR_THRESHOLD:
        risk, action = "MEDIUM", "MONITOR"
    else:
        risk, action = "LOW", "NO_ACTION"

    return {
        "risk_level": risk,
        "recommended_action": action,
        "expected_avoided_return_cost": expected_avoided,
        "intervention_cost": INTERVENTION_COST,
        "net_expected_value": net_value,
    }


def predict_order(artifacts: dict[str, Any], order: dict[str, Any]) -> dict[str, Any]:
    frame = pd.DataFrame([order], columns=FEATURES)
    probability = float(artifacts["model"].predict_proba(frame)[0, 1])
    return {"return_probability": probability, **decision(probability)}


def _source_feature(transformed_name: str) -> str:
    # sklearn ColumnTransformer prefixes transformed names with num__/cat__.
    name = transformed_name.split("__", 1)[-1]
    categorical = [
        "category", "payment_type", "discount_bucket", "price_bucket"
    ]
    for base in categorical:
        if name == base or name.startswith(base + "_"):
            return base
    return name


def explain_order(artifacts: dict[str, Any], order: dict[str, Any], top_n: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame([order], columns=FEATURES)
    transformed = artifacts["preprocessor"].transform(frame)
    values = np.asarray(artifacts["explainer"].shap_values(transformed))
    values = values[0] if values.ndim == 2 else values.reshape(-1)

    names = artifacts["feature_names"]
    rows = []
    for idx in np.argsort(np.abs(values))[::-1][:top_n]:
        name = names[idx]
        source = _source_feature(name)
        rows.append({
            "feature": name,
            "source_feature": source,
            "feature_value": order.get(source),
            "shap_value": float(values[idx]),
            "absolute_shap": abs(float(values[idx])),
            "direction": "Increases risk" if values[idx] > 0 else "Reduces risk",
        })
    return pd.DataFrame(rows)


def batch_decisions(artifacts: dict[str, Any]) -> pd.DataFrame:
    test = artifacts["test"].copy()
    probabilities = artifacts["model"].predict_proba(test[FEATURES])[:, 1]
    test["return_probability"] = probabilities
    test["risk_level"] = np.select(
        [probabilities >= INTERVENTION_THRESHOLD, probabilities >= MONITOR_THRESHOLD],
        ["HIGH", "MEDIUM"],
        default="LOW",
    )
    test["recommended_action"] = np.select(
        [probabilities >= INTERVENTION_THRESHOLD, probabilities >= MONITOR_THRESHOLD],
        ["INTERVENE", "MONITOR"],
        default="NO_ACTION",
    )
    test["expected_avoided_return_cost"] = probabilities * EFFECTIVENESS * AVOIDED_RETURN_COST
    test["net_expected_value"] = test["expected_avoided_return_cost"] - INTERVENTION_COST
    return test


def summary_metrics(batch: pd.DataFrame) -> dict[str, float | int]:
    intervened = batch[batch["recommended_action"] == "INTERVENE"]
    return {
        "orders": int(len(batch)),
        "observed_return_rate": float(batch["returned"].mean()),
        "avg_predicted_return_probability": float(batch["return_probability"].mean()),
        "high_risk_orders": int((batch["risk_level"] == "HIGH").sum()),
        "intervene_orders": int((batch["recommended_action"] == "INTERVENE").sum()),
        "monitor_orders": int((batch["recommended_action"] == "MONITOR").sum()),
        "no_action_orders": int((batch["recommended_action"] == "NO_ACTION").sum()),
        "modeled_intervention_cost": float(intervened["intervention_cost"].sum()) if "intervention_cost" in intervened else float(len(intervened) * INTERVENTION_COST),
        "modeled_net_value": float(intervened["net_expected_value"].sum()),
    }

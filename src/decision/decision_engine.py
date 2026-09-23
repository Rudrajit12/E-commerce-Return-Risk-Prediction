"""
Step 11 — Business Decision Engine

Converts model output + business economics + SHAP reasons into an
operational recommendation.

Decision policy:
    probability >= high-risk threshold -> INTERVENE
    probability >= monitor threshold   -> MONITOR
    otherwise                          -> NO_ACTION

Default thresholds:
    production intervention threshold = 0.21 (frozen in Step 9)
    monitor threshold = 0.10

Business assumptions:
    intervention cost = ₹100
    avoided return cost = ₹800
    intervention effectiveness = 60%

Expected value of intervention:
    expected_avoided_cost = p(return) * effectiveness * avoided_return_cost
    net_expected_value = expected_avoided_cost - intervention_cost

The engine is deterministic and does not use post-order information.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INTERVENTION_THRESHOLD = 0.21
DEFAULT_MONITOR_THRESHOLD = 0.10
DEFAULT_INTERVENTION_COST = 100.0
DEFAULT_AVOIDED_RETURN_COST = 800.0
DEFAULT_EFFECTIVENESS = 0.60


def classify_risk(probability: float, intervention_threshold: float,
                  monitor_threshold: float) -> str:
    if probability >= intervention_threshold:
        return "HIGH"
    if probability >= monitor_threshold:
        return "MEDIUM"
    return "LOW"


def recommend_action(probability: float, intervention_threshold: float,
                      monitor_threshold: float, intervention_cost: float,
                      avoided_return_cost: float,
                      effectiveness: float) -> dict[str, Any]:
    expected_avoided_cost = probability * effectiveness * avoided_return_cost
    net_expected_value = expected_avoided_cost - intervention_cost
    risk_level = classify_risk(
        probability, intervention_threshold, monitor_threshold
    )

    # Threshold is the operational policy. The economic EV is surfaced as
    # supporting evidence rather than silently overriding the frozen policy.
    if risk_level == "HIGH":
        action = "INTERVENE"
        reason = (
            "Return risk is at or above the frozen production threshold; "
            "the order is eligible for proactive intervention."
        )
    elif risk_level == "MEDIUM":
        action = "MONITOR"
        reason = (
            "Return risk is below the intervention threshold but above the "
            "monitoring threshold; monitor without proactive intervention."
        )
    else:
        action = "NO_ACTION"
        reason = "Return risk is below the monitoring threshold."

    return {
        "risk_level": risk_level,
        "recommended_action": action,
        "probability": probability,
        "intervention_threshold": intervention_threshold,
        "monitor_threshold": monitor_threshold,
        "expected_avoided_return_cost": expected_avoided_cost,
        "intervention_cost": intervention_cost,
        "net_expected_value": net_expected_value,
        "policy_reason": reason,
    }


def build_explanation(row: pd.Series, local_shap: pd.DataFrame,
                      decision: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    for _, r in local_shap.sort_values("absolute_shap", ascending=False).iterrows():
        reasons.append({
            "feature": r["feature"],
            "source_feature": r["source_feature"],
            "feature_value": r["feature_value"],
            "shap_value_log_odds": float(r["shap_value_log_odds"]),
            "direction": r["direction"],
        })

    positive = [
        r for r in reasons if r["shap_value_log_odds"] > 0
    ][:5]
    negative = [
        r for r in reasons if r["shap_value_log_odds"] < 0
    ][:5]

    return {
        "order_id": row["order_id"],
        "model_prediction": {
            "return_probability": float(decision["probability"]),
            "risk_level": decision["risk_level"],
        },
        "business_decision": {
            "recommended_action": decision["recommended_action"],
            "expected_avoided_return_cost": decision["expected_avoided_return_cost"],
            "intervention_cost": decision["intervention_cost"],
            "net_expected_value": decision["net_expected_value"],
        },
        "explanation": {
            "top_risk_increasing_factors": positive,
            "top_risk_reducing_factors": negative,
            "policy_reason": decision["policy_reason"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="data/processed/model_selection/logistic_regression_test_predictions.csv",
    )
    parser.add_argument(
        "--shap-local",
        default="data/processed/explainability/shap_local_explanations.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/decision_engine",
    )
    parser.add_argument("--intervention-threshold", type=float,
                        default=DEFAULT_INTERVENTION_THRESHOLD)
    parser.add_argument("--monitor-threshold", type=float,
                        default=DEFAULT_MONITOR_THRESHOLD)
    parser.add_argument("--intervention-cost", type=float,
                        default=DEFAULT_INTERVENTION_COST)
    parser.add_argument("--avoided-return-cost", type=float,
                        default=DEFAULT_AVOIDED_RETURN_COST)
    parser.add_argument("--effectiveness", type=float,
                        default=DEFAULT_EFFECTIVENESS)
    args = parser.parse_args()

    if not 0 <= args.monitor_threshold < args.intervention_threshold <= 1:
        raise ValueError(
            "Require 0 <= monitor_threshold < intervention_threshold <= 1."
        )
    if args.intervention_cost < 0 or args.avoided_return_cost < 0:
        raise ValueError("Costs cannot be negative.")
    if not 0 <= args.effectiveness <= 1:
        raise ValueError("Effectiveness must be between 0 and 1.")

    root = Path(__file__).resolve().parents[2]
    predictions = pd.read_csv(root / args.predictions)
    shap_local = pd.read_csv(root / args.shap_local)
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    required = {"order_id", "predicted_probability"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Prediction file missing columns: {required - set(predictions.columns)}")

    # Build decisions for every final-test order.
    decisions = []
    for _, row in predictions.iterrows():
        d = recommend_action(
            float(row["predicted_probability"]),
            args.intervention_threshold,
            args.monitor_threshold,
            args.intervention_cost,
            args.avoided_return_cost,
            args.effectiveness,
        )
        decisions.append({
            "order_id": row["order_id"],
            "order_date": row.get("order_date"),
            "actual_returned": row.get("returned"),
            "return_probability": d["probability"],
            "risk_level": d["risk_level"],
            "recommended_action": d["recommended_action"],
            "intervention_threshold": d["intervention_threshold"],
            "monitor_threshold": d["monitor_threshold"],
            "expected_avoided_return_cost": d["expected_avoided_return_cost"],
            "intervention_cost": d["intervention_cost"],
            "net_expected_value": d["net_expected_value"],
        })

    decision_df = pd.DataFrame(decisions)
    decision_df.to_csv(output_dir / "decision_output.csv", index=False)

    # Use the deterministic order already selected by Step 10 so the
    # decision record and SHAP explanation refer to the same order.
    shap_order_ids = shap_local["order_id"].astype(str).unique().tolist()
    if not shap_order_ids:
        raise ValueError("No SHAP local explanation order IDs found.")

    selected_candidates = decision_df[
        decision_df["order_id"].astype(str).isin(shap_order_ids)
    ].copy()
    if selected_candidates.empty:
        raise ValueError("SHAP explanation order is missing from prediction output.")

    selected = selected_candidates.sort_values(
        ["return_probability", "order_id"],
        ascending=[False, True],
    ).iloc[0]

    local = shap_local[
        shap_local["order_id"].astype(str) == str(selected["order_id"])
    ].copy()

    explanation = build_explanation(
        selected,
        local,
        {
            "probability": float(selected["return_probability"]),
            "risk_level": selected["risk_level"],
            "recommended_action": selected["recommended_action"],
            "intervention_threshold": float(args.intervention_threshold),
            "monitor_threshold": float(args.monitor_threshold),
            "expected_avoided_return_cost": float(selected["expected_avoided_return_cost"]),
            "intervention_cost": float(args.intervention_cost),
            "net_expected_value": float(selected["net_expected_value"]),
            "policy_reason": (
                "Return risk is at or above the frozen production threshold; "
                "the order is eligible for proactive intervention."
                if selected["risk_level"] == "HIGH"
                else "Risk does not meet the intervention policy."
            ),
        },
    )
    (output_dir / "example_decision.json").write_text(
        json.dumps(explanation, indent=2, default=str), encoding="utf-8"
    )

    # Operational summary.
    counts = decision_df["recommended_action"].value_counts().to_dict()
    summary = {
        "policy": {
            "intervention_threshold": args.intervention_threshold,
            "monitor_threshold": args.monitor_threshold,
            "intervention_cost": args.intervention_cost,
            "avoided_return_cost": args.avoided_return_cost,
            "intervention_effectiveness": args.effectiveness,
        },
        "population": {
            "orders_evaluated": int(len(decision_df)),
            "high_risk_orders": int((decision_df["risk_level"] == "HIGH").sum()),
            "medium_risk_orders": int((decision_df["risk_level"] == "MEDIUM").sum()),
            "low_risk_orders": int((decision_df["risk_level"] == "LOW").sum()),
            "intervene_orders": int(counts.get("INTERVENE", 0)),
            "monitor_orders": int(counts.get("MONITOR", 0)),
            "no_action_orders": int(counts.get("NO_ACTION", 0)),
        },
        "expected_economics": {
            "total_intervention_cost_if_all_flagged": float(
                decision_df.loc[
                    decision_df["recommended_action"] == "INTERVENE",
                    "intervention_cost"
                ].sum()
            ),
            "sum_expected_avoided_return_cost": float(
                decision_df.loc[
                    decision_df["recommended_action"] == "INTERVENE",
                    "expected_avoided_return_cost"
                ].sum()
            ),
            "sum_net_expected_value": float(
                decision_df.loc[
                    decision_df["recommended_action"] == "INTERVENE",
                    "net_expected_value"
                ].sum()
            ),
        },
        "example_order": explanation,
        "interpretation_note": (
            "Expected economics are model-based estimates using assumed "
            "intervention effectiveness. They are not realized savings or "
            "causal estimates."
        ),
    }
    (output_dir / "decision_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    # LLM-ready JSONL: one compact decision record per order.
    with open(output_dir / "llm_decision_records.jsonl", "w", encoding="utf-8") as f:
        for _, row in decision_df.iterrows():
            record = {
                "order_id": row["order_id"],
                "return_probability": float(row["return_probability"]),
                "risk_level": row["risk_level"],
                "recommended_action": row["recommended_action"],
                "intervention_threshold": float(row["intervention_threshold"]),
                "expected_avoided_return_cost": float(row["expected_avoided_return_cost"]),
                "intervention_cost": float(row["intervention_cost"]),
                "net_expected_value": float(row["net_expected_value"]),
            }
            f.write(json.dumps(record, default=str) + "\n")

    print("Decision engine completed.")
    print(f"Orders evaluated: {len(decision_df):,}")
    print(f"INTERVENE: {counts.get('INTERVENE', 0):,}")
    print(f"MONITOR: {counts.get('MONITOR', 0):,}")
    print(f"NO_ACTION: {counts.get('NO_ACTION', 0):,}")
    print(f"Example order: {selected['order_id']}")
    print(f"Example probability: {selected['return_probability']:.2%}")
    print(f"Example action: {selected['recommended_action']}")
    print(f"Example net expected value: ₹{selected['net_expected_value']:,.2f}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()

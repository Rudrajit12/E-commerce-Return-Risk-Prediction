"""
Step 12 — LLM Explanation Layer

Takes deterministic decision-engine output + SHAP evidence and turns it into
a concise business explanation.

Modes:
    mock   : deterministic, offline, reproducible test mode.
    openai : uses the OpenAI Responses API with Structured Outputs.

The LLM is NOT allowed to change the model probability, risk classification,
recommended action, thresholds, or economics. It only verbalizes the supplied
evidence.

Environment variables for openai mode:
    OPENAI_API_KEY
    OPENAI_MODEL (default: gpt-5.6-luna)

OpenAI's Responses API and Structured Outputs are used so the generated
explanation conforms to a JSON schema.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_input_payload(decision: dict[str, Any]) -> dict[str, Any]:
    # Pass only decision-engine facts and SHAP evidence to the LLM.
    return {
        "order_id": decision["order_id"],
        "return_probability": decision["model_prediction"]["return_probability"],
        "risk_level": decision["model_prediction"]["risk_level"],
        "recommended_action": decision["business_decision"]["recommended_action"],
        "expected_avoided_return_cost": decision["business_decision"]["expected_avoided_return_cost"],
        "intervention_cost": decision["business_decision"]["intervention_cost"],
        "net_expected_value": decision["business_decision"]["net_expected_value"],
        "top_risk_increasing_factors": decision["explanation"]["top_risk_increasing_factors"],
        "top_risk_reducing_factors": decision["explanation"]["top_risk_reducing_factors"],
        "policy_reason": decision["explanation"]["policy_reason"],
    }


def mock_explanation(payload: dict[str, Any]) -> dict[str, Any]:
    p = payload["return_probability"]
    action = payload["recommended_action"]
    risk = payload["risk_level"]

    positive = payload["top_risk_increasing_factors"][:3]
    negative = payload["top_risk_reducing_factors"][:2]

    key_factors = []
    for item in positive:
        key_factors.append({
            "factor": str(item["source_feature"]),
            "direction": "increases_risk",
            "impact": f"SHAP contribution {float(item['shap_value_log_odds']):+.3f} log-odds.",
        })
    for item in negative:
        key_factors.append({
            "factor": str(item["source_feature"]),
            "direction": "reduces_risk",
            "impact": f"SHAP contribution {float(item['shap_value_log_odds']):+.3f} log-odds.",
        })

    if action == "INTERVENE":
        headline = f"High return risk: {p:.1%} — proactive intervention recommended."
    elif action == "MONITOR":
        headline = f"Medium return risk: {p:.1%} — monitor the order."
    else:
        headline = f"Low return risk: {p:.1%} — no proactive action recommended."

    factor_text = ", ".join(str(x["source_feature"]) for x in positive) or "the supplied SHAP factors"
    explanation = (
        f"The model estimates a {p:.1%} probability of return and classifies "
        f"the order as {risk.lower()} risk. The main model-attribution signals "
        f"increasing risk are {factor_text}. "
        f"The decision engine recommends {action} based on the frozen policy."
    )

    business = (
        f"Expected avoided return cost is ₹{payload['expected_avoided_return_cost']:,.2f} "
        f"against an intervention cost of ₹{payload['intervention_cost']:,.2f}, "
        f"giving an estimated net value of ₹{payload['net_expected_value']:,.2f} "
        f"under the stated intervention-effectiveness assumption."
    )

    return {
        "headline": headline,
        "explanation": explanation,
        "key_factors": key_factors,
        "recommended_action": action,
        "business_context": business,
        "caveat": (
            "SHAP describes model behavior, not causality. Expected economics "
            "use an assumed intervention effectiveness and should be validated "
            "with intervention or experiment data."
        ),
    }


SYSTEM_PROMPT = """You are the explanation layer for an e-commerce return-risk
decision system.

Your job is ONLY to verbalize deterministic inputs supplied by the decision
engine. Do not make or change decisions.

Hard rules:
1. Never change return_probability, risk_level, recommended_action, thresholds,
   or business economics.
2. Use only supplied evidence. Never invent customer, product, operational,
   or causal facts.
3. Treat SHAP as model attribution, not causality.
4. Describe expected economics as model-based estimates under assumptions,
   not realized savings.
5. Keep the explanation concise and useful to an operations manager.
6. The recommended_action in your output MUST exactly match the input.
"""


def openai_explanation(payload: dict[str, Any], schema: dict[str, Any],
                       model_name: str) -> dict[str, Any]:
    try:
        # Import lazily so non-OpenAI modes do not require the optional SDK.
        OpenAI = getattr(__import__("openai", fromlist=["OpenAI"]), "OpenAI")
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI SDK is not installed. Install requirements.txt before using --mode openai."
        ) from exc

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for --mode openai.")

    client = OpenAI()
    response = client.responses.create(
        model=model_name,
        instructions=SYSTEM_PROMPT,
        input=json.dumps(payload, default=str),
        text={
            "format": {
                "type": "json_schema",
                "name": "return_risk_explanation",
                "description": "Structured business explanation of a deterministic return-risk decision.",
                "strict": True,
                "schema": schema,
            }
        },
        store=False,
    )

    text = response.output_text
    if not text:
        raise RuntimeError("OpenAI returned no output text.")

    result = json.loads(text)
    return result


def validate_output(result: dict[str, Any], payload: dict[str, Any]) -> None:
    required = {
        "headline", "explanation", "key_factors",
        "recommended_action", "business_context", "caveat"
    }
    missing = required - result.keys()
    if missing:
        raise ValueError(f"LLM explanation missing fields: {sorted(missing)}")

    if result["recommended_action"] != payload["recommended_action"]:
        raise ValueError(
            "LLM changed the deterministic recommended_action. "
            f"Expected {payload['recommended_action']}, got {result['recommended_action']}."
        )

    if not isinstance(result["key_factors"], list):
        raise ValueError("key_factors must be a list.")

    for factor in result["key_factors"]:
        if factor["direction"] not in {"increases_risk", "reduces_risk"}:
            raise ValueError("Invalid factor direction.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/processed/decision_engine/example_decision.json",
    )
    parser.add_argument(
        "--schema",
        default="src/llm/explanation_schema.json",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/llm_explanations",
    )
    parser.add_argument("--mode", choices=["mock", "openai"], default="mock")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    decision = load_json(root / args.input)
    schema = load_json(root / args.schema)
    payload = build_input_payload(decision)

    if args.mode == "mock":
        result = mock_explanation(payload)
    else:
        result = openai_explanation(payload, schema, args.model)

    validate_output(result, payload)

    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    envelope = {
        "mode": args.mode,
        "model": args.model if args.mode == "openai" else None,
        "order_id": payload["order_id"],
        "source_decision": payload,
        "llm_explanation": result,
    }

    (output_dir / "example_llm_explanation.json").write_text(
        json.dumps(envelope, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "example_llm_explanation.txt").write_text(
        "\n".join([
            result["headline"],
            "",
            result["explanation"],
            "",
            "Key factors:",
            *[
                f"- {x['factor']}: {x['direction']} — {x['impact']}"
                for x in result["key_factors"]
            ],
            "",
            f"Recommended action: {result['recommended_action']}",
            "",
            result["business_context"],
            "",
            f"Caveat: {result['caveat']}",
        ]) + "\n",
        encoding="utf-8",
    )

    print("LLM explanation layer completed.")
    print(f"Mode: {args.mode}")
    print(f"Order: {payload['order_id']}")
    print(f"Action preserved: {result['recommended_action']}")
    print(f"Headline: {result['headline']}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()

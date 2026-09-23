"""
Optimize the return-intervention threshold using business economics.

Usage:
    python src/decision/threshold_optimization.py

Inputs:
    data/processed/xgboost_predictions.csv

Default business assumptions:
    intervention_cost = ₹100 per targeted order
    avoided_return_cost = ₹800 per successfully prevented return
    intervention_effectiveness = 60%

For each threshold, expected net benefit is:

    expected_benefit =
        targeted_orders
        * avoided_return_cost
        * intervention_effectiveness
        * predicted_return_precision

    expected_cost =
        targeted_orders * intervention_cost

    net_benefit = expected_benefit - expected_cost

This is a decision-analysis model, not a causal estimate. The
intervention_effectiveness assumption must eventually be validated with
an experiment or historical intervention data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INTERVENTION_COST = 100.0
DEFAULT_AVOIDED_RETURN_COST = 800.0
DEFAULT_EFFECTIVENESS = 0.60


def calculate_threshold_table(
    predictions: pd.DataFrame,
    intervention_cost: float,
    avoided_return_cost: float,
    effectiveness: float,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """Calculate classification and economic metrics for each threshold."""
    rows = []

    actual_returns = predictions["returned"].astype(int)
    probabilities = predictions["predicted_probability"]

    total_orders = len(predictions)
    total_returns = int(actual_returns.sum())

    for threshold in thresholds:
        targeted = probabilities >= threshold
        predicted_positive = int(targeted.sum())

        tp = int(((targeted) & (actual_returns == 1)).sum())
        fp = int(((targeted) & (actual_returns == 0)).sum())
        fn = int(((~targeted) & (actual_returns == 1)).sum())
        tn = int(((~targeted) & (actual_returns == 0)).sum())

        precision = tp / predicted_positive if predicted_positive else 0.0
        recall = tp / total_returns if total_returns else 0.0

        # Economic model:
        # precision = estimated fraction of targeted orders that would return.
        # effectiveness = assumed fraction of those returns prevented.
        expected_prevented_returns = (
            predicted_positive * precision * effectiveness
        )

        intervention_cost_total = predicted_positive * intervention_cost
        avoided_cost_total = (
            expected_prevented_returns * avoided_return_cost
        )
        net_benefit = avoided_cost_total - intervention_cost_total

        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "targeted_orders": predicted_positive,
                "target_rate": predicted_positive / total_orders,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "precision": precision,
                "recall": recall,
                "expected_prevented_returns": expected_prevented_returns,
                "intervention_cost": intervention_cost_total,
                "avoided_return_cost": avoided_cost_total,
                "net_benefit": net_benefit,
                "roi": (
                    net_benefit / intervention_cost_total
                    if intervention_cost_total
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="data/processed/xgboost_predictions.csv",
    )
    parser.add_argument(
        "--output",
        default="data/processed/threshold_analysis.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="data/processed/decision_summary.json",
    )
    parser.add_argument(
        "--intervention-cost",
        type=float,
        default=DEFAULT_INTERVENTION_COST,
    )
    parser.add_argument(
        "--avoided-return-cost",
        type=float,
        default=DEFAULT_AVOIDED_RETURN_COST,
    )
    parser.add_argument(
        "--effectiveness",
        type=float,
        default=DEFAULT_EFFECTIVENESS,
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.01,
    )
    args = parser.parse_args()

    if args.intervention_cost < 0:
        raise ValueError("intervention cost cannot be negative.")

    if args.avoided_return_cost < 0:
        raise ValueError("avoided return cost cannot be negative.")

    if not 0 <= args.effectiveness <= 1:
        raise ValueError("effectiveness must be between 0 and 1.")

    if args.step <= 0 or args.step > 1:
        raise ValueError("step must be > 0 and <= 1.")

    project_dir = Path(__file__).resolve().parents[2]
    input_path = project_dir / args.predictions
    output_path = project_dir / args.output
    summary_path = project_dir / args.summary_output

    if not input_path.exists():
        raise FileNotFoundError(f"Predictions not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(input_path)

    required = {
        "order_id",
        "returned",
        "predicted_probability",
        "predicted_return",
    }

    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if predictions["order_id"].duplicated().any():
        raise ValueError("Duplicate order IDs found.")

    if (
        (predictions["predicted_probability"] < 0)
        | (predictions["predicted_probability"] > 1)
    ).any():
        raise ValueError("Invalid prediction probabilities found.")

    thresholds = np.round(
        np.arange(0.01, 1.00 + args.step / 2, args.step),
        4,
    )

    analysis = calculate_threshold_table(
        predictions=predictions,
        intervention_cost=args.intervention_cost,
        avoided_return_cost=args.avoided_return_cost,
        effectiveness=args.effectiveness,
        thresholds=thresholds,
    )

    # F1-optimal threshold.
    analysis["f1"] = np.where(
        (analysis["precision"] + analysis["recall"]) > 0,
        2
        * analysis["precision"]
        * analysis["recall"]
        / (analysis["precision"] + analysis["recall"]),
        0,
    )

    best_net = analysis.loc[analysis["net_benefit"].idxmax()]
    best_f1 = analysis.loc[analysis["f1"].idxmax()]

    analysis.to_csv(output_path, index=False)

    summary = {
        "business_assumptions": {
            "intervention_cost": args.intervention_cost,
            "avoided_return_cost": args.avoided_return_cost,
            "intervention_effectiveness": args.effectiveness,
        },
        "test_orders": int(len(predictions)),
        "actual_returns": int(predictions["returned"].sum()),
        "actual_return_rate": float(predictions["returned"].mean()),
        "recommended_threshold_by_net_benefit": float(best_net["threshold"]),
        "max_net_benefit": float(best_net["net_benefit"]),
        "recommended_threshold_by_f1": float(best_f1["threshold"]),
        "max_f1": float(best_f1["f1"]),
        "decision_at_net_benefit_threshold": {
            "targeted_orders": int(best_net["targeted_orders"]),
            "target_rate": float(best_net["target_rate"]),
            "precision": float(best_net["precision"]),
            "recall": float(best_net["recall"]),
            "expected_prevented_returns": float(
                best_net["expected_prevented_returns"]
            ),
            "intervention_cost": float(best_net["intervention_cost"]),
            "avoided_return_cost": float(best_net["avoided_return_cost"]),
            "net_benefit": float(best_net["net_benefit"]),
            "roi": float(best_net["roi"]),
        },
        "note": (
            "Economic results depend on the assumed intervention effectiveness "
            "and avoided return cost. These assumptions are not causal estimates "
            "and should be validated experimentally."
        ),
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Threshold optimization completed.")
    print("\nBusiness assumptions:")
    print(f"Intervention cost: ₹{args.intervention_cost:,.2f}")
    print(f"Avoided return cost: ₹{args.avoided_return_cost:,.2f}")
    print(f"Intervention effectiveness: {args.effectiveness:.0%}")

    print("\nNet-benefit threshold:")
    print(f"Threshold: {best_net['threshold']:.2f}")
    print(f"Targeted orders: {int(best_net['targeted_orders']):,}")
    print(f"Target rate: {best_net['target_rate']:.2%}")
    print(f"Precision: {best_net['precision']:.2%}")
    print(f"Recall: {best_net['recall']:.2%}")
    print(
        f"Expected prevented returns: "
        f"{best_net['expected_prevented_returns']:.1f}"
    )
    print(f"Intervention cost: ₹{best_net['intervention_cost']:,.2f}")
    print(f"Avoided return cost: ₹{best_net['avoided_return_cost']:,.2f}")
    print(f"Net benefit: ₹{best_net['net_benefit']:,.2f}")
    print(f"ROI: {best_net['roi']:.2f}x")

    print("\nF1 threshold:")
    print(f"Threshold: {best_f1['threshold']:.2f}")
    print(f"F1: {best_f1['f1']:.4f}")

    print(f"\nAnalysis: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

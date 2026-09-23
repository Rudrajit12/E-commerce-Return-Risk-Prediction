"""
Select model and intervention threshold using the validation period, then
evaluate the frozen choices once on the untouched test period.

Business assumptions:
    intervention cost = ₹100/order
    avoided return cost = ₹800/return
    intervention effectiveness = 60%

Selection rule:
    maximize realized modeled net benefit on validation data.

Test evaluation uses the selected validation threshold without re-optimizing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss, precision_score, recall_score,
    roc_auc_score,
)


DEFAULT_INTERVENTION_COST = 100.0
DEFAULT_AVOIDED_RETURN_COST = 800.0
DEFAULT_EFFECTIVENESS = 0.60


def threshold_metrics(df: pd.DataFrame, threshold: float,
                      intervention_cost: float,
                      avoided_return_cost: float,
                      effectiveness: float) -> dict:
    y = df["returned"].astype(int).to_numpy()
    p = df["predicted_probability"].to_numpy()
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    targeted = int(pred.sum())

    precision = precision_score(y, pred, zero_division=0)
    recall = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)

    # Retrospective modeled economics: only true returns among targeted
    # orders create avoidable-return value; effectiveness is an assumption.
    expected_prevented = tp * effectiveness
    intervention_total = targeted * intervention_cost
    avoided_total = expected_prevented * avoided_return_cost
    net = avoided_total - intervention_total

    return {
        "threshold": float(threshold),
        "targeted_orders": targeted,
        "target_rate": targeted / len(df),
        "true_positives": int(tp), "false_positives": int(fp),
        "false_negatives": int(fn), "true_negatives": int(tn),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "expected_prevented_returns": float(expected_prevented),
        "intervention_cost": float(intervention_total),
        "avoided_return_cost": float(avoided_total),
        "net_benefit": float(net),
        "roi": float(net / intervention_total) if intervention_total else 0.0,
    }


def probability_metrics(df: pd.DataFrame, threshold: float) -> dict:
    y = df["returned"].astype(int)
    p = df["predicted_probability"]
    pred = (p >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
    }


def analyze_model(df: pd.DataFrame, model_name: str,
                  intervention_cost: float, avoided_return_cost: float,
                  effectiveness: float, thresholds: np.ndarray) -> tuple[pd.DataFrame, dict]:
    rows = []
    for t in thresholds:
        row = threshold_metrics(df, float(t), intervention_cost, avoided_return_cost, effectiveness)
        row["model"] = model_name
        rows.append(row)
    table = pd.DataFrame(rows)
    best = table.loc[table["net_benefit"].idxmax()].to_dict()
    best_f1 = table.loc[table["f1"].idxmax()].to_dict()
    best["selection_basis"] = "maximum validation net benefit"
    best["f1_threshold"] = float(best_f1["threshold"])
    best["f1_at_f1_threshold"] = float(best_f1["f1"])
    return table, best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", default="data/processed/model_selection")
    parser.add_argument("--output-dir", default="data/processed/model_selection")
    parser.add_argument("--intervention-cost", type=float, default=DEFAULT_INTERVENTION_COST)
    parser.add_argument("--avoided-return-cost", type=float, default=DEFAULT_AVOIDED_RETURN_COST)
    parser.add_argument("--effectiveness", type=float, default=DEFAULT_EFFECTIVENESS)
    parser.add_argument("--step", type=float, default=0.01)
    args = parser.parse_args()

    if args.intervention_cost < 0 or args.avoided_return_cost < 0:
        raise ValueError("Costs cannot be negative.")
    if not 0 <= args.effectiveness <= 1:
        raise ValueError("Effectiveness must be between 0 and 1.")

    root = Path(__file__).resolve().parents[2]
    pred_dir = root / args.predictions_dir
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = np.round(np.arange(0.01, 1.0001, args.step), 4)
    all_validation = []
    selection = {}

    for model in ["logistic_regression", "xgboost"]:
        val = pd.read_csv(pred_dir / f"{model}_validation_predictions.csv")
        test = pd.read_csv(pred_dir / f"{model}_test_predictions.csv")

        val_table, best = analyze_model(
            val, model, args.intervention_cost, args.avoided_return_cost,
            args.effectiveness, thresholds
        )
        val_table.to_csv(out_dir / f"{model}_validation_thresholds.csv", index=False)
        all_validation.append(val_table)

        selected_threshold = float(best["threshold"])
        test_decision = threshold_metrics(
            test, selected_threshold, args.intervention_cost,
            args.avoided_return_cost, args.effectiveness
        )
        test_prob = probability_metrics(test, selected_threshold)

        selection[model] = {
            "validation": best,
            "test_at_frozen_threshold": {**test_decision, **test_prob},
        }

    validation_table = pd.concat(all_validation, ignore_index=True)
    validation_table.to_csv(out_dir / "validation_threshold_comparison.csv", index=False)

    # Production candidate is selected solely by validation net benefit.
    selected_model = max(
        selection,
        key=lambda m: selection[m]["validation"]["net_benefit"]
    )
    selected_threshold = selection[selected_model]["validation"]["threshold"]

    model_comparison = []
    for model, result in selection.items():
        model_comparison.append({
            "model": model,
            "validation_best_threshold": result["validation"]["threshold"],
            "validation_net_benefit": result["validation"]["net_benefit"],
            "validation_precision": result["validation"]["precision"],
            "validation_recall": result["validation"]["recall"],
            "validation_f1": result["validation"]["f1"],
            "test_frozen_threshold": result["test_at_frozen_threshold"]["threshold"],
            "test_net_benefit": result["test_at_frozen_threshold"]["net_benefit"],
            "test_precision": result["test_at_frozen_threshold"]["precision"],
            "test_recall": result["test_at_frozen_threshold"]["recall"],
            "test_f1": result["test_at_frozen_threshold"]["f1"],
            "test_roc_auc": result["test_at_frozen_threshold"]["roc_auc"],
            "test_pr_auc": result["test_at_frozen_threshold"]["pr_auc"],
            "test_log_loss": result["test_at_frozen_threshold"]["log_loss"],
            "test_brier_score": result["test_at_frozen_threshold"]["brier_score"],
        })
    comparison = pd.DataFrame(model_comparison)
    comparison.to_csv(out_dir / "model_selection_comparison.csv", index=False)

    summary = {
        "business_assumptions": {
            "intervention_cost": args.intervention_cost,
            "avoided_return_cost": args.avoided_return_cost,
            "intervention_effectiveness": args.effectiveness,
        },
        "selection_rule": "Choose the model whose validation-period optimized threshold produces the highest modeled net benefit; freeze that model and threshold before final test evaluation.",
        "selected_model": selected_model,
        "selected_threshold": selected_threshold,
        "model_results": selection,
        "test_evaluation_note": "The test threshold is frozen from validation. No test-set threshold optimization was used.",
        "effectiveness_note": "Intervention effectiveness is an explicit business assumption, not a causal estimate; validate with an experiment or historical intervention data.",
    }
    (out_dir / "model_selection_summary.json").write_text(json.dumps(summary, indent=2))

    print("Model selection completed.")
    print(f"Selected model: {selected_model}")
    print(f"Frozen threshold: {selected_threshold:.2f}")
    print("\nValidation net benefit:")
    for model in selection:
        print(f"  {model}: ₹{selection[model]['validation']['net_benefit']:,.2f}")
    print("\nFinal test evaluation at frozen thresholds:")
    for model in selection:
        t = selection[model]["test_at_frozen_threshold"]
        print(
            f"  {model}: threshold={t['threshold']:.2f}, "
            f"net benefit=₹{t['net_benefit']:,.2f}, "
            f"precision={t['precision']:.2%}, recall={t['recall']:.2%}, "
            f"F1={t['f1']:.4f}"
        )


if __name__ == "__main__":
    main()

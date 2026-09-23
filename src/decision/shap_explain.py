"""
Step 10: SHAP explainability for the selected Logistic Regression model.

Explains the frozen production candidate selected in Step 9:
    model: models/model_selection/logistic_regression.joblib
    threshold: 0.21

Outputs:
    shap_global_importance.csv
    shap_local_explanations.csv
    shap_summary.json
    shap_explanation_report.txt

The model is a sklearn Pipeline. SHAP is applied to the transformed feature
matrix used by Logistic Regression. Contributions are therefore expressed in
log-odds space, which is the native additive space of logistic regression.

No target-derived or post-order information is introduced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


DEFAULT_THRESHOLD = 0.21
RANDOM_STATE = 42

FEATURES = [
    "product_price", "product_rating", "product_historical_return_rate",
    "customer_age", "customer_previous_orders", "customer_previous_returns",
    "customer_return_rate", "quantity", "discount_pct",
    "delivery_distance_km", "expected_delivery_days", "order_month",
    "order_day_of_week",
    "category", "payment_type", "discount_bucket", "price_bucket",
]


def sigmoid(x):
    x = np.clip(x, -700, 700)
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="models/model_selection/logistic_regression.joblib",
    )
    parser.add_argument(
        "--train",
        default="data/processed/train_model_selection.csv",
    )
    parser.add_argument(
        "--test",
        default="data/processed/test_model_selection.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/explainability",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--background-size", type=int, default=1000)
    parser.add_argument("--explanation-size", type=int, default=2000)
    parser.add_argument("--top-features", type=int, default=15)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    model_path = root / args.model
    train_path = root / args.train
    test_path = root / args.test
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model = joblib.load(model_path)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    missing = [f for f in FEATURES if f not in train.columns or f not in test.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    X_train = train[FEATURES].copy()
    X_test = test[FEATURES].copy()

    # Deterministic samples for reproducibility.
    rng = np.random.default_rng(RANDOM_STATE)
    bg_n = min(args.background_size, len(X_train))
    ex_n = min(args.explanation_size, len(X_test))

    bg_idx = rng.choice(len(X_train), size=bg_n, replace=False)
    ex_idx = rng.choice(len(X_test), size=ex_n, replace=False)

    background = X_train.iloc[bg_idx].copy()
    explain_df = test.iloc[ex_idx].copy()
    X_background = background[FEATURES]
    X_explain = explain_df[FEATURES]

    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]

    X_background_t = preprocessor.transform(X_background)
    X_explain_t = preprocessor.transform(X_explain)
    feature_names = list(preprocessor.get_feature_names_out())

    # LinearExplainer gives additive contributions in logistic regression's
    # native log-odds space.
    explainer = shap.LinearExplainer(estimator, X_background_t)
    shap_values = explainer.shap_values(X_explain_t)

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_values = np.asarray(shap_values)

    if shap_values.ndim != 2:
        raise RuntimeError(f"Unexpected SHAP shape: {shap_values.shape}")

    # Native model output (log-odds) for additivity validation.
    linear_output = estimator.decision_function(X_explain_t)
    expected_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    reconstructed = expected_value + shap_values.sum(axis=1)
    max_additivity_error = float(np.max(np.abs(reconstructed - linear_output)))
    if max_additivity_error > 1e-5:
        raise RuntimeError(
            f"SHAP additivity check failed: max error={max_additivity_error}"
        )

    probabilities = sigmoid(linear_output)
    predictions = (probabilities >= args.threshold).astype(int)

    # Global importance across the deterministic explanation sample.
    global_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap_log_odds": np.mean(np.abs(shap_values), axis=0),
        "mean_shap_log_odds": np.mean(shap_values, axis=0),
    })
    global_df["rank"] = (
        global_df["mean_abs_shap_log_odds"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    global_df = global_df.sort_values(
        ["mean_abs_shap_log_odds", "feature"],
        ascending=[False, True],
    ).reset_index(drop=True)
    global_df.to_csv(output_dir / "shap_global_importance.csv", index=False)

    # Select the highest-risk order in the explanation sample.
    selected_pos = int(np.argmax(probabilities))
    selected_order = explain_df.iloc[selected_pos]
    selected_shap = shap_values[selected_pos]

    # User-friendly local explanation. For one-hot encoded features, retain
    # the transformed feature name but also expose its base source column.
    rows = []
    order_id = selected_order["order_id"]
    for idx in np.argsort(np.abs(selected_shap))[::-1][:args.top_features]:
        transformed_name = feature_names[idx]
        base_feature = transformed_name.split("_", 1)[0] if "_" in transformed_name else transformed_name
        value = None

        # Try to recover a human-readable value from the original row.
        if base_feature in explain_df.columns:
            value = selected_order[base_feature]

        contribution = float(selected_shap[idx])
        rows.append({
            "order_id": order_id,
            "predicted_probability": float(probabilities[selected_pos]),
            "decision_threshold": float(args.threshold),
            "predicted_return": int(predictions[selected_pos]),
            "feature": transformed_name,
            "source_feature": base_feature,
            "feature_value": value,
            "shap_value_log_odds": contribution,
            "direction": "increases_return_risk" if contribution > 0 else "decreases_return_risk",
            "absolute_shap": abs(contribution),
        })

    local_df = pd.DataFrame(rows)
    local_df.to_csv(output_dir / "shap_local_explanations.csv", index=False)

    # A compact explanation JSON suitable for a future LLM layer.
    positive = local_df[local_df["shap_value_log_odds"] > 0].head(10)
    negative = local_df[local_df["shap_value_log_odds"] < 0].head(10)

    summary = {
        "step": 10,
        "purpose": "Explain the selected production return-risk model.",
        "model": str(args.model),
        "model_type": "Logistic Regression",
        "threshold": float(args.threshold),
        "background_size": int(bg_n),
        "explanation_sample_size": int(ex_n),
        "random_state": RANDOM_STATE,
        "shap_version": shap.__version__,
        "explanation_space": "log_odds",
        "expected_log_odds": expected_value,
        "expected_probability": float(sigmoid(expected_value)),
        "max_additivity_error": max_additivity_error,
        "selected_order": {
            "order_id": order_id,
            "actual_returned": int(selected_order["returned"]),
            "predicted_probability": float(probabilities[selected_pos]),
            "predicted_return": int(predictions[selected_pos]),
            "above_threshold": bool(probabilities[selected_pos] >= args.threshold),
        },
        "top_global_features": global_df.head(15).to_dict(orient="records"),
        "top_positive_local_contributions": positive.to_dict(orient="records"),
        "top_negative_local_contributions": negative.to_dict(orient="records"),
        "llm_ready_fields": [
            "order_id",
            "predicted_probability",
            "decision_threshold",
            "feature",
            "source_feature",
            "feature_value",
            "shap_value_log_odds",
            "direction",
        ],
    }
    (output_dir / "shap_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    # Human-readable report.
    report = []
    report.append("STEP 10 — SHAP EXPLAINABILITY REPORT")
    report.append("=" * 42)
    report.append(f"Model: Logistic Regression ({args.model})")
    report.append(f"Decision threshold: {args.threshold:.2f}")
    report.append(f"Explanation sample: {ex_n:,} test orders")
    report.append(f"Background sample: {bg_n:,} training orders")
    report.append("")
    report.append("Global feature importance (mean absolute SHAP, log-odds):")
    for _, row in global_df.head(10).iterrows():
        report.append(
            f"  {int(row['rank']):>2}. {row['feature']}: "
            f"{row['mean_abs_shap_log_odds']:.6f}"
        )
    report.append("")
    report.append("Selected highest-risk order:")
    report.append(f"  Order ID: {order_id}")
    report.append(f"  Actual returned: {int(selected_order['returned'])}")
    report.append(f"  Predicted probability: {probabilities[selected_pos]:.2%}")
    report.append(f"  Predicted return at threshold: {int(predictions[selected_pos])}")
    report.append("")
    report.append("Top local contributions:")
    for _, row in local_df.iterrows():
        report.append(
            f"  {row['feature']}: {row['shap_value_log_odds']:+.6f} "
            f"({row['direction']})"
        )
    report.append("")
    report.append(
        "Interpretation: positive SHAP values increase the model's return-risk "
        "log-odds; negative values decrease them. SHAP describes model behavior, "
        "not causality."
    )
    report.append(
        f"Additivity check: maximum absolute error = {max_additivity_error:.12g}"
    )
    (output_dir / "shap_explanation_report.txt").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print("SHAP explainability completed.")
    print(f"Selected order: {order_id}")
    print(f"Predicted return probability: {probabilities[selected_pos]:.2%}")
    print(f"Decision threshold: {args.threshold:.2f}")
    print(f"Predicted return: {int(predictions[selected_pos])}")
    print(f"Max SHAP additivity error: {max_additivity_error:.12g}")
    print("\nTop global features:")
    print(global_df.head(10).to_string(index=False))
    print("\nTop local contributions:")
    print(local_df.to_string(index=False))
    print(f"\nOutputs: {output_dir}")


if __name__ == "__main__":
    main()

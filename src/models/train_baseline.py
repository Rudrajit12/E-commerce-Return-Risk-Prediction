"""
Train and evaluate the Logistic Regression baseline.

Usage:
    python src/models/train_baseline.py

Inputs:
    data/processed/train.csv
    data/processed/test.csv

Outputs:
    models/logistic_regression_baseline.joblib
    data/processed/baseline_predictions.csv
    data/processed/baseline_metrics.json
    data/processed/baseline_coefficients.csv

Important:
- Preprocessing is fitted only on the training data.
- Test data is used only for final evaluation.
- The model predicts return probability at order placement time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "returned"

ID_COLUMNS = [
    "order_id",
    "customer_id",
    "product_id",
]

NUMERIC_FEATURES = [
    "product_price",
    "product_rating",
    "product_historical_return_rate",
    "customer_age",
    "customer_previous_orders",
    "customer_previous_returns",
    "customer_return_rate",
    "quantity",
    "discount_pct",
    "delivery_distance_km",
    "expected_delivery_days",
    "order_month",
    "order_day_of_week",
]

CATEGORICAL_FEATURES = [
    "category",
    "payment_type",
    "discount_bucket",
    "price_bucket",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_pipeline() -> Pipeline:
    """Build a leakage-safe preprocessing + Logistic Regression pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> tuple[dict, np.ndarray]:
    """Calculate classification and probability-quality metrics."""
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "log_loss": float(log_loss(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "positive_rate_actual": float(y_test.mean()),
        "positive_rate_predicted": float(predictions.mean()),
    }

    return metrics, probabilities


def save_coefficients(model: Pipeline, output_path: Path) -> pd.DataFrame:
    """Save interpretable standardized Logistic Regression coefficients."""
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]

    coefficient_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "odds_ratio": np.exp(coefficients),
            "absolute_coefficient": np.abs(coefficients),
        }
    ).sort_values(
        "absolute_coefficient",
        ascending=False,
    ).reset_index(drop=True)

    coefficient_df.to_csv(output_path, index=False)
    return coefficient_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument(
        "--model-output",
        default="models/logistic_regression_baseline.joblib",
    )
    parser.add_argument(
        "--predictions-output",
        default="data/processed/baseline_predictions.csv",
    )
    parser.add_argument(
        "--metrics-output",
        default="data/processed/baseline_metrics.json",
    )
    parser.add_argument(
        "--coefficients-output",
        default="data/processed/baseline_coefficients.csv",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[2]

    train_path = project_dir / args.train
    test_path = project_dir / args.test
    model_path = project_dir / args.model_output
    predictions_path = project_dir / args.predictions_output
    metrics_path = project_dir / args.metrics_output
    coefficients_path = project_dir / args.coefficients_output

    for path in [train_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")

    for path in [
        model_path,
        predictions_path,
        metrics_path,
        coefficients_path,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(train_path, parse_dates=["order_date"])
    test = pd.read_csv(test_path, parse_dates=["order_date"])

    missing_features = sorted(set(FEATURES) - set(train.columns))
    if missing_features:
        raise ValueError(f"Missing model features: {missing_features}")

    if set(train[ID_COLUMNS].columns) != set(ID_COLUMNS):
        raise ValueError("Required identifiers missing from train data.")

    X_train = train[FEATURES]
    y_train = train[TARGET].astype(int)

    X_test = test[FEATURES]
    y_test = test[TARGET].astype(int)

    # Fit ALL preprocessing only on training data.
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    metrics, probabilities = evaluate_model(
        pipeline,
        X_test,
        y_test,
        threshold=0.5,
    )

    predictions = (probabilities >= 0.5).astype(int)

    prediction_df = test[
        ["order_id", "customer_id", "product_id", "order_date", TARGET]
    ].copy()

    prediction_df["predicted_probability"] = probabilities
    prediction_df["predicted_return"] = predictions

    prediction_df.to_csv(predictions_path, index=False)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    coefficient_df = save_coefficients(
        pipeline,
        coefficients_path,
    )

    joblib.dump(pipeline, model_path)

    print("Logistic Regression baseline trained successfully.")
    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print("\nTest metrics @ threshold 0.50:")
    for key in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "log_loss",
        "brier_score",
    ]:
        print(f"{key:22s}: {metrics[key]:.4f}")

    print("\nConfusion matrix:")
    print(
        f"TN={metrics['true_negative']}, "
        f"FP={metrics['false_positive']}, "
        f"FN={metrics['false_negative']}, "
        f"TP={metrics['true_positive']}"
    )

    print("\nTop positive/negative standardized coefficients:")
    print(coefficient_df.head(10)[["feature", "coefficient", "odds_ratio"]].to_string(index=False))

    print(f"\nModel: {model_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Coefficients: {coefficients_path}")


if __name__ == "__main__":
    main()

"""
Train and evaluate the XGBoost return prediction model.

Usage:
    python src/models/train_xgboost.py

Inputs:
    data/processed/train.csv
    data/processed/test.csv

Outputs:
    models/xgboost_return_model.joblib
    data/processed/xgboost_predictions.csv
    data/processed/xgboost_metrics.json
    data/processed/xgboost_feature_importance.csv

Design:
- Chronological train/test split is inherited from Step 5.
- Preprocessing is fitted only on training data.
- Categorical variables are one-hot encoded.
- The test set is used only for final evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
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
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


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


def build_pipeline(scale_pos_weight: float) -> Pipeline:
    """Build preprocessing + XGBoost pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = XGBClassifier(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=2,
        scale_pos_weight=scale_pos_weight,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> tuple[dict, np.ndarray]:
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


def save_feature_importance(
    model: Pipeline,
    output_path: Path,
) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importance = classifier.feature_importances_

    result = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importance,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    result["importance_share"] = (
        result["importance"] / result["importance"].sum()
    )

    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument(
        "--model-output",
        default="models/xgboost_return_model.joblib",
    )
    parser.add_argument(
        "--predictions-output",
        default="data/processed/xgboost_predictions.csv",
    )
    parser.add_argument(
        "--metrics-output",
        default="data/processed/xgboost_metrics.json",
    )
    parser.add_argument(
        "--importance-output",
        default="data/processed/xgboost_feature_importance.csv",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[2]

    train_path = project_dir / args.train
    test_path = project_dir / args.test
    model_path = project_dir / args.model_output
    predictions_path = project_dir / args.predictions_output
    metrics_path = project_dir / args.metrics_output
    importance_path = project_dir / args.importance_output

    train = pd.read_csv(train_path, parse_dates=["order_date"])
    test = pd.read_csv(test_path, parse_dates=["order_date"])

    X_train = train[FEATURES]
    y_train = train[TARGET].astype(int)

    X_test = test[FEATURES]
    y_test = test[TARGET].astype(int)

    # Training-data-only class weighting.
    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    scale_pos_weight = negative / positive

    pipeline = build_pipeline(scale_pos_weight)
    pipeline.fit(X_train, y_train)

    metrics, probabilities = evaluate(
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

    model_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    prediction_df.to_csv(predictions_path, index=False)

    metrics["scale_pos_weight"] = float(scale_pos_weight)
    metrics["xgboost_version"] = __import__("xgboost").__version__

    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    importance_df = save_feature_importance(
        pipeline,
        importance_path,
    )

    joblib.dump(pipeline, model_path)

    print("XGBoost model trained successfully.")
    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Scale positive weight: {scale_pos_weight:.4f}")
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

    print("\nTop features:")
    print(importance_df.head(15).to_string(index=False))

    print(f"\nModel: {model_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Importance: {importance_path}")


if __name__ == "__main__":
    main()

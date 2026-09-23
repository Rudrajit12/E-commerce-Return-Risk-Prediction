"""
Train Logistic Regression and XGBoost on the training period and generate
probabilities for validation and final test periods.

This script intentionally fits preprocessing/models only on the training
partition. Validation is used for model + threshold selection; test is held
out for final evaluation.
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


TARGET = "returned"

NUMERIC_FEATURES = [
    "product_price", "product_rating", "product_historical_return_rate",
    "customer_age", "customer_previous_orders", "customer_previous_returns",
    "customer_return_rate", "quantity", "discount_pct",
    "delivery_distance_km", "expected_delivery_days", "order_month",
    "order_day_of_week",
]
CATEGORICAL_FEATURES = [
    "category", "payment_type", "discount_bucket", "price_bucket",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_logistic() -> Pipeline:
    return Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)),
    ])


def make_xgboost(scale_pos_weight: float) -> Pipeline:
    return Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", XGBClassifier(
            n_estimators=350, max_depth=4, learning_rate=0.05,
            min_child_weight=3, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.05, reg_lambda=1.0, tree_method="hist",
            random_state=42, n_jobs=2, scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
        )),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train_model_selection.csv")
    parser.add_argument("--validation", default="data/processed/validation_model_selection.csv")
    parser.add_argument("--test", default="data/processed/test_model_selection.csv")
    parser.add_argument("--output-dir", default="data/processed/model_selection")
    parser.add_argument("--model-dir", default="models/model_selection")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    train = pd.read_csv(root / args.train)
    validation = pd.read_csv(root / args.validation)
    test = pd.read_csv(root / args.test)

    X_train, y_train = train[FEATURES], train[TARGET].astype(int)
    X_val, y_val = validation[FEATURES], validation[TARGET].astype(int)
    X_test, y_test = test[FEATURES], test[TARGET].astype(int)

    output_dir = root / args.output_dir
    model_dir = root / args.model_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "logistic_regression": make_logistic(),
        "xgboost": make_xgboost(float((y_train == 0).sum() / (y_train == 1).sum())),
    }

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        val_prob = pipeline.predict_proba(X_val)[:, 1]
        test_prob = pipeline.predict_proba(X_test)[:, 1]

        for split_name, frame, probs in [
            ("validation", validation, val_prob),
            ("test", test, test_prob),
        ]:
            out = frame[["order_id", "order_date", "returned"]].copy()
            out["predicted_probability"] = probs
            out.to_csv(output_dir / f"{name}_{split_name}_predictions.csv", index=False)

        joblib.dump(pipeline, model_dir / f"{name}.joblib")

    metadata = {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "features": FEATURES,
        "models": list(models.keys()),
        "random_state": 42,
        "xgboost_scale_pos_weight": float((y_train == 0).sum() / (y_train == 1).sum()),
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2))

    print("Both candidate models trained successfully.")
    print(f"Outputs: {output_dir}")
    print(f"Models: {model_dir}")


if __name__ == "__main__":
    main()

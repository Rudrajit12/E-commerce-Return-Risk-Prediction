"""
Build the model-ready feature dataset for e-commerce return prediction.

Usage:
    python src/features/build_features.py

Input:
    data/processed/orders_clean_reproducible.csv

Outputs:
    data/processed/model_features.csv
    data/processed/feature_metadata.json

Design rules:
- Features must be available at order time.
- Identifier columns are retained for traceability but are not model features.
- No target-derived or post-event variables are engineered.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


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

# Explicit allowlist protects the model from accidentally consuming
# identifiers or future/post-event columns.
ALLOWED_MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the model-ready dataset without introducing target leakage."""
    required = set(ID_COLUMNS + ["order_date", TARGET] + ALLOWED_MODEL_FEATURES)
    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    features = df.copy()

    # Recalculate order-time derived variables from authoritative fields.
    denominator = features["customer_previous_orders"].replace(0, pd.NA)
    features["customer_return_rate"] = pd.to_numeric(
        features["customer_previous_returns"] / denominator,
        errors="coerce",
    ).fillna(0).clip(0, 1)

    features["discount_bucket"] = pd.cut(
        features["discount_pct"],
        bins=[-0.01, 10, 25, 40, float("inf")],
        labels=["Low", "Medium", "High", "Very High"],
    ).astype(str)

    features["price_bucket"] = pd.cut(
        features["product_price"],
        bins=[0, 500, 1500, 5000, float("inf")],
        labels=["Budget", "Mid", "Premium", "Luxury"],
    ).astype(str)

    features["order_month"] = features["order_date"].dt.month
    features["order_day_of_week"] = features["order_date"].dt.dayofweek

    # Preserve identifiers and date for traceability/splitting.
    output_columns = ID_COLUMNS + ["order_date"] + ALLOWED_MODEL_FEATURES + [TARGET]

    output = (
        features[output_columns]
        .sort_values(["order_date", "order_id"])
        .reset_index(drop=True)
    )

    # Hard leakage guard.
    forbidden_terms = {
        "return_reason",
        "return_requested_date",
        "refund_amount",
        "actual_delivery_days",
        "delivered_date",
        "post_delivery_rating",
        "refund_date",
        "return_date",
    }
    leaked = forbidden_terms.intersection(output.columns)
    if leaked:
        raise ValueError(f"Leakage columns found in output: {sorted(leaked)}")

    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/processed/orders_clean.csv",
    )
    parser.add_argument(
        "--output",
        default="data/processed/model_features.csv",
    )
    parser.add_argument(
        "--metadata",
        default="data/processed/feature_metadata.json",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[2]
    input_path = project_dir / args.input
    output_path = project_dir / args.output
    metadata_path = project_dir / args.metadata

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_path, parse_dates=["order_date"])
    model_df = build_features(raw)

    model_df.to_csv(output_path, index=False)

    metadata = {
        "target": TARGET,
        "id_columns": ID_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "model_features": ALLOWED_MODEL_FEATURES,
        "prediction_point": "Immediately after order placement",
        "leakage_policy": "Only information available at order time may be used.",
        "rows": len(model_df),
        "columns": len(model_df.columns),
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("Feature engineering completed successfully.")
    print(f"Rows: {len(model_df):,}")
    print(f"Model features: {len(ALLOWED_MODEL_FEATURES)}")
    print(f"Numeric features: {len(NUMERIC_FEATURES)}")
    print(f"Categorical features: {len(CATEGORICAL_FEATURES)}")
    print(f"Missing cells: {int(model_df.isna().sum().sum()):,}")
    print(f"Output: {output_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()

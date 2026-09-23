"""
Clean the validated e-commerce return dataset.

Usage:
    python src/data/cleaning.py

Input:
    data/raw/orders.csv

Output:
    data/processed/orders_clean.csv

The raw dataset is never modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = [
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

CATEGORICAL_COLUMNS = [
    "category",
    "payment_type",
    "discount_bucket",
    "price_bucket",
]

REQUIRED_COLUMNS = [
    "order_id",
    "order_date",
    "customer_id",
    "product_id",
    "returned",
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw dataset."""
    cleaned = df.copy()

    # Standardize date.
    cleaned["order_date"] = pd.to_datetime(
        cleaned["order_date"],
        errors="coerce",
    )

    # Remove exact duplicate rows.
    cleaned = cleaned.drop_duplicates().copy()

    # Remove duplicate order IDs while keeping the first occurrence.
    cleaned = cleaned.drop_duplicates(
        subset=["order_id"],
        keep="first",
    ).copy()

    # Required identifiers/target should never be imputed.
    cleaned = cleaned.dropna(subset=REQUIRED_COLUMNS).copy()

    # Numeric coercion.
    for column in NUMERIC_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(
                cleaned[column],
                errors="coerce",
            )

    # Median imputation for continuous/numeric fields.
    for column in NUMERIC_COLUMNS:
        if column not in cleaned.columns:
            continue

        if cleaned[column].isna().any():
            median_value = cleaned[column].median()
            cleaned[column] = cleaned[column].fillna(median_value)

    # Mode imputation for categorical fields.
    for column in CATEGORICAL_COLUMNS:
        if column not in cleaned.columns:
            continue

        if cleaned[column].isna().any():
            mode = cleaned[column].mode(dropna=True)
            if not mode.empty:
                cleaned[column] = cleaned[column].fillna(mode.iloc[0])

    # Recalculate derived fields from cleaned values.
    # This prevents stale derived values after imputation.
    cleaned["customer_return_rate"] = (
        cleaned["customer_previous_returns"]
        / cleaned["customer_previous_orders"].replace(0, np.nan)
    ).fillna(0).clip(0, 1).round(4)

    cleaned["discount_bucket"] = pd.cut(
        cleaned["discount_pct"],
        bins=[-0.01, 10, 25, 40, float("inf")],
        labels=["Low", "Medium", "High", "Very High"],
    ).astype(str)

    cleaned["price_bucket"] = pd.cut(
        cleaned["product_price"],
        bins=[0, 500, 1500, 5000, float("inf")],
        labels=["Budget", "Mid", "Premium", "Luxury"],
    ).astype(str)

    # Recreate calendar features from the authoritative date.
    cleaned["order_month"] = cleaned["order_date"].dt.month
    cleaned["order_day_of_week"] = cleaned["order_date"].dt.dayofweek

    # Enforce useful dtypes.
    cleaned["returned"] = cleaned["returned"].astype(int)
    cleaned["quantity"] = cleaned["quantity"].astype(int)
    cleaned["expected_delivery_days"] = (
        cleaned["expected_delivery_days"].round().astype(int)
    )
    cleaned["order_month"] = cleaned["order_month"].astype(int)
    cleaned["order_day_of_week"] = cleaned["order_day_of_week"].astype(int)

    # Final ordering.
    cleaned = cleaned.sort_values(
        ["order_date", "order_id"]
    ).reset_index(drop=True)

    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/raw/orders.csv",
        help="Path to raw CSV",
    )
    parser.add_argument(
        "--output",
        default="data/processed/orders_clean.csv",
        help="Path to cleaned CSV",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[2]
    input_path = project_dir / args.input
    output_path = project_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw = pd.read_csv(input_path)
    cleaned = clean_data(raw)
    cleaned.to_csv(output_path, index=False)

    print("Cleaning completed successfully.")
    print(f"Raw rows:     {len(raw):,}")
    print(f"Cleaned rows: {len(cleaned):,}")
    print(f"Rows removed: {len(raw) - len(cleaned):,}")
    print(f"Missing cells before: {int(raw.isna().sum().sum()):,}")
    print(f"Missing cells after:  {int(cleaned.isna().sum().sum()):,}")
    print(f"Return rate: {cleaned['returned'].mean():.2%}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()

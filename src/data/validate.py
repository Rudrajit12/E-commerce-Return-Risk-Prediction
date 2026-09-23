"""
Validate the raw e-commerce return dataset.

Usage:
    python src/data/validate.py

The validator performs structural, quality, range, consistency, target,
and leakage checks. It does not modify the raw dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = [
    "order_id",
    "order_date",
    "customer_id",
    "product_id",
    "category",
    "product_price",
    "product_rating",
    "product_historical_return_rate",
    "customer_age",
    "customer_previous_orders",
    "customer_previous_returns",
    "customer_return_rate",
    "quantity",
    "discount_pct",
    "payment_type",
    "delivery_distance_km",
    "expected_delivery_days",
    "discount_bucket",
    "price_bucket",
    "order_month",
    "order_day_of_week",
    "returned",
]

EXPECTED_CATEGORIES = {
    "category": {
        "Fashion",
        "Electronics",
        "Home & Kitchen",
        "Beauty",
        "Footwear",
        "Sports",
        "Books",
    },
    "payment_type": {
        "Credit Card",
        "Debit Card",
        "UPI",
        "COD",
        "Wallet",
    },
    "discount_bucket": {"Low", "Medium", "High", "Very High"},
    "price_bucket": {"Budget", "Mid", "Premium", "Luxury"},
}

# Anything in this set would be suspicious for an order-time prediction model.
POST_EVENT_OR_LEAKAGE_COLUMNS = {
    "return_reason",
    "return_requested_date",
    "refund_amount",
    "actual_delivery_days",
    "delivered_date",
    "post_delivery_rating",
    "refund_date",
    "return_date",
}

RANGE_RULES = {
    "product_price": (0, None),
    "product_rating": (0, 5),
    "product_historical_return_rate": (0, 1),
    "customer_age": (0, 120),
    "customer_previous_orders": (0, None),
    "customer_previous_returns": (0, None),
    "customer_return_rate": (0, 1),
    "quantity": (1, None),
    "discount_pct": (0, 100),
    "delivery_distance_km": (0, None),
    "expected_delivery_days": (1, None),
    "order_month": (1, 12),
    "order_day_of_week": (0, 6),
    "returned": (0, 1),
}


def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of validation errors. Empty list means PASS."""
    errors: list[str] = []

    # Schema
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    unexpected_columns = sorted(set(df.columns) - set(EXPECTED_COLUMNS))

    if missing_columns:
        errors.append(f"Missing columns: {missing_columns}")

    if unexpected_columns:
        errors.append(f"Unexpected columns: {unexpected_columns}")

    # Duplicate primary key
    if "order_id" in df.columns:
        duplicate_orders = int(df["order_id"].duplicated().sum())
        if duplicate_orders:
            errors.append(f"Duplicate order_id values: {duplicate_orders}")

    # Required identifiers
    for column in ["order_id", "customer_id", "product_id"]:
        if column in df.columns and df[column].isna().any():
            errors.append(f"Null values found in required identifier: {column}")

    # Date
    if "order_date" in df.columns:
        parsed_dates = pd.to_datetime(df["order_date"], errors="coerce")
        invalid_dates = int(parsed_dates.isna().sum())
        if invalid_dates:
            errors.append(f"Invalid order_date values: {invalid_dates}")

    # Numeric ranges
    for column, (lower, upper) in RANGE_RULES.items():
        if column not in df.columns:
            continue

        numeric = pd.to_numeric(df[column], errors="coerce")

        if lower is not None:
            violations = int((numeric.dropna() < lower).sum())
            if violations:
                errors.append(
                    f"{column}: {violations} values below minimum {lower}"
                )

        if upper is not None:
            violations = int((numeric.dropna() > upper).sum())
            if violations:
                errors.append(
                    f"{column}: {violations} values above maximum {upper}"
                )

    # Logical customer history
    if {
        "customer_previous_returns",
        "customer_previous_orders",
    }.issubset(df.columns):
        invalid_history = (
            df["customer_previous_returns"]
            > df["customer_previous_orders"]
        ).sum()
        if invalid_history:
            errors.append(
                "customer_previous_returns exceeds "
                f"customer_previous_orders in {invalid_history} rows"
            )

    # Categorical values
    for column, allowed in EXPECTED_CATEGORIES.items():
        if column not in df.columns:
            continue

        observed = set(df[column].dropna().unique())
        invalid = sorted(observed - allowed)

        if invalid:
            errors.append(f"{column}: unexpected values {invalid}")

    # Target
    if "returned" in df.columns:
        target_values = set(df["returned"].dropna().unique())
        if not target_values.issubset({0, 1}):
            errors.append(
                f"Target 'returned' contains invalid values: {sorted(target_values)}"
            )

    # Leakage
    leakage_columns = sorted(
        POST_EVENT_OR_LEAKAGE_COLUMNS.intersection(df.columns)
    )
    if leakage_columns:
        errors.append(
            "Potential post-event/leakage columns detected: "
            f"{leakage_columns}"
        )

    return errors


def build_report(df: pd.DataFrame, errors: list[str]) -> str:
    """Build a human-readable validation report."""
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    report = [
        "E-COMMERCE RETURN DATASET VALIDATION REPORT",
        "=" * 48,
        f"Rows: {len(df):,}",
        f"Columns: {len(df.columns):,}",
        f"Duplicate order IDs: {df['order_id'].duplicated().sum():,}",
        f"Total missing cells: {int(df.isna().sum().sum()):,}",
        f"Return rate: {df['returned'].mean():.2%}",
        f"Validation status: {'PASS' if not errors else 'FAIL'}",
        "",
        "Missing values:",
    ]

    if missing.empty:
        report.append("  None")
    else:
        for column, count in missing.items():
            report.append(
                f"  {column}: {int(count):,} ({count / len(df):.2%})"
            )

    report.extend(["", "Errors:"])
    if errors:
        report.extend(f"  - {error}" for error in errors)
    else:
        report.append("  None")

    return "\n".join(report)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/raw/orders.csv",
        help="Path to raw CSV",
    )
    parser.add_argument(
        "--report",
        default="data/processed/validation_report.txt",
        help="Path to validation report",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[2]
    input_path = project_dir / args.input
    report_path = project_dir / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return 1

    df = pd.read_csv(input_path)
    errors = validate(df)
    report = build_report(df, errors)
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nReport saved to: {report_path}")

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())

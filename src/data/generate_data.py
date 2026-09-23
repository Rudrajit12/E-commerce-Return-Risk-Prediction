"""
Generate a realistic synthetic e-commerce order-return dataset.

Usage:
    python generate_data.py

The script is deterministic by default:
    python generate_data.py --seed 42

Outputs:
    data/raw/orders.csv

Design principle:
All predictive features represent information available at order time.
Post-delivery / post-return information is not used as a model feature.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CATEGORIES = [
    "Fashion",
    "Electronics",
    "Home & Kitchen",
    "Beauty",
    "Footwear",
    "Sports",
    "Books",
]

CATEGORY_BASE_RETURN_RATE = {
    "Fashion": 0.24,
    "Footwear": 0.21,
    "Electronics": 0.10,
    "Home & Kitchen": 0.12,
    "Beauty": 0.09,
    "Sports": 0.11,
    "Books": 0.06,
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Convert log-odds to probability."""
    return 1.0 / (1.0 + np.exp(-x))


def generate_dataset(
    n_orders: int = 40_000,
    n_customers: int = 8_000,
    n_products: int = 2_500,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a deterministic synthetic order-level dataset."""
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Customer master data
    # ------------------------------------------------------------------
    customer_ids = np.array(
        [f"C{idx:05d}" for idx in range(1, n_customers + 1)]
    )

    customer_age = rng.integers(18, 65, n_customers)

    customer_previous_orders = np.clip(
        rng.poisson(8, n_customers),
        0,
        40,
    )

    # Latent variable used only to simulate realistic historical behavior.
    # It is NOT included in the final dataset.
    customer_return_propensity = rng.beta(2.0, 8.0, n_customers)

    customer_previous_returns = np.array(
        [
            rng.binomial(order_count, propensity)
            for order_count, propensity in zip(
                customer_previous_orders,
                customer_return_propensity,
            )
        ]
    )

    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "customer_age": customer_age,
            "customer_previous_orders": customer_previous_orders,
            "customer_previous_returns": customer_previous_returns,
        }
    )

    # ------------------------------------------------------------------
    # 2. Product master data
    # ------------------------------------------------------------------
    product_ids = np.array(
        [f"P{idx:05d}" for idx in range(1, n_products + 1)]
    )

    product_category = rng.choice(
        CATEGORIES,
        size=n_products,
        p=[0.24, 0.14, 0.16, 0.12, 0.12, 0.10, 0.12],
    )

    product_price = np.clip(
        np.exp(rng.normal(np.log(1200), 0.85, n_products)),
        150,
        50_000,
    ).round(2)

    product_rating = np.clip(
        rng.normal(4.0, 0.45, n_products),
        2.2,
        5.0,
    ).round(2)

    product_return_rate = np.array(
        [
            np.clip(
                CATEGORY_BASE_RETURN_RATE[category] + rng.normal(0, 0.035),
                0.02,
                0.45,
            )
            for category in product_category
        ]
    ).round(4)

    products = pd.DataFrame(
        {
            "product_id": product_ids,
            "category": product_category,
            "product_price": product_price,
            "product_rating": product_rating,
            "product_historical_return_rate": product_return_rate,
        }
    )

    # ------------------------------------------------------------------
    # 3. Order data
    # ------------------------------------------------------------------
    order_customer_idx = rng.integers(0, n_customers, n_orders)
    order_product_idx = rng.integers(0, n_products, n_orders)

    order_dates = pd.date_range(
        start="2025-01-01",
        end="2025-12-31",
        periods=n_orders,
    )

    orders = pd.DataFrame(
        {
            "order_id": [
                f"O{idx:06d}" for idx in range(1, n_orders + 1)
            ],
            "order_date": order_dates,
            "customer_id": customer_ids[order_customer_idx],
            "product_id": product_ids[order_product_idx],
        }
    )

    orders = orders.merge(customers, on="customer_id", how="left")
    orders = orders.merge(products, on="product_id", how="left")

    # ------------------------------------------------------------------
    # 4. Order-time features
    # ------------------------------------------------------------------
    orders["quantity"] = rng.choice(
        [1, 2, 3, 4],
        n_orders,
        p=[0.72, 0.20, 0.06, 0.02],
    )

    orders["discount_pct"] = np.clip(
        rng.normal(18, 12, n_orders),
        0,
        60,
    ).round(1)

    orders["payment_type"] = rng.choice(
        ["Credit Card", "Debit Card", "UPI", "COD", "Wallet"],
        n_orders,
        p=[0.22, 0.18, 0.36, 0.18, 0.06],
    )

    orders["delivery_distance_km"] = np.clip(
        rng.gamma(shape=2.2, scale=7.0, size=n_orders),
        1,
        100,
    ).round(1)

    orders["expected_delivery_days"] = np.clip(
        np.round(
            2.5
            + orders["delivery_distance_km"] / 35
            + rng.normal(0, 0.7, n_orders)
        ),
        1,
        10,
    ).astype(int)

    orders["customer_return_rate"] = np.where(
        orders["customer_previous_orders"] > 0,
        orders["customer_previous_returns"]
        / orders["customer_previous_orders"],
        0,
    ).round(4)

    orders["discount_bucket"] = pd.cut(
        orders["discount_pct"],
        bins=[-0.01, 10, 25, 40, 60],
        labels=["Low", "Medium", "High", "Very High"],
    ).astype(str)

    orders["price_bucket"] = pd.cut(
        orders["product_price"],
        bins=[0, 500, 1500, 5000, np.inf],
        labels=["Budget", "Mid", "Premium", "Luxury"],
    ).astype(str)

    orders["order_month"] = orders["order_date"].dt.month
    orders["order_day_of_week"] = orders["order_date"].dt.dayofweek

    # ------------------------------------------------------------------
    # 5. Target generation
    # ------------------------------------------------------------------
    # All variables below are available at order time.
    logit = (
        -2.75
        + 2.10 * orders["customer_return_rate"]
        + 2.30 * orders["product_historical_return_rate"]
        + 0.75
        * orders["category"].isin(["Fashion", "Footwear"]).astype(int)
        + 0.012 * orders["discount_pct"]
        + 0.000015 * orders["product_price"]
        + 0.22 * (orders["payment_type"] == "COD").astype(int)
        - 0.35 * (orders["product_rating"] - 3.5)
        + 0.008 * orders["delivery_distance_km"]
        + 0.12 * (orders["quantity"] > 1).astype(int)
        + 0.12 * (orders["expected_delivery_days"] >= 5).astype(int)
        + rng.normal(0, 0.55, n_orders)
    )

    return_probability = sigmoid(logit)
    orders["returned"] = rng.binomial(1, return_probability)

    # ------------------------------------------------------------------
    # 6. Realistic missingness
    # ------------------------------------------------------------------
    missingness = {
        "product_rating": 0.012,
        "delivery_distance_km": 0.008,
        "discount_pct": 0.006,
    }

    for column, fraction in missingness.items():
        mask = rng.random(n_orders) < fraction
        orders.loc[mask, column] = np.nan

    # ------------------------------------------------------------------
    # 7. Final schema
    # ------------------------------------------------------------------
    columns = [
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

    return (
        orders[columns]
        .sort_values("order_date")
        .reset_index(drop=True)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic e-commerce return prediction data."
    )
    parser.add_argument("--orders", type=int, default=40_000)
    parser.add_argument("--customers", type=int, default=8_000)
    parser.add_argument("--products", type=int, default=2_500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.orders <= 0 or args.customers <= 0 or args.products <= 0:
        raise ValueError("orders, customers and products must all be positive.")

    project_dir = Path(__file__).resolve().parents[2]
    output_dir = project_dir / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "orders.csv"

    df = generate_dataset(
        n_orders=args.orders,
        n_customers=args.customers,
        n_products=args.products,
        seed=args.seed,
    )

    df.to_csv(output_path, index=False)

    print("Dataset generated successfully.")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Return rate: {df['returned'].mean():.2%}")
    print(f"Missing cells: {int(df.isna().sum().sum()):,}")
    print(f"Date range: {df['order_date'].min().date()} → {df['order_date'].max().date()}")
    print(f"Seed: {args.seed}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()

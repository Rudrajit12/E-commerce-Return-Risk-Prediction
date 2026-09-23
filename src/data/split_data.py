"""
Create a chronological train/test split for return prediction.

Usage:
    python src/data/split_data.py

Default:
    Train: all orders before 2025-10-01
    Test:  2025-10-01 onward

This simulates the real prediction setting:
train on historical orders -> predict future orders.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def chronological_split(
    df: pd.DataFrame,
    test_start: str = "2025-10-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data chronologically without shuffling."""
    if "order_date" not in df.columns:
        raise ValueError("order_date column is required.")

    data = df.copy()
    data["order_date"] = pd.to_datetime(data["order_date"], errors="coerce")

    if data["order_date"].isna().any():
        raise ValueError("Invalid/missing order_date values detected.")

    cutoff = pd.Timestamp(test_start)

    train = data[data["order_date"] < cutoff].copy()
    test = data[data["order_date"] >= cutoff].copy()

    if train.empty or test.empty:
        raise ValueError(
            f"Split produced an empty partition. Cutoff: {test_start}"
        )

    if train["order_date"].max() >= test["order_date"].min():
        raise ValueError("Temporal leakage detected between train and test.")

    return (
        train.sort_values(["order_date", "order_id"]).reset_index(drop=True),
        test.sort_values(["order_date", "order_id"]).reset_index(drop=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/processed/model_features.csv",
    )
    parser.add_argument(
        "--train-output",
        default="data/processed/train.csv",
    )
    parser.add_argument(
        "--test-output",
        default="data/processed/test.csv",
    )
    parser.add_argument(
        "--test-start",
        default="2025-10-01",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[2]
    input_path = project_dir / args.input
    train_path = project_dir / args.train_output
    test_path = project_dir / args.test_output

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    train_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, parse_dates=["order_date"])
    train, test = chronological_split(df, args.test_start)

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    print("Chronological split completed successfully.")
    print(f"Cutoff: {args.test_start}")
    print(
        f"Train: {len(train):,} rows | "
        f"{train['order_date'].min().date()} → {train['order_date'].max().date()} | "
        f"return rate: {train['returned'].mean():.2%}"
    )
    print(
        f"Test:  {len(test):,} rows | "
        f"{test['order_date'].min().date()} → {test['order_date'].max().date()} | "
        f"return rate: {test['returned'].mean():.2%}"
    )
    print(f"Train output: {train_path}")
    print(f"Test output: {test_path}")


if __name__ == "__main__":
    main()

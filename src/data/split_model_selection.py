"""
Create chronological train/validation/test splits for model selection.

Default:
    Train      : 2025-01-01 to 2025-08-31
    Validation : 2025-09-01 to 2025-09-30
    Test       : 2025-10-01 to 2025-12-31

The validation period is used for model/threshold selection.
The test period remains untouched until final evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def chronological_train_val_test(
    df: pd.DataFrame,
    validation_start: str = "2025-09-01",
    test_start: str = "2025-10-01",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "order_date" not in df.columns:
        raise ValueError("order_date column is required.")

    data = df.copy()
    data["order_date"] = pd.to_datetime(data["order_date"], errors="coerce")
    if data["order_date"].isna().any():
        raise ValueError("Invalid/missing order_date values detected.")

    val_cutoff = pd.Timestamp(validation_start)
    test_cutoff = pd.Timestamp(test_start)

    if val_cutoff >= test_cutoff:
        raise ValueError("validation_start must be before test_start.")

    train = data[data["order_date"] < val_cutoff].copy()
    validation = data[
        (data["order_date"] >= val_cutoff) & (data["order_date"] < test_cutoff)
    ].copy()
    test = data[data["order_date"] >= test_cutoff].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("One or more partitions are empty.")

    partitions = [train, validation, test]
    for left, right in zip(partitions, partitions[1:]):
        if left["order_date"].max() >= right["order_date"].min():
            raise ValueError("Temporal leakage detected between partitions.")

    sort_cols = ["order_date", "order_id"]
    return tuple(
        part.sort_values(sort_cols).reset_index(drop=True)
        for part in partitions
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/model_features.csv")
    parser.add_argument("--train-output", default="data/processed/train_model_selection.csv")
    parser.add_argument("--validation-output", default="data/processed/validation_model_selection.csv")
    parser.add_argument("--test-output", default="data/processed/test_model_selection.csv")
    parser.add_argument("--validation-start", default="2025-09-01")
    parser.add_argument("--test-start", default="2025-10-01")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[2]
    input_path = project_dir / args.input
    outputs = [project_dir / p for p in
               [args.train_output, args.validation_output, args.test_output]]

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path, parse_dates=["order_date"])
    train, validation, test = chronological_train_val_test(
        df, args.validation_start, args.test_start
    )

    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)

    train.to_csv(outputs[0], index=False)
    validation.to_csv(outputs[1], index=False)
    test.to_csv(outputs[2], index=False)

    print("Three-way chronological split completed.")
    for name, part, path in zip(["Train", "Validation", "Test"], [train, validation, test], outputs):
        print(
            f"{name}: {len(part):,} rows | "
            f"{part.order_date.min().date()} -> {part.order_date.max().date()} | "
            f"return rate: {part.returned.mean():.2%} | {path}"
        )


if __name__ == "__main__":
    main()

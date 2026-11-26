from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"


def summarize(frame: pd.DataFrame) -> dict:
    actual = frame["actual"].to_numpy(dtype=float)
    lower = frame["p10"].to_numpy(dtype=float)
    median = frame["p50"].to_numpy(dtype=float)
    upper = frame["p90"].to_numpy(dtype=float)

    positive = actual > 0
    covered = (actual >= lower) & (actual <= upper)
    errors = np.abs(actual - median)
    total_actual = float(actual.sum())

    return {
        "observations": len(frame),
        "actual_units": total_actual,
        "zero_sales_percent": float(100 * (actual == 0).mean()),
        "p50_mae": float(errors.mean()),
        "p50_wape_percent": (
            float(100 * errors.sum() / total_actual)
            if total_actual > 0 else None
        ),
        "coverage_percent": float(100 * covered.mean()),
        "positive_demand_coverage_percent": (
            float(100 * covered[positive].mean())
            if positive.any() else None
        ),
        "above_p90_percent": float(100 * (actual > upper).mean()),
        "below_p10_percent": float(100 * (actual < lower).mean()),
        "mean_interval_width": float((upper - lower).mean()),
        "units_above_p90": float(
            np.maximum(actual - upper, 0).sum()
        ),
    }


def main():
    data = pd.read_csv(
        REPORTS / "quantile_predictions.csv",
        parse_dates=["date", "origin"],
    )

    if data.duplicated(["fold", "date", "series_id"]).any():
        raise ValueError("Duplicate forecast observations.")

    values = data[
        ["actual", "p10", "p50", "p90"]
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Missing or non-finite forecast values.")

    if (
        (data["p10"] > data["p50"])
        | (data["p50"] > data["p90"])
    ).any():
        raise ValueError("Adjusted quantiles are not ordered.")

    product_rows = []

    for series_id, group in data.groupby("series_id"):
        product_rows.append({
            "series_id": series_id,
            **summarize(group),
        })

    products = pd.DataFrame(product_rows).sort_values(
        ["above_p90_percent", "units_above_p90"],
        ascending=False,
    )

    products.to_csv(
        REPORTS / "diagnostics_by_product.csv",
        index=False,
    )

    fold_product_rows = []

    for (fold, series_id), group in data.groupby(
        ["fold", "series_id"]
    ):
        fold_product_rows.append({
            "fold": fold,
            "series_id": series_id,
            **summarize(group),
        })

    pd.DataFrame(fold_product_rows).to_csv(
        REPORTS / "diagnostics_by_fold_product.csv",
        index=False,
    )

    columns = [
        "series_id",
        "zero_sales_percent",
        "p50_mae",
        "coverage_percent",
        "positive_demand_coverage_percent",
        "above_p90_percent",
        "units_above_p90",
    ]

    print("PRODUCTS WITH THE HIGHEST UPPER-BOUND MISS RATE")
    print(
        products[columns]
        .head(10)
        .round(2)
        .to_string(index=False)
    )

    print("\nSaved:")
    print("reports/diagnostics_by_product.csv")
    print("reports/diagnostics_by_fold_product.csv")
    print("\nReserved test data was not used.")


if __name__ == "__main__":
    main()
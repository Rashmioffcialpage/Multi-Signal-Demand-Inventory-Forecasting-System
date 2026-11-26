from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"

HORIZON = 28
N_FOLDS = 3
SEASON = 7


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    data = pd.read_parquet(
        ROOT / "data" / "processed" / "daily_sales.parquet"
    )

    sales = (
        data.pivot(
            index="date",
            columns="series_id",
            values="units",
        )
        .sort_index()
    )

    if sales.isna().any().any():
        raise ValueError("Missing series-date observations.")

    dates = sales.index
    holdout_start = len(dates) - HORIZON
    first_validation = holdout_start - N_FOLDS * HORIZON

    if first_validation < SEASON:
        raise ValueError("Insufficient history for evaluation.")

    predictions = []
    fold_metrics = []

    for fold in range(1, N_FOLDS + 1):
        start = first_validation + (fold - 1) * HORIZON
        stop = start + HORIZON

        # All forecast inputs precede this fold's forecast dates.
        history = sales.iloc[:start]
        actual = sales.iloc[start:stop]

        last_week = history.iloc[-SEASON:].to_numpy(dtype=float)

        # Repeat the same observed week across the full horizon.
        # Do not use actual sales from inside the forecast window.
        forecast = np.tile(
            last_week,
            (int(np.ceil(HORIZON / SEASON)), 1),
        )[:HORIZON]

        observed = actual.to_numpy(dtype=float)
        absolute_error = np.abs(observed - forecast)
        total_actual = float(observed.sum())

        metrics = {
            "fold": fold,
            "origin": str(history.index[-1].date()),
            "validation_start": str(actual.index[0].date()),
            "validation_end": str(actual.index[-1].date()),
            "mae": float(absolute_error.mean()),
            "wape_percent": (
                float(100 * absolute_error.sum() / total_actual)
                if total_actual > 0 else None
            ),
        }
        fold_metrics.append(metrics)

        for column, series_id in enumerate(sales.columns):
            predictions.append(
                pd.DataFrame({
                    "fold": fold,
                    "origin": history.index[-1],
                    "date": actual.index,
                    "horizon_day": np.arange(1, HORIZON + 1),
                    "series_id": series_id,
                    "actual": observed[:, column],
                    "prediction": forecast[:, column],
                    "model": "seasonal_naive_7",
                })
            )

        print(
            f"Fold {fold}: "
            f"{metrics['validation_start']} to "
            f"{metrics['validation_end']} | "
            f"MAE: {metrics['mae']:.3f} | "
            f"WAPE: {metrics['wape_percent']}"
        )

    results = pd.concat(predictions, ignore_index=True)
    results.to_csv(
        REPORTS / "baseline_predictions.csv",
        index=False,
    )
    pd.DataFrame(fold_metrics).to_csv(
        REPORTS / "baseline_fold_metrics.csv",
        index=False,
    )

    errors = (
        results["actual"] - results["prediction"]
    ).abs()
    total_actual = float(results["actual"].sum())

    summary = {
        "model": "seasonal_naive_7",
        "folds": N_FOLDS,
        "horizon_days": HORIZON,
        "forecast_rows": len(results),
        "pooled_mae": float(errors.mean()),
        "pooled_wape_percent": (
            float(100 * errors.sum() / total_actual)
            if total_actual > 0 else None
        ),
        "reserved_test_start": str(dates[holdout_start].date()),
        "reserved_test_end": str(dates[-1].date()),
        "test_evaluated": False,
    }

    (REPORTS / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nBASELINE SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nSaved predictions and metrics in reports/.")


if __name__ == "__main__":
    main()
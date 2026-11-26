from pathlib import Path
import sqlite3

import pandas as pd

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
DATABASE = ROOT / "data" / "demandlens.db"


def main():
    sales = pd.read_parquet(
        PROCESSED / "daily_sales.parquet"
    )

    forecasts = pd.read_csv(
        REPORTS / "quantile_predictions.csv",
        parse_dates=["date", "origin"],
    )

    calibrated = pd.read_csv(
        REPORTS / "calibrated_validation_predictions.csv",
        parse_dates=["date", "origin"],
    )

    keys = ["fold", "date", "series_id"]

    if forecasts.duplicated(keys).any():
        raise ValueError("Duplicate forecast rows.")

    if calibrated.duplicated(keys).any():
        raise ValueError("Duplicate calibrated rows.")

    forecasts = forecasts.merge(
        calibrated[
            keys + ["p90_adjusted", "upper_adjustment_units"]
        ],
        on=keys,
        how="left",
        validate="one_to_one",
    )

    # Fold 1 has no earlier calibration history.
    # Its adjusted-bound fields intentionally remain missing.
    sales["date"] = sales["date"].dt.strftime("%Y-%m-%d")
    forecasts["date"] = forecasts["date"].dt.strftime("%Y-%m-%d")
    forecasts["origin"] = forecasts["origin"].dt.strftime("%Y-%m-%d")

    daily_columns = [
        "series_id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "date",
        "units",
        "sell_price",
        "price_missing",
    ]

    forecast_columns = [
        "fold",
        "series_id",
        "origin",
        "date",
        "horizon_day",
        "actual",
        "p10",
        "p50",
        "p90",
        "quantile_crossed",
        "p90_adjusted",
        "upper_adjustment_units",
    ]

    DATABASE.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE) as connection:
        sales[daily_columns].to_sql(
            "daily_sales",
            connection,
            if_exists="replace",
            index=False,
        )

        forecasts[forecast_columns].to_sql(
            "validation_forecasts",
            connection,
            if_exists="replace",
            index=False,
        )

        connection.execute("""
            CREATE UNIQUE INDEX idx_daily_series_date
            ON daily_sales(series_id, date)
        """)

        connection.execute("""
            CREATE UNIQUE INDEX idx_forecast_fold_series_date
            ON validation_forecasts(fold, series_id, date)
        """)

        summary = pd.read_sql_query("""
            SELECT
                fold,
                COUNT(*) AS forecast_rows,
                ROUND(AVG(ABS(actual - p50)), 3) AS mae,
                ROUND(
                    100.0 * AVG(
                        CASE
                            WHEN actual BETWEEN p10 AND p90
                            THEN 1.0 ELSE 0.0
                        END
                    ),
                    2
                ) AS coverage_percent,
                ROUND(
                    100.0 * AVG(
                        CASE WHEN actual > p90
                        THEN 1.0 ELSE 0.0 END
                    ),
                    2
                ) AS upper_miss_percent
            FROM validation_forecasts
            GROUP BY fold
            ORDER BY fold
        """, connection)

        daily_count = connection.execute(
            "SELECT COUNT(*) FROM daily_sales"
        ).fetchone()[0]

        forecast_count = connection.execute(
            "SELECT COUNT(*) FROM validation_forecasts"
        ).fetchone()[0]

        if daily_count != len(sales):
            raise ValueError("Daily sales row count mismatch.")

        if forecast_count != len(forecasts):
            raise ValueError("Forecast row count mismatch.")

    print("DATABASE READY")
    print(f"Daily sales rows: {daily_count:,}")
    print(f"Validation forecast rows: {forecast_count:,}")

    print("\nSQL VALIDATION SUMMARY")
    print(summary.to_string(index=False))

    print(f"\nSaved: {DATABASE}")
    print(
        "Historical sales include the reserved period, "
        "but this step does not train or evaluate on it."
    )


if __name__ == "__main__":
    main()
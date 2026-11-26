from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

HORIZON = 28


def main():
    data = pd.read_parquet(
        PROCESSED / "daily_sales.parquet"
    ).sort_values(["series_id", "date"]).reset_index(drop=True)

    sales = data.groupby("series_id", sort=False)["units"]

    # These observations are available for every day of a
    # 28-day forecast issued at a single origin.
    for lag in [28, 35, 42, 56]:
        data[f"sales_lag_{lag}"] = sales.shift(lag)

    for window in [7, 28, 56]:
        data[f"sales_mean_{window}"] = sales.transform(
            lambda values: (
                values.shift(HORIZON)
                .rolling(window, min_periods=window)
                .mean()
            )
        )

    data["sales_std_28"] = sales.transform(
        lambda values: (
            values.shift(HORIZON)
            .rolling(28, min_periods=28)
            .std(ddof=0)
        )
    )

    data["zero_share_28"] = sales.transform(
        lambda values: (
            values.eq(0)
            .astype(float)
            .shift(HORIZON)
            .rolling(28, min_periods=28)
            .mean()
        )
    )

    # Historical price only: future realized prices are excluded.
    data["price_lag_28"] = (
        data.groupby("series_id", sort=False)["sell_price"]
        .shift(HORIZON)
    )

    # Calendar features known in advance.
    data["day_of_week"] = data["date"].dt.day_of_week
    data["day_of_month"] = data["date"].dt.day
    data["month_number"] = data["date"].dt.month
    data["year_number"] = data["date"].dt.year
    data["time_index"] = (
        data["date"] - data["date"].min()
    ).dt.days

    data["weekday_sin"] = np.sin(
        2 * np.pi * data["day_of_week"] / 7
    )
    data["weekday_cos"] = np.cos(
        2 * np.pi * data["day_of_week"] / 7
    )

    categorical = [
        "series_id",
        "dept_id",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
    ]

    # Keep labels as strings; encoders will be fitted within each fold.
    for column in categorical:
        data[column] = (
            data[column].fillna("None").astype(str)
        )

    numeric = [
        "sales_lag_28",
        "sales_lag_35",
        "sales_lag_42",
        "sales_lag_56",
        "sales_mean_7",
        "sales_mean_28",
        "sales_mean_56",
        "sales_std_28",
        "zero_share_28",
        "price_lag_28",
        "day_of_week",
        "day_of_month",
        "month_number",
        "year_number",
        "time_index",
        "weekday_sin",
        "weekday_cos",
        "snap_CA",
    ]

    # Remove only the initial rows lacking sufficient sales history.
    # Missing historical prices remain NaN for LightGBM to handle.
    required_history = [
        column for column in numeric
        if column.startswith(("sales_", "zero_share_"))
    ]

    features = data.dropna(
        subset=required_history
    ).copy()

    output_columns = list(dict.fromkeys(
        ["date", "series_id", "units"] + categorical + numeric
    ))
    features = features[output_columns]

    if features.duplicated(["series_id", "date"]).any():
        raise ValueError("Duplicate series-date rows.")

    features.to_parquet(
        PROCESSED / "features.parquet",
        index=False,
    )

    config = {
        "horizon_days": HORIZON,
        "categorical_features": categorical,
        "numeric_features": numeric,
        "target": "units",
        "future_prices_used": False,
    }

    (PROCESSED / "feature_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    summary = {
        "rows": len(features),
        "series": int(features["series_id"].nunique()),
        "features": len(categorical) + len(numeric),
        "first_date": str(features["date"].min().date()),
        "last_date": str(features["date"].max().date()),
        "missing_historical_prices": int(
            features["price_lag_28"].isna().sum()
        ),
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "feature_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("FEATURE SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nSaved features.parquet and feature_config.json")


if __name__ == "__main__":
    main()
from pathlib import Path
import json
import sqlite3

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_forecasts(
    filename: str,
    expected_rows: int,
    expected_start: str,
    expected_end: str,
    expected_folds: int | None = None,
) -> pd.DataFrame:
    path = REPORTS / filename
    require(path.is_file(), f"Missing file: {path}")

    frame = pd.read_csv(path)

    required = {"series_id", "date", "actual", "p10", "p50", "p90"}
    require(
        required.issubset(frame.columns),
        f"{filename}: missing columns {required - set(frame.columns)}",
    )

    frame["date"] = pd.to_datetime(frame["date"], errors="raise")

    require(len(frame) == expected_rows, f"{filename}: incorrect row count")
    require(frame["series_id"].nunique() == 20, f"{filename}: expected 20 series")

    keys = ["series_id", "date"]
    require(
        not bool(frame.duplicated(keys).any()),
        f"{filename}: duplicate product-date rows",
    )

    values = frame[["actual", "p10", "p50", "p90"]].to_numpy(
        dtype=np.float64
    )

    require(bool(np.isfinite(values).all()), f"{filename}: invalid numbers")
    require(bool((values >= 0).all()), f"{filename}: negative values")
    require(
        bool((frame["p10"] <= frame["p50"]).all()),
        f"{filename}: P10 exceeds P50",
    )
    require(
        bool((frame["p50"] <= frame["p90"]).all()),
        f"{filename}: P50 exceeds P90",
    )

    require(
        frame["date"].min() == pd.Timestamp(expected_start),
        f"{filename}: incorrect start date",
    )
    require(
        frame["date"].max() == pd.Timestamp(expected_end),
        f"{filename}: incorrect end date",
    )

    if expected_folds is not None:
        require("fold" in frame.columns, f"{filename}: missing fold column")
        require(
            set(frame["fold"].unique()) == set(range(1, expected_folds + 1)),
            f"{filename}: unexpected folds",
        )

        for fold, group in frame.groupby("fold"):
            require(len(group) == 560, f"Fold {fold}: expected 560 rows")
            require(
                bool((group.groupby("series_id").size() == 28).all()),
                f"Fold {fold}: expected 28 days per product",
            )
    else:
        require(
            bool((frame.groupby("series_id").size() == 28).all()),
            f"{filename}: expected 28 days per product",
        )

    if "origin" in frame.columns:
        origins = pd.to_datetime(frame["origin"], errors="raise")
        horizon = (frame["date"] - origins).dt.days

        require(
            bool(horizon.between(1, 28).all()),
            f"{filename}: forecast dates must be 1–28 days after origin",
        )

    print(f"PASS — {filename}")
    return frame


def check_holdout_metrics(frame: pd.DataFrame) -> None:
    path = REPORTS / "holdout_summary.json"
    require(path.is_file(), "Missing holdout_summary.json")

    with path.open(encoding="utf-8") as file:
        summary = json.load(file)

    actual = frame["actual"].to_numpy(dtype=np.float64)
    predicted = frame["p50"].to_numpy(dtype=np.float64)

    mae = float(np.mean(np.abs(actual - predicted)))
    total_actual = float(actual.sum())
    require(total_actual > 0, "Holdout actual sales total must be positive")

    wape = float(100 * np.abs(actual - predicted).sum() / total_actual)

    coverage = float(
        100
        * (
            (frame["actual"] >= frame["p10"])
            & (frame["actual"] <= frame["p90"])
        ).mean()
    )

    calculated = {
        "lightgbm_mae": mae,
        "lightgbm_wape_percent": wape,
        "coverage_percent": coverage,
    }

    for name, value in calculated.items():
        require(name in summary, f"Holdout summary missing {name}")
        require(
            bool(np.isclose(value, float(summary[name]), atol=1e-8, rtol=1e-8)),
            f"Holdout summary mismatch: {name}",
        )

    require(summary.get("test_evaluated") is True, "Holdout status is incorrect")
    require(summary.get("forecast_rows") == 560, "Holdout summary row mismatch")

    print("PASS — holdout metrics match saved predictions")


def check_database() -> None:
    path = ROOT / "data" / "demandlens.db"
    require(path.is_file(), "Missing data/demandlens.db")

    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        require(integrity == ("ok",), "SQLite integrity check failed")

        sales_rows = connection.execute(
            "SELECT COUNT(*) FROM daily_sales"
        ).fetchone()

        forecast_rows = connection.execute(
            "SELECT COUNT(*) FROM validation_forecasts"
        ).fetchone()

    require(sales_rows == (38260,), "Database sales row count mismatch")
    require(forecast_rows == (1680,), "Database forecast row count mismatch")

    print("PASS — SQLite integrity and row counts")


def main() -> None:
    validation = check_forecasts(
        filename="quantile_predictions.csv",
        expected_rows=1680,
        expected_start="2016-01-04",
        expected_end="2016-03-27",
        expected_folds=3,
    )

    holdout = check_forecasts(
        filename="holdout_predictions.csv",
        expected_rows=560,
        expected_start="2016-03-28",
        expected_end="2016-04-24",
    )

    require(
        validation["date"].max() < holdout["date"].min(),
        "Validation and holdout dates overlap",
    )
    print("PASS — validation and holdout dates do not overlap")

    check_holdout_metrics(holdout)
    check_database()

    print("\nAll project checks passed.")
    print("These checks inspect saved outputs; they do not retrain models.")
    print("Full pipeline reproducibility and training-time leakage need separate checks.")


if __name__ == "__main__":
    main()
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

FOLD = 3
LEAD_TIME = 7
REVIEW_PERIOD = 7
HISTORY_DAYS = 364

# Demonstration assumptions, not observed inventory records.
ON_HAND = 5
ON_ORDER = 0
BACKORDERS = 0


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    sales = pd.read_parquet(
        PROCESSED / "daily_sales.parquet"
    )
    sales["date"] = pd.to_datetime(sales["date"])

    forecasts = pd.read_csv(
        REPORTS / "quantile_predictions.csv",
        parse_dates=["date", "origin"],
    )

    forecasts = forecasts.loc[
        forecasts["fold"] == FOLD
    ].copy()

    if forecasts.empty:
        raise ValueError("Selected validation fold is missing.")

    origins = forecasts["origin"].drop_duplicates()

    if len(origins) != 1:
        raise ValueError("Expected one forecast origin.")

    origin = pd.Timestamp(str(origins.iloc[0]))
    protection_days = LEAD_TIME + REVIEW_PERIOD

    if protection_days > 28:
        raise ValueError("Protection period exceeds forecast horizon.")

    history_start = origin - pd.Timedelta(
        days=HISTORY_DAYS - 1
    )

    rows = []

    for series_id, group in forecasts.groupby("series_id"):
        group = group.sort_values("date")

        future = group.loc[
            group["date"].between(
                origin + pd.Timedelta(days=1),
                origin + pd.Timedelta(days=protection_days),
            )
        ].copy()

        expected_dates = pd.date_range(
            origin + pd.Timedelta(days=1),
            periods=protection_days,
            freq="D",
        )

        if not pd.DatetimeIndex(future["date"]).equals(
            expected_dates
        ):
            raise ValueError(
                f"Incomplete forecast horizon: {series_id}"
            )

        historical = sales.loc[
            (sales["series_id"] == series_id)
            & sales["date"].between(history_start, origin)
        ].sort_values("date")

        history_dates = pd.date_range(
            history_start,
            origin,
            freq="D",
        )

        if not pd.DatetimeIndex(historical["date"]).equals(
            history_dates
        ):
            raise ValueError(
                f"Incomplete historical window: {series_id}"
            )

        values = historical["units"].to_numpy(dtype=float)

        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Invalid historical sales.")

        # Form non-overlapping blocks ending at the origin.
        block_count = len(values) // protection_days
        usable = values[-block_count * protection_days:]

        block_totals = usable.reshape(
            block_count,
            protection_days,
        ).sum(axis=1)

        historical_p90 = float(
            np.quantile(block_totals, 0.90, method="higher")
        )

        central_total = float(future["p50"].sum())
        lead_time_total = float(
            future.iloc[:LEAD_TIME]["p50"].sum()
        )

        # Summing daily medians is a central planning estimate,
        # not the median of total protection-period demand.
        target_stock = math.ceil(
            max(central_total, historical_p90)
        )

        inventory_position = ON_HAND + ON_ORDER - BACKORDERS
        available_now = max(ON_HAND - BACKORDERS, 0)

        recommended_order = max(
            target_stock - inventory_position,
            0,
        )

        rows.append({
            "series_id": series_id,
            "origin": origin.strftime("%Y-%m-%d"),
            "lead_time_days": LEAD_TIME,
            "review_period_days": REVIEW_PERIOD,
            "protection_days": protection_days,
            "historical_blocks": block_count,
            "central_protection_units": central_total,
            "historical_block_p90_units": historical_p90,
            "target_stock_units": target_stock,
            "assumed_on_hand": ON_HAND,
            "assumed_on_order": ON_ORDER,
            "assumed_backorders": BACKORDERS,
            "inventory_position": inventory_position,
            "recommended_order_units": recommended_order,
            "central_lead_time_units": lead_time_total,
            "pre_arrival_risk_flag": (
                available_now < lead_time_total
            ),
        })

    result = pd.DataFrame(rows).sort_values(
        "recommended_order_units",
        ascending=False,
    )

    result.to_csv(
        REPORTS / "inventory_recommendations.csv",
        index=False,
    )

    assumptions = {
        "scenario": "Illustrative inventory planning snapshot",
        "validation_fold": FOLD,
        "origin": origin.strftime("%Y-%m-%d"),
        "lead_time_days": LEAD_TIME,
        "review_period_days": REVIEW_PERIOD,
        "history_days": HISTORY_DAYS,
        "assumed_on_hand_per_product": ON_HAND,
        "assumed_on_order_per_product": ON_ORDER,
        "assumed_backorders_per_product": BACKORDERS,
        "policy": (
            "Order up to the ceiling of the larger of summed "
            "daily P50 forecasts and historical block P90."
        ),
        "limitations": [
            "Inventory inputs are assumptions, not observed stock.",
            "Historical block P90 is not a forecast demand quantile.",
            "No guaranteed service level.",
            "No costs, capacity limits, or minimum order quantities.",
            "Recorded sales may understate unconstrained demand.",
            "This is a recommendation snapshot, not a dynamic simulation.",
        ],
        "test_evaluated": False,
    }

    (REPORTS / "inventory_assumptions.json").write_text(
        json.dumps(assumptions, indent=2),
        encoding="utf-8",
    )

    columns = [
        "series_id",
        "central_protection_units",
        "historical_block_p90_units",
        "target_stock_units",
        "recommended_order_units",
        "pre_arrival_risk_flag",
    ]

    print("INVENTORY PLANNING SCENARIO")
    print(f"Origin: {origin.date()}")
    print(f"Protection period: {protection_days} days")
    print(f"Assumed stock per product: {ON_HAND} units\n")

    print(result[columns].round(2).to_string(index=False))

    print("\nSaved recommendations and assumptions in reports/.")
    print("Reserved test outcomes were not used.")


if __name__ == "__main__":
    main()
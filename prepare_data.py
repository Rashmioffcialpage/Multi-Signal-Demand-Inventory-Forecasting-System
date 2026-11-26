from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

STORE = "CA_1"
CATEGORY = "FOODS"
N_PRODUCTS = 20


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    print("Reading sales...")
    sales = pd.read_csv(RAW / "sales_train_validation.csv")

    candidates = sales.loc[
        (sales["store_id"] == STORE)
        & (sales["cat_id"] == CATEGORY)
    ].copy()

    # Select deterministically by item ID, not future sales performance.
    selected = (
        candidates.sort_values("item_id")
        .head(N_PRODUCTS)
        .copy()
    )

    require(
        len(selected) == N_PRODUCTS,
        f"Expected {N_PRODUCTS} products; found {len(selected)}.",
    )
    require(selected["id"].is_unique, "Duplicate series IDs.")

    day_columns = sorted(
        [
            column
            for column in selected.columns
            if column.startswith("d_")
        ],
        key=lambda column: int(column.split("_")[1]),
    )

    require(len(day_columns) == 1913, "Expected 1,913 sales days.")

    daily = selected.melt(
        id_vars=[
            "id", "item_id", "dept_id",
            "cat_id", "store_id", "state_id",
        ],
        value_vars=day_columns,
        var_name="d",
        value_name="units",
    )

    del sales, candidates

    print("Joining calendar...")
    calendar = pd.read_csv(
        RAW / "calendar.csv",
        parse_dates=["date"],
    )

    require(calendar["d"].is_unique, "Duplicate calendar day IDs.")

    daily = daily.merge(
        calendar,
        on="d",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    require(
        daily["_merge"].eq("both").all(),
        "Some sales days have no calendar match.",
    )
    daily = daily.drop(columns="_merge")

    print("Reading prices...")
    prices = pd.read_csv(
        RAW / "sell_prices.csv",
        dtype={"sell_price": "float32"},
    )

    prices = prices.loc[
        (prices["store_id"] == STORE)
        & prices["item_id"].isin(selected["item_id"])
    ].copy()

    price_keys = ["store_id", "item_id", "wm_yr_wk"]

    require(
        not prices.duplicated(price_keys).any(),
        "Duplicate item-store-week price records.",
    )

    daily = daily.merge(
        prices,
        on=price_keys,
        how="left",
        validate="many_to_one",
    )

    daily = daily.rename(columns={"id": "series_id"})
    daily["units"] = pd.to_numeric(daily["units"], errors="raise")

    require(daily["units"].notna().all(), "Missing sales values.")
    require(daily["units"].ge(0).all(), "Negative sales values.")

    # Missing prices remain missing; no backward filling from future weeks.
    daily["price_missing"] = daily["sell_price"].isna()

    require(
        daily.loc[
            ~daily["price_missing"], "sell_price"
        ].gt(0).all(),
        "Non-positive recorded prices.",
    )

    require(
        not daily.duplicated(["series_id", "date"]).any(),
        "Duplicate series-date rows.",
    )

    require(
        len(daily) == N_PRODUCTS * len(day_columns),
        "Unexpected row count after joins.",
    )

    daily = daily.sort_values(
        ["series_id", "date"]
    ).reset_index(drop=True)

    gaps = daily.groupby("series_id")["date"].diff().dropna()
    require(
        gaps.eq(pd.Timedelta(days=1)).all(),
        "A series has missing or nonconsecutive dates.",
    )

    output = PROCESSED / "daily_sales.parquet"
    daily.to_parquet(output, index=False)

    report = {
        "store": STORE,
        "category": CATEGORY,
        "selection_rule": "First 20 item IDs alphabetically",
        "series": int(daily["series_id"].nunique()),
        "rows": len(daily),
        "days_per_series": len(day_columns),
        "start_date": str(daily["date"].min().date()),
        "end_date": str(daily["date"].max().date()),
        "zero_sales_rows": int(daily["units"].eq(0).sum()),
        "missing_price_rows": int(daily["price_missing"].sum()),
        "positive_sales_with_missing_price": int(
            (
                daily["units"].gt(0)
                & daily["price_missing"]
            ).sum()
        ),
    }

    (REPORTS / "data_preparation.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nDATA PREPARATION")
    print(json.dumps(report, indent=2))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
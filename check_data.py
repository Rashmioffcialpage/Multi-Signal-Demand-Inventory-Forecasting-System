from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"

REQUIRED = {
    "sales_train_validation.csv": {
        "id", "item_id", "dept_id", "cat_id",
        "store_id", "state_id", "d_1", "d_1913",
    },
    "calendar.csv": {
        "date", "wm_yr_wk", "d",
        "event_name_1", "event_type_1",
        "snap_CA", "snap_TX", "snap_WI",
    },
    "sell_prices.csv": {
        "store_id", "item_id", "wm_yr_wk", "sell_price",
    },
}


def main():
    errors = []

    for filename, expected_columns in REQUIRED.items():
        path = RAW / filename

        if not path.is_file():
            errors.append(f"Missing file: {path}")
            continue

        try:
            sample = pd.read_csv(path, nrows=5)
        except Exception as exc:
            errors.append(f"{filename}: cannot read CSV — {exc}")
            continue

        missing = expected_columns - set(sample.columns)

        if missing:
            errors.append(
                f"{filename}: missing columns {sorted(missing)}"
            )
            continue

        if sample.empty:
            errors.append(f"{filename}: file contains no data rows")
            continue

        size_mb = path.stat().st_size / 1_000_000
        print(f"PASS — {filename} ({size_mb:.1f} MB)")

    if errors:
        print("\nPlease fix:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print("\nAll files passed the initial format check.")
    print("Next: prepare a small selection of item-store series.")


if __name__ == "__main__":
    main()
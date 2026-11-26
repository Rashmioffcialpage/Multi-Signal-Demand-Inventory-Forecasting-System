from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.preprocessing import OrdinalEncoder

from evaluate_lightgbm import calculate_metrics
from evaluate_quantiles import interval_metrics

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

ORIGIN = pd.Timestamp("2016-03-27")
TEST_START = pd.Timestamp("2016-03-28")
TEST_END = pd.Timestamp("2016-04-24")
HORIZON = 28

PARAMETERS = {
    "objective": "quantile",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 50,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": 2,
    "verbosity": -1,
}


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    data = pd.read_parquet(PROCESSED / "features.parquet")
    data["date"] = pd.to_datetime(data["date"])

    sales = pd.read_parquet(PROCESSED / "daily_sales.parquet")
    sales["date"] = pd.to_datetime(sales["date"])

    config = json.loads(
        (PROCESSED / "feature_config.json").read_text(
            encoding="utf-8"
        )
    )

    if int(config["horizon_days"]) != HORIZON:
        raise ValueError("Feature horizon does not match evaluation.")

    categorical = config["categorical_features"]
    feature_names = categorical + config["numeric_features"]

    train = data.loc[data["date"] <= ORIGIN].copy()
    test = data.loc[
        data["date"].between(TEST_START, TEST_END)
    ].copy()

    if train.empty or test.empty:
        raise ValueError("Missing training or test data.")

    if test.duplicated(["series_id", "date"]).any():
        raise ValueError("Duplicate test observations.")

    expected_dates = pd.date_range(
        TEST_START,
        TEST_END,
        freq="D",
    )

    for series_id, group in test.groupby("series_id"):
        observed_dates = pd.DatetimeIndex(
            group["date"].sort_values()
        )

        if not observed_dates.equals(expected_dates):
            raise ValueError(f"Incomplete test series: {series_id}")

    if (
        test["date"] - pd.Timedelta(days=HORIZON) > ORIGIN
    ).any():
        raise ValueError("Features use sales after the forecast origin.")

    # Record the evaluation specification before fitting.
    specification = {
        "origin": str(ORIGIN.date()),
        "test_start": str(TEST_START.date()),
        "test_end": str(TEST_END.date()),
        "model_parameters": PARAMETERS,
        "quantiles": [0.1, 0.5, 0.9],
        "features": feature_names,
        "conservative_upper_adjustment_used": False,
    }

    (REPORTS / "holdout_specification.json").write_text(
        json.dumps(specification, indent=2),
        encoding="utf-8",
    )

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.int32,
    )

    x_train = train[feature_names].copy()
    x_test = test[feature_names].copy()

    x_train[categorical] = encoder.fit_transform(
        train[categorical]
    )
    x_test[categorical] = encoder.transform(
        test[categorical]
    )

    results = test[
        ["series_id", "date", "units"]
    ].rename(columns={"units": "actual"}).copy()

    trained_models = {}

    for quantile, name in [
        (0.1, "p10"),
        (0.5, "p50"),
        (0.9, "p90"),
    ]:
        print(f"Training {name.upper()}...")

        model = LGBMRegressor(
            **PARAMETERS,
            alpha=quantile,
        )

        model.fit(
            x_train,
            train["units"],
            categorical_feature=categorical,
        )

        raw = model.predict(x_test)

        if not isinstance(raw, np.ndarray):
            raise TypeError("Expected a NumPy prediction array.")

        prediction = raw.astype(np.float64)

        if prediction.ndim != 1 or len(prediction) != len(test):
            raise ValueError("Unexpected prediction shape.")

        if not np.isfinite(prediction).all():
            raise ValueError("Non-finite predictions.")

        results[f"raw_{name}"] = np.maximum(prediction, 0.0)
        trained_models[name] = model

    raw_quantiles = results[
        ["raw_p10", "raw_p50", "raw_p90"]
    ].to_numpy(dtype=float)

    crossed = (
        (raw_quantiles[:, 0] > raw_quantiles[:, 1])
        | (raw_quantiles[:, 1] > raw_quantiles[:, 2])
    )

    # Same adjustment used during validation.
    results["p10"] = raw_quantiles.min(axis=1)
    results["p50"] = raw_quantiles[:, 1]
    results["p90"] = raw_quantiles.max(axis=1)
    results["quantile_crossed"] = crossed

    # Seasonal-naive baseline: repeat the last observed week.
    baseline_rows = []

    last_week_dates = pd.date_range(
        ORIGIN - pd.Timedelta(days=6),
        ORIGIN,
        freq="D",
    )

    for series_id in sorted(test["series_id"].unique()):
        history = sales.loc[
            (sales["series_id"] == series_id)
            & sales["date"].between(
                last_week_dates[0],
                ORIGIN,
            )
        ].sort_values("date")

        if not pd.DatetimeIndex(history["date"]).equals(
            last_week_dates
        ):
            raise ValueError(f"Missing baseline history: {series_id}")

        last_week = history["units"].to_numpy(dtype=float)

        baseline_rows.append(pd.DataFrame({
            "series_id": series_id,
            "date": expected_dates,
            "baseline_prediction": np.tile(last_week, 4),
        }))

    baseline = pd.concat(baseline_rows, ignore_index=True)

    results = results.merge(
        baseline,
        on=["series_id", "date"],
        how="left",
        validate="one_to_one",
    )

    if results["baseline_prediction"].isna().any():
        raise ValueError("Missing baseline predictions.")

    actual = results["actual"].to_numpy(dtype=float)

    model_metrics = calculate_metrics(
        actual,
        results["p50"].to_numpy(dtype=float),
    )
    baseline_metrics = calculate_metrics(
        actual,
        results["baseline_prediction"].to_numpy(dtype=float),
    )

    model_mae = model_metrics["mae"]
    baseline_mae = baseline_metrics["mae"]

    improvement = None

    if (
        model_mae is not None
        and baseline_mae is not None
        and baseline_mae > 0
    ):
        improvement = (
            100 * (baseline_mae - model_mae) / baseline_mae
        )

    summary = {
        "evaluation": "Reserved final holdout",
        "origin": str(ORIGIN.date()),
        "test_start": str(TEST_START.date()),
        "test_end": str(TEST_END.date()),
        "series": int(results["series_id"].nunique()),
        "forecast_rows": len(results),
        "lightgbm_mae": model_mae,
        "lightgbm_wape_percent": model_metrics["wape_percent"],
        "baseline_mae": baseline_mae,
        "baseline_wape_percent": baseline_metrics["wape_percent"],
        "mae_improvement_percent": improvement,
        "raw_crossing_percent": float(100 * crossed.mean()),
        **interval_metrics(results),
        "test_evaluated": True,
    }

    results["origin"] = ORIGIN
    results["horizon_day"] = (
        results["date"] - ORIGIN
    ).dt.days

    results.to_csv(
        REPORTS / "holdout_predictions.csv",
        index=False,
    )

    (REPORTS / "holdout_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    # Save the fitted pre-holdout models and category encoder.
    joblib.dump(
        {
            "models": trained_models,
            "encoder": encoder,
            "feature_names": feature_names,
            "categorical_features": categorical,
            "origin": ORIGIN,
            "parameters": PARAMETERS,
        },
        MODELS / "holdout_model_bundle.joblib",
    )

    print("\nFINAL HOLDOUT RESULTS")
    print(json.dumps(summary, indent=2))
    print("\nSaved predictions, summary, and model bundle.")


if __name__ == "__main__":
    main()
from pathlib import Path
import json

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def pinball_loss(
    actual: np.ndarray,
    prediction: np.ndarray,
    quantile: float,
) -> float:
    error = actual - prediction
    return float(
        np.maximum(
            quantile * error,
            (quantile - 1) * error,
        ).mean()
    )


def interval_metrics(frame: pd.DataFrame) -> dict:
    actual = frame["actual"].to_numpy(dtype=float)
    lower = frame["p10"].to_numpy(dtype=float)
    median = frame["p50"].to_numpy(dtype=float)
    upper = frame["p90"].to_numpy(dtype=float)

    covered = (actual >= lower) & (actual <= upper)
    width = upper - lower

    # Proper interval score for a nominal central 80% interval.
    alpha = 0.20
    score = (
        width
        + (2 / alpha) * np.maximum(lower - actual, 0)
        + (2 / alpha) * np.maximum(actual - upper, 0)
    )

    return {
        "coverage_percent": float(100 * covered.mean()),
        "mean_interval_width_units": float(width.mean()),
        "mean_interval_score": float(score.mean()),
        "p10_pinball_loss": pinball_loss(actual, lower, 0.1),
        "p50_pinball_loss": pinball_loss(actual, median, 0.5),
        "p90_pinball_loss": pinball_loss(actual, upper, 0.9),
        "below_p10_percent": float(100 * (actual < lower).mean()),
        "above_p90_percent": float(100 * (actual > upper).mean()),
    }


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    data = pd.read_parquet(PROCESSED / "features.parquet")
    data["date"] = pd.to_datetime(data["date"])

    config = json.loads(
        (PROCESSED / "feature_config.json").read_text(
            encoding="utf-8"
        )
    )

    categorical = config["categorical_features"]
    feature_names = categorical + config["numeric_features"]
    horizon = int(config["horizon_days"])

    folds = pd.read_csv(
        REPORTS / "baseline_fold_metrics.csv",
        parse_dates=[
            "origin",
            "validation_start",
            "validation_end",
        ],
    ).sort_values("fold")

    # Reuse the P50 forecasts already evaluated.
    p50_results = pd.read_csv(
        REPORTS / "lightgbm_predictions.csv",
        parse_dates=["date", "origin"],
    )

    keys = ["fold", "date", "series_id"]

    if p50_results.duplicated(keys).any():
        raise ValueError("Duplicate P50 prediction rows.")

    all_results = []
    fold_reports = []

    for fold in folds.itertuples(index=False):
        fold_number = int(str(fold.fold))
        origin = pd.Timestamp(str(fold.origin))
        start = pd.Timestamp(str(fold.validation_start))
        end = pd.Timestamp(str(fold.validation_end))

        train = data.loc[data["date"] <= origin].copy()
        valid = data.loc[
            data["date"].between(start, end)
        ].copy()

        if train.empty or valid.empty:
            raise ValueError("Empty training or validation data.")

        if train["date"].max() >= valid["date"].min():
            raise ValueError("Training and validation overlap.")

        latest_input = (
            valid["date"] - pd.Timedelta(days=horizon)
        )

        if (latest_input > origin).any():
            raise ValueError("Unavailable sales used in features.")

        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int32,
        )

        x_train = train[feature_names].copy()
        x_valid = valid[feature_names].copy()

        x_train[categorical] = encoder.fit_transform(
            train[categorical]
        )
        x_valid[categorical] = encoder.transform(
            valid[categorical]
        )

        result = valid[
            ["date", "series_id", "units"]
        ].rename(columns={"units": "actual"}).copy()

        result["fold"] = fold_number

        for quantile, column in [(0.1, "raw_p10"), (0.9, "raw_p90")]:
            model = LGBMRegressor(
                objective="quantile",
                alpha=quantile,
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=15,
                min_child_samples=50,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=2,
                verbosity=-1,
            )

            model.fit(
                x_train,
                train["units"],
                categorical_feature=categorical,
            )

            raw = model.predict(x_valid)

            if not isinstance(raw, np.ndarray):
                raise TypeError("Expected a NumPy prediction array.")

            prediction = raw.astype(np.float64)

            if prediction.ndim != 1 or len(prediction) != len(valid):
                raise ValueError("Unexpected prediction shape.")

            if not np.isfinite(prediction).all():
                raise ValueError("Non-finite predictions.")

            result[column] = np.maximum(prediction, 0.0)

        saved = p50_results.loc[
            p50_results["fold"] == fold_number,
            keys + ["actual", "prediction", "origin"],
        ].rename(columns={
            "actual": "saved_actual",
            "prediction": "raw_p50",
            "origin": "saved_origin",
        })

        result = result.merge(
            saved,
            on=keys,
            how="outer",
            validate="one_to_one",
            indicator=True,
        )

        if not result["_merge"].eq("both").all():
            raise ValueError("P50 and interval forecast rows differ.")

        if not np.allclose(
            result["actual"].to_numpy(dtype=float),
            result["saved_actual"].to_numpy(dtype=float),
        ):
            raise ValueError("Actual values differ from saved P50 run.")

        if not result["saved_origin"].eq(origin).all():
            raise ValueError("Saved P50 forecast origin differs.")

        raw_quantiles = result[
            ["raw_p10", "raw_p50", "raw_p90"]
        ].to_numpy(dtype=float)

        if not np.isfinite(raw_quantiles).all():
            raise ValueError("Non-finite quantile forecasts.")

        # Independent quantile models can cross.
        crossing = (
            (raw_quantiles[:, 0] > raw_quantiles[:, 1])
            | (raw_quantiles[:, 1] > raw_quantiles[:, 2])
        )
        result["quantile_crossed"] = crossing

        # Expand bounds around the existing P50, preserving its forecast.
        # Retain raw predictions so this adjustment remains transparent.
        result["p10"] = raw_quantiles.min(axis=1)
        result["p50"] = raw_quantiles[:, 1]
        result["p90"] = raw_quantiles.max(axis=1)

        result["origin"] = origin
        result["horizon_day"] = (
            result["date"] - origin
        ).dt.days

        result = result.drop(
            columns=["saved_actual", "saved_origin", "_merge"]
        )

        metrics = interval_metrics(result)

        fold_reports.append({
            "fold": fold_number,
            "origin": str(origin.date()),
            "raw_crossing_percent": float(100 * crossing.mean()),
            **metrics,
        })

        all_results.append(result)

        print(
            f"Fold {fold_number} | "
            f"Coverage: {metrics['coverage_percent']:.2f}% | "
            f"Width: {metrics['mean_interval_width_units']:.3f} | "
            f"Crossing: {100 * crossing.mean():.2f}%"
        )

    if not all_results:
        raise ValueError("No folds evaluated.")

    results = pd.concat(all_results, ignore_index=True)

    if len(results) != len(p50_results):
        raise ValueError("Forecast count differs from P50 evaluation.")

    results.to_csv(
        REPORTS / "quantile_predictions.csv",
        index=False,
    )

    pd.DataFrame(fold_reports).to_csv(
        REPORTS / "quantile_fold_metrics.csv",
        index=False,
    )

    # Also inspect whether performance changes with forecast lead time.
    horizon_reports = []

    for day in range(1, horizon + 1):
        subset = results.loc[results["horizon_day"] == day]
        if not subset.empty:
            horizon_reports.append({
                "horizon_day": day,
                **interval_metrics(subset),
            })

    pd.DataFrame(horizon_reports).to_csv(
        REPORTS / "quantile_horizon_metrics.csv",
        index=False,
    )

    summary = {
        "forecast_rows": len(results),
        "nominal_coverage_percent": 80.0,
        "raw_crossing_percent": float(
            100 * results["quantile_crossed"].mean()
        ),
        "crossing_adjustment": (
            "Expand bounds to include all three forecasts; preserve P50"
        ),
        "metrics_use_adjusted_bounds": True,
        **interval_metrics(results),
        "test_evaluated": False,
    }

    (REPORTS / "quantile_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nQUANTILE SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nThe reserved test period remains unevaluated.")


if __name__ == "__main__":
    main()
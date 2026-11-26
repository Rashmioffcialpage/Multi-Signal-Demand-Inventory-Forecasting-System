from pathlib import Path
import json

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def calculate_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float | None]:
    errors = np.abs(actual - prediction)
    denominator = float(actual.sum())

    return {
        "mae": float(errors.mean()),
        "wape_percent": (
            float(100 * errors.sum() / denominator)
            if denominator > 0
            else None
        ),
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
    numeric = config["numeric_features"]
    feature_names = categorical + numeric
    horizon = int(config["horizon_days"])

    # Use exactly the same validation windows as the baseline.
    folds = pd.read_csv(
        REPORTS / "baseline_fold_metrics.csv",
        parse_dates=[
            "origin",
            "validation_start",
            "validation_end",
        ],
    ).sort_values("fold")

    baseline = pd.read_csv(
        REPORTS / "baseline_predictions.csv",
        parse_dates=["date"],
    )

    all_predictions = []
    fold_metrics = []

    for fold in folds.itertuples(index=False):
        fold_number = int(str(fold.fold))
        origin = pd.Timestamp(str(fold.origin))
        validation_start = pd.Timestamp(
            str(fold.validation_start)
        )
        validation_end = pd.Timestamp(
            str(fold.validation_end)
        )

        train = data.loc[
            data["date"] <= origin
        ].copy()

        valid = data.loc[
            data["date"].between(
                validation_start,
                validation_end,
            )
        ].copy()

        if train.empty or valid.empty:
            raise ValueError(
                f"Fold {fold_number}: empty training or validation data."
            )

        if train["date"].max() >= valid["date"].min():
            raise ValueError(
                f"Fold {fold_number}: training and validation overlap."
            )

        # Every sales-based feature uses data at least 28 days old.
        latest_sales_input = (
            valid["date"] - pd.Timedelta(days=horizon)
        )

        if (latest_sales_input > origin).any():
            raise ValueError(
                f"Fold {fold_number}: unavailable sales used in features."
            )

        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int32,
        )

        x_train = train[feature_names].copy()
        x_valid = valid[feature_names].copy()

        # Learn category mappings from training history only.
        x_train[categorical] = encoder.fit_transform(
            train[categorical]
        )
        x_valid[categorical] = encoder.transform(
            valid[categorical]
        )

        model = LGBMRegressor(
            objective="quantile",
            alpha=0.5,
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

        raw_prediction = model.predict(x_valid)

        if not isinstance(raw_prediction, np.ndarray):
            raise TypeError(
                "Expected LightGBM to return a NumPy prediction array."
            )

        prediction = np.maximum(
            raw_prediction.astype(np.float64),
            0.0,
        )

        if prediction.ndim != 1:
            raise ValueError("Expected one prediction per row.")

        if len(prediction) != len(valid):
            raise ValueError("Prediction count does not match validation.")

        if not np.isfinite(prediction).all():
            raise ValueError("Predictions contain NaN or infinity.")

        result = valid[
            ["date", "series_id", "units"]
        ].rename(columns={"units": "actual"}).copy()

        result["fold"] = fold_number
        result["origin"] = origin
        result["horizon_day"] = (
            result["date"] - origin
        ).dt.days
        result["prediction"] = prediction
        result["model"] = "lightgbm_p50"

        all_predictions.append(result)

        metrics = calculate_metrics(
            result["actual"].to_numpy(dtype=np.float64),
            prediction,
        )

        fold_metrics.append({
            "fold": fold_number,
            "origin": str(origin.date()),
            "validation_start": str(validation_start.date()),
            "validation_end": str(validation_end.date()),
            "train_rows": len(train),
            "validation_rows": len(valid),
            **metrics,
        })

        mae = metrics["mae"]
        wape = metrics["wape_percent"]

        mae_text = f"{mae:.3f}" if mae is not None else "N/A"
        wape_text = f"{wape:.2f}%" if wape is not None else "N/A"

        print(
            f"Fold {fold_number} | "
            f"Train rows: {len(train):,} | "
            f"MAE: {mae_text} | "
            f"WAPE: {wape_text}"
        )

    if not all_predictions:
        raise ValueError("No validation folds were evaluated.")

    results = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    keys = ["fold", "date", "series_id"]

    if results.duplicated(keys).any():
        raise ValueError("Duplicate model forecast rows.")

    # Confirm a fair comparison on identical observations.
    matched = results.merge(
        baseline[keys + ["actual"]],
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "_baseline"),
    )

    if not matched["_merge"].eq("both").all():
        raise ValueError(
            "Baseline and LightGBM forecast rows differ."
        )

    if not np.allclose(
        matched["actual"].to_numpy(dtype=np.float64),
        matched["actual_baseline"].to_numpy(dtype=np.float64),
    ):
        raise ValueError(
            "Baseline and LightGBM actual values differ."
        )

    model_metrics = calculate_metrics(
        results["actual"].to_numpy(dtype=np.float64),
        results["prediction"].to_numpy(dtype=np.float64),
    )

    baseline_metrics = calculate_metrics(
        baseline["actual"].to_numpy(dtype=np.float64),
        baseline["prediction"].to_numpy(dtype=np.float64),
    )

    baseline_mae = baseline_metrics["mae"]
    model_mae = model_metrics["mae"]

    improvement = None

    if (
        baseline_mae is not None
        and model_mae is not None
        and baseline_mae > 0
    ):
        improvement = (
            100 * (baseline_mae - model_mae) / baseline_mae
        )

    summary = {
        "model": "lightgbm_p50",
        "folds": len(fold_metrics),
        "forecast_rows": len(results),
        "pooled_mae": model_mae,
        "pooled_wape_percent": model_metrics["wape_percent"],
        "baseline_mae": baseline_mae,
        "baseline_wape_percent": baseline_metrics["wape_percent"],
        "mae_improvement_percent": improvement,
        "test_evaluated": False,
    }

    results.to_csv(
        REPORTS / "lightgbm_predictions.csv",
        index=False,
    )

    pd.DataFrame(fold_metrics).to_csv(
        REPORTS / "lightgbm_fold_metrics.csv",
        index=False,
    )

    (REPORTS / "lightgbm_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nLIGHTGBM SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nSaved predictions and metrics in reports/.")
    print("The reserved test period remains unevaluated.")


if __name__ == "__main__":
    main()
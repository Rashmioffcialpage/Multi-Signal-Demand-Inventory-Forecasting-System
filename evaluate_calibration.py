from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"


def main():
    data = pd.read_csv(
        REPORTS / "quantile_predictions.csv",
        parse_dates=["date", "origin"],
    )

    if data.duplicated(["fold", "date", "series_id"]).any():
        raise ValueError("Duplicate forecast observations.")

    outputs = []
    metrics = []

    folds = sorted(data["fold"].unique())

    # The first fold supplies calibration history only.
    for fold in folds[1:]:
        history = data.loc[data["fold"] < fold].copy()
        current = data.loc[data["fold"] == fold].copy()

        if history.empty or current.empty:
            raise ValueError("Missing calibration or evaluation data.")

        origin = current["origin"].min()

        # Calibration outcomes must already be observed.
        if history["date"].max() > origin:
            raise ValueError("Calibration uses unavailable outcomes.")

        # Positive residual means actual demand exceeded the upper bound.
        residuals = (
            history["actual"] - history["p90"]
        ).to_numpy(dtype=float)

        if not np.isfinite(residuals).all():
            raise ValueError("Non-finite calibration residuals.")

        # Widen only; retain the original bound if no increase is needed.
        adjustment = max(
            0.0,
            float(np.quantile(residuals, 0.90, method="higher")),
        )

        current["p90_adjusted"] = current["p90"] + adjustment
        current["upper_adjustment_units"] = adjustment

        actual = current["actual"].to_numpy(dtype=float)
        lower = current["p10"].to_numpy(dtype=float)
        original = current["p90"].to_numpy(dtype=float)
        adjusted = current["p90_adjusted"].to_numpy(dtype=float)

        row = {
            "fold": int(fold),
            "calibration_rows": len(history),
            "evaluation_rows": len(current),
            "upper_adjustment_units": adjustment,
            "original_above_p90_percent": float(
                100 * (actual > original).mean()
            ),
            "adjusted_above_p90_percent": float(
                100 * (actual > adjusted).mean()
            ),
            "original_coverage_percent": float(
                100 * (
                    (actual >= lower) & (actual <= original)
                ).mean()
            ),
            "adjusted_coverage_percent": float(
                100 * (
                    (actual >= lower) & (actual <= adjusted)
                ).mean()
            ),
            "original_mean_width": float(
                (original - lower).mean()
            ),
            "adjusted_mean_width": float(
                (adjusted - lower).mean()
            ),
        }

        metrics.append(row)
        outputs.append(current)

        print(
            f"Fold {fold} | "
            f"Adjustment: {adjustment:.3f} units | "
            f"Upper misses: "
            f"{row['original_above_p90_percent']:.2f}% → "
            f"{row['adjusted_above_p90_percent']:.2f}% | "
            f"Coverage: {row['adjusted_coverage_percent']:.2f}%"
        )

    if not outputs:
        raise ValueError("Need at least two validation folds.")

    pd.concat(outputs, ignore_index=True).to_csv(
        REPORTS / "calibrated_validation_predictions.csv",
        index=False,
    )

    pd.DataFrame(metrics).to_csv(
        REPORTS / "calibration_metrics.csv",
        index=False,
    )

    summary = {
        "method": "Historical pooled additive upper adjustment",
        "evaluation_folds": [int(fold) for fold in folds[1:]],
        "p50_changed": False,
        "p10_changed": False,
        "test_evaluated": False,
        "limitation": (
            "Pooled daily adjustment; no guaranteed per-product "
            "or inventory lead-time coverage."
        ),
        "fold_results": metrics,
    }

    (REPORTS / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nSaved calibration predictions and metrics in reports/.")
    print("The reserved test period remains unevaluated.")


if __name__ == "__main__":
    main()
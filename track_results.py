from pathlib import Path
import hashlib
import json
import math

import mlflow

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
TRACKING = ROOT / "tracking"

JOBS = [
    (
        "seasonal_naive_validation",
        "baseline_summary.json",
        [
            "baseline_fold_metrics.csv",
            "baseline_predictions.csv",
        ],
    ),
    (
        "lightgbm_p50_validation",
        "lightgbm_summary.json",
        [
            "lightgbm_fold_metrics.csv",
            "lightgbm_predictions.csv",
        ],
    ),
    (
        "lightgbm_quantiles_validation",
        "quantile_summary.json",
        [
            "quantile_fold_metrics.csv",
            "quantile_horizon_metrics.csv",
            "quantile_predictions.csv",
        ],
    ),
    (
        "upper_bound_adjustment",
        "calibration_summary.json",
        [
            "calibration_metrics.csv",
            "calibrated_validation_predictions.csv",
        ],
    ),
    (
        "inventory_scenario",
        "inventory_assumptions.json",
        ["inventory_recommendations.csv"],
    ),
]


def main():
    # Check all inputs before creating any tracking runs.
    required = {
        filename
        for _, summary_file, artifacts in JOBS
        for filename in [summary_file, *artifacts]
    }

    missing = [
        filename
        for filename in sorted(required)
        if not (REPORTS / filename).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"Missing reports: {', '.join(missing)}"
        )

    TRACKING.mkdir(parents=True, exist_ok=True)
    artifacts_dir = TRACKING / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    tracking_uri = (
        f"sqlite:///{(TRACKING / 'mlflow.db').as_posix()}"
    )
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = "DemandLens"
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=artifacts_dir.as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id

    for run_name, summary_file, artifacts in JOBS:
        summary_path = REPORTS / summary_file
        summary = json.loads(
            summary_path.read_text(encoding="utf-8")
        )

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
        ) as run:
            mlflow.set_tags({
                "project": "DemandLens",
                "logging_mode": "retrospective_report_import",
                "test_evaluated": "false",
                "model_artifact_saved": "false",
            })

            # Log numeric top-level summary fields as metrics.
            # Booleans, text, and nested results stay in the JSON artifact.
            for key, value in summary.items():
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                ):
                    mlflow.log_metric(key, float(value))

            for filename in [summary_file, *artifacts]:
                path = REPORTS / filename

                mlflow.log_artifact(
                    str(path),
                    artifact_path="reports",
                )

                digest = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()

                mlflow.set_tag(
                    f"sha256_{path.stem}",
                    digest,
                )

            config = ROOT / "data" / "processed" / "feature_config.json"

            if config.exists():
                mlflow.log_artifact(
                    str(config),
                    artifact_path="configuration",
                )

            print(f"Logged: {run_name}")
            print(f"Run ID: {run.info.run_id}\n")

    print("All five report groups logged.")
    print(f"Tracking URI: {tracking_uri}")
    print(
        "This imports existing reports; it does not retrain "
        "models or evaluate the reserved test period."
    )


if __name__ == "__main__":
    main()
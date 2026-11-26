from pathlib import Path
import sqlite3
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATABASE = ROOT / "data" / "demandlens.db"

st.set_page_config(
    page_title="DemandLens",
    page_icon="📈",
    layout="wide",
)

st.title("DemandLens")
st.caption(
    "Demand forecasting · Validation results · "
    "LightGBM quantile models · SQLite"
)

if not DATABASE.exists():
    st.error("Database missing. Run python build_database.py first.")
    st.stop()


@st.cache_data
def load_forecasts(database_modified: int) -> pd.DataFrame:
    with sqlite3.connect(
        f"{DATABASE.as_uri()}?mode=ro",
        uri=True,
    ) as connection:
        return pd.read_sql_query(
            """
            SELECT *
            FROM validation_forecasts
            ORDER BY fold, series_id, date
            """,
            connection,
            parse_dates=["date", "origin"],
        )


@st.cache_data
def load_history(
    database_modified: int,
    series_id: str,
    origin: str,
) -> pd.DataFrame:
    with sqlite3.connect(
        f"{DATABASE.as_uri()}?mode=ro",
        uri=True,
    ) as connection:
        return pd.read_sql_query(
            """
            SELECT date, units
            FROM daily_sales
            WHERE series_id = ?
              AND date <= ?
            ORDER BY date DESC
            LIMIT 56
            """,
            connection,
            params=(series_id, origin),
            parse_dates=["date"],
        ).sort_values("date")


modified = DATABASE.stat().st_mtime_ns
forecasts = load_forecasts(modified)

st.sidebar.header("Explore forecasts")

fold = st.sidebar.selectbox(
    "Validation fold",
    sorted(forecasts["fold"].unique().tolist()),
)

fold_data = forecasts.loc[forecasts["fold"] == fold].copy()
series_id = st.sidebar.selectbox(
    "Product series",
    sorted(fold_data["series_id"].unique().tolist()),
)

if series_id is None:
    st.info("Select a product series to continue.")
    st.stop()

series_id = str(series_id)

selected = fold_data.loc[
    fold_data["series_id"] == series_id
].sort_values("date").copy()

selected = fold_data.loc[
    fold_data["series_id"] == series_id
].sort_values("date").copy()

origin = selected["origin"].iloc[0]
origin_text = origin.strftime("%Y-%m-%d")

st.sidebar.write(f"Forecast origin: **{origin_text}**")
st.sidebar.caption(
    "Each forecast covers 28 days. The final reserved test "
    "period is not displayed here."
)

scope = st.sidebar.radio(
    "Metric scope",
    ["Selected product", "All products in this fold"],
)

metric_data = (
    selected
    if scope == "Selected product"
    else fold_data
)

actual = metric_data["actual"]
errors = (actual - metric_data["p50"]).abs()
denominator = float(actual.sum())

mae = float(errors.mean())
wape = (
    100 * float(errors.sum()) / denominator
    if denominator > 0 else None
)

coverage = 100 * (
    (actual >= metric_data["p10"])
    & (actual <= metric_data["p90"])
).mean()

upper_misses = 100 * (
    actual > metric_data["p90"]
).mean()

st.subheader(series_id.replace("_validation", ""))
st.caption(
    f"Fold {fold} · Origin {origin_text} · "
    f"Metrics: {scope.lower()}"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("P50 MAE", f"{mae:.3f} units")
c2.metric("WAPE", f"{wape:.2f}%" if wape is not None else "N/A")
c3.metric("P10–P90 coverage", f"{coverage:.2f}%")
c4.metric("Above P90", f"{upper_misses:.2f}%")

st.caption(
    "Metrics above use the original bounds after crossing "
    "adjustment. Coverage is descriptive validation performance."
)
forecast_tab, results_tab, inventory_tab, notes_tab = st.tabs([
    "Forecast",
    "Product results",
    "Inventory",
    "Method & limitations",
])

with forecast_tab:
    show_adjusted = st.checkbox(
        "Show conservative upper-bound scenario",
        value=False,
        disabled=bool(selected["p90_adjusted"].isna().all()),
    )

    history = load_history(modified, series_id, origin_text)

    fig = go.Figure()

    # Add lower then upper traces to shade the interval.
    fig.add_trace(go.Scatter(
        x=selected["date"],
        y=selected["p10"],
        mode="lines",
        line=dict(width=0),
        name="P10",
        hovertemplate="%{x|%Y-%m-%d}<br>P10: %{y:.2f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=selected["date"],
        y=selected["p90"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(55, 110, 180, 0.18)",
        name="P10–P90 range",
        hovertemplate="%{x|%Y-%m-%d}<br>P90: %{y:.2f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=history["date"],
        y=history["units"],
        mode="lines",
        name="Observed history",
        line=dict(color="#8795a5", width=1.5),
    ))

    fig.add_trace(go.Scatter(
        x=selected["date"],
        y=selected["actual"],
        mode="lines+markers",
        name="Actual validation demand",
        line=dict(color="#18354e", width=2),
        marker=dict(size=4),
    ))

    fig.add_trace(go.Scatter(
        x=selected["date"],
        y=selected["p50"],
        mode="lines",
        name="P50 forecast",
        line=dict(color="#3679bf", width=2.5),
    ))

    if show_adjusted:
        fig.add_trace(go.Scatter(
            x=selected["date"],
            y=selected["p90_adjusted"],
            mode="lines",
            name="Conservative upper scenario",
            line=dict(color="#c08b38", dash="dash", width=2),
        ))

    fig.update_layout(
        template="plotly_white",
        height=460,
        xaxis_title="Date",
        yaxis_title="Daily units",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.22),
        margin=dict(l=15, r=15, t=20, b=90),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "The shaded range is nominally 80%; actual coverage "
        "varies by product and time. It is not an inventory "
        "service-level guarantee."
    )

    st.download_button(
        "Download selected forecast",
        data=selected.to_csv(index=False).encode("utf-8"),
        file_name=f"demandlens_fold_{fold}_{series_id}.csv",
        mime="text/csv",
    )

with results_tab:
    rows = []

    for product, group in fold_data.groupby("series_id"):
        covered = (
            (group["actual"] >= group["p10"])
            & (group["actual"] <= group["p90"])
        )

        rows.append({
            "Product": product,
            "Actual units": float(group["actual"].sum()),
            "MAE": float(
                (group["actual"] - group["p50"]).abs().mean()
            ),
            "Coverage %": float(100 * covered.mean()),
            "Above P90 %": float(
                100 * (group["actual"] > group["p90"]).mean()
            ),
            "Mean interval width": float(
                (group["p90"] - group["p10"]).mean()
            ),
        })

    table = pd.DataFrame(rows).sort_values(
        "Above P90 %",
        ascending=False,
    )

    st.dataframe(
        table.round(2),
        hide_index=True,
        use_container_width=True,
    )
with inventory_tab:
    st.subheader("Inventory planning scenario")

    inventory_file = ROOT / "reports" / "inventory_recommendations.csv"
    assumptions_file = ROOT / "reports" / "inventory_assumptions.json"

    if not inventory_file.exists() or not assumptions_file.exists():
        st.info(
            "Run python inventory_recommendations.py "
            "to generate this scenario."
        )
    else:
        inventory = pd.read_csv(inventory_file)
        assumptions = json.loads(
            assumptions_file.read_text(encoding="utf-8")
        )

        scenario_fold = int(assumptions["validation_fold"])
        scenario_origin = str(assumptions["origin"])

        st.warning(
            "Illustrative recommendations using assumed stock levels. "
            "These are not observed inventory records or guaranteed "
            "service levels."
        )

        st.caption(
            f"Scenario: Fold {scenario_fold} · "
            f"Origin: {scenario_origin} · "
            f"Lead time: {assumptions['lead_time_days']} days · "
            f"Review period: {assumptions['review_period_days']} days"
        )

        # The CSV currently contains only the Fold 3 scenario.
        if str(fold) != str(scenario_fold):
            st.info(
                f"Select Fold {scenario_fold} in the sidebar "
                "to view its inventory recommendations."
            )

        elif scenario_origin != origin_text:
            st.error(
                "Scenario origin does not match the selected forecast. "
                "Regenerate inventory_recommendations.csv."
            )

        else:
            product_inventory = inventory.loc[
                inventory["series_id"] == series_id
            ]

            if len(product_inventory) != 1:
                st.error(
                    "Expected exactly one recommendation "
                    "for the selected product."
                )
            else:
                row = product_inventory.iloc[0]

                stock = int(str(row["assumed_on_hand"]))
                target = int(str(row["target_stock_units"]))
                order = int(str(row["recommended_order_units"]))

                a, b, c = st.columns(3)
                a.metric("Assumed stock on hand", stock)
                b.metric("Target stock", target)
                c.metric("Recommended order", order)

                st.write(
                    "Central 14-day planning estimate: "
                    f"**{float(row['central_protection_units']):.2f} units**"
                )

                st.write(
                    "Historical 14-day block P90: "
                    f"**{float(row['historical_block_p90_units']):.2f} units**"
                )

                risk_flag = (
                    str(row["pre_arrival_risk_flag"]).strip().lower()
                    == "true"
                )

                if risk_flag:
                    st.warning(
                        "Assumed available stock is below the central "
                        "lead-time sales estimate. An order arriving "
                        "after seven days may not prevent an earlier shortage."
                    )
                else:
                    st.info(
                        "Assumed available stock covers the central "
                        "lead-time estimate. This does not rule out shortages."
                    )

                st.caption(
                    "Target stock is the rounded-up maximum of the "
                    "summed daily P50 forecasts and historical block P90. "
                    "Neither is a calibrated lead-time demand quantile."
                )

            st.divider()
            st.subheader("All product recommendations")
            st.caption(
                "This table covers every product in the scenario, "
                "regardless of the sidebar metric scope."
            )

            display_columns = [
                "series_id",
                "central_protection_units",
                "historical_block_p90_units",
                "target_stock_units",
                "assumed_on_hand",
                "recommended_order_units",
                "pre_arrival_risk_flag",
            ]

            st.dataframe(
                inventory[display_columns].round(2),
                hide_index=True,
                use_container_width=True,
            )

            st.download_button(
                "Download inventory recommendations",
                data=inventory.to_csv(index=False).encode("utf-8"),
                file_name="demandlens_inventory_scenario.csv",
                mime="text/csv",
                key="inventory_download",
            )

            with st.expander("Scenario assumptions"):
                st.json(assumptions)
with notes_tab:
    st.markdown("""
- **Scope:** 20 food-product series from store CA_1.
- **Validation:** three rolling-origin windows, 28 days each.
- **Features:** sales lags of at least 28 days, historical
  prices, calendar events, and SNAP indicators.
- **Models:** independently trained P10, P50, and P90 models.
- **Crossing adjustment:** expand bounds to contain all three
  predictions while preserving P50.
- **Conservative scenario:** an additive upper adjustment
  learned only from earlier validation folds.
- **Demand limitation:** recorded sales may differ from
  unconstrained demand when stockouts occur.
- **Inventory:** recommendations use assumed stock and lead
  times. Daily quantiles cannot simply be summed to obtain
  lead-time demand quantiles.
- **Final test:** March 28–April 24, 2016 was evaluated
  separately using the model settings retained from validation.
- **Dashboard scope:** the forecast chart displays validation
  folds, not final holdout predictions.
""")

    holdout_file = ROOT / "reports" / "holdout_summary.json"

    if holdout_file.exists():
        holdout = json.loads(
            holdout_file.read_text(encoding="utf-8")
        )

        st.subheader("Final holdout results")

        st.caption(
            f"{holdout['test_start']} to {holdout['test_end']} · "
            f"{holdout['forecast_rows']} forecasts · "
            f"{holdout['series']} product series"
        )

        h1, h2, h3 = st.columns(3)

        h1.metric(
            "LightGBM MAE",
            f"{holdout['lightgbm_mae']:.3f} units",
        )

        h2.metric(
            "Seasonal-naive MAE",
            f"{holdout['baseline_mae']:.3f} units",
        )

        improvement = holdout.get("mae_improvement_percent")

        h3.metric(
            "MAE reduction",
            f"{improvement:.2f}%"
            if improvement is not None
            else "N/A",
        )

        st.info(
            f"The nominal 80% interval achieved "
            f"{holdout['coverage_percent']:.2f}% coverage "
            "on this holdout, indicating conservative intervals. "
            "These results apply to the selected products "
            "and evaluation period."
        )

        comparison = pd.DataFrame([
            {
                "Model": "Seasonal naive",
                "MAE": holdout["baseline_mae"],
                "WAPE (%)": holdout["baseline_wape_percent"],
            },
            {
                "Model": "LightGBM P50",
                "MAE": holdout["lightgbm_mae"],
                "WAPE (%)": holdout["lightgbm_wape_percent"],
            },
        ])

        st.dataframe(
            comparison.round(3),
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "Download holdout summary",
            data=json.dumps(holdout, indent=2),
            file_name="demandlens_holdout_summary.json",
            mime="application/json",
            key="holdout_summary_download",
        )

    else:
        st.warning(
            "Holdout summary not found. Place the generated "
            "holdout_summary.json file in reports/ to display results."
        )
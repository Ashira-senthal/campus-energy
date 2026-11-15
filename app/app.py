"""
Campus Energy Intelligence - Streamlit dashboard.

Run from the project root (the app needs ``data/processed/*.parquet``):

    .venv/bin/python -m streamlit run app/app.py

Design notes
------------
* The app never trains a model. It reads small parquet files written by
  ``run_pipeline.py``, so it loads in a couple of seconds.
* Anomaly flags are recomputed live from the stored residuals, and tagging uses
  the *same* ``src.explain.tag_flagged_hours`` function the evaluation used, so
  the dashboard can never disagree with the reported metrics.
* Every widget is defensive: an empty selection shows a message instead of a
  traceback.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.anomaly import apply_persistence, flatline_flags  # noqa: E402
from src.config import (  # noqa: E402
    BUILDING_LABELS,
    CO2_KG_PER_KWH,
    MODEL_LABELS,
    PERSIST_K,
    PERSIST_N,
    TARIFF_INR_PER_KWH,
    Z_THRESHOLD,
)
from src.explain import estimate_impact, tag_flagged_hours  # noqa: E402

PROCESSED = ROOT / "data" / "processed"

METHOD_LABELS = {
    "residual": "Residual (robust z)",
    "residual+persistence": "Residual + persistence",
    "isolation_forest": "Isolation Forest",
    "hybrid": "Hybrid (persistence + flatline guard)",
    "adaptive_hybrid": "Adaptive hybrid (30-day recalibration)",
}

st.set_page_config(page_title="Campus Energy Intelligence", page_icon="⚡", layout="wide")


# ---------------------------------------------------------------------------
# Data loading (cached: parquet reads happen once per session)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_tables() -> dict[str, pd.DataFrame]:
    """Read every dashboard input; raise a readable error if the pipeline has not run."""
    files = {
        "hourly": "hourly.parquet",
        "predictions": "predictions.parquet",
        "residuals": "residuals.parquet",
        "metrics": "metrics.parquet",
        "hour_errors": "errors_by_hour.parquet",
        "detection": "detection.parquet",
        "sensitivity": "sensitivity.parquet",
        "importance": "importance.parquet",
        "shap": "shap.parquet",
        "buildings": "buildings.parquet",
    }
    missing = [f for f in files.values() if not (PROCESSED / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dashboard inputs: " + ", ".join(missing) + ". Run `python run_pipeline.py` first."
        )
    tables = {key: pd.read_parquet(PROCESSED / name) for key, name in files.items()}
    for key in ("hourly", "predictions", "residuals"):
        tables[key]["timestamp"] = pd.to_datetime(tables[key]["timestamp"])
    return tables


@st.cache_data(show_spinner=False)
def cached_tags(residuals: pd.DataFrame, method: str, z: float) -> pd.DataFrame:
    """Recompute flags + tags per building using ``method`` and ``z``.

    Grouping by building is essential, not cosmetic: the persistence rule and the
    flatline guard look at *previous hours*, so if four buildings were concatenated
    into one series each building would inherit its neighbour's last hour as
    "the previous hour" and the flags would be wrong.
    """
    if residuals.empty:
        return residuals.iloc[0:0]

    tagged_parts: list[pd.DataFrame] = []
    for _, group in residuals.groupby("building_id"):
        frame = group.sort_values("timestamp").reset_index(drop=True)
        z_values = frame["z"].to_numpy(dtype="float64")
        raw = np.abs(z_values) > z
        persisted = apply_persistence(raw, PERSIST_K, PERSIST_N)
        flat = flatline_flags(frame["electricity"])
        isolation = frame["if_score"].to_numpy() < frame["if_threshold_used"].to_numpy()
        adaptive_z = frame["adaptive_z"].to_numpy(dtype="float64")
        adaptive = apply_persistence(np.abs(adaptive_z) > z, PERSIST_K, PERSIST_N) | flat

        flags = {
            "residual": raw,
            "residual+persistence": persisted,
            "isolation_forest": isolation,
            "hybrid": persisted | flat,
            "adaptive_hybrid": adaptive,
        }[method]
        tagged_parts.append(tag_flagged_hours(frame, flags, z_values))

    return pd.concat(tagged_parts, ignore_index=True)


def kpi_row(items: list[tuple[str, str]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value) in zip(columns, items):
        column.metric(label, value)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
def sidebar(tables: dict[str, pd.DataFrame]) -> dict:
    st.sidebar.title("⚡ Campus Energy")
    # Options are building IDs; labels are only for display, because every table
    # is keyed on building_id.
    available_ids = sorted(tables["hourly"]["building_id"].dropna().unique())
    label_of = lambda bid: BUILDING_LABELS.get(bid, bid)  # noqa: E731
    selected = st.sidebar.multiselect(
        "Buildings",
        available_ids,
        default=available_ids,
        format_func=label_of,
        help="Empty selection shows a message, not a crash.",
    )

    residuals = tables["residuals"]
    min_date = residuals["timestamp"].min()
    max_date = residuals["timestamp"].max()
    date_range = st.sidebar.date_input(
        "Test-period date range",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )

    model = st.sidebar.selectbox(
        "Forecast model",
        list(MODEL_LABELS),
        index=list(MODEL_LABELS).index("random_forest"),
        format_func=lambda key: MODEL_LABELS[key],
    )
    method = st.sidebar.selectbox(
        "Detection method",
        list(METHOD_LABELS),
        index=list(METHOD_LABELS).index("hybrid"),
        format_func=lambda key: METHOD_LABELS[key],
    )
    z_value = st.sidebar.slider(
        "Anomaly sensitivity (robust z threshold)",
        min_value=2.0,
        max_value=6.0,
        value=Z_THRESHOLD,
        step=0.25,
        help=(
            f"Lower = more alerts. Persistence rule: {PERSIST_K} of the last {PERSIST_N} hours. "
            "Isolation Forest ignores this slider (its cut is learned on validation)."
        ),
    )
    tariff = st.sidebar.number_input("Tariff (INR per kWh)", 0.0, 100.0, TARIFF_INR_PER_KWH, 0.5)
    co2 = st.sidebar.number_input("Grid CO₂ factor (kg per kWh)", 0.0, 2.0, CO2_KG_PER_KWH, 0.01)
    st.sidebar.caption(
        "Tariff and CO₂ factor are assumptions: verify them against your campus bill and the "
        "current CEA grid factor."
    )

    st.sidebar.divider()
    primary = st.sidebar.selectbox(
        "Building for detail views",
        selected if selected else available_ids[:1],
        format_func=label_of,
        disabled=not selected,
    )

    return {
        "buildings": selected,
        "date_range": date_range,
        "model": model,
        "method": method,
        "z": z_value,
        "tariff": tariff,
        "co2": co2,
        "primary": primary,
    }


def filter_by_date(frame: pd.DataFrame, date_range) -> pd.DataFrame:
    if len(date_range) != 2:
        return frame
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
    return frame[(frame["timestamp"] >= start) & (frame["timestamp"] < end)]


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def tab_overview(tables, controls, res_view, tags) -> None:
    st.subheader("Overview")
    if res_view.empty:
        st.info("No data for this selection. Widen the date range or add a building.")
        return

    total_kwh = float(res_view["electricity"].sum())
    peak = float(res_view["electricity"].max())
    average = float(res_view["electricity"].mean())
    impact = estimate_impact(tags, controls["tariff"], controls["co2"])
    hours = len(res_view)

    kpi_row(
        [
            ("Total consumption", f"{total_kwh:,.0f} kWh"),
            ("Peak hourly load", f"{peak:,.0f} kWh"),
            ("Average load", f"{average:,.0f} kWh"),
            ("Anomalous hours", f"{len(tags):,} ({100 * len(tags) / max(hours, 1):.1f}%)"),
        ]
    )
    kpi_row(
        [
            ("Excess (wasted) energy", f"{impact['excess_kwh']:,.0f} kWh"),
            ("Estimated cost", f"₹{impact['cost_inr']:,.0f}"),
            ("Estimated CO₂", f"{impact['co2_kg']:,.0f} kg"),
            ("Detection method", METHOD_LABELS[controls["method"]].split(" (")[0]),
        ]
    )

    st.markdown(
        "**Campus comparison** - mean hourly load by building. CV(RMSE) is shown because "
        "raw kWh is not comparable across a 17,000 m² laboratory and a 1,000 m² auditorium."
    )
    comparison = (
        res_view.groupby("building_label")
        .agg(
            mean_kwh=("electricity", "mean"),
            peak_kwh=("electricity", "max"),
            total_kwh=("electricity", "sum"),
        )
        .reset_index()
        .sort_values("mean_kwh", ascending=False)
    )
    test_metrics = tables["metrics"][
        (tables["metrics"]["fold"] == "test") & (tables["metrics"]["model"] == controls["model"])
    ][["building_id", "building_label", "cv_rmse_pct", "nmbe_pct"]]
    comparison = comparison.merge(
        test_metrics.drop(columns="building_id"), on="building_label", how="left"
    )
    fig = px.bar(
        comparison,
        x="building_label",
        y="mean_kwh",
        text=comparison["cv_rmse_pct"].map(lambda v: f"CV(RMSE) {v:.1f}%" if pd.notna(v) else ""),
        labels={"building_label": "", "mean_kwh": "mean hourly kWh"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=380, margin=dict(t=30, b=10))
    st.plotly_chart(fig, width="stretch")


def tab_patterns(tables, controls, res_view) -> None:
    st.subheader("Normal consumption patterns")
    hourly = tables["hourly"]
    hourly = hourly[hourly["building_id"].isin(controls["buildings"])]
    hourly = filter_by_date(hourly, controls["date_range"])
    if hourly.empty or not controls["buildings"]:
        st.info("No data for this selection.")
        return

    primary = controls["primary"]
    subset = hourly[hourly["building_id"] == primary].dropna(subset=["electricity"])
    if subset.empty:
        st.info("This building has no readings in the selected range.")
        return

    left, right = st.columns(2)
    with left:
        st.markdown(f"**Hour x weekday heatmap - {BUILDING_LABELS.get(primary, primary)}**")
        pivot = subset.pivot_table(
            index="dayofweek", columns="hour", values="electricity", aggfunc="mean"
        ).reindex(range(7))
        figure = go.Figure(
            go.Heatmap(
                z=pivot.to_numpy(),
                x=list(range(24)),
                y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                colorscale="Magma",
                colorbar=dict(title="kWh"),
            )
        )
        figure.update_layout(height=360, margin=dict(t=20, b=20), xaxis_title="hour of day")
        st.plotly_chart(figure, width="stretch")

    with right:
        st.markdown("**Weekday vs weekend daily profile**")
        profile = (
            subset.assign(kind=np.where(subset["is_weekend"] == 1, "weekend", "weekday"))
            .groupby(["kind", "hour"])["electricity"]
            .mean()
            .reset_index()
        )
        figure = px.line(profile, x="hour", y="electricity", color="kind", markers=True)
        figure.update_layout(height=360, margin=dict(t=20, b=20), yaxis_title="mean kWh")
        st.plotly_chart(figure, width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("**Load vs outdoor temperature**")
        sample = subset.dropna(subset=["airTemperature"])
        if len(sample) > 4000:
            sample = sample.sample(4000, random_state=42)
        figure = px.scatter(
            sample,
            x="airTemperature",
            y="electricity",
            opacity=0.35,
            labels={"airTemperature": "air temperature (°C)", "electricity": "kWh"},
        )
        figure.update_traces(marker=dict(size=4))
        figure.update_layout(height=360, margin=dict(t=20, b=20))
        st.plotly_chart(figure, width="stretch")

    with right:
        st.markdown("**Monthly mean load (all selected buildings)**")
        trend = (
            hourly.dropna(subset=["electricity"])
            .assign(month=lambda d: d["timestamp"].dt.to_period("M").dt.to_timestamp())
            .groupby(["month", "building_label"])["electricity"]
            .mean()
            .reset_index()
        )
        if trend.empty:
            st.info("Nothing to plot.")
        else:
            figure = px.line(trend, x="month", y="electricity", color="building_label", markers=True)
            figure.update_layout(
                height=360, margin=dict(t=20, b=20), yaxis_title="mean kWh", legend_title=""
            )
            st.plotly_chart(figure, width="stretch")


def tab_forecast(tables, controls, res_view) -> None:
    st.subheader("Forecast vs actual")
    if not controls["buildings"]:
        st.info("Select at least one building.")
        return

    predictions = tables["predictions"]
    predictions = predictions[predictions["building_id"].isin(controls["buildings"])]
    predictions = filter_by_date(predictions, controls["date_range"])
    selected_model = controls["model"]
    model_frame = predictions[predictions["model"] == selected_model]

    building = controls["primary"]
    building_frame = model_frame[model_frame["building_id"] == building].sort_values("timestamp")
    if building_frame.empty:
        st.info("No predictions for this building in the selected range.")
        return

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=building_frame["timestamp"],
            y=building_frame["actual"],
            name="actual",
            line=dict(color="#1f77b4", width=1.2),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=building_frame["timestamp"],
            y=building_frame["expected"],
            name="expected (model)",
            line=dict(color="#d62728", width=1.4, dash="dot"),
        )
    )
    figure.update_layout(
        height=380,
        margin=dict(t=20, b=10),
        yaxis_title="kWh",
        title=f"{BUILDING_LABELS.get(building, building)} - {MODEL_LABELS[selected_model]}",
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(figure, width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("**Model comparison on the test fold** (lower is better)")
        table = tables["metrics"][tables["metrics"]["fold"] == "test"]
        if controls["buildings"]:
            table = table[table["building_id"].isin(controls["buildings"])]
        table = (
            table.groupby("model")[["mae", "rmse", "cv_rmse_pct", "nmbe_pct", "smape_pct"]]
            .mean()
            .round(2)
            .reset_index()
        )
        table["model"] = table["model"].map(MODEL_LABELS)
        st.dataframe(
            table.rename(
                columns={
                    "model": "model",
                    "mae": "MAE (kWh)",
                    "rmse": "RMSE (kWh)",
                    "cv_rmse_pct": "CV(RMSE) %",
                    "nmbe_pct": "NMBE %",
                    "smape_pct": "sMAPE %",
                }
            ),
            hide_index=True,
        )
        st.caption(
            "CV(RMSE) and NMBE follow ASHRAE Guideline 14: scale-free, so buildings of "
            "different size can share a table."
        )

    with right:
        st.markdown("**Error by hour of day** (mean over selected buildings)")
        errors = tables["hour_errors"]
        errors = errors[
            (errors["model"] == selected_model) & (errors["building_id"].isin(controls["buildings"]))
        ]
        if errors.empty:
            st.info("No error breakdown available.")
        else:
            by_hour = errors.groupby("hour")[["mae", "bias"]].mean().reset_index()
            figure = px.bar(by_hour, x="hour", y="mae", labels={"mae": "MAE (kWh)"})
            figure.update_layout(height=300, margin=dict(t=20, b=10))
            st.plotly_chart(figure, width="stretch")
            st.caption(
                "Bias = mean(actual − expected); negative bias means the model over-predicts "
                "at that hour."
            )


def tab_anomalies(tables, controls, res_view, tags) -> None:
    st.subheader("Detected anomalies")
    if res_view.empty:
        st.info("No data for this selection.")
        return

    # The timeline is per building, so a four-way overlay would be unreadable.
    primary = controls["primary"]
    res_view = res_view[res_view["building_id"] == primary]
    tags = tags[tags["building_id"] == primary]
    if res_view.empty:
        st.info("No data for this building in the selected range.")
        return

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=res_view["timestamp"],
            y=res_view["electricity"],
            mode="lines",
            name="actual",
            line=dict(color="#9aa5b1", width=1),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=res_view["timestamp"],
            y=res_view["expected"],
            mode="lines",
            name="expected",
            line=dict(color="#1f77b4", width=1, dash="dot"),
        )
    )
    if not tags.empty:
        figure.add_trace(
            go.Scatter(
                x=tags["timestamp"],
                y=tags["electricity"],
                mode="markers",
                name="flagged",
                marker=dict(color="#d62728", size=8, symbol="x"),
                text=tags["anomaly_type"],
                hovertemplate="%{x}<br>%{y:.1f} kWh<br>%{text}<extra></extra>",
            )
        )
    figure.update_layout(
        height=380,
        margin=dict(t=20, b=10),
        yaxis_title="kWh",
        legend=dict(orientation="h", y=1.05),
        title=f"{BUILDING_LABELS.get(controls['primary'], controls['primary'])} - "
        f"{METHOD_LABELS[controls['method']]} at z={controls['z']:.2f}",
    )
    st.plotly_chart(figure, width="stretch")

    if tags.empty:
        st.success(
            "Nothing flagged for this selection. Try lowering the z threshold or picking a more "
            "sensitive method."
        )
        return

    impact = estimate_impact(tags, controls["tariff"], controls["co2"])
    kpi_row(
        [
            ("Flagged hours", f"{len(tags):,}"),
            ("Excess energy", f"{impact['excess_kwh']:,.1f} kWh"),
            ("Cost", f"₹{impact['cost_inr']:,.0f}"),
            ("CO₂", f"{impact['co2_kg']:,.0f} kg"),
        ]
    )

    display = tags.copy()
    display["timestamp"] = display["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    display = display.rename(
        columns={
            "timestamp": "timestamp",
            "building_id": "building",
            "electricity": "actual kWh",
            "expected": "expected kWh",
            "excess_kwh": "excess kWh",
            "z": "robust z",
            "severity": "severity",
            "anomaly_type": "type",
            "reason": "plain-English reason",
        }
    )[
        [
            "timestamp",
            "building",
            "actual kWh",
            "expected kWh",
            "excess kWh",
            "robust z",
            "severity",
            "type",
            "plain-English reason",
        ]
    ]
    st.dataframe(display.sort_values("timestamp", ascending=False), hide_index=True, height=360)
    st.download_button(
        "Download flagged hours as CSV",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="campus_anomalies.csv",
        mime="text/csv",
    )


def tab_explain(tables, controls, res_view, tags) -> None:
    st.subheader("Why the model expects what it expects")
    primary = controls["primary"]
    importance = tables["importance"]
    building_importance = importance[importance["building_id"] == primary].sort_values(
        "mean_abs_shap", ascending=False
    )
    if building_importance.empty:
        st.info("No SHAP values for this building.")
        return

    left, right = st.columns(2)
    with left:
        st.markdown("**Global feature importance (mean |SHAP|, test period)**")
        figure = px.bar(
            building_importance.head(12).sort_values("mean_abs_shap"),
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            labels={"mean_abs_shap": "mean |SHAP| (kWh)", "feature": ""},
        )
        figure.update_layout(height=420, margin=dict(t=20, b=10))
        st.plotly_chart(figure, width="stretch")

    with right:
        st.markdown("**SHAP distribution for the top features**")
        shap_values = tables["shap"]
        shap_values = shap_values[shap_values["building_id"] == primary]
        top = building_importance.head(8)["feature"].tolist()
        sample = shap_values[shap_values["feature"].isin(top)]
        if sample.empty:
            st.info("No SHAP detail for this building.")
        else:
            # px.strip has no continuous colour scale, so a jittered scatter is used
            # to get the beeswarm shape with the feature value as colour.
            figure = px.scatter(
                sample,
                x="shap",
                y="feature",
                color="value",
                color_continuous_scale="RdBu_r",
                labels={"shap": "SHAP contribution (kWh)", "feature": "", "value": "feature value"},
            )
            figure.update_traces(marker=dict(size=5, opacity=0.6))
            figure.update_layout(height=420, margin=dict(t=20, b=10))
            st.plotly_chart(figure, width="stretch")

    st.divider()
    st.markdown("**Why was this specific hour flagged?**")
    if tags.empty:
        st.info("No flagged hours to explain for this selection.")
        return

    options = tags.sort_values("timestamp", ascending=False)["timestamp"].tolist()
    chosen = st.selectbox(
        "Flagged hour",
        options,
        format_func=lambda ts: f"{pd.Timestamp(ts):%Y-%m-%d %H:%M}",
    )
    row = tags[tags["timestamp"] == chosen].iloc[0]
    st.markdown(
        f"**{row['anomaly_type']}** - {row['reason']}\n\n"
        f"Actual **{row['electricity']:.1f} kWh** vs expected **{row['expected']:.1f} kWh** "
        f"(residual {row['residual']:+.1f} kWh, robust z {row['z']:.2f}, severity {row['severity']})."
    )

    shap_frame = shap_values[shap_values["timestamp"] == pd.Timestamp(chosen)]
    if shap_frame.empty:
        st.info("No SHAP row stored for this hour.")
        return
    shap_frame = shap_frame.reindex(shap_frame["shap"].abs().sort_values(ascending=False).index).head(8)
    figure = px.bar(
        shap_frame.sort_values("shap"),
        x="shap",
        y="feature",
        orientation="h",
        color="shap",
        color_continuous_scale="RdBu_r",
        labels={"shap": "contribution to the prediction (kWh)", "feature": ""},
    )
    figure.update_layout(height=340, margin=dict(t=20, b=10), coloraxis_showscale=False)
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Positive bars pushed the expected value up (more consumption); negative bars pushed it "
        "down. SHAP values add up to the prediction minus the base value."
    )


def tab_method(tables, controls) -> None:
    st.subheader("Methodology and limitations")
    st.markdown(
        f"""
**Data.** Building Data Genome Project 2 (BDG2), hourly electricity meters for the
`{tables['buildings']['building_id'].iloc[0].split('_')[0]}` site, 2016-2017, four education buildings
chosen for coverage and usage diversity.

**Task 1 - forecasting (supervised regression).** Target: measured hourly kWh. Features: calendar
(hour, weekday, month, cyclical encodings, US public holidays), weather (temperature, dew point,
cloud cover, wind, precipitation, 24 h mean temperature) and consumption history. Every history
feature lags by **at least 24 h** so the model describes what the building *should* be doing rather
than copying the anomaly currently happening.

**Task 2 - anomaly detection (unsupervised + residual rules).** The residual (actual − expected) is
converted to a robust z-score with median/MAD learned on the validation fold. Three ideas are
compared: the raw residual, the residual with a persistence rule ({PERSIST_K} of the last
{PERSIST_N} hours), Isolation Forest on a multivariate feature set, and a hybrid that adds a
flatline guard for frozen meter readings. A fifth variant recalibrates the median/MAD on a trailing
30-day window to follow seasonal drift.

**Evaluation of detectors without labels.** Known faults (spike, after-hours load, flatline, drift)
are injected into the test period, lags are recomputed exactly as a live system would, and
precision/recall/F1 are reported. Validation data is never contaminated.

**Limitations you should state in the viva.**
- The detector's sensitivity depends on the residual yardstick: one building's term-time occupancy
  change inflates its residuals and produces more false alarms (see the sensitivity table).
- Slow drifts below the noise level are largely missed; a change-point or trend detector would be
  needed.
- Synthetic anomalies are simpler than reality: they are clean, isolated and of known shape.
- Tariff (₹{controls['tariff']:.2f}/kWh) and CO₂ factor ({controls['co2']:.2f} kg/kWh) are
  assumptions, adjustable above.
- One-step-ahead forecasting only; multi-day forecasts would need weather forecasts as inputs.
        """
    )

    st.markdown("**Detector comparison (from `reports/detection_scores.csv`)**")
    overall = tables["detection"]
    overall = overall[overall["anomaly_type"] == "ALL"]
    if not overall.empty:
        table = (
            overall.groupby("method")[["precision", "recall", "f1", "false_alarm_rate_pct"]]
            .mean()
            .round(3)
            .reset_index()
        )
        table["method"] = table["method"].map(lambda m: METHOD_LABELS.get(m, m))
        st.dataframe(table, hide_index=True)

    st.markdown("**Sensitivity to the z threshold (mean over buildings)**")
    sensitivity = tables["sensitivity"]
    if not sensitivity.empty:
        table = (
            sensitivity.groupby(["z_threshold", "method"])[
                ["precision", "recall", "f1", "false_alarm_rate_pct"]
            ]
            .mean()
            .round(3)
            .reset_index()
        )
        figure = px.line(
            table,
            x="z_threshold",
            y="f1",
            color="method",
            markers=True,
            labels={"z_threshold": "z threshold", "f1": "F1 (injected anomalies)"},
        )
        figure.update_layout(height=340, margin=dict(t=20, b=10), legend_title="")
        st.plotly_chart(figure, width="stretch")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("Campus Energy Consumption Intelligence & Anomaly Detection")
    st.caption(
        "BDG2 hourly meters · 4 education buildings · forecast (supervised) + anomaly detection "
        "(residual rules and Isolation Forest)"
    )

    try:
        tables = load_tables()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    controls = sidebar(tables)

    if not controls["buildings"]:
        st.warning("No buildings selected. Use the sidebar to add at least one.")
        st.stop()

    residuals = tables["residuals"]
    residuals = residuals[residuals["building_id"].isin(controls["buildings"])]
    residuals = filter_by_date(residuals, controls["date_range"])

    thresholds = tables["buildings"][["building_id", "if_threshold"]]
    residuals = residuals.merge(thresholds, on="building_id", how="left")
    residuals = residuals.rename(columns={"if_threshold": "if_threshold_used"})

    tags = cached_tags(residuals, controls["method"], controls["z"])

    overview, patterns, forecast, anomalies, explain, method = st.tabs(
        ["Overview", "Patterns", "Forecast", "Anomalies", "Explainability", "Methodology"]
    )
    with overview:
        tab_overview(tables, controls, residuals, tags)
    with patterns:
        tab_patterns(tables, controls, residuals)
    with forecast:
        tab_forecast(tables, controls, residuals)
    with anomalies:
        tab_anomalies(tables, controls, residuals, tags)
    with explain:
        tab_explain(tables, controls, residuals, tags)
    with method:
        tab_method(tables, controls)


if __name__ == "__main__":
    main()

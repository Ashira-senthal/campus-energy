"""
Campus Energy Intelligence - Streamlit dashboard.

Run from the project root (needs ``data/processed/*.parquet``):

    .venv/bin/python -m streamlit run app/app.py

Visual design lives in ``app/theme.py`` (glass panels, gradient pills, glow) and
``.streamlit/config.toml`` (Streamlit's own palette). This file only decides
*what* is shown, not how it is painted.

Functionality is unchanged from the data-first version: nothing here trains a
model, it reads precomputed parquet and recomputes flags from stored residuals
with the same ``src.explain`` tagging used by the evaluation, so the UI can never
disagree with the reported metrics.
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
APP_DIR = Path(__file__).resolve().parent
for candidate in (ROOT, APP_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import theme  # noqa: E402  (app/theme.py, resolved via APP_DIR)
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

TYPE_GLYPHS = {
    "short spike": "▲",
    "after-hours load": "●",
    "meter fault (flatline)": "■",
    "slow drift": "↗",
    "sustained high load": "◆",
    "unexpected drop": "▼",
    "unusual load": "?",
}

st.set_page_config(
    page_title="Campus Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Data
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

    Grouping by building is essential: persistence and the flatline guard look at
    previous hours, so a concatenated series would let one building inherit its
    neighbour's last hour.
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


def filter_by_date(frame: pd.DataFrame, date_range) -> pd.DataFrame:
    if len(date_range) != 2:
        return frame
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
    return frame[(frame["timestamp"] >= start) & (frame["timestamp"] < end)]


def panel(key: str):
    """A container that the theme paints as a glass card (key prefix matters)."""
    return st.container(key=f"panel_{key}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def sidebar(tables: dict[str, pd.DataFrame]) -> dict:
    st.sidebar.markdown(
        '<p class="panel-title">⚡ Campus Energy</p>'
        '<p class="panel-sub">BDG2 · Fox site · 4 education buildings</p>',
        unsafe_allow_html=True,
    )

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
    min_date = residuals["timestamp"].min().date()
    max_date = residuals["timestamp"].max().date()
    date_range = st.sidebar.date_input(
        "Test-period date range", value=(min_date, max_date),
        min_value=min_date, max_value=max_date,
    )

    st.sidebar.markdown('<p class="panel-sub">Forecast &amp; detection</p>', unsafe_allow_html=True)
    model = st.sidebar.selectbox(
        "Forecast model", list(MODEL_LABELS),
        index=list(MODEL_LABELS).index("random_forest"),
        format_func=lambda key: MODEL_LABELS[key],
    )
    method = st.sidebar.selectbox(
        "Detection method", list(METHOD_LABELS),
        index=list(METHOD_LABELS).index("hybrid"),
        format_func=lambda key: METHOD_LABELS[key],
    )
    z_value = st.sidebar.slider(
        "Sensitivity (robust z)",
        min_value=2.0, max_value=6.0, value=Z_THRESHOLD, step=0.25,
        help=(
            f"Lower = more alerts. Persistence rule: {PERSIST_K} of the last {PERSIST_N} hours. "
            "Isolation Forest ignores this slider (its cut is learned on validation)."
        ),
    )

    st.sidebar.markdown('<p class="panel-sub">Cost assumptions</p>', unsafe_allow_html=True)
    tariff = st.sidebar.number_input("Tariff (INR per kWh)", 0.0, 100.0, TARIFF_INR_PER_KWH, 0.5)
    co2 = st.sidebar.number_input("Grid CO₂ factor (kg per kWh)", 0.0, 2.0, CO2_KG_PER_KWH, 0.01)

    primary = st.sidebar.selectbox(
        "Detail building",
        selected if selected else available_ids[:1],
        format_func=label_of,
        disabled=not selected,
    )

    st.sidebar.markdown(
        '<p class="panel-sub" style="margin-top:1.2rem">Tariff and CO₂ factor are assumptions — '
        "verify them against your campus bill and the current CEA grid factor.</p>",
        unsafe_allow_html=True,
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


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def tab_overview(tables, controls, res_view, tags) -> None:
    if res_view.empty:
        st.info("No data for this selection. Widen the date range or add a building.")
        return

    primary = controls["primary"]
    impact = estimate_impact(tags, controls["tariff"], controls["co2"])
    total_kwh = float(res_view["electricity"].sum())
    expected_kwh = float(res_view["expected"].sum())
    avg = float(res_view["electricity"].mean())
    peak = float(res_view["electricity"].max())
    delta_pct = 100.0 * (total_kwh - expected_kwh) / expected_kwh if expected_kwh else 0.0
    flagged_share = 100.0 * len(tags) / max(len(res_view), 1)

    left, mid, right = st.columns([1.0, 1.25, 1.05], gap="medium")

    with left:
        with panel("total"):
            st.markdown('<p class="panel-sub">Total consumption</p>', unsafe_allow_html=True)
            st.markdown(
                f'<p class="kpi-value" style="font-size:2.1rem;font-weight:600;color:#fff;'
                f'margin:.15rem 0 .4rem 0">{total_kwh:,.0f} kWh</p>',
                unsafe_allow_html=True,
            )
            arrow = "▲" if delta_pct > 0 else "▼"
            colour = theme.PALETTE["negative"] if delta_pct > 0 else theme.PALETTE["positive"]
            st.markdown(
                f'<p style="font-size:.8rem;color:{theme.PALETTE["muted"]};margin:0">'
                f'vs {expected_kwh:,.0f} kWh expected '
                f'<span style="color:{colour}">{arrow} {abs(delta_pct):.1f}%</span></p>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="promo">
              <h4>Avoidable spend</h4>
              <p>{impact['excess_kwh']:,.0f} kWh above expectation across {impact['flagged_hours']:,}
                 flagged hours, priced at ₹{controls['tariff']:.2f}/kWh.</p>
              <div class="pill">₹{impact['cost_inr']:,.0f} · {impact['co2_kg']:,.0f} kg CO₂</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with mid:
        with panel("top"):
            theme.panel_title(
                "Flagged hours",
                f"{METHOD_LABELS[controls['method']].split(' (')[0]} · z {controls['z']:.2f}",
            )
            ranked = tags.sort_values("excess_kwh", ascending=False).head(6)
            rows = []
            for _, row in ranked.iterrows():
                rows.append(
                    {
                        "glyph": TYPE_GLYPHS.get(row["anomaly_type"], "?"),
                        "title": row["anomaly_type"],
                        "sub": f"{pd.Timestamp(row['timestamp']):%d %b %H:%M} · {BUILDING_LABELS.get(row['building_id'], '').split(' (')[0]}",
                        "value": f"{row['excess_kwh']:,.0f} kWh",
                        "delta": f"z {row['z']:.1f}",
                    }
                )
            theme.anomaly_rows(rows)

    with right:
        with panel("buildings"):
            theme.panel_title("Buildings", "mean load · CV(RMSE)")
            test_metrics = tables["metrics"][
                (tables["metrics"]["fold"] == "test")
                & (tables["metrics"]["model"] == controls["model"])
            ].set_index("building_id")
            cards = []
            for building_id in controls["buildings"]:
                subset = res_view[res_view["building_id"] == building_id]
                if subset.empty:
                    continue
                cv = test_metrics["cv_rmse_pct"].get(building_id, float("nan"))
                cards.append(
                    {
                        "name": BUILDING_LABELS.get(building_id, building_id),
                        "value": f"{subset['electricity'].mean():,.0f} kWh",
                        "tag": f"{cv:.1f}% CV(RMSE)" if pd.notna(cv) else "no metric",
                    }
                )
            theme.building_cards(cards)

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    with panel("performance"):
        title_col, pill_col = st.columns([3, 1], vertical_alignment="center")
        with title_col:
            theme.panel_title(
                "Consumption vs expectation",
                f"{BUILDING_LABELS.get(primary, primary)} · dotted line is what the model expects",
            )
        with pill_col:
            window = st.segmented_control(
                "Window", ["1W", "1M", "3M", "ALL"], default="1M",
                label_visibility="collapsed", key="overview_window",
            )

        series = res_view[res_view["building_id"] == primary].sort_values("timestamp")
        days = {"1W": 7, "1M": 30, "3M": 90}.get(window or "1M")
        if days and not series.empty:
            cutoff = series["timestamp"].max() - pd.Timedelta(days=days)
            series = series[series["timestamp"] >= cutoff]

        if series.empty:
            st.info("No data for this building in the selected range.")
        else:
            figure = go.Figure()
            figure.add_trace(
                go.Scatter(
                    x=series["timestamp"], y=series["electricity"], name="actual",
                    line=dict(color=theme.PALETTE["accent"], width=1.8, shape="spline"),
                    fill="tozeroy",
                    fillgradient=dict(
                        type="vertical",
                        colorscale=[[0, "rgba(201,162,255,0)"], [1, "rgba(201,162,255,0.28)"]],
                    ),
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=series["timestamp"], y=series["expected"], name="expected",
                    line=dict(color=theme.PALETTE["pink"], width=1.1, dash="dot"),
                )
            )
            peak_row = series.loc[series["residual"].idxmax()]
            if peak_row["residual"] > 0:
                figure.add_vline(
                    x=peak_row["timestamp"], line_dash="dash",
                    line_color="rgba(233,184,255,.45)", line_width=1,
                )
                figure.add_annotation(
                    x=peak_row["timestamp"], y=peak_row["electricity"],
                    text=f"peak excess +{peak_row['residual']:,.0f} kWh",
                    showarrow=True, arrowhead=0, ax=0, ay=-34,
                    font=dict(size=10, color=theme.PALETTE["pink"]),
                    bgcolor="rgba(27,19,36,.9)", bordercolor="rgba(233,184,255,.3)",
                    borderpad=6,
                )
            figure.update_layout(yaxis_title="kWh")
            theme.style_figure(figure, height=330)
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    theme.kpi_cards(
        [
            ("Peak load", f"{peak:,.0f} kWh", "single highest hour", ""),
            ("Average load", f"{avg:,.0f} kWh", "per hour in window", ""),
            ("Flagged hours", f"{len(tags):,}", f"{flagged_share:.1f}% of hours", ""),
            ("CO₂ from excess", f"{impact['co2_kg']:,.0f} kg", "at your grid factor", ""),
        ]
    )


def tab_patterns(tables, controls, res_view) -> None:
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

    left, right = st.columns(2, gap="medium")
    with left:
        with panel("heat"):
            theme.panel_title("Weekly rhythm", "mean kWh by hour and weekday")
            pivot = subset.pivot_table(
                index="dayofweek", columns="hour", values="electricity", aggfunc="mean"
            ).reindex(range(7))
            figure = go.Figure(
                go.Heatmap(
                    z=pivot.to_numpy(), x=list(range(24)),
                    y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                    colorscale=[[0, "#150e1d"], [0.5, "#5b3a8e"], [1, "#e9b8ff"]],
                    colorbar=dict(title="kWh", thickness=8, outlinewidth=0),
                    hovertemplate="%{y} %{x}:00 — %{z:.1f} kWh<extra></extra>",
                )
            )
            figure.update_layout(xaxis_title="hour of day", yaxis_title="")
            theme.style_figure(figure, height=330, legend=False)
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    with right:
        with panel("profile"):
            theme.panel_title("Weekday vs weekend", "average daily shape")
            profile = (
                subset.assign(kind=np.where(subset["is_weekend"] == 1, "weekend", "weekday"))
                .groupby(["kind", "hour"])["electricity"].mean().reset_index()
            )
            figure = px.line(profile, x="hour", y="electricity", color="kind", markers=True)
            figure.update_traces(line=dict(width=2, shape="spline"), marker=dict(size=4))
            figure.update_layout(yaxis_title="mean kWh", xaxis_title="hour of day")
            theme.style_figure(figure, height=330)
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    left, right = st.columns(2, gap="medium")
    with left:
        with panel("weather"):
            theme.panel_title("Load vs temperature", "each dot is one hour")
            sample = subset.dropna(subset=["airTemperature"])
            if len(sample) > 4000:
                sample = sample.sample(4000, random_state=42)
            figure = px.scatter(
                sample, x="airTemperature", y="electricity", opacity=0.45,
                labels={"airTemperature": "air temperature (°C)", "electricity": "kWh"},
            )
            figure.update_traces(marker=dict(size=4, color=theme.PALETTE["accent"]))
            theme.style_figure(figure, height=320, legend=False)
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    with right:
        with panel("monthly"):
            theme.panel_title("Monthly trend", "all selected buildings")
            trend = (
                hourly.dropna(subset=["electricity"])
                .assign(month=lambda d: d["timestamp"].dt.to_period("M").dt.to_timestamp())
                .groupby(["month", "building_label"])["electricity"].mean().reset_index()
            )
            if trend.empty:
                st.info("Nothing to plot.")
            else:
                figure = px.line(trend, x="month", y="electricity", color="building_label", markers=True)
                figure.update_traces(line=dict(width=2))
                figure.update_layout(yaxis_title="mean kWh", xaxis_title="", legend_title="")
                theme.style_figure(figure, height=320)
                st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def tab_forecast(tables, controls, res_view) -> None:
    if not controls["buildings"]:
        st.info("Select at least one building.")
        return

    predictions = tables["predictions"]
    predictions = predictions[predictions["building_id"].isin(controls["buildings"])]
    predictions = filter_by_date(predictions, controls["date_range"])
    selected_model = controls["model"]
    building = controls["primary"]
    frame = predictions[
        (predictions["model"] == selected_model) & (predictions["building_id"] == building)
    ].sort_values("timestamp")

    with panel("forecast_chart"):
        theme.panel_title(
            "Forecast vs actual",
            f"{BUILDING_LABELS.get(building, building)} · {MODEL_LABELS[selected_model]}",
        )
        if frame.empty:
            st.info("No predictions for this building in the selected range.")
        else:
            figure = go.Figure()
            figure.add_trace(
                go.Scatter(
                    x=frame["timestamp"], y=frame["actual"], name="actual",
                    line=dict(color=theme.PALETTE["accent"], width=1.7),
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=frame["timestamp"], y=frame["expected"], name="expected",
                    line=dict(color=theme.PALETTE["pink"], width=1.3, dash="dot"),
                )
            )
            figure.update_layout(yaxis_title="kWh")
            theme.style_figure(figure, height=340)
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    left, right = st.columns([1.15, 1.0], gap="medium")
    with left:
        with panel("models"):
            theme.panel_title("Model comparison", "test fold, lower is better")
            table = tables["metrics"][tables["metrics"]["fold"] == "test"]
            if controls["buildings"]:
                table = table[table["building_id"].isin(controls["buildings"])]
            table = (
                table.groupby("model")[["mae", "rmse", "cv_rmse_pct", "nmbe_pct", "smape_pct"]]
                .mean().round(2).reset_index()
            )
            table["model"] = table["model"].map(MODEL_LABELS)
            st.dataframe(
                table.rename(columns={
                    "model": "model", "mae": "MAE (kWh)", "rmse": "RMSE (kWh)",
                    "cv_rmse_pct": "CV(RMSE) %", "nmbe_pct": "NMBE %", "smape_pct": "sMAPE %",
                }),
                hide_index=True, width="stretch",
            )
            st.caption(
                "CV(RMSE) and NMBE follow ASHRAE Guideline 14: scale-free, so buildings of "
                "different size share a table."
            )

    with right:
        with panel("hourly_error"):
            theme.panel_title("Error by hour of day", "mean absolute error")
            errors = tables["hour_errors"]
            errors = errors[
                (errors["model"] == selected_model) & (errors["building_id"].isin(controls["buildings"]))
            ]
            if errors.empty:
                st.info("No error breakdown available.")
            else:
                by_hour = errors.groupby("hour")[["mae", "bias"]].mean().reset_index()
                figure = px.bar(by_hour, x="hour", y="mae", labels={"mae": "MAE (kWh)"})
                figure.update_traces(marker=dict(color=theme.PALETTE["accent"], opacity=0.85,
                                                 line=dict(width=0)))
                theme.style_figure(figure, height=300, legend=False)
                st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
                st.caption("Bias = mean(actual − expected); negative bias means over-prediction.")


def tab_anomalies(tables, controls, res_view, tags) -> None:
    if res_view.empty:
        st.info("No data for this selection.")
        return

    primary = controls["primary"]
    res_view = res_view[res_view["building_id"] == primary]
    tags = tags[tags["building_id"] == primary]
    if res_view.empty:
        st.info("No data for this building in the selected range.")
        return

    with panel("timeline"):
        theme.panel_title(
            "Anomaly timeline",
            f"{BUILDING_LABELS.get(primary, primary)} · {METHOD_LABELS[controls['method']]} at z {controls['z']:.2f}",
        )
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=res_view["timestamp"], y=res_view["electricity"], name="actual",
                line=dict(color="rgba(201,162,255,.75)", width=1.2),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=res_view["timestamp"], y=res_view["expected"], name="expected",
                line=dict(color=theme.PALETTE["pink"], width=1, dash="dot"),
            )
        )
        if not tags.empty:
            figure.add_trace(
                go.Scatter(
                    x=tags["timestamp"], y=tags["electricity"], mode="markers", name="flagged",
                    marker=dict(color="#ffa3b3", size=8, symbol="x", line=dict(width=0)),
                    text=tags["anomaly_type"],
                    hovertemplate="%{x}<br>%{y:.1f} kWh<br>%{text}<extra></extra>",
                )
            )
        figure.update_layout(yaxis_title="kWh")
        theme.style_figure(figure, height=330)
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    if tags.empty:
        st.success(
            "Nothing flagged for this selection. Lower the z threshold or choose a more "
            "sensitive method."
        )
        return

    impact = estimate_impact(tags, controls["tariff"], controls["co2"])
    theme.kpi_cards(
        [
            ("Flagged hours", f"{len(tags):,}", "in this window", ""),
            ("Excess energy", f"{impact['excess_kwh']:,.0f} kWh", "above expectation", ""),
            ("Cost", f"₹{impact['cost_inr']:,.0f}", f"at ₹{controls['tariff']:.2f}/kWh", ""),
            ("CO₂", f"{impact['co2_kg']:,.0f} kg", f"at {controls['co2']:.2f} kg/kWh", ""),
        ]
    )

    with panel("table"):
        theme.panel_title("Flagged hours", "newest first · download as CSV below")
        display = tags.copy()
        display["timestamp"] = display["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        display = display.rename(columns={
            "timestamp": "timestamp", "building_id": "building", "electricity": "actual kWh",
            "expected": "expected kWh", "excess_kwh": "excess kWh", "z": "robust z",
            "severity": "severity", "anomaly_type": "type", "reason": "plain-English reason",
        })[
            ["timestamp", "building", "actual kWh", "expected kWh", "excess kWh",
             "robust z", "severity", "type", "plain-English reason"]
        ]
        st.dataframe(
            display.sort_values("timestamp", ascending=False),
            hide_index=True, height=340, width="stretch",
        )
        st.download_button(
            "Download flagged hours as CSV",
            data=display.to_csv(index=False).encode("utf-8"),
            file_name="campus_anomalies.csv",
            mime="text/csv",
        )


def tab_explain(tables, controls, res_view, tags) -> None:
    primary = controls["primary"]
    importance = tables["importance"]
    building_importance = importance[importance["building_id"] == primary].sort_values(
        "mean_abs_shap", ascending=False
    )
    if building_importance.empty:
        st.info("No SHAP values for this building.")
        return

    left, right = st.columns(2, gap="medium")
    with left:
        with panel("importance"):
            theme.panel_title("What drives the forecast", "mean |SHAP| in kWh, test period")
            figure = px.bar(
                building_importance.head(12).sort_values("mean_abs_shap"),
                x="mean_abs_shap", y="feature", orientation="h",
                labels={"mean_abs_shap": "mean |SHAP| (kWh)", "feature": ""},
            )
            figure.update_traces(marker=dict(color=theme.PALETTE["accent"], opacity=0.9,
                                             line=dict(width=0)))
            theme.style_figure(figure, height=400, legend=False)
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    with right:
        with panel("shap"):
            theme.panel_title("SHAP distribution", "colour = feature value")
            shap_values = tables["shap"]
            shap_values = shap_values[shap_values["building_id"] == primary]
            top = building_importance.head(8)["feature"].tolist()
            sample = shap_values[shap_values["feature"].isin(top)]
            if sample.empty:
                st.info("No SHAP detail for this building.")
            else:
                figure = px.scatter(
                    sample, x="shap", y="feature", color="value",
                    color_continuous_scale=[[0, "#ffa3b3"], [0.5, "#2a1a3a"], [1, "#c9a2ff"]],
                    labels={"shap": "SHAP contribution (kWh)", "feature": "", "value": "feature value"},
                )
                figure.update_traces(marker=dict(size=4, opacity=0.55))
                theme.style_figure(figure, height=400, legend=False)
                st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    with panel("local"):
        theme.panel_title("Why was this hour flagged?", "pick a flag to see its drivers")
        if tags.empty:
            st.info("No flagged hours to explain for this selection.")
            return

        options = tags.sort_values("timestamp", ascending=False)["timestamp"].tolist()
        chosen = st.selectbox(
            "Flagged hour", options, format_func=lambda ts: f"{pd.Timestamp(ts):%Y-%m-%d %H:%M}"
        )
        row = tags[tags["timestamp"] == chosen].iloc[0]
        st.markdown(
            f"**{row['anomaly_type']}** — {row['reason']}  \n"
            f"Actual **{row['electricity']:.1f} kWh** vs expected **{row['expected']:.1f} kWh** "
            f"(residual {row['residual']:+.1f} kWh, robust z {row['z']:.2f}, severity {row['severity']})."
        )

        shap_frame = shap_values[shap_values["timestamp"] == pd.Timestamp(chosen)]
        if shap_frame.empty:
            st.info("No SHAP row stored for this hour.")
            return
        shap_frame = shap_frame.reindex(
            shap_frame["shap"].abs().sort_values(ascending=False).index
        ).head(8)
        figure = px.bar(
            shap_frame.sort_values("shap"), x="shap", y="feature", orientation="h",
            color="shap",
            color_continuous_scale=[[0, "#ffa3b3"], [0.5, "#2a1a3a"], [1, "#c9a2ff"]],
            labels={"shap": "contribution to the prediction (kWh)", "feature": ""},
        )
        figure.update_traces(marker_line_width=0)
        theme.style_figure(figure, height=300, legend=False)
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
        st.caption(
            "Positive bars pushed the expectation up; negative pushed it down. SHAP values add "
            "up to the prediction minus the base value."
        )


def tab_method(tables, controls) -> None:
    with panel("method_text"):
        theme.panel_title("Methodology and limitations")
        st.markdown(
            f"""
**Data.** Building Data Genome Project 2, hourly meters for four education
buildings at one site, 2016–2017.

**Task 1 — forecasting (supervised regression).** Target: measured hourly kWh.
Features: calendar (hour, weekday, month, cyclical encodings, US holidays),
weather (temperature, dew point, cloud, wind, precipitation) and consumption
history, with every history feature lagged by **at least 24 h** so the model
describes what the building *should* be doing instead of copying the anomaly.

**Task 2 — anomaly detection.** The residual (actual − expected) becomes a robust
z-score using median/MAD learned on the validation fold. Compared: raw residual,
residual with a persistence rule ({PERSIST_K} of the last {PERSIST_N} hours),
Isolation Forest on a multivariate feature set, a hybrid that adds a flatline
guard for frozen meters, and an adaptive variant that recalibrates on a trailing
30-day window.

**Evaluation without labels.** Known faults (spike, after-hours, flatline, drift)
are injected into the test period, lags are recomputed exactly as a live system
would, and precision/recall/F1 are reported.

**Limitations.** Detector sensitivity depends on the residual yardstick — one
building's term-time shift inflates false alarms; slow drifts below the noise
floor are missed; synthetic faults are simpler than reality; tariff
(₹{controls['tariff']:.2f}/kWh) and CO₂ factor ({controls['co2']:.2f} kg/kWh) are
assumptions; forecasts are one step ahead only.
            """
        )

    left, right = st.columns(2, gap="medium")
    with left:
        with panel("detectors"):
            theme.panel_title("Detector comparison", "injected faults, mean over buildings")
            overall = tables["detection"]
            overall = overall[overall["anomaly_type"] == "ALL"]
            if not overall.empty:
                table = (
                    overall.groupby("method")[["precision", "recall", "f1", "false_alarm_rate_pct"]]
                    .mean().round(3).reset_index()
                )
                table["method"] = table["method"].map(lambda m: METHOD_LABELS.get(m, m))
                st.dataframe(
                    table.rename(columns={
                        "method": "detector", "precision": "precision", "recall": "recall",
                        "f1": "F1", "false_alarm_rate_pct": "false alarms %",
                    }),
                    hide_index=True, width="stretch",
                )

    with right:
        with panel("sensitivity"):
            theme.panel_title("Sensitivity to the z threshold", "F1 on injected faults")
            sensitivity = tables["sensitivity"]
            if sensitivity.empty:
                st.info("No sensitivity table.")
            else:
                table = (
                    sensitivity.groupby(["z_threshold", "method"])[
                        ["precision", "recall", "f1", "false_alarm_rate_pct"]
                    ].mean().round(3).reset_index()
                )
                figure = px.line(
                    table, x="z_threshold", y="f1", color="method", markers=True,
                    labels={"z_threshold": "z threshold", "f1": "F1"},
                )
                figure.update_traces(line=dict(width=2), marker=dict(size=5))
                theme.style_figure(figure, height=300)
                st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    theme.inject_css()

    try:
        tables = load_tables()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    controls = sidebar(tables)

    hero_chips = [
        f'<span class="dot"></span><b>{len(controls["buildings"])}</b> buildings',
        f'<b>{METHOD_LABELS[controls["method"]].split(" (")[0]}</b> · z {controls["z"]:.2f}',
        f'<b>{MODEL_LABELS[controls["model"]]}</b> model',
    ]
    theme.hero(
        "Campus energy",
        "intelligence",
        "Four metered buildings — expected consumption, flagged deviations, and what they cost.",
        hero_chips,
    )

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

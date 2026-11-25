"""
Page 5: Compare.
Side-by-side comparative analysis across buildings with normalization.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

from components import (
    ce_page_header,
    ce_table,
    ce_source_note,
    ce_empty_message,
)
from theme import (
    fig_compare_overlay,
    fig_daily_profile,
    chart_layout,
    CHART_THEME,
    render_chart,
)
from data import (
    get_building_catalog,
    load_parquet,
    load_csv_report,
    build_event_registry,
    fmt_kwh,
    fmt_inr,
    fmt_pct,
    building_id_to_short,
)
import plotly.graph_objects as go

def render(ctx: dict):
    st.markdown(
        ce_page_header(
            "Compare",
            "Multi-facility benchmark: normalized consumption overlays, correlation dynamics, and comparative efficiency.",
        ),
        unsafe_allow_html=True,
    )

    catalog = get_building_catalog()
    all_labels = catalog["building_label"].tolist()

    # Controls Bar
    with st.container():
        ccol1, ccol2 = st.columns([3, 2])
        with ccol1:
            sel_labels = st.multiselect(
                "Select Buildings to Compare (2–4 buildings)",
                all_labels,
                default=all_labels[:3],
                key="compare_bld_select",
            )
        with ccol2:
            norm_mode = st.radio(
                "Normalization Scale",
                ["absolute", "sqm", "index"],
                index=0,
                format_func=lambda m: {"absolute": "Absolute kWh", "sqm": "Per m² (kWh/m²)", "index": "Indexed (Mean = 100)"}[m],
                horizontal=True,
                key="compare_norm_mode",
            )

    if len(sel_labels) < 2:
        st.info("Please select at least 2 buildings to enable side-by-side comparison.")
        return

    sel_bids = catalog[catalog["building_label"].isin(sel_labels)]["building_id"].tolist()
    start_ts = pd.to_datetime(ctx.get("start_date", "2017-09-01"))
    end_ts = pd.to_datetime(ctx.get("end_date", "2017-12-31"))
    model = ctx.get("model", "random_forest")
    tariff = float(ctx.get("tariff", 8.0))

    hourly_df = load_parquet("hourly")
    if hourly_df.empty:
        st.markdown(ce_empty_message("No hourly telemetry available."), unsafe_allow_html=True)
        return

    hourly_df["timestamp"] = pd.to_datetime(hourly_df["timestamp"])
    h_sub = hourly_df[
        (hourly_df["building_id"].isin(sel_bids)) &
        (hourly_df["timestamp"] >= start_ts) &
        (hourly_df["timestamp"] <= end_ts)
    ].copy()

    if h_sub.empty:
        st.markdown(ce_empty_message("No telemetry data for the selected range."), unsafe_allow_html=True)
        return

    # Merge sqm
    sqm_map = {r["building_id"]: r["sqm"] for _, r in catalog.iterrows()}
    h_sub["sqm"] = h_sub["building_id"].map(sqm_map)

    # Compute normalized values
    if norm_mode == "sqm":
        h_sub["val"] = h_sub["electricity"] / h_sub["sqm"]
    elif norm_mode == "index":
        mean_loads = h_sub.groupby("building_id")["electricity"].transform("mean")
        h_sub["val"] = (h_sub["electricity"] / mean_loads.replace(0, np.nan)) * 100.0
    else:
        h_sub["val"] = h_sub["electricity"]

    # Primary Chart: Overlaid lines for selected buildings
    fig_comp = fig_compare_overlay(h_sub, norm_mode=norm_mode)
    render_chart(fig_comp, key="compare_primary")

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    # Tabs: Summary Table | Profiles | Distribution | Correlation
    tab_sum, tab_prof, tab_dist, tab_corr = st.tabs([
        "Summary Table",
        "Profiles Overlay",
        "Load Distribution",
        "Correlation Matrix",
    ])

    with tab_sum:
        st.markdown("<h4 style='font-size: 14px; margin-bottom: 8px;'>Comparative Operational Metrics</h4>", unsafe_allow_html=True)

        forecast_metrics = load_csv_report("forecast_metrics")
        events_df = build_event_registry(tariff=tariff)

        table_data = []
        for bid in sel_bids:
            b_info = catalog[catalog["building_id"] == bid].iloc[0]
            b_sub = h_sub[h_sub["building_id"] == bid]
            elec = b_sub["electricity"].dropna()

            mean_l = float(elec.mean()) if not elec.empty else 0.0
            peak_l = float(elec.max()) if not elec.empty else 0.0
            load_factor = (mean_l / peak_l) if peak_l > 0 else 0.0
            kwh_m2 = (mean_l * 1000.0 / b_info["sqm"]) if b_info["sqm"] > 0 else 0.0

            # Flags per 1,000 hours
            hours_count = len(elec)
            b_ev = events_df[events_df["building_id"] == bid] if not events_df.empty else pd.DataFrame()
            flagged_h = int(b_ev["duration_h"].sum()) if not b_ev.empty else 0
            flags_per_1k = (flagged_h / hours_count * 1000.0) if hours_count > 0 else 0.0

            # Excess Cost
            excess_cost = float(b_ev["est_cost_INR"].sum()) if not b_ev.empty else 0.0

            # CV(RMSE)
            cv_rmse = np.nan
            if not forecast_metrics.empty:
                m_sub = forecast_metrics[
                    (forecast_metrics["building_id"] == bid) &
                    (forecast_metrics["fold"] == "test") &
                    (forecast_metrics["model"] == model)
                ]
                if not m_sub.empty:
                    cv_rmse = float(m_sub.iloc[0]["cv_rmse_pct"])

            table_data.append({
                "bid": bid,
                "label": b_info["building_label"],
                "mean": mean_l,
                "peak": peak_l,
                "lf": load_factor,
                "sqm_intensity": kwh_m2,
                "flags_1k": flags_per_1k,
                "cv_rmse": cv_rmse,
                "excess_inr": excess_cost,
            })

        t_df = pd.DataFrame(table_data)

        # Highlight highest and lowest per column using font weight and a small marker (▲ / ▼), not color alone!
        num_cols = ["mean", "peak", "lf", "sqm_intensity", "flags_1k", "cv_rmse", "excess_inr"]
        min_vals = {c: t_df[c].min() for c in num_cols}
        max_vals = {c: t_df[c].max() for c in num_cols}

        headers = ["Building", "Mean Load", "Peak Load", "Load Factor", "Intensity (W/m²)", "Flags / 1k h", f"CV(RMSE) ({model})", "Excess Cost"]
        rows = []

        for _, r in t_df.iterrows():
            def mark(val, col, fmt_fn):
                if pd.isna(val):
                    return "—"
                formatted = fmt_fn(val)
                if val == max_vals[col] and len(t_df) > 1:
                    return f"<b>▲ {formatted}</b>"
                elif val == min_vals[col] and len(t_df) > 1:
                    return f"<span style='font-weight:600;'>▼ {formatted}</span>"
                return formatted

            rows.append([
                r["label"],
                mark(r["mean"], "mean", lambda v: f"{v:,.1f} kWh"),
                mark(r["peak"], "peak", lambda v: f"{v:,.1f} kWh"),
                mark(r["lf"], "lf", lambda v: f"{v:.2f}"),
                mark(r["sqm_intensity"], "sqm_intensity", lambda v: f"{v:,.1f}"),
                mark(r["flags_1k"], "flags_1k", lambda v: f"{v:.1f}"),
                mark(r["cv_rmse"], "cv_rmse", lambda v: f"{v:.2f}%"),
                mark(r["excess_inr"], "excess_inr", lambda v: f"₹{v:,.0f}"),
            ])

        st.markdown(
            ce_table(headers, rows, sort_keys=["bld", "mean", "peak", "lf", "sqm", "flags", "cv", "cost"], source="hourly.parquet + reports/forecast_metrics.csv"),
            unsafe_allow_html=True,
        )
        st.caption("▲ Highest value in cohort | ▼ Lowest value in cohort. Highlighted using font weight and markers (WCAG AA).")

    with tab_prof:
        st.markdown("<h4 style='font-size: 14px; margin-bottom: 8px;'>Daily 24-Hour Profile Overlay</h4>", unsafe_allow_html=True)
        fig_p = go.Figure()
        grp_prof = h_sub.groupby(["building_label", "hour"])["val"].mean().reset_index()

        colors = CHART_THEME["compare"]
        for i, b_lbl in enumerate(grp_prof["building_label"].unique()):
            sub_p = grp_prof[grp_prof["building_label"] == b_lbl].sort_values("hour")
            fig_p.add_trace(go.Scatter(
                x=sub_p["hour"],
                y=sub_p["val"],
                mode="lines+markers",
                name=b_lbl,
                line={"color": colors[i % len(colors)], "width": 2},
                marker={"size": 5},
                hovertemplate=f"<b>{b_lbl}</b><br>Hour %{{x}}:00: %{{y:.2f}}<extra></extra>",
            ))

        fig_p.update_layout(chart_layout(
            title="Hourly Mean Profile Overlay",
            subtitle=f"Average profile across 24h ({norm_mode} scale).",
            x_title="Hour of Day (0–23)",
            y_title="Normalized Load",
            height=340,
        ))
        render_chart(fig_p, key="compare_profile")

    with tab_dist:
        st.markdown("<h4 style='font-size: 14px; margin-bottom: 8px;'>Hourly Load Spread (Box Plot)</h4>", unsafe_allow_html=True)
        fig_box = go.Figure()
        colors = CHART_THEME["compare"]

        for i, b_lbl in enumerate(h_sub["building_label"].unique()):
            b_vals = h_sub[h_sub["building_label"] == b_lbl]["val"].dropna()
            fig_box.add_trace(go.Box(
                y=b_vals,
                name=b_lbl,
                marker_color=colors[i % len(colors)],
                boxpoints="outliers",
            ))

        fig_box.update_layout(chart_layout(
            title="Hourly Consumption Variance & Outlier Spread",
            subtitle="Box plots show quartiles and extreme outlier hours across cohort.",
            x_title="Building",
            y_title=f"Load ({norm_mode})",
            height=340,
        ))
        render_chart(fig_box, key="compare_box")

    with tab_corr:
        st.markdown("<h4 style='font-size: 14px; margin-bottom: 8px;'>Pearson Correlation Matrix</h4>", unsafe_allow_html=True)
        pivot_corr = h_sub.pivot_table(index="timestamp", columns="building_label", values="electricity").corr()

        fig_corr = go.Figure(data=go.Heatmap(
            z=pivot_corr.values,
            x=pivot_corr.columns,
            y=pivot_corr.index,
            colorscale=[[0.0, "#f7f5f2"], [0.5, "#c9c2b8"], [1.0, "#55514b"]],
            zmin=0,
            zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in pivot_corr.values],
            texttemplate="%{text}",
            textfont={"size": 9, "color": "#1b1a18"},
            colorbar={
                "title": {"text": "r", "font": {"size": 10, "color": CHART_THEME["text_muted"]}},
                "thickness": 10,
                "outlinewidth": 0,
                "tickfont": {"size": 10, "color": CHART_THEME["text_muted"]},
                "bgcolor": "rgba(0,0,0,0)",
            },
        ))
        fig_corr.update_layout(chart_layout(
            title="Inter-Building Load Correlation",
            subtitle="High correlation (r > 0.8) indicates shared campus schedules and weather response.",
            height=340,
        ))
        render_chart(fig_corr, key="compare_corr")

"""
Page 2: Live Monitor.
The main actual-vs-expected graph and time-series analysis.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

from components import (
    ce_page_header,
    ce_kpi_strip,
    ce_metric_row,
    ce_source_note,
    ce_empty_message,
)
from theme import (
    fig_actual_vs_expected,
    fig_residuals_time,
    fig_residuals_hist,
    fig_daily_profile,
    fig_heatmap,
    fig_weather_overlay,
    fig_compare_overlay,
    render_chart,
)
from data import (
    get_building_catalog,
    load_parquet,
    build_event_registry,
    get_series_data,
    get_residual_scale,
    fmt_kwh,
    building_id_to_short,
)

def render(ctx: dict):
    st.markdown(
        ce_page_header(
            "Live Monitor",
            "High-resolution telemetry inspection: actual load vs model expectation with anomaly triage markers.",
        ),
        unsafe_allow_html=True,
    )

    catalog = get_building_catalog()
    all_bids = catalog["building_id"].tolist()
    model = ctx.get("model", "random_forest")
    tariff = float(ctx.get("tariff", 8.0))

    # Slim Filter Bar inside Streamlit Columns
    with st.container(key="ce_filters"):
        fcol1, fcol2, fcol3, fcol4 = st.columns([3, 2, 2, 2])

        with fcol1:
            bld_options = ["Portfolio total"] + catalog["building_label"].tolist()
            default_bld = "Portfolio total"
            selected_bld = st.selectbox(
                "Building Selection",
                bld_options,
                index=0,
                key="live_bld_select",
                label_visibility="collapsed",
            )

        with fcol2:
            quick_range = st.radio(
                "Range",
                ["1W", "1M", "3M", "All"],
                index=3,
                horizontal=True,
                key="live_quick_range",
                label_visibility="collapsed",
            )

        with fcol3:
            resolution = st.radio(
                "Resolution",
                ["Hourly", "Daily"],
                index=0,
                horizontal=True,
                key="live_resolution",
                label_visibility="collapsed",
            )

        with fcol4:
            show_by_building = st.checkbox(
                "Show by building",
                value=False,
                key="live_show_by_bld",
                help="Overlay individual building series when viewing Portfolio total",
            )

    # Determine building IDs and date limits
    if selected_bld == "Portfolio total":
        sel_bids = all_bids
    else:
        sel_bids = [catalog[catalog["building_label"] == selected_bld]["building_id"].iloc[0]]

    # Compute Date Range from quick buttons
    max_test_date = pd.Timestamp("2017-12-31 23:00")
    if quick_range == "1W":
        start_ts = max_test_date - pd.Timedelta(days=7)
    elif quick_range == "1M":
        start_ts = max_test_date - pd.Timedelta(days=30)
    elif quick_range == "3M":
        start_ts = max_test_date - pd.Timedelta(days=90)
    else:
        start_ts = pd.Timestamp("2017-09-01 00:00")
    end_ts = max_test_date

    # Fetch Series Data
    raw_series = get_series_data(sel_bids, model=model, start_ts=start_ts, end_ts=end_ts)

    if resolution == "Daily" and not raw_series.empty:
        # Resample to daily
        daily_df = raw_series.set_index("timestamp").resample("D").agg({
            "actual": "sum",
            "expected": "sum",
            "residual": "sum",
            "z": "mean",
        }).reset_index()
        series_df = daily_df
    else:
        series_df = raw_series

    # Fetch Events
    events_df = build_event_registry(tariff=tariff, co2_factor=float(ctx.get("co2_factor", 0.71)))
    if not events_df.empty:
        ev_sub = events_df[
            (events_df["building_id"].isin(sel_bids)) &
            (events_df["start"] <= end_ts) &
            (events_df["end"] >= start_ts)
        ].copy()
    else:
        ev_sub = pd.DataFrame()

    # Compute KPI Strip (4 numbers)
    if not series_df.empty and "actual" in series_df.columns:
        tot_kwh = float(series_df["actual"].sum())
        peak_idx = int(series_df["actual"].idxmax())
        peak_load = float(series_df.loc[peak_idx, "actual"])
        peak_time = series_df.loc[peak_idx, "timestamp"].strftime("%Y-%m-%d %H:%M")
        avg_load = float(series_df["actual"].mean())
    else:
        tot_kwh = 0.0
        peak_load = 0.0
        peak_time = "—"
        avg_load = 0.0

    flagged_hours = int(ev_sub["duration_h"].sum()) if not ev_sub.empty else 0

    kpis_html = "".join([
        ce_metric_row("Total Consumption", f"{tot_kwh:,.0f}", unit="kWh", source="hourly.parquet"),
        ce_metric_row("Peak Load", f"{peak_load:,.1f}", unit="kWh", delta=f"@ {peak_time}", delta_kind="neutral", source="hourly.parquet"),
        ce_metric_row("Average Load", f"{avg_load:,.1f}", unit="kWh/h", delta=f"{len(sel_bids)} building(s)", delta_kind="neutral", source="hourly.parquet"),
        ce_metric_row("Flagged Hours", f"{flagged_hours:,} h", delta=f"{len(ev_sub)} anomalies", delta_kind="warn" if flagged_hours > 0 else "pass", source="anomalies_clean_test.csv"),
    ])
    st.markdown(ce_kpi_strip(kpis_html), unsafe_allow_html=True)

    # Primary Chart: Large, full width
    if not series_df.empty:
        # Determine scale for uncertainty band
        if len(sel_bids) == 1:
            u_scale = get_residual_scale(sel_bids[0])
        else:
            u_scale = float(np.sum([get_residual_scale(b) for b in sel_bids])) * 0.7

        if show_by_building and selected_bld == "Portfolio total":
            # Multi-building overlay view
            hourly_all = load_parquet("hourly")
            hourly_all["timestamp"] = pd.to_datetime(hourly_all["timestamp"])
            h_sub = hourly_all[
                (hourly_all["building_id"].isin(all_bids)) &
                (hourly_all["timestamp"] >= start_ts) &
                (hourly_all["timestamp"] <= end_ts)
            ].copy()
            h_sub["val"] = h_sub["electricity"]
            fig = fig_compare_overlay(h_sub, norm_mode="absolute")
            render_chart(fig, key="live_overlay")
        else:
            fig = fig_actual_vs_expected(
                df=series_df,
                events=ev_sub if resolution == "Hourly" else None,
                title=f"{selected_bld}: Actual vs Model Expectation",
                subtitle=f"Resolution: {resolution} | Model: {model} | Uncertainty: expected ± 3.0σ robust scale",
                show_uncertainty=(resolution == "Hourly"),
                uncertainty_scale=u_scale,
                show_rangeslider=False,
                height=480,
            )
            render_chart(fig, key="live_primary")

        # Chart download buttons
        dcol1, dcol2, _ = st.columns([2, 2, 6])
        with dcol1:
            csv_data = series_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download chart data (CSV)",
                data=csv_data,
                file_name=f"campus_energy_{selected_bld.lower().replace(' ', '_')}_{quick_range.lower()}.csv",
                mime="text/csv",
                key="download_live_csv",
            )
        with dcol2:
            st.caption("📷 For PNG: hover over chart and click the camera icon.")

        st.markdown(
            ce_source_note("data/processed/hourly.parquet + data/processed/predictions.parquet + reports/anomalies_clean_test.csv"),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(ce_empty_message("No consumption telemetry in this range."), unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    # Tabs under the chart (peer views, one visible at a time)
    tab_res, tab_prof, tab_heat, tab_wx = st.tabs([
        "Residuals",
        "Daily Profile",
        "Heatmap (Hour × Weekday)",
        "Weather Overlay",
    ])

    with tab_res:
        rcol1, rcol2 = st.columns(2)
        with rcol1:
            fig_r1 = fig_residuals_time(series_df)
            render_chart(fig_r1, key="live_residual_time")
        with rcol2:
            fig_r2 = fig_residuals_hist(series_df)
            render_chart(fig_r2, key="live_residual_hist")

    with tab_prof:
        hourly_df = load_parquet("hourly")
        if not hourly_df.empty:
            hourly_df["timestamp"] = pd.to_datetime(hourly_df["timestamp"])
            h_sub = hourly_df[
                (hourly_df["building_id"].isin(sel_bids)) &
                (hourly_df["timestamp"] >= start_ts) &
                (hourly_df["timestamp"] <= end_ts)
            ]
            fig_prof = fig_daily_profile(h_sub)
            render_chart(fig_prof, key="live_profile")
        else:
            st.markdown(ce_empty_message("No profile data available."), unsafe_allow_html=True)

    with tab_heat:
        hourly_df = load_parquet("hourly")
        if not hourly_df.empty:
            hourly_df["timestamp"] = pd.to_datetime(hourly_df["timestamp"])
            h_sub = hourly_df[
                (hourly_df["building_id"].isin(sel_bids)) &
                (hourly_df["timestamp"] >= start_ts) &
                (hourly_df["timestamp"] <= end_ts)
            ]
            fig_hm = fig_heatmap(h_sub)
            render_chart(fig_hm, key="live_heatmap")
        else:
            st.markdown(ce_empty_message("No heatmap data available."), unsafe_allow_html=True)

    with tab_wx:
        hourly_df = load_parquet("hourly")
        if not hourly_df.empty:
            hourly_df["timestamp"] = pd.to_datetime(hourly_df["timestamp"])
            h_sub = hourly_df[
                (hourly_df["building_id"].isin(sel_bids)) &
                (hourly_df["timestamp"] >= start_ts) &
                (hourly_df["timestamp"] <= end_ts)
            ]
            fig_wx = fig_weather_overlay(h_sub)
            render_chart(fig_wx, key="live_weather")
        else:
            st.markdown(ce_empty_message("No weather data available."), unsafe_allow_html=True)

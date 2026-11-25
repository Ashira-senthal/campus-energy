"""
Page 1: Overview (Health Snapshot).
Answers: "Is anything wrong right now?"
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

from components import (
    ce_page_header,
    ce_kpi_strip,
    ce_metric_row,
    ce_table,
    ce_tag,
    ce_card,
    ce_source_note,
    ce_empty_message,
)
from theme import fig_actual_vs_expected, render_chart
from data import (
    get_building_catalog,
    load_csv_report,
    build_event_registry,
    get_series_data,
    fmt_kwh,
    fmt_inr,
    fmt_pct,
    building_id_to_short,
)

def render(ctx: dict):
    st.markdown(
        ce_page_header(
            "Overview",
            "At-a-glance health snapshot of campus energy consumption and unreviewed anomalies.",
        ),
        unsafe_allow_html=True,
    )

    catalog = get_building_catalog()
    all_bids = catalog["building_id"].tolist()
    sel_bids = ctx.get("buildings", all_bids)
    start_ts = pd.to_datetime(ctx.get("start_date", "2017-09-01"))
    end_ts = pd.to_datetime(ctx.get("end_date", "2017-12-31"))
    model = ctx.get("model", "random_forest")
    tariff = float(ctx.get("tariff", 8.0))

    # Load data
    series_df = get_series_data(sel_bids, model=model, start_ts=start_ts, end_ts=end_ts)
    events_df = build_event_registry(tariff=tariff, co2_factor=float(ctx.get("co2_factor", 0.71)))

    # Filter events in range and buildings
    if not events_df.empty:
        ev_sub = events_df[
            (events_df["building_id"].isin(sel_bids)) &
            (events_df["start"] <= end_ts) &
            (events_df["end"] >= start_ts)
        ].copy()
    else:
        ev_sub = pd.DataFrame()

    # KPI 1: Consumption vs Expected (% difference)
    if not series_df.empty and "actual" in series_df.columns and "expected" in series_df.columns:
        tot_act = float(series_df["actual"].sum())
        tot_exp = float(series_df["expected"].dropna().sum())
        cons_unit = "kWh"
        if tot_exp > 0:
            diff_pct = ((tot_act - tot_exp) / tot_exp) * 100.0
            delta_str = f"{diff_pct:+.1f}% vs model"
            delta_kind = "fail" if diff_pct > 5.0 else ("warn" if diff_pct > 0 else "pass")
            cons_val = f"{tot_act:,.0f}"
        else:
            cons_val = f"{tot_act:,.0f}"
            delta_str = "No expectation"
            delta_kind = "neutral"
    else:
        cons_val = "—"
        cons_unit = None
        delta_str = None
        delta_kind = "neutral"

    # KPI 2: Flagged Hours in range
    flagged_hours = int(ev_sub["duration_h"].sum()) if not ev_sub.empty else 0

    # KPI 3: Open (unreviewed) anomalies
    open_events = int((ev_sub["review_status"] == "New").sum()) if not ev_sub.empty else 0

    # KPI 4: Avoidable cost in INR
    avoidable_cost = float(ev_sub["est_cost_INR"].sum()) if not ev_sub.empty else 0.0

    # Render KPI Strip (exactly 4 numbers)
    kpis_html = "".join([
        ce_metric_row(
            label="Total Consumption",
            value=cons_val,
            unit=cons_unit,
            delta=delta_str,
            delta_kind=delta_kind,
            source="hourly.parquet + predictions.parquet",
        ),
        ce_metric_row(
            label="Flagged Hours",
            value=f"{flagged_hours:,} h",
            delta=f"{len(ev_sub)} events" if not ev_sub.empty else "0 events",
            delta_kind="warn" if flagged_hours > 0 else "pass",
            source="anomalies_clean_test.csv",
        ),
        ce_metric_row(
            label="Open Anomalies",
            value=open_events,
            delta="Needs triage" if open_events > 0 else "All reviewed",
            delta_kind="fail" if open_events > 0 else "pass",
            source="anomaly_reviews.csv",
        ),
        ce_metric_row(
            label="Avoidable Cost",
            value=fmt_inr(avoidable_cost),
            unit="",
            delta=f"Tariff ₹{tariff:.2f}/kWh",
            delta_kind="warn" if avoidable_cost > 0 else "neutral",
            source="anomalies_clean_test.csv (excess × tariff)",
        ),
    ])
    st.markdown(ce_kpi_strip(kpis_html), unsafe_allow_html=True)

    # Primary View: Portfolio Actual vs Expected line chart with flagged hours marked
    if not series_df.empty:
        fig = fig_actual_vs_expected(
            df=series_df,
            events=ev_sub,
            title="Portfolio Electricity Consumption: Actual vs Expected",
            subtitle=f"Aggregated load for {len(sel_bids)} building(s) with model: {model}.",
            show_uncertainty=True,
            uncertainty_scale=15.0 if len(sel_bids) > 1 else 10.0,
            show_rangeslider=False,
            height=380,
        )
        render_chart(fig, key="overview_primary")
        st.markdown(
            ce_source_note("hourly.parquet (actual) + predictions.parquet (expected) + anomalies_clean_test.csv (flags)"),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(ce_empty_message("No telemetry data found for the selected buildings and range."), unsafe_allow_html=True)

    # Secondary Views: full-width blocks so the tables stay readable
    st.markdown("<h3 class='ce-section-title'>Latest Anomalies</h3>", unsafe_allow_html=True)
    if not ev_sub.empty:
        latest = ev_sub.sort_values("start", ascending=False).head(5)
        headers = ["Event ID", "Building", "Start", "Type", "Severity", "Cost (₹)"]
        rows = []
        for _, r in latest.iterrows():
            link_html = f'<a href="?page=anomalies&event={r["event_id"]}" target="_self" style="font-weight:600">{r["event_id"]}</a>'
            b_short = building_id_to_short(r["building_id"])
            start_str = r["start"].strftime("%Y-%m-%d %H:%M")
            type_tag = ce_tag(r["type"], r["type"])
            sev_tag = ce_tag(r["severity"], r["severity"])
            cost_str = f"₹{r['est_cost_INR']:,.0f}"
            rows.append([link_html, b_short, start_str, type_tag, sev_tag, cost_str])

        st.markdown(
            ce_table(headers, rows, sort_keys=["id", "bld", "start", "type", "sev", "cost"], source="reports/anomalies_clean_test.csv"),
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="margin-top: 10px; font-size: 14.5px;"><a href="?page=anomalies" target="_self">View all anomalies in registry ›</a></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(ce_card("Latest Anomalies", ce_empty_message("No anomalies detected in selected range.")), unsafe_allow_html=True)

    st.markdown("<h3 class='ce-section-title'>Building Status</h3>", unsafe_allow_html=True)
    forecast_metrics = load_csv_report("forecast_metrics")
    tot_mean_load = sum(b["mean_kwh"] for b in catalog.to_dict("records"))

    b_headers = ["Building", "Mean Load", "Share", "Flags", "CV(RMSE)"]
    b_rows = []

    for b in catalog.to_dict("records"):
        bid = b["building_id"]
        short_lbl = b["short_name"]
        mean_load = b["mean_kwh"]
        share_pct = (mean_load / tot_mean_load * 100.0) if tot_mean_load > 0 else 0.0

        # Count flags for this building in range
        b_flags = int(ev_sub[ev_sub["building_id"] == bid]["duration_h"].sum()) if not ev_sub.empty else 0

        # Lookup CV(RMSE) for test fold and active model
        cv_rmse_str = "—"
        if not forecast_metrics.empty:
            m_sub = forecast_metrics[
                (forecast_metrics["building_id"] == bid) &
                (forecast_metrics["fold"] == "test") &
                (forecast_metrics["model"] == model)
            ]
            if not m_sub.empty and pd.notna(m_sub.iloc[0]["cv_rmse_pct"]):
                cv_rmse_str = f"{m_sub.iloc[0]['cv_rmse_pct']:.2f}%"

        b_rows.append([
            short_lbl,
            f"{mean_load:,.1f} kWh",
            f"{share_pct:.1f}%",
            f"{b_flags:,} h",
            cv_rmse_str,
        ])

    st.markdown(
        ce_table(b_headers, b_rows, sort_keys=["bld", "load", "share", "flags", "cvrmse"], source="reports/forecast_metrics.csv (test fold)"),
        unsafe_allow_html=True,
    )

"""
Page 4: Buildings.
Single-building deep dive into telemetry, forecast accuracy, anomalies, and carbon cost.
"""

from __future__ import annotations

import html

import streamlit as st
import pandas as pd
import numpy as np

from components import (
    ce_page_header,
    ce_table,
    ce_tag,
    ce_source_note,
    ce_empty_message,
)
from theme import (
    fig_actual_vs_expected,
    fig_daily_profile,
    fig_heatmap,
    chart_layout,
    CHART_THEME,
    render_chart,
)
from data import (
    get_building_catalog,
    load_parquet,
    load_csv_report,
    build_event_registry,
    get_series_data,
    get_residual_scale,
    fmt_kwh,
    fmt_inr,
    fmt_co2,
    fmt_pct,
)
import plotly.graph_objects as go

def render(ctx: dict):
    catalog = get_building_catalog()
    all_bids = catalog["building_id"].tolist()
    labels = catalog["building_label"].tolist()

    st.markdown(
        ce_page_header(
            "Buildings",
            "Single-facility engineering audit: baseload, peak intensity, model performance, and monthly cost impact.",
        ),
        unsafe_allow_html=True,
    )

    # Top Building Selector
    sel_label = st.selectbox(
        "Select Building to Inspect",
        labels,
        index=0,
        key="building_page_select",
    )
    b_info = catalog[catalog["building_label"] == sel_label].iloc[0]
    bid = b_info["building_id"]

    start_ts = pd.to_datetime(ctx.get("start_date", "2017-09-01"))
    end_ts = pd.to_datetime(ctx.get("end_date", "2017-12-31"))
    model = ctx.get("model", "random_forest")
    tariff = float(ctx.get("tariff", 8.0))
    co2_factor = float(ctx.get("co2_factor", 0.71))

    # Fetch Data
    series_df = get_series_data([bid], model=model, start_ts=start_ts, end_ts=end_ts)
    hourly_df = load_parquet("hourly")
    quality_df = load_csv_report("building_quality")

    # Metrics for Header Card
    sqm = float(b_info["sqm"])
    mean_load = float(b_info["mean_kwh"])
    peak_load = float(series_df["actual"].max()) if not series_df.empty else 0.0
    load_per_sqm = (mean_load * 1000.0 / sqm) if sqm > 0 else 0.0  # W/m^2 or kWh/m^2

    # Data Coverage
    cov_str = "100.0%"
    if not quality_df.empty:
        q_row = quality_df[quality_df["building_id"] == bid]
        if not q_row.empty:
            cov_str = f"{q_row.iloc[0]['pct_complete_after_repair']:.2f}%"

    # Header Card
    st.markdown(f"""
<div class="ce-building-card">
  <div class="ce-building-card__head">
    <div>
      <span class="ce-building-card__name">{html.escape(b_info['building_label'])}</span>
      <span class="ce-building-card__usage">Usage: <b>{html.escape(str(b_info['usage']))}</b></span>
    </div>
    <span class="ce-building-card__badge">Site Fox · BDG2</span>
  </div>
  <div class="ce-building-card__grid">
    <div>
      <div class="ce-building-card__label">Floor Area</div>
      <div class="ce-building-card__value">{sqm:,.0f} m²</div>
    </div>
    <div>
      <div class="ce-building-card__label">Mean Load</div>
      <div class="ce-building-card__value">{mean_load:,.1f} kWh/h</div>
    </div>
    <div>
      <div class="ce-building-card__label">Peak Observed</div>
      <div class="ce-building-card__value">{peak_load:,.1f} kWh/h</div>
    </div>
    <div>
      <div class="ce-building-card__label">Intensity</div>
      <div class="ce-building-card__value">{load_per_sqm:,.2f} W/m²</div>
    </div>
    <div>
      <div class="ce-building-card__label">Data Coverage</div>
      <div class="ce-building-card__value ce-building-card__value--positive">{cov_str}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Primary Chart: Actual vs Expected with flags marked
    events_df = build_event_registry(tariff=tariff, co2_factor=co2_factor)
    b_events = events_df[
        (events_df["building_id"] == bid) &
        (events_df["start"] <= end_ts) &
        (events_df["end"] >= start_ts)
    ] if not events_df.empty else pd.DataFrame()

    if not series_df.empty:
        u_scale = get_residual_scale(bid)
        fig_main = fig_actual_vs_expected(
            df=series_df,
            events=b_events,
            title=f"{b_info['building_label']} — Actual vs Expected Telemetry",
            subtitle=f"Model: {model} | Uncertainty: expected ± 3.0σ ({u_scale:.1f} kWh)",
            show_uncertainty=True,
            uncertainty_scale=u_scale,
            show_rangeslider=False,
            height=420,
        )
        render_chart(fig_main, key="building_primary")
        st.markdown(
            ce_source_note("data/processed/hourly.parquet + data/processed/predictions.parquet + reports/anomalies_clean_test.csv"),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(ce_empty_message("No telemetry available for this building."), unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    # Tabs: Profile | Forecast Accuracy | Anomalies | Cost & Carbon
    tab_prof, tab_acc, tab_anom, tab_cost = st.tabs([
        "Profile (Daily & Weekly)",
        "Forecast Accuracy (All Models)",
        f"Anomalies ({len(b_events)})",
        "Cost & Carbon Breakdown",
    ])

    with tab_prof:
        if not hourly_df.empty:
            b_hourly = hourly_df[hourly_df["building_id"] == bid]
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                fig_p = fig_daily_profile(b_hourly)
                render_chart(fig_p, key="building_profile")
            with pcol2:
                fig_h = fig_heatmap(b_hourly)
                render_chart(fig_h, key="building_heatmap")
        else:
            st.markdown(ce_empty_message("No profile telemetry available."), unsafe_allow_html=True)

    with tab_acc:
        st.markdown("<h4 style='font-size: 17px; margin-bottom: 8px;'>Candidate Model Benchmark for this Building (Test Fold: Sep–Dec 2017)</h4>", unsafe_allow_html=True)
        forecast_metrics = load_csv_report("forecast_metrics")
        if not forecast_metrics.empty:
            b_metrics = forecast_metrics[
                (forecast_metrics["building_id"] == bid) &
                (forecast_metrics["fold"] == "test")
            ].copy()

            if not b_metrics.empty:
                headers = ["Model", "MAE (kWh)", "RMSE (kWh)", "CV(RMSE) %", "NMBE %", "sMAPE %", "ASHRAE 14 Acceptance"]
                rows = []
                for _, r in b_metrics.iterrows():
                    m_name = r["model"]
                    is_active = (m_name == model)
                    name_str = f"<b>{m_name} (active)</b>" if is_active else m_name
                    cv = float(r["cv_rmse_pct"])
                    nmbe = float(r["nmbe_pct"])

                    # ASHRAE Guideline 14 hourly criteria: CV(RMSE) <= 30%, |NMBE| <= 10%
                    pass_cv = cv <= 30.0
                    pass_nmbe = abs(nmbe) <= 10.0
                    if pass_cv and pass_nmbe:
                        ashrae_tag = ce_tag("PASS", "confirmed")
                    else:
                        ashrae_tag = ce_tag("FAIL", "new")

                    rows.append([
                        name_str,
                        f"{r['mae']:.3f}",
                        f"{r['rmse']:.3f}",
                        f"{cv:.2f}%",
                        f"{nmbe:+.2f}%",
                        f"{r['smape_pct']:.2f}%",
                        ashrae_tag,
                    ])

                st.markdown(
                    ce_table(headers, rows, sort_keys=["model", "mae", "rmse", "cv", "nmbe", "smape", "ashrae"], source="reports/forecast_metrics.csv"),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(ce_empty_message("No model metrics for this building in test fold."), unsafe_allow_html=True)
        else:
            st.markdown(ce_empty_message("reports/forecast_metrics.csv not found."), unsafe_allow_html=True)

    with tab_anom:
        st.markdown(f"<h4 style='font-size: 17px; margin-bottom: 8px;'>Flagged Incidents for {b_info['short_name']}</h4>", unsafe_allow_html=True)
        if not b_events.empty:
            a_headers = ["Event ID", "Start", "End", "Dur", "Type", "Severity", "Peak z", "Excess kWh", "Cost (₹)", "Status"]
            a_rows = []
            for _, r in b_events.iterrows():
                link_html = f'<a href="?page=anomalies&event={r["event_id"]}" target="_self"><b>{r["event_id"]}</b></a>'
                s_str = r["start"].strftime("%Y-%m-%d %H:%M")
                e_str = r["end"].strftime("%Y-%m-%d %H:%M")
                t_tag = ce_tag(r["type"], r["type"])
                sev_tag = ce_tag(r["severity"], r["severity"])
                stat_tag = ce_tag(r["review_status"], r["review_status"])

                a_rows.append([
                    link_html,
                    s_str,
                    e_str,
                    f"{r['duration_h']}h",
                    t_tag,
                    sev_tag,
                    f"{r['peak_z']:.2f}",
                    f"{r['excess_kWh']:,.1f}",
                    f"₹{r['est_cost_INR']:,.0f}",
                    stat_tag,
                ])
            st.markdown(
                ce_table(a_headers, a_rows, sort_keys=["id", "start", "end", "dur", "type", "sev", "z", "excess", "cost", "status"], source="reports/anomalies_clean_test.csv"),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(ce_empty_message("No anomalies flagged for this building in selected range."), unsafe_allow_html=True)

    with tab_cost:
        st.markdown("<h4 style='font-size: 17px; margin-bottom: 8px;'>Monthly Excess Waste, Tariff Consequence, and Emissions</h4>", unsafe_allow_html=True)
        if not b_events.empty:
            b_events_copy = b_events.copy()
            b_events_copy["month"] = b_events_copy["start"].dt.strftime("%Y-%m")
            m_grp = b_events_copy.groupby("month").agg(
                events=("event_id", "count"),
                hours=("duration_h", "sum"),
                excess_kwh=("excess_kWh", "sum"),
                cost_inr=("est_cost_INR", "sum"),
                co2_kg=("est_CO2_kg", "sum"),
            ).reset_index()

            c_headers = ["Month", "Events", "Hours", "Excess Energy", f"Cost (₹{tariff:.2f}/kWh)", f"Carbon ({co2_factor:.2f} kg/kWh)"]
            c_rows = []
            for _, r in m_grp.iterrows():
                c_rows.append([
                    r["month"],
                    f"{r['events']:,}",
                    f"{r['hours']:,} h",
                    fmt_kwh(r["excess_kwh"]),
                    fmt_inr(r["cost_inr"]),
                    fmt_co2(r["co2_kg"]),
                ])

            st.markdown(
                ce_table(c_headers, c_rows, sort_keys=["month", "events", "hours", "excess", "cost", "co2"], source="reports/anomaly_impact.csv"),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(ce_empty_message("No excess load recorded."), unsafe_allow_html=True)

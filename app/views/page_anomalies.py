"""
Page 3: Anomalies (Registry and Triage).
Audit, review, and mark detected energy anomalies.
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
    ce_source_note,
    ce_empty_message,
)
from theme import (
    fig_event_zoom,
    chart_layout,
    CHART_THEME,
    ANOMALY_MARKERS,
    SEVERITY_SIZES,
    render_chart,
    time_bounds,
)
from data import (
    get_building_catalog,
    build_event_registry,
    save_anomaly_review,
    get_series_data,
    fmt_kwh,
    fmt_inr,
    fmt_co2,
    building_id_to_short,
)
import plotly.graph_objects as go

def render(ctx: dict):
    st.markdown(
        ce_page_header(
            "Anomalies",
            "Unified anomaly event registry: investigate, audit, triage, and confirm flagged operational incidents.",
        ),
        unsafe_allow_html=True,
    )

    catalog = get_building_catalog()
    all_bids = catalog["building_id"].tolist()
    tariff = float(ctx.get("tariff", 8.0))
    co2_factor = float(ctx.get("co2_factor", 0.71))

    # Pre-select event if passed via query params
    url_event_id = st.query_params.get("event")

    # Filter Bar
    with st.container(key="ce_filters"):
        fcol1, fcol2, fcol3, fcol4, fcol5 = st.columns([2, 2, 2, 2, 2])

        with fcol1:
            b_opts = ["All buildings"] + catalog["building_label"].tolist()
            sel_b_label = st.selectbox("Building", b_opts, index=0, key="anom_bld")

        with fcol2:
            type_opts = ["All types", "spike", "after-hours", "flatline", "drift"]
            sel_type = st.selectbox("Type", type_opts, index=0, key="anom_type")

        with fcol3:
            sev_opts = ["All severities", "high", "medium", "low"]
            sel_sev = st.selectbox("Severity", sev_opts, index=0, key="anom_sev")

        with fcol4:
            status_opts = ["All statuses", "New", "Confirmed", "False positive", "Resolved"]
            sel_status = st.selectbox("Status", status_opts, index=0, key="anom_status")

        with fcol5:
            min_z = st.number_input("Min |z| score", min_value=1.0, max_value=20.0, value=3.0, step=0.5, key="anom_minz")

    # Load Full Registry
    events_df = build_event_registry(tariff=tariff, co2_factor=co2_factor)

    if events_df.empty:
        st.markdown(ce_empty_message("No anomaly events recorded."), unsafe_allow_html=True)
        return

    # Filter registry
    filtered = events_df.copy()
    if sel_b_label != "All buildings":
        sel_bid = catalog[catalog["building_label"] == sel_b_label]["building_id"].iloc[0]
        filtered = filtered[filtered["building_id"] == sel_bid]

    if sel_type != "All types":
        filtered = filtered[filtered["type"] == sel_type]

    if sel_sev != "All severities":
        filtered = filtered[filtered["severity"] == sel_sev]

    if sel_status != "All statuses":
        filtered = filtered[filtered["review_status"] == sel_status]

    if min_z > 0:
        filtered = filtered[filtered["peak_z"] >= min_z]

    # KPI Strip (4 numbers)
    tot_events = len(filtered)
    open_events = int((filtered["review_status"] == "New").sum()) if not filtered.empty else 0
    tot_excess = float(filtered["excess_kWh"].sum()) if not filtered.empty else 0.0
    tot_cost = float(filtered["est_cost_INR"].sum()) if not filtered.empty else 0.0

    kpis_html = "".join([
        ce_metric_row("Total Events", f"{tot_events:,}", delta=f"{len(events_df)} total in registry", delta_kind="neutral", source="reports/anomalies_clean_test.csv"),
        ce_metric_row("Open (Unreviewed)", f"{open_events:,}", delta="Requires review" if open_events > 0 else "All resolved", delta_kind="warn" if open_events > 0 else "pass", source="data/processed/anomaly_reviews.csv"),
        ce_metric_row("Excess Energy", f"{tot_excess:,.1f}", unit="kWh", delta=f"~{tot_excess * co2_factor:,.1f} kg CO₂", delta_kind="neutral", source="anomalies_clean_test.csv"),
        ce_metric_row("Estimated Impact", fmt_inr(tot_cost), delta=f"Tariff ₹{tariff:.2f}/kWh", delta_kind="warn" if tot_cost > 0 else "neutral", source="excess_kWh × tariff"),
    ])
    st.markdown(ce_kpi_strip(kpis_html), unsafe_allow_html=True)

    # Primary Views Tabs
    tab_events, tab_by_type, tab_timeline, tab_dist = st.tabs([
        "Events (Registry & Triage)",
        "By Anomaly Type",
        "Timeline Axis",
        "Distributions",
    ])

    with tab_events:
        st.markdown("<h4 style='margin: 0 0 10px 0; font-size: 17px;'>Filtered Incident Records</h4>", unsafe_allow_html=True)

        if not filtered.empty:
            # Event selection dropdown for investigation
            event_ids = filtered["event_id"].tolist()
            default_idx = 0
            if url_event_id and url_event_id in event_ids:
                default_idx = event_ids.index(url_event_id)

            sel_col, page_col = st.columns([4, 1])
            with sel_col:
                selected_eid = st.selectbox(
                    "Select Event to Inspect & Review",
                    event_ids,
                    index=default_idx,
                    key="sel_event_box",
                    format_func=lambda eid: f"{eid} — {filtered[filtered['event_id']==eid]['building_label'].iloc[0]} ({filtered[filtered['event_id']==eid]['type'].iloc[0]}, {filtered[filtered['event_id']==eid]['severity'].iloc[0]})",
                )

            # Pagination setup (25 rows)
            page_size = 25
            total_pages = max(1, (len(filtered) - 1) // page_size + 1)
            with page_col:
                page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1, key="anom_page_num")

            start_idx = (page_num - 1) * page_size
            page_rows = filtered.iloc[start_idx : start_idx + page_size]

            # Format Table
            headers = ["Event ID", "Building", "Start", "Dur", "Type", "Sev", "Peak z", "Excess", "₹ Impact", "Status"]
            rows = []
            for _, r in page_rows.iterrows():
                is_active = (r["event_id"] == selected_eid)
                prefix = "▶ " if is_active else ""
                id_cell = f"<span style='font-family:var(--font-mono);font-weight:600;'>{prefix}{r['event_id']}</span>"
                b_short = building_id_to_short(r["building_id"])
                start_str = r["start"].strftime("%Y-%m-%d %H:%M")
                dur_str = f"{r['duration_h']}h"
                t_tag = ce_tag(r["type"], r["type"])
                s_tag = ce_tag(r["severity"], r["severity"])
                z_str = f"{r['peak_z']:.2f}"
                exc_str = f"{r['excess_kWh']:,.1f}"
                cost_str = f"₹{r['est_cost_INR']:,.0f}"
                stat_tag = ce_tag(r["review_status"], r["review_status"])

                rows.append([id_cell, b_short, start_str, dur_str, t_tag, s_tag, z_str, exc_str, cost_str, stat_tag])

            st.markdown(
                ce_table(headers, rows, sort_keys=["id", "bld", "start", "dur", "type", "sev", "z", "excess", "cost", "status"]),
                unsafe_allow_html=True,
            )
            st.caption(f"Showing {start_idx+1}–{min(start_idx+page_size, len(filtered))} of {len(filtered)} events. Select an event above to open detail panel.")

            # Export CSV button
            csv_bytes = filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download filtered registry (CSV)",
                data=csv_bytes,
                file_name="campus_energy_anomaly_registry.csv",
                mime="text/csv",
                key="export_registry_csv",
            )
        else:
            st.markdown(ce_empty_message("No events match the selected filters."), unsafe_allow_html=True)

        st.markdown("<hr style='border:none;border-top:1px solid var(--border-subtle);margin:24px 0;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin: 0 0 12px 0; font-size: 17px;'>Incident Investigation & Triage</h4>", unsafe_allow_html=True)

        if not filtered.empty and selected_eid:
            ev_row = filtered[filtered["event_id"] == selected_eid].iloc[0]

            detail_col, review_col = st.columns([3, 2])

            with detail_col:
                # Plain-English Explanation Card
                st.markdown(f"""
<div style="background:var(--surface-2);border-left:3px solid var(--accent);padding:14px 18px;border-radius:0 var(--radius-md) var(--radius-md) 0;font-size:15px;margin-bottom:16px;line-height:1.6;color:var(--text-secondary);">
  <b style="color:var(--text-primary);">Diagnostic Summary:</b> {ev_row['explanation']}
</div>
""", unsafe_allow_html=True)

                # 48-Hour Zoom Chart around event
                zoom_start = ev_row["start"] - pd.Timedelta(hours=24)
                zoom_end = ev_row["end"] + pd.Timedelta(hours=24)
                zoom_df = get_series_data([ev_row["building_id"]], model=ctx.get("model", "random_forest"), start_ts=zoom_start, end_ts=zoom_end)

                if not zoom_df.empty:
                    fig_z = fig_event_zoom(zoom_df, ev_row.to_dict())
                    render_chart(fig_z, key="anomaly_event_zoom")

                # Event Metrics Grid
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Peak z", f"{ev_row['peak_z']:.2f}")
                with m2:
                    st.metric("Excess Load", fmt_kwh(ev_row["excess_kWh"]))
                with m3:
                    st.metric("Cost Consequence", fmt_inr(ev_row["est_cost_INR"]))

            with review_col:
                st.markdown("<b style='font-size:16px;'>Auditor Review Decision</b>", unsafe_allow_html=True)

                current_status = ev_row.get("review_status", "New")
                current_note = ev_row.get("reviewer_note", "")

                status_choices = ["Confirmed", "False positive", "Resolved", "New"]
                st_idx = status_choices.index(current_status) if current_status in status_choices else 0

                new_status = st.selectbox(
                    "Triage Status",
                    status_choices,
                    index=st_idx,
                    key=f"status_sel_{selected_eid}",
                )

                new_note = st.text_area(
                    "Auditor Notes / Action Taken",
                    value=current_note if pd.notna(current_note) else "",
                    placeholder="e.g. Schedule override identified on AHU-2; verified with facility log.",
                    key=f"note_area_{selected_eid}",
                    height=110,
                )

                if st.button("Save Review Label", key=f"save_btn_{selected_eid}", type="primary"):
                    ok = save_anomaly_review(selected_eid, new_status, new_note)
                    if ok:
                        st.success(f"Saved review for {selected_eid} as '{new_status}' in data/processed/anomaly_reviews.csv.")
                        st.cache_data.clear()
                        st.rerun()

                if ev_row.get("last_updated"):
                    st.caption(f"Last updated: {ev_row['last_updated']}")

    with tab_by_type:
        st.markdown("<h4 style='font-size: 17px; margin-bottom: 8px;'>Anomaly Volume and Energy Waste by Failure Mode</h4>", unsafe_allow_html=True)
        if not filtered.empty:
            type_grp = filtered.groupby("type").agg(
                count=("event_id", "count"),
                hours=("duration_h", "sum"),
                excess_kwh=("excess_kWh", "sum"),
                cost_inr=("est_cost_INR", "sum"),
                mean_z=("peak_z", "mean"),
            ).reset_index()

            t_headers = ["Anomaly Type", "Event Count", "Total Hours", "Excess kWh", "Est. Cost (₹)", "Mean Peak z"]
            t_rows = []
            for _, r in type_grp.iterrows():
                t_tag = ce_tag(r["type"], r["type"])
                t_rows.append([
                    t_tag,
                    f"{r['count']:,}",
                    f"{r['hours']:,} h",
                    f"{r['excess_kwh']:,.1f} kWh",
                    f"₹{r['cost_inr']:,.0f}",
                    f"{r['mean_z']:.2f}",
                ])

            st.markdown(ce_table(t_headers, t_rows, sort_keys=["type", "count", "hours", "excess", "cost", "z"]), unsafe_allow_html=True)
        else:
            st.markdown(ce_empty_message("No events in this category."), unsafe_allow_html=True)

    with tab_timeline:
        st.markdown("<h4 style='font-size: 17px; margin-bottom: 8px;'>Incident Timeline Across Buildings</h4>", unsafe_allow_html=True)
        if not filtered.empty:
            fig_tl = go.Figure()
            colors = {
                "spike": CHART_THEME["type_spike"],
                "after-hours": CHART_THEME["type_after_hours"],
                "flatline": CHART_THEME["type_flatline"],
                "drift": CHART_THEME["type_drift"],
            }

            for anom_t, group in filtered.groupby("type"):
                fig_tl.add_trace(go.Scatter(
                    x=group["start"],
                    y=group["building_label"],
                    mode="markers",
                    name=anom_t,
                    marker={
                        "symbol": ANOMALY_MARKERS.get(anom_t, "circle"),
                        "size": [SEVERITY_SIZES.get(s, 8) + 2 for s in group["severity"]],
                        "color": colors.get(anom_t, CHART_THEME["actual"]),
                    },
                    hovertemplate="<b>%{y}</b><br>Start: %{x|%Y-%m-%d %H:%M}<extra></extra>",
                ))

            fig_tl.update_layout(chart_layout(
                title="Chronological Incident Scatter per Building",
                subtitle="Concentrations in late autumn correspond to term starts and weather transitions.",
                x_title="Date",
                y_title="Building",
                height=320,
                x_bounds=time_bounds(filtered),
            ))
            render_chart(fig_tl, key="anomaly_timeline")
        else:
            st.markdown(ce_empty_message("No events on timeline."), unsafe_allow_html=True)

    with tab_dist:
        d1, d2 = st.columns(2)
        with d1:
            fig_z_dist = go.Figure()
            fig_z_dist.add_trace(go.Histogram(
                x=filtered["peak_z"],
                nbinsx=30,
                marker_color=CHART_THEME["actual"],
            ))
            fig_z_dist.update_layout(chart_layout(
                title="Peak |z| Score Distribution",
                subtitle="Most anomalies cluster around z=3–5; extreme spikes reach |z| > 10.",
                x_title="Peak |z| Score",
                y_title="Count of Events",
                height=320,
            ))
            render_chart(fig_z_dist, key="anomaly_z_dist")

        with d2:
            fig_dur_dist = go.Figure()
            fig_dur_dist.add_trace(go.Histogram(
                x=filtered["duration_h"],
                nbinsx=30,
                marker_color=CHART_THEME["expected"],
            ))
            fig_dur_dist.update_layout(chart_layout(
                title="Incident Duration Distribution (Hours)",
                subtitle="Short spikes are 1–2h; flatlines and drift persist for multiple days.",
                x_title="Duration (Hours)",
                y_title="Count of Events",
                height=320,
            ))
            render_chart(fig_dur_dist, key="anomaly_duration_dist")

    st.markdown(ce_source_note("reports/anomalies_clean_test.csv + data/processed/anomaly_reviews.csv"), unsafe_allow_html=True)

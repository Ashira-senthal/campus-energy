"""
Page 6: Explainability.
SHAP feature attributions, single-hour breakdowns, and feature audit glossary.
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
    fig_shap_bar,
    fig_shap_dependence,
    fig_single_hour_contributions,
    render_chart,
)
from data import (
    get_building_catalog,
    load_parquet,
    load_csv_report,
    get_building_thresholds,
    build_event_registry,
    building_id_to_short,
)

def render(ctx: dict):
    st.markdown(
        ce_page_header(
            "Explainability",
            "Interpretability engine: Shapley Additive exPlanations (SHAP) decomposing predictive drivers and single-incident attribution.",
        ),
        unsafe_allow_html=True,
    )

    catalog = get_building_catalog()
    all_bids = catalog["building_id"].tolist()
    labels = catalog["building_label"].tolist()

    # Top Selector Bar
    with st.container():
        ecol1, ecol2 = st.columns([3, 2])
        with ecol1:
            sel_label = st.selectbox(
                "Select Building",
                labels,
                index=0,
                key="explain_bld_select",
            )
        with ecol2:
            model_options = ["lightgbm", "random_forest"]
            sel_model = st.selectbox(
                "Tree Model (TreeExplainer)",
                model_options,
                index=0,
                key="explain_model_select",
                help="SHAP exact TreeExplainer is computed for gradient-boosted and random forest tree models.",
            )

    bid = catalog[catalog["building_label"] == sel_label]["building_id"].iloc[0]

    # Interpretation Caution Alert
    st.markdown("""
<div style="background:var(--surface-alt);border-left:3px solid var(--info);padding:10px 14px;border-radius:var(--radius-md);font-size:12px;margin:12px 0 20px 0;color:var(--text-muted);">
  <b>Interpretation Caution:</b> SHAP explains what drove the statistical model's prediction given its training features; it does not prove physical causation or mechanical causality in the facility.
</div>
""", unsafe_allow_html=True)

    # Primary Chart: Global SHAP Importance
    shap_importance_df = load_csv_report("shap_importance")
    if shap_importance_df.empty:
        shap_importance_df = load_parquet("importance")

    if not shap_importance_df.empty:
        b_imp = shap_importance_df[shap_importance_df["building_id"] == bid].copy()
        fig_bar = fig_shap_bar(b_imp)
        render_chart(fig_bar, key="explain_global")
        st.markdown(ce_source_note("reports/shap_importance.csv (TreeExplainer over test fold)"), unsafe_allow_html=True)
    else:
        st.markdown(ce_empty_message("No global SHAP importance scores available."), unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    # Tabs: Feature Dependence | Single Hour Explanation | Feature Glossary
    tab_dep, tab_single, tab_gloss = st.tabs([
        "Feature Dependence",
        "Single Hour Explanation",
        "Feature Engineering Glossary",
    ])

    with tab_dep:
        st.markdown("<h4 style='font-size: 14px; margin-bottom: 8px;'>Marginal Feature Dependence (Attribution vs Value)</h4>", unsafe_allow_html=True)

        shap_df = load_parquet("shap")
        if not shap_df.empty:
            b_shap = shap_df[shap_df["building_id"] == bid].copy()
            feat_list = b_shap["feature"].unique().tolist()
            default_feat = "kwh_lag_24" if "kwh_lag_24" in feat_list else (feat_list[0] if feat_list else "")

            sel_feat = st.selectbox("Select Feature to Plot", feat_list, index=feat_list.index(default_feat) if default_feat in feat_list else 0, key="explain_dep_feat")

            fig_dep = fig_shap_dependence(b_shap, sel_feat)
            render_chart(fig_dep, key="explain_dependence")
        else:
            st.markdown(ce_empty_message("Detailed hourly SHAP attribution dataset not loaded."), unsafe_allow_html=True)

    with tab_single:
        st.markdown("<h4 style='font-size: 14px; margin-bottom: 8px;'>Single Incident Hour Decomposition</h4>", unsafe_allow_html=True)

        # Lookup base value
        thresh = get_building_thresholds()
        base_val = 100.0
        if not thresh.empty and "shap_base_value" in thresh.columns:
            t_sub = thresh[thresh["building_id"] == bid]
            if not t_sub.empty:
                base_val = float(t_sub.iloc[0]["shap_base_value"])

        events_df = build_event_registry()
        b_events = events_df[events_df["building_id"] == bid] if not events_df.empty else pd.DataFrame()

        shap_df = load_parquet("shap")

        if not shap_df.empty:
            b_shap = shap_df[shap_df["building_id"] == bid].copy()
            b_shap["timestamp"] = pd.to_datetime(b_shap["timestamp"])
            avail_timestamps = b_shap["timestamp"].drop_duplicates().sort_values(ascending=False).tolist()

            # Offer to pick from flagged anomaly timestamps
            pick_mode = st.radio("Selection Source", ["Flagged Anomaly Event", "Arbitrary Timestamp"], horizontal=True, key="single_pick_mode")

            selected_ts = None
            if pick_mode == "Flagged Anomaly Event" and not b_events.empty:
                ev_options = b_events["event_id"].tolist()
                sel_eid = st.selectbox(
                    "Choose Anomaly Event",
                    ev_options,
                    format_func=lambda eid: f"{eid} — {b_events[b_events['event_id']==eid]['type'].iloc[0]} ({b_events[b_events['event_id']==eid]['start'].iloc[0]:%Y-%m-%d %H:%M})",
                    key="single_ev_sel",
                )
                ev_start = b_events[b_events["event_id"] == sel_eid]["start"].iloc[0]
                # Find closest timestamp in b_shap
                if avail_timestamps:
                    selected_ts = min(avail_timestamps, key=lambda t: abs(t - ev_start))
            else:
                if avail_timestamps:
                    selected_ts = st.selectbox(
                        "Choose Hourly Timestamp",
                        avail_timestamps[:50],
                        format_func=lambda t: t.strftime("%Y-%m-%d %H:%M"),
                        key="single_ts_sel",
                    )

            if selected_ts:
                hour_shap = b_shap[b_shap["timestamp"] == selected_ts].copy()
                if not hour_shap.empty:
                    net_pred = base_val + float(hour_shap["shap"].sum())
                    # Sort top 10 contributors
                    hour_shap["abs_shap"] = hour_shap["shap"].abs()
                    top_contrib = hour_shap.sort_values("abs_shap", ascending=False).head(10)

                    fig_waterfall = fig_single_hour_contributions(top_contrib, base_val=base_val, pred_val=net_pred)
                    render_chart(fig_waterfall, key="explain_single_hour")
                else:
                    st.markdown(ce_empty_message("No SHAP values found for this specific timestamp."), unsafe_allow_html=True)
        else:
            st.markdown(ce_empty_message("data/processed/shap.parquet not available."), unsafe_allow_html=True)

    with tab_gloss:
        st.markdown("<h4 style='font-size: 14px; margin-bottom: 8px;'>Engineered Input Feature Registry (Lag ≥ 24h Guard)</h4>", unsafe_allow_html=True)

        features_data = [
            ("hour", "Calendar", "Hour of day (0–23)", "int32", "Knowable at t=0"),
            ("dayofweek", "Calendar", "Day of week (0=Mon, 6=Sun)", "int32", "Knowable at t=0"),
            ("month", "Calendar", "Month of year (1–12)", "int32", "Knowable at t=0"),
            ("is_weekend", "Calendar", "Indicator for Saturday or Sunday", "binary", "Knowable at t=0"),
            ("is_holiday", "Calendar", "Federal university holiday flag (US calendar)", "binary", "Knowable at t=0"),
            ("hour_sin", "Calendar", "sin(2π · hour / 24) cyclical encoding", "float64", "Continuous periodic representation"),
            ("hour_cos", "Calendar", "cos(2π · hour / 24) cyclical encoding", "float64", "Continuous periodic representation"),
            ("doy_sin", "Calendar", "sin(2π · dayofyear / 365) seasonal encoding", "float64", "Annual solar seasonality"),
            ("doy_cos", "Calendar", "cos(2π · dayofyear / 365) seasonal encoding", "float64", "Annual solar seasonality"),
            ("airTemperature", "Weather", "Dry bulb outdoor air temperature (°C)", "float64", "Exogenous measurement at t"),
            ("dewTemperature", "Weather", "Dew point temperature (°C)", "float64", "Exogenous measurement at t"),
            ("cloudCoverage", "Weather", "Oktas cloud cover (0–8 scale)", "float64", "Exogenous measurement at t"),
            ("windSpeed", "Weather", "Wind speed (m/s)", "float64", "Exogenous measurement at t"),
            ("precipDepth1HR", "Weather", "Precipitation in last hour (mm)", "float64", "Exogenous measurement at t"),
            ("temp_roll_24", "Weather", "Trailing 24h mean outdoor air temperature", "float64", "Thermal inertia proxy"),
            ("kwh_lag_24", "History", "Measured load at t - 24 hours (yesterday same hour)", "float64", "Strict lag ≥ 24h floor"),
            ("kwh_lag_48", "History", "Measured load at t - 48 hours (day-before-yesterday)", "float64", "Strict lag ≥ 24h floor"),
            ("kwh_lag_168", "History", "Measured load at t - 168 hours (same hour last week)", "float64", "Weekly seasonality"),
            ("kwh_roll_mean_24", "History", "Mean load over trailing [t-48h, t-24h] window", "float64", "Yesterday baseload baseline"),
            ("kwh_roll_mean_168", "History", "Mean load over trailing [t-192h, t-24h] window", "float64", "Weekly baseline"),
            ("kwh_roll_std_24", "History", "Standard deviation over [t-48h, t-24h]", "float64", "Volatility proxy"),
            ("kwh_roll_std_168", "History", "Standard deviation over [t-192h, t-24h]", "float64", "Weekly volatility proxy"),
        ]

        g_headers = ["Feature Name", "Group", "Definition & Rationale", "Data Type", "Leakage Guard"]
        g_rows = [[f"<span style='font-family:var(--font-mono);font-weight:600;'>{f[0]}</span>", f[1], f[2], f[3], f[4]] for f in features_data]

        st.markdown(
            ce_table(g_headers, g_rows, sort_keys=["name", "grp", "def", "type", "guard"], source="src/features.py"),
            unsafe_allow_html=True,
        )
        st.markdown("""
<div style="font-size: 12px; color: var(--text-muted); margin-top: 8px;">
  <b>The Lag ≥ 24h Rule:</b> No electricity feature may use information from less than 24 hours prior to prediction time.
  If the model were allowed to use <code>kWh(t-1)</code>, ongoing faults would be learned as normal behavior, causing the residual to collapse to zero exactly when an anomaly alarm is required.
</div>
""", unsafe_allow_html=True)

"""
Page 10: Settings & Assumptions.
Configurable economic parameters, detection sensitivity, and operational thresholds.
"""

from __future__ import annotations

import json
import streamlit as st
import pandas as pd

from components import (
    ce_page_header,
    ce_table,
    ce_source_note,
)
from data import get_building_catalog

def render(ctx: dict):
    st.markdown(
        ce_page_header(
            "Settings & Assumptions",
            "Operational configuration: customize energy tariffs, carbon conversion factors, and anomaly thresholds.",
        ),
        unsafe_allow_html=True,
    )

    # Defaults from src/config.py
    DEFAULT_TARIFF = 8.00
    DEFAULT_CO2 = 0.71
    DEFAULT_Z = 3.0
    DEFAULT_MODEL = "random_forest"
    DEFAULT_DETECTOR = "hybrid"
    DEFAULT_HOLIDAY = "US"

    st.markdown("<h4 style='font-size: 14px; margin-bottom: 12px;'>Editable Assumptions and Thresholds</h4>", unsafe_allow_html=True)

    with st.form("settings_form"):
        col1, col2 = st.columns(2)

        with col1:
            tariff = st.number_input(
                "Electricity Tariff (₹ / kWh)",
                min_value=0.50,
                max_value=100.0,
                value=float(ctx.get("tariff", DEFAULT_TARIFF)),
                step=0.50,
                help="Tariff rate used to compute avoidable financial waste.",
            )
            st.caption("🔍 **Verify:** *Placeholder. Verify against campus utility electricity tariff bill (typically HT-commercial / institutional rate).*")

            co2 = st.number_input(
                "Grid Carbon Factor (kg CO₂ / kWh)",
                min_value=0.05,
                max_value=2.00,
                value=float(ctx.get("co2_factor", DEFAULT_CO2)),
                step=0.01,
                help="Average grid emissions intensity factor per kWh consumed.",
            )
            st.caption("🔍 **Verify:** *Placeholder. Verify with regional Central Electricity Authority (CEA) CO₂ Baseline Database for the national grid.*")

            holiday_country = st.selectbox(
                "Holiday Calendar Country",
                ["US", "IN", "UK", "SG"],
                index=["US", "IN", "UK", "SG"].index(ctx.get("holiday_country", DEFAULT_HOLIDAY)),
                help="Determines university calendar closures.",
            )
            st.caption("🔍 **Verify:** *US matches BDG2 site Fox geographic coordinates. Switch to IN if testing on local Indian campus meters.*")

        with col2:
            z_thresh = st.slider(
                "Anomaly Robust z Threshold",
                min_value=2.0,
                max_value=6.0,
                value=float(ctx.get("z_threshold", DEFAULT_Z)),
                step=0.5,
                help="Number of scaled MADs above median required to flag an hour.",
            )
            st.caption("🔍 **Verify:** *Default z=3.0 was selected from validation fold sweep to guarantee validation false alarms ≤ 2.0%.*")

            model = st.selectbox(
                "Default Forecasting Model",
                ["random_forest", "ridge", "lightgbm", "snaive_168", "snaive_24"],
                index=["random_forest", "ridge", "lightgbm", "snaive_168", "snaive_24"].index(ctx.get("model", DEFAULT_MODEL)),
                help="Primary regression model used to compute residuals.",
            )
            st.caption("🔍 **Verify:** *Random forest and LightGBM provide the lowest CV(RMSE) across all four buildings.*")

            detector = st.selectbox(
                "Default Detection Engine",
                ["hybrid", "robust_z", "residual_persistence", "isolation_forest", "adaptive_hybrid"],
                index=["hybrid", "robust_z", "residual_persistence", "isolation_forest", "adaptive_hybrid"].index(ctx.get("detector", DEFAULT_DETECTOR)),
                help="Engine used to isolate abnormal deviations.",
            )
            st.caption("🔍 **Verify:** *Hybrid balances recall and precision; switch to adaptive_hybrid if operating automated pager alerts.*")

        st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
        btn_save, btn_reset, _ = st.columns([2, 2, 6])
        with btn_save:
            submitted = st.form_submit_button("Save & Apply Settings", type="primary")
        with btn_reset:
            reset = st.form_submit_button("Reset to Defaults")

    if submitted:
        ctx["tariff"] = tariff
        ctx["co2_factor"] = co2
        ctx["z_threshold"] = z_thresh
        ctx["model"] = model
        ctx["detector"] = detector
        ctx["holiday_country"] = holiday_country
        st.cache_data.clear()
        st.success("Settings updated and applied across all monitoring views.")
        st.rerun()

    if reset:
        ctx["tariff"] = DEFAULT_TARIFF
        ctx["co2_factor"] = DEFAULT_CO2
        ctx["z_threshold"] = DEFAULT_Z
        ctx["model"] = DEFAULT_MODEL
        ctx["detector"] = DEFAULT_DETECTOR
        ctx["holiday_country"] = DEFAULT_HOLIDAY
        st.cache_data.clear()
        st.info("Reset all settings to repository defaults.")
        st.rerun()

    st.markdown("<hr style='border:none;border-top:1px solid var(--border);margin:24px 0;'>", unsafe_allow_html=True)
    st.markdown("<h4 style='font-size: 14px; margin-bottom: 8px;'>Active Configuration State (JSON)</h4>", unsafe_allow_html=True)

    config_view = {
        "site_id": "Fox",
        "buildings": ctx.get("buildings", ["Fox_education_Andre", "Fox_education_Elvira", "Fox_education_Claire", "Fox_education_Kendrick"]),
        "start_date": str(ctx.get("start_date", "2017-09-01")),
        "end_date": str(ctx.get("end_date", "2017-12-31")),
        "tariff_inr_per_kwh": ctx.get("tariff", DEFAULT_TARIFF),
        "co2_kg_per_kwh": ctx.get("co2_factor", DEFAULT_CO2),
        "z_threshold": ctx.get("z_threshold", DEFAULT_Z),
        "primary_model": ctx.get("model", DEFAULT_MODEL),
        "detector": ctx.get("detector", DEFAULT_DETECTOR),
        "holiday_country": ctx.get("holiday_country", DEFAULT_HOLIDAY),
    }
    st.json(config_view)
    st.markdown(ce_source_note("src/config.py"), unsafe_allow_html=True)

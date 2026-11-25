"""
Campus Energy Monitoring Console — Main Application Shell and Router.

Audience: Engineers and facility analysts.
Dense-with-order, progressive disclosure, 100% traceable to reports/ and data/processed/.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import pandas as pd

from theme import load_css
from components import ce_sidebar_html, ce_context_bar
from data import (
    get_pipeline_status,
    get_building_catalog,
    building_id_to_short,
)

# Page Registry matching Section 2 Navigation
PAGES = [
    # Group 1: PROJECTS
    {"slug": "overview", "label": "Overview", "icon": "●", "group": "Projects"},
    {"slug": "live", "label": "Live Monitor", "icon": "◉", "group": "Projects"},
    {"slug": "anomalies", "label": "Anomalies", "icon": "◆", "group": "Projects"},

    # Group 2: STATUS
    {"slug": "buildings", "label": "Buildings", "icon": "▤", "group": "Status"},
    {"slug": "compare", "label": "Compare", "icon": "▽", "group": "Status"},
    {"slug": "explain", "label": "Explainability", "icon": "◻", "group": "Status"},

    # Group 3: HISTORY
    {"slug": "model", "label": "Model Performance", "icon": "◁", "group": "History"},
    {"slug": "detector", "label": "Detector Evaluation", "icon": "▷", "group": "History"},

    # Group 4: SYSTEM
    {"slug": "data", "label": "Data & Quality", "icon": "▥", "group": "System"},
    {"slug": "settings", "label": "Settings & Assumptions", "icon": "⚙", "group": "System"},
    {"slug": "method", "label": "Methodology", "icon": "○", "group": "System"},
]

def init_session_state():
    catalog = get_building_catalog()
    all_bids = catalog["building_id"].tolist()

    if "ctx" not in st.session_state:
        st.session_state["ctx"] = {
            "buildings": all_bids,
            "start_date": pd.Timestamp("2017-09-01"),
            "end_date": pd.Timestamp("2017-12-31"),
            "model": "random_forest",
            "detector": "hybrid",
            "tariff": 8.0,
            "co2_factor": 0.71,
            "z_threshold": 3.0,
        }

def main():
    st.set_page_config(
        page_title="Campus Energy — Monitoring console",
        page_icon=str(Path(__file__).resolve().parent / "assets" / "logo.png"),
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    ctx = st.session_state["ctx"]

    # Handle filter reset query param
    if st.query_params.get("reset_filters"):
        catalog = get_building_catalog()
        ctx["buildings"] = catalog["building_id"].tolist()
        ctx["start_date"] = pd.Timestamp("2017-09-01")
        ctx["end_date"] = pd.Timestamp("2017-12-31")
        ctx["model"] = "random_forest"
        ctx["detector"] = "hybrid"
        ctx["tariff"] = 8.0
        ctx["co2_factor"] = 0.71
        ctx["z_threshold"] = 3.0
        # Preserve page if present
        current_page = st.query_params.get("page", "overview")
        st.query_params.clear()
        st.query_params["page"] = current_page
        st.rerun()

    # Determine active page from URL query params
    active_page = st.query_params.get("page", "overview")
    valid_slugs = [p["slug"] for p in PAGES]
    if active_page not in valid_slugs:
        active_page = "overview"
        st.query_params["page"] = "overview"

    # Inject the design system
    css_content = load_css()
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

    # Render Sidebar
    run_time, window = get_pipeline_status()
    with st.sidebar:
        st.markdown(
            ce_sidebar_html(PAGES, active_page, run_time, window),
            unsafe_allow_html=True,
        )

    # Render Sticky Global Context Bar
    catalog = get_building_catalog()
    all_bids = catalog["building_id"].tolist()
    sel_bids = ctx.get("buildings", all_bids)

    if set(sel_bids) == set(all_bids):
        blds_str = "All 4 buildings"
    else:
        blds_str = ", ".join([building_id_to_short(b) for b in sel_bids]) if sel_bids else "None"

    start_d = pd.to_datetime(ctx.get("start_date", "2017-09-01")).strftime("%Y-%m-%d")
    end_d = pd.to_datetime(ctx.get("end_date", "2017-12-31")).strftime("%Y-%m-%d")
    range_str = f"{start_d} → {end_d}"

    model_label_map = {
        "random_forest": "Random forest",
        "ridge": "Ridge",
        "lightgbm": "LightGBM",
        "snaive_168": "Seasonal naive (168h)",
        "snaive_24": "Seasonal naive (24h)",
    }
    model_str = model_label_map.get(ctx.get("model", "random_forest"), ctx.get("model", "random_forest"))

    detector_label_map = {
        "hybrid": "Hybrid (persistence + flatline)",
        "robust_z": "Residual (robust z)",
        "residual_persistence": "Residual + persistence",
        "isolation_forest": "Isolation Forest",
        "adaptive_hybrid": "Adaptive hybrid (30-day)",
    }
    detector_str = detector_label_map.get(ctx.get("detector", "hybrid"), ctx.get("detector", "hybrid"))

    st.markdown(
        ce_context_bar(blds_str, range_str, model_str, detector_str, current_page=active_page),
        unsafe_allow_html=True,
    )

    # Route to individual page module
    if active_page == "overview":
        from views.page_overview import render
        render(ctx)
    elif active_page == "live":
        from views.page_live import render
        render(ctx)
    elif active_page == "anomalies":
        from views.page_anomalies import render
        render(ctx)
    elif active_page == "buildings":
        from views.page_buildings import render
        render(ctx)
    elif active_page == "compare":
        from views.page_compare import render
        render(ctx)
    elif active_page == "explain":
        from views.page_explain import render
        render(ctx)
    elif active_page == "model":
        from views.page_model import render
        render(ctx)
    elif active_page == "detector":
        from views.page_detector import render
        render(ctx)
    elif active_page == "data":
        from views.page_data import render
        render(ctx)
    elif active_page == "settings":
        from views.page_settings import render
        render(ctx)
    elif active_page == "method":
        from views.page_method import render
        render(ctx)

if __name__ == "__main__":
    main()

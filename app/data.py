"""
Data helpers and anomaly review persistence for Campus Energy Monitoring Console.

Reads from data/processed/ (parquet) and reports/ (csv, json).
Provides cached accessors, building catalog helpers, anomaly event merging,
and atomic review persistence in data/processed/anomaly_reviews.csv.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
REVIEWS_PATH = PROCESSED_DIR / "anomaly_reviews.csv"

# --------------------------------------------------------------------------
# Building catalog constants
# --------------------------------------------------------------------------
BUILDINGS_INFO = [
    {
        "building_id": "Fox_education_Andre",
        "building_label": "Andre (Laboratory)",
        "short_name": "Andre",
        "usage": "College Laboratory",
        "sqm": 17044.0,
        "mean_kwh": 454.46,
    },
    {
        "building_id": "Fox_education_Elvira",
        "building_label": "Elvira (Student Union)",
        "short_name": "Elvira",
        "usage": "Student Union",
        "sqm": 13826.7,
        "mean_kwh": 187.90,
    },
    {
        "building_id": "Fox_education_Claire",
        "building_label": "Claire (Classroom)",
        "short_name": "Claire",
        "usage": "College Classroom",
        "sqm": 8145.1,
        "mean_kwh": 60.08,
    },
    {
        "building_id": "Fox_education_Kendrick",
        "building_label": "Kendrick (Auditorium)",
        "short_name": "Kendrick",
        "usage": "Auditorium",
        "sqm": 1052.5,
        "mean_kwh": 12.45,
    },
]

ID_TO_LABEL = {b["building_id"]: b["building_label"] for b in BUILDINGS_INFO}
LABEL_TO_ID = {b["building_label"]: b["building_id"] for b in BUILDINGS_INFO}
ID_TO_SHORT = {b["building_id"]: b["short_name"] for b in BUILDINGS_INFO}

# --------------------------------------------------------------------------
# Cached File Loaders
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_parquet(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_csv_report(name: str) -> pd.DataFrame:
    path = REPORTS_DIR / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_pipeline_summary() -> dict:
    path = REPORTS_DIR / "pipeline_summary.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def get_pipeline_status() -> tuple[str, str]:
    """Return (last_run_iso, data_window_str)."""
    summary = load_pipeline_summary()
    run_time = summary.get("generated")
    if not run_time:
        p_path = REPORTS_DIR / "pipeline_summary.json"
        if p_path.exists():
            run_time = pd.Timestamp(os.path.getmtime(p_path), unit="s").isoformat()[:16]
        else:
            run_time = "Pipeline not run"
    else:
        run_time = run_time[:16].replace("T", " ")

    window = "2016-01-01 → 2017-12-31"
    return run_time, window

def get_building_catalog() -> pd.DataFrame:
    return pd.DataFrame(BUILDINGS_INFO)

def building_id_to_label(bid: str) -> str:
    return ID_TO_LABEL.get(bid, bid)

def building_label_to_id(label: str) -> str:
    return LABEL_TO_ID.get(label, label)

def building_id_to_short(bid: str) -> str:
    return ID_TO_SHORT.get(bid, bid)

# --------------------------------------------------------------------------
# Robust Scale & Thresholds
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_building_thresholds() -> pd.DataFrame:
    df = load_csv_report("building_thresholds")
    if not df.empty:
        return df
    # Fallback to buildings.parquet
    bp = load_parquet("buildings")
    return bp

def get_residual_scale(building_id: str) -> float:
    thresh = get_building_thresholds()
    if not thresh.empty and "scale" in thresh.columns:
        row = thresh[thresh["building_id"] == building_id]
        if not row.empty and pd.notna(row.iloc[0]["scale"]):
            return float(row.iloc[0]["scale"])
    res = load_parquet("residuals")
    if not res.empty and "residual" in res.columns:
        sub = res[res["building_id"] == building_id]["residual"].dropna()
        if not sub.empty:
            med = sub.median()
            return float(np.median(np.abs(sub - med)) * 1.4826)
    return 10.0

# --------------------------------------------------------------------------
# Anomaly Reviews Persistence (Atomic file write)
# --------------------------------------------------------------------------
def load_anomaly_reviews() -> pd.DataFrame:
    if not REVIEWS_PATH.exists():
        return pd.DataFrame(columns=["event_id", "review_status", "reviewer_note", "last_updated"])
    try:
        df = pd.read_csv(REVIEWS_PATH, dtype=str)
        for col in ["event_id", "review_status", "reviewer_note", "last_updated"]:
            if col not in df.columns:
                df[col] = ""
        return df
    except Exception:
        return pd.DataFrame(columns=["event_id", "review_status", "reviewer_note", "last_updated"])

def save_anomaly_review(event_id: str, review_status: str, reviewer_note: str) -> bool:
    """Atomically update review status for an event in anomaly_reviews.csv."""
    try:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        now_ts = pd.Timestamp.now().isoformat(timespec="minutes")
        reviews = load_anomaly_reviews()

        if event_id in reviews["event_id"].values:
            mask = reviews["event_id"] == event_id
            reviews.loc[mask, "review_status"] = review_status
            reviews.loc[mask, "reviewer_note"] = reviewer_note
            reviews.loc[mask, "last_updated"] = now_ts
        else:
            new_row = pd.DataFrame([{
                "event_id": event_id,
                "review_status": review_status,
                "reviewer_note": reviewer_note,
                "last_updated": now_ts,
            }])
            reviews = pd.concat([reviews, new_row], ignore_index=True)

        tmp_path = PROCESSED_DIR / "anomaly_reviews.csv.tmp"
        reviews.to_csv(tmp_path, index=False)
        os.replace(tmp_path, REVIEWS_PATH)
        return True
    except Exception as e:
        st.error(f"Failed to save review: {e}")
        return False

def get_review_counts() -> dict[str, int]:
    reviews = load_anomaly_reviews()
    if reviews.empty:
        return {"total": 0, "open": 0, "confirmed": 0, "false_positive": 0, "resolved": 0}
    vc = reviews["review_status"].value_counts()
    return {
        "total": len(reviews),
        "open": int(vc.get("New", 0)),
        "confirmed": int(vc.get("Confirmed", 0)),
        "false_positive": int(vc.get("False positive", 0)),
        "resolved": int(vc.get("Resolved", 0)),
    }

# --------------------------------------------------------------------------
# Event Registry Builder (Merged consecutive hours, Section 5 format)
# --------------------------------------------------------------------------
ANOMALY_TYPE_MAP = {
    "short spike": "spike",
    "spike": "spike",
    "after-hours load": "after-hours",
    "after_hours": "after-hours",
    "meter fault (flatline)": "flatline",
    "flatline": "flatline",
    "slow drift": "drift",
    "sustained high load": "drift",
    "drift": "drift",
    "unexpected drop": "after-hours",
    "unusual load": "drift",
}

@st.cache_data(show_spinner=False)
def build_event_registry(tariff: float = 8.0, co2_factor: float = 0.71) -> pd.DataFrame:
    """Build unified event registry from anomalies_clean_test.csv and merge reviews."""
    raw = load_csv_report("anomalies_clean_test")
    if raw.empty or "timestamp" not in raw.columns:
        return pd.DataFrame()

    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["z"] = pd.to_numeric(df["z"], errors="coerce").fillna(0.0)
    df["electricity"] = pd.to_numeric(df["electricity"], errors="coerce").fillna(0.0)
    df["expected"] = pd.to_numeric(df["expected"], errors="coerce").fillna(0.0)
    df["excess_kwh"] = pd.to_numeric(df["excess_kwh"], errors="coerce").fillna(0.0)
    df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)

    events: list[dict] = []
    event_seq = 1

    for bid, bgroup in df.groupby("building_id"):
        bgroup = bgroup.sort_values("timestamp").reset_index(drop=True)
        # Identify consecutive runs (gap <= 1.5 hours)
        time_diff = (bgroup["timestamp"] - bgroup["timestamp"].shift(1)).dt.total_seconds() / 3600.0
        bgroup["run_id"] = (time_diff > 1.5).cumsum()

        for _, run in bgroup.groupby("run_id"):
            if run.empty:
                continue
            start_ts = run["timestamp"].min()
            end_ts = run["timestamp"].max()
            duration_h = int(len(run))

            # Determine dominant canonical anomaly type
            types = run["anomaly_type"].map(ANOMALY_TYPE_MAP).dropna().tolist()
            if not types:
                types = ["drift"]
            dom_type = Counter(types).most_common(1)[0][0]

            peak_z = float(run["z"].abs().max())
            actual_kwh = float(run["electricity"].sum())
            expected_kwh = float(run["expected"].sum())
            excess_kwh = float(run["excess_kwh"].sum())

            # Severity thresholds: high if |z| >= 6.0 or excess >= 500 kWh
            # medium if |z| >= 4.5 or excess >= 150 kWh; otherwise low
            if peak_z >= 6.0 or excess_kwh >= 500.0:
                severity = "high"
            elif peak_z >= 4.5 or excess_kwh >= 150.0:
                severity = "medium"
            else:
                severity = "low"

            est_cost_inr = round(excess_kwh * tariff, 0)
            est_co2_kg = round(excess_kwh * co2_factor, 1)

            # Pick explanation
            reasons = run["reason"].dropna().tolist()
            explanation = reasons[0] if reasons else f"Anomalous {dom_type} detected with peak z={peak_z:.1f}."

            event_id = f"ANM-2017-{event_seq:06d}"
            event_seq += 1

            events.append({
                "event_id": event_id,
                "building_id": bid,
                "building_label": building_id_to_label(bid),
                "start": start_ts,
                "end": end_ts,
                "duration_h": duration_h,
                "type": dom_type,
                "severity": severity,
                "peak_z": round(peak_z, 2),
                "actual_kWh": round(actual_kwh, 1),
                "expected_kWh": round(expected_kwh, 1),
                "excess_kWh": round(excess_kwh, 1),
                "est_cost_INR": est_cost_inr,
                "est_CO2_kg": est_co2_kg,
                "detector": "hybrid",
                "explanation": explanation,
                "review_status": "New",
                "reviewer_note": "",
                "last_updated": "",
            })

    reg = pd.DataFrame(events)
    if reg.empty:
        return reg

    # Merge persisted reviews
    reviews = load_anomaly_reviews()
    if not reviews.empty and "event_id" in reviews.columns:
        reviews_dict = reviews.set_index("event_id").to_dict(orient="index")
        for idx, row in reg.iterrows():
            eid = row["event_id"]
            if eid in reviews_dict:
                rev = reviews_dict[eid]
                status = rev.get("review_status")
                note = rev.get("reviewer_note")
                updated = rev.get("last_updated")
                if status and pd.notna(status) and status != "":
                    reg.at[idx, "review_status"] = status
                if note and pd.notna(note):
                    reg.at[idx, "reviewer_note"] = str(note)
                if updated and pd.notna(updated):
                    reg.at[idx, "last_updated"] = str(updated)

    return reg

# --------------------------------------------------------------------------
# Portfolio & Timeseries Series Extractors
# --------------------------------------------------------------------------
def get_series_data(
    building_ids: list[str],
    model: str = "random_forest",
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return aggregated hourly DataFrame with timestamp, actual, expected, residual, z."""
    hourly = load_parquet("hourly")
    predictions = load_parquet("predictions")
    residuals = load_parquet("residuals")

    if hourly.empty:
        return pd.DataFrame()

    hourly = hourly.copy()
    hourly["timestamp"] = pd.to_datetime(hourly["timestamp"])
    if building_ids:
        hourly = hourly[hourly["building_id"].isin(building_ids)]
    if start_ts:
        hourly = hourly[hourly["timestamp"] >= start_ts]
    if end_ts:
        hourly = hourly[hourly["timestamp"] <= end_ts]

    if hourly.empty:
        return pd.DataFrame()

    # Aggregate actual
    actual_agg = hourly.groupby("timestamp")["electricity"].sum().reset_index()
    actual_agg.rename(columns={"electricity": "actual"}, inplace=True)

    # Get predictions
    expected_agg = pd.DataFrame()
    if not predictions.empty:
        p_sub = predictions.copy()
        p_sub["timestamp"] = pd.to_datetime(p_sub["timestamp"])
        if building_ids:
            p_sub = p_sub[p_sub["building_id"].isin(building_ids)]
        if model:
            p_sub = p_sub[p_sub["model"] == model]
        if start_ts:
            p_sub = p_sub[p_sub["timestamp"] >= start_ts]
        if end_ts:
            p_sub = p_sub[p_sub["timestamp"] <= end_ts]

        if not p_sub.empty:
            expected_agg = p_sub.groupby("timestamp")["expected"].sum().reset_index()

    # Merge
    if not expected_agg.empty:
        merged = actual_agg.merge(expected_agg, on="timestamp", how="left")
    else:
        merged = actual_agg
        merged["expected"] = np.nan

    merged["residual"] = merged["actual"] - merged["expected"]

    # Merge z from residuals if single building
    merged["z"] = np.nan
    if len(building_ids) == 1 and not residuals.empty:
        r_sub = residuals[residuals["building_id"] == building_ids[0]].copy()
        r_sub["timestamp"] = pd.to_datetime(r_sub["timestamp"])
        merged = merged.merge(r_sub[["timestamp", "z"]], on="timestamp", how="left", suffixes=("", "_res"))
        if "z_res" in merged.columns:
            merged["z"] = merged["z_res"]
            merged.drop(columns=["z_res"], inplace=True)

    return merged.sort_values("timestamp").reset_index(drop=True)

# --------------------------------------------------------------------------
# Formatting Utilities
# --------------------------------------------------------------------------
def fmt_kwh(val: float | None) -> str:
    if val is None or pd.isna(val):
        return "—"
    return f"{val:,.1f} kWh"

def fmt_inr(val: float | None) -> str:
    if val is None or pd.isna(val):
        return "—"
    return f"₹{val:,.0f}"

def fmt_co2(val: float | None) -> str:
    if val is None or pd.isna(val):
        return "—"
    return f"{val:,.1f} kg"

def fmt_pct(val: float | None) -> str:
    if val is None or pd.isna(val):
        return "—"
    return f"{val:.2f}%"

def fmt_num(val: float | None, dec: int = 2) -> str:
    if val is None or pd.isna(val):
        return "—"
    return f"{val:,.{dec}f}"

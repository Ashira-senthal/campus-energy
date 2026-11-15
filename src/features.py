"""
Cleaning and feature engineering.

Design rule that shapes this whole file: **no feature may use information from
less than 24 hours before the target hour**. If we used ``kWh(t-1)`` the model
would copy the anomaly currently happening and "explain it away"; the residual
(actual - expected) would then be near zero exactly when we need it to be large.
With a 24 h floor, the model describes what the building *should* be doing based
on yesterday, last week, the calendar and the weather.

Weather is exogenous: the temperature at hour t is known at hour t and is not
affected by the building, so it can be used without a lag. Electricity history
is endogenous and always lags by >= 24 h.
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from src.config import (
    HOLIDAY_COUNTRY,
    HOLIDAY_YEARS,
    LAG_HOURS,
    MAX_INTERP_GAP_HOURS,
    MIN_LAG_HOURS,
    ROLL_WINDOWS_HOURS,
)

# Columns that are inputs to the models (everything else is identifier, target
# or bookkeeping). Kept as one list so training and the dashboard agree.
CALENDAR_FEATURES = [
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "is_holiday",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]
WEATHER_FEATURES = [
    "airTemperature",
    "dewTemperature",
    "cloudCoverage",
    "windSpeed",
    "precipDepth1HR",
    "temp_roll_24",
]
HISTORY_FEATURES = (
    [f"kwh_lag_{lag}" for lag in LAG_HOURS]
    + [f"kwh_roll_mean_{w}" for w in ROLL_WINDOWS_HOURS]
    + [f"kwh_roll_std_{w}" for w in ROLL_WINDOWS_HOURS]
)
FEATURE_COLUMNS = CALENDAR_FEATURES + WEATHER_FEATURES + HISTORY_FEATURES


def clean_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Put every building on a complete hourly grid and repair short gaps.

    Steps and reasons:

    1. Reindex to a full hourly calendar so every hour exists (a missing row is
       the same as a missing reading, but explicit rows make lag maths safe).
    2. Negative electricity is physically impossible -> treat as missing.
    3. Interpolate gaps of at most ``MAX_INTERP_GAP_HOURS`` linearly: a 1-3 hour
       comms hiccup carries no information, and dropping those rows would punch
       holes in the lags for the following day. Longer gaps stay NaN and are
       dropped later, because inventing a day of readings would be fabrication.
    4. Keep flags (``was_missing``, ``was_imputed``) so the dashboard and the
       limitations section can report how much of the data is repaired.

    Returns the cleaned panel and a per-building data-quality table.
    """
    panel = panel.sort_values(["building_id", "timestamp"]).copy()
    full_index = pd.date_range(
        panel["timestamp"].min(), panel["timestamp"].max(), freq="h", name="timestamp"
    )

    cleaned_frames: list[pd.DataFrame] = []
    quality_rows: list[dict] = []

    for building_id, group in panel.groupby("building_id", sort=True):
        g = group.set_index("timestamp").sort_index().reindex(full_index)
        g["building_id"] = building_id

        # Static attributes and weather arrive pre-merged; forward-fill them
        # across grid cells that only exist because of reindexing.
        for col in ("sqm", "primaryspaceusage", "sub_primaryspaceusage"):
            if col in g.columns:
                g[col] = g[col].ffill().bfill()

        raw = g["electricity"].astype("float64")
        raw = raw.mask(raw < 0)                      # impossible values -> missing
        g["was_missing"] = raw.isna()

        repaired = raw.interpolate(
            method="time", limit=MAX_INTERP_GAP_HOURS, limit_area="inside"
        )
        g["was_imputed"] = raw.isna() & repaired.notna()
        g["electricity"] = repaired

        # Weather: keep NaN (LightGBM handles missing natively, Ridge imputes)
        # but smooth temperature over the past day for thermal inertia.
        g["temp_roll_24"] = (
            g["airTemperature"].rolling(24, min_periods=12).mean()
        )

        kept = int(g["electricity"].notna().sum())
        quality_rows.append(
            {
                "building_id": building_id,
                "hours_total": len(g),
                "hours_raw_present": int((~g["was_missing"]).sum()),
                "hours_imputed": int(g["was_imputed"].sum()),
                "hours_still_missing": int(g["electricity"].isna().sum()),
                "pct_complete_after_repair": round(100 * kept / len(g), 3),
                "zero_hours": int((g["electricity"] == 0).sum()),
            }
        )
        cleaned_frames.append(g.reset_index())

    cleaned = pd.concat(cleaned_frames, ignore_index=True)
    return cleaned, pd.DataFrame(quality_rows)


def add_calendar_features(
    df: pd.DataFrame,
    country: str = HOLIDAY_COUNTRY,
    years: list[int] = HOLIDAY_YEARS,
) -> pd.DataFrame:
    """Add time-of-day, weekday, seasonal and holiday features.

    Cyclical sin/cos encoding is used for hour and day-of-year because 23:00 and
    00:00 are one hour apart but 23 apart as raw integers; sin/cos keeps that
    neighbour relationship, which a linear model can exploit and trees can split
    on cleanly.
    """
    out = df.copy()
    ts = out["timestamp"]
    out["hour"] = ts.dt.hour
    out["dayofweek"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype("int8")

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    # 365.25 keeps leap years from shifting the phase.
    out["doy_sin"] = np.sin(2 * np.pi * ts.dt.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * ts.dt.dayofyear / 365.25)

    public_holidays = holidays.country_holidays(country, years=years)
    out["is_holiday"] = ts.dt.date.isin(set(public_holidays)).astype("int8")
    return out


def add_history_features(
    df: pd.DataFrame,
    lags: tuple[int, ...] = LAG_HOURS,
    windows: tuple[int, ...] = ROLL_WINDOWS_HOURS,
    min_lag: int = MIN_LAG_HOURS,
) -> pd.DataFrame:
    """Add lagged and rolling statistics of consumption, all >= ``min_lag`` old.

    Two details matter for correctness:

    * Lags are taken per building (``groupby``), never across buildings.
    * Rolling windows are computed on the series **shifted by ``min_lag``**, so
      the window ending at t-24 h contains only hours <= t-24. Shifting *after*
      the rolling mean would silently include hours newer than the lag floor and
      would let the model see the anomaly it is supposed to detect.
    """
    out = df.sort_values(["building_id", "timestamp"]).copy()
    grouped = out.groupby("building_id", sort=False)

    for lag in lags:
        out[f"kwh_lag_{lag}"] = grouped["electricity"].shift(lag)

    # shift first, then roll: guarantees every value in the window is >= min_lag old
    shifted = grouped["electricity"].shift(min_lag)
    for window in windows:
        roll = shifted.groupby(out["building_id"], sort=False).rolling(
            window, min_periods=max(6, window // 4)
        )
        out[f"kwh_roll_mean_{window}"] = roll.mean().reset_index(level=0, drop=True)
        out[f"kwh_roll_std_{window}"] = roll.std().reset_index(level=0, drop=True)

    return out


def build_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full cleaning + feature pipeline. Returns (features, quality report)."""
    clean, quality = clean_panel(panel)
    featured = add_calendar_features(clean)
    featured = add_history_features(featured)
    featured = featured.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    return featured, quality


def rebuild_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute lag/rolling features after the meter series was modified.

    Used by the synthetic-anomaly evaluation: if we secretly change an hour's
    consumption, a live system would only see that change tomorrow. Rebuilding
    the lags reproduces exactly that, and proves the >= 24 h rule holds.
    """
    stale = [c for c in frame.columns if c.startswith("kwh_")]
    return add_history_features(frame.drop(columns=stale, errors="ignore"))


def modelling_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that cannot be trained or scored (missing target or lags).

    The first 168 h of each series have no weekly lag, and long meter gaps leave
    NaN targets. Dropping is honest here: we know the feature vector is
    incomplete, so predicting from it would mean inventing inputs.
    """
    needed = ["electricity", *FEATURE_COLUMNS]
    return df.dropna(subset=needed).reset_index(drop=True)

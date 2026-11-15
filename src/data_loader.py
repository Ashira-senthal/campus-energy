"""
Load the Building Data Genome Project 2 (BDG2) files into tidy DataFrames.

Three raw files are used (all downloaded into ``data/raw/``):

* ``electricity_cleaned.csv`` -- WIDE format: one row per hour, one column per
  building, values in kWh (empty cells = missing reading).
* ``weather.csv`` -- LONG format: one row per hour per site.
* ``metadata.csv`` -- one row per building (area, usage type, site, ...).

Everything here only *loads and reshapes*. Cleaning rules and imputation live in
``src/features.py`` / the Phase 3 notebook so that each concern is testable on
its own.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Paths are derived from this file's location, never hard-coded, so the project
# runs the same on Windows, macOS and Linux.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ELECTRICITY_PATH = RAW_DIR / "electricity_cleaned.csv"
WEATHER_PATH = RAW_DIR / "weather.csv"
METADATA_PATH = RAW_DIR / "metadata.csv"

# Weather columns we actually model with (others are dropped to keep it light).
WEATHER_COLUMNS = [
    "airTemperature",
    "dewTemperature",
    "cloudCoverage",
    "windSpeed",
    "precipDepth1HR",
]

# Metadata fields we keep for modelling and for the dashboard.
METADATA_COLUMNS = [
    "building_id",
    "site_id",
    "primaryspaceusage",
    "sub_primaryspaceusage",
    "sqm",
    "timezone",
    "yearbuilt",
    "numberoffloors",
    "occupants",
]


def load_metadata(path: Path | str = METADATA_PATH) -> pd.DataFrame:
    """Read building metadata, keeping only the columns the project uses."""
    meta = pd.read_csv(path)
    missing = [c for c in METADATA_COLUMNS if c not in meta.columns]
    if missing:
        raise ValueError(f"metadata is missing expected columns: {missing}")
    return meta[METADATA_COLUMNS].copy()


def load_electricity_header(path: Path | str = ELECTRICITY_PATH) -> list[str]:
    """Return the building ids (i.e. all CSV columns except ``timestamp``).

    Reading only the header is cheap and lets us intersect metadata with the
    meter file before loading 167 MB of values.
    """
    header = pd.read_csv(path, nrows=0)
    return [c for c in header.columns if c != "timestamp"]


def load_electricity(
    building_ids: list[str],
    path: Path | str = ELECTRICITY_PATH,
) -> pd.DataFrame:
    """Load selected buildings and return tidy long format.

    Returns columns ``timestamp``, ``building_id``, ``electricity`` (kWh).
    Only the requested columns are parsed, which keeps memory small.
    """
    available = set(load_electricity_header(path))
    ids = [b for b in building_ids if b in available]
    unknown = sorted(set(building_ids) - available)
    if unknown:
        raise ValueError(f"building ids not found in meter file: {unknown}")

    wide = pd.read_csv(
        path,
        usecols=["timestamp", *ids],
        parse_dates=["timestamp"],
    )
    long = wide.melt(
        id_vars="timestamp",
        value_vars=ids,
        var_name="building_id",
        value_name="electricity",
    )
    # Sort so every downstream shift/rolling operation is time-ordered.
    return long.sort_values(["building_id", "timestamp"]).reset_index(drop=True)


def load_weather(
    site_id: str,
    path: Path | str = WEATHER_PATH,
) -> pd.DataFrame:
    """Load hourly weather for one site, keeping the modelled columns."""
    weather = pd.read_csv(
        path,
        usecols=["timestamp", "site_id", *WEATHER_COLUMNS],
        parse_dates=["timestamp"],
    )
    weather = weather[weather["site_id"] == site_id].copy()
    weather = weather.drop(columns="site_id")
    weather = weather.drop_duplicates(subset="timestamp").sort_values("timestamp")
    return weather.reset_index(drop=True)


def build_panel(
    building_ids: list[str],
    site_id: str,
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join meters + weather + building attributes into one modelling table.

    The result is the single input every later step (features, models,
    anomalies) works from, so column names are stable and documented here.
    """
    meta = load_metadata() if metadata is None else metadata
    meta = meta[meta["building_id"].isin(building_ids)]

    panel = load_electricity(building_ids)
    weather = load_weather(site_id)
    panel = panel.merge(weather, on="timestamp", how="left")
    panel = panel.merge(
        meta[["building_id", "sqm", "primaryspaceusage", "sub_primaryspaceusage"]],
        on="building_id",
        how="left",
    )
    return panel.sort_values(["building_id", "timestamp"]).reset_index(drop=True)


def coverage_report(
    building_ids: list[str],
    path: Path | str = ELECTRICITY_PATH,
) -> pd.DataFrame:
    """Data-quality summary per building: present hours, zeros, mean load.

    Used to justify building selection: a building with 40% missing readings
    cannot support hourly forecasting, however interesting its usage type is.
    """
    available = set(load_electricity_header(path))
    ids = [b for b in building_ids if b in available]
    frame = pd.read_csv(
        path,
        usecols=["timestamp", *ids],
        parse_dates=["timestamp"],
    )
    total = len(frame)
    rows = []
    for building in ids:
        series = frame[building]
        present = series.notna()
        rows.append(
            {
                "building_id": building,
                "hours_present": int(present.sum()),
                "pct_present": round(100 * present.mean(), 2),
                "pct_zero": round(100 * (series.fillna(0) == 0).mean(), 2),
                "mean_kwh": round(float(series.mean()), 2) if present.any() else float("nan"),
                "first_timestamp": frame.loc[present, "timestamp"].min(),
                "last_timestamp": frame.loc[present, "timestamp"].max(),
            }
        )
    return pd.DataFrame(rows).sort_values("pct_present", ascending=False)

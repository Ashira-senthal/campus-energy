"""
Project constants: one place to change buildings, dates, thresholds and money.

Nothing in this file is computed; it is pure configuration so that the numbers
used in the viva report can be traced to a single source. Paths are relative
(resolved from this file), so the project runs on any machine.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# --------------------------------------------------------------------------
# Data selection (justified in PROJECT_GUIDE.md section 4)
# --------------------------------------------------------------------------
# BDG2 site with the most well-covered education buildings across distinct
# usage types (56 usable buildings, 5 space usages) -- see reports/building_coverage.csv.
SITE_ID = "Fox"

# Four buildings, deliberately different usage types and very different sizes,
# so model comparison and CV(RMSE) can be read across scales.
BUILDINGS: dict[str, str] = {
    "Fox_education_Andre": "College Laboratory",      # large lab, ~455 kWh mean
    "Fox_education_Elvira": "Student Union",          # ~189 kWh mean
    "Fox_education_Claire": "College Classroom",      # ~61 kWh mean
    "Fox_education_Kendrick": "Auditorium",           # ~12 kWh mean (baseload dominated)
}

# Short labels for charts and tables.
BUILDING_LABELS: dict[str, str] = {
    "Fox_education_Andre": "Andre (Laboratory)",
    "Fox_education_Elvira": "Elvira (Student Union)",
    "Fox_education_Claire": "Claire (Classroom)",
    "Fox_education_Kendrick": "Kendrick (Auditorium)",
}

# Holidays follow the site's country (Fox metadata timezone is US/*), because
# applying Indian holidays to a US campus would be wrong. Change this to "IN"
# if you later run the pipeline on your own campus meters.
HOLIDAY_COUNTRY = "US"
HOLIDAY_YEARS = [2016, 2017]

# --------------------------------------------------------------------------
# Chronological split: train earliest -> validate -> test latest.
# NO random shuffling anywhere: that would leak the future into the past.
# --------------------------------------------------------------------------
SPLITS: dict[str, tuple[str, str]] = {
    "train": ("2016-01-01 00:00", "2017-04-30 23:00"),
    "val": ("2017-05-01 00:00", "2017-08-31 23:00"),
    "test": ("2017-09-01 00:00", "2017-12-31 23:00"),
}

RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
# Minimum lag allowed for any autoregressive feature. 24 h = one day, so the
# model never sees the hour it is trying to explain -- otherwise an anomaly
# would be "learned" as normal input and cancel itself out in the residual.
MIN_LAG_HOURS = 24
LAG_HOURS = (24, 48, 168)              # yesterday, day-before, same hour last week
ROLL_WINDOWS_HOURS = (24, 168)         # short and weekly history windows
MAX_INTERP_GAP_HOURS = 3               # interpolate only short meter gaps

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
MODEL_NAMES = ["snaive_24", "snaive_168", "ridge", "random_forest", "lightgbm"]
MODEL_LABELS = {
    "snaive_24": "Seasonal naive (t-24h)",
    "snaive_168": "Seasonal naive (t-168h)",
    "ridge": "Ridge regression",
    "random_forest": "Random forest",
    "lightgbm": "LightGBM",
}
# The model whose residuals drive anomaly detection and SHAP explanations.
PRIMARY_MODEL = "lightgbm"

# --------------------------------------------------------------------------
# Anomaly detection
# --------------------------------------------------------------------------
# Selection rule fixed BEFORE looking at test-set detection scores: take the most
# sensitive (lowest) threshold whose false-alarm rate on the validation fold stays
# at or below 2%. In the Phase 6 sweep z=3.0 flags 1.69% of validation hours and
# is therefore the default. The dashboard slider exposes 2-6.
Z_THRESHOLD = 3.0            # robust z on residuals; dashboard slider changes this
PERSIST_K = 2                # require K flagged hours ...
PERSIST_N = 3                # ... out of the last N hours
IF_VALID_QUANTILE = 99.0     # Isolation Forest threshold = this percentile of validation scores

# Adaptive calibration: trailing window used to recalibrate the robust z-score.
# 720 h = 30 days, long enough to be stable, short enough to follow a season.
ADAPTIVE_WINDOW_HOURS = 720
ADAPTIVE_MIN_PERIODS = 168

# --------------------------------------------------------------------------
# Synthetic anomaly injection (labels for evaluating detectors)
# --------------------------------------------------------------------------
SYNTHETIC_SEED = 2024
N_EVENTS_PER_TYPE = 6        # events per anomaly type per building
SYNTHETIC_SETTINGS = {
    "spike":        {"hours": (1, 2),   "scale": (2.0, 3.5)},
    "after_hours":  {"hours": (6, 10),  "scale": (0.4, 0.9)},    "flatline":     {"hours": (8, 24), "scale": None},
    "drift":        {"hours": (48, 72), "scale": (0.15, 0.30)},
}

# --------------------------------------------------------------------------
# Money and emissions (ASSUMPTIONS -- verify against your campus bill and the
# current CEA grid emission factor; both are editable in the dashboard).
# --------------------------------------------------------------------------
TARIFF_INR_PER_KWH = 8.0
CO2_KG_PER_KWH = 0.71

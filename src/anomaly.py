"""
Anomaly detection: robust residual z-scores, Isolation Forest, and synthetic
anomaly injection used to create ground truth for evaluation.

Why residuals: a supervised forecaster already models what is normal given hour,
weekday, season and weather. Actual minus expected is therefore a direct,
interpretable measure of surprise, in kWh.

Why robust (median/MAD) statistics: the thing we are hunting is the outlier.
If we scaled residuals by their mean and standard deviation, a few large
anomalies would inflate the yardstick and hide themselves. Median and MAD have a
breakdown point of ~50%, so they survive contaminated data.

Why MAD is scaled by 1.4826 (0.6745 in the inverted form used here): for
normally distributed data MAD is ~0.6745 sigma, so dividing by (1.4826 * MAD)
makes the robust z comparable to an ordinary z-score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.config import (
    ADAPTIVE_MIN_PERIODS,
    ADAPTIVE_WINDOW_HOURS,
    IF_VALID_QUANTILE,
    N_EVENTS_PER_TYPE,
    PERSIST_K,
    PERSIST_N,
    SYNTHETIC_SEED,
    SYNTHETIC_SETTINGS,
    Z_THRESHOLD,
)

MAD_TO_SIGMA = 1.4826


# --------------------------------------------------------------------------
# Residual statistics
# --------------------------------------------------------------------------
def robust_location_scale(residuals: np.ndarray | pd.Series) -> tuple[float, float]:
    """Return (median, scaled MAD) of the residuals.

    The scale is floored at a tiny positive number so a perfectly constant
    validation residual cannot cause a divide-by-zero.
    """
    values = np.asarray(residuals, dtype="float64")
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) * MAD_TO_SIGMA
    if mad < 1e-6:
        # Fall back to a small fraction of the mean load rather than exploding.
        mad = float(np.std(values)) or 1e-3
    return median, mad


def robust_z(
    residuals: np.ndarray | pd.Series,
    median: float,
    scale: float,
) -> np.ndarray:
    """Robust z-score ``(x - median) / (1.4826 * MAD)``."""
    return (np.asarray(residuals, dtype="float64") - median) / scale


def fit_residual_thresholds(val_residuals: np.ndarray | pd.Series) -> dict[str, float]:
    """Learn residual centre and spread from the VALIDATION fold only.

    The test period never contributes to these numbers, so reported test
    performance is not tuned on the data it is scored on.
    """
    median, scale = robust_location_scale(val_residuals)
    return {"median": median, "scale": scale}


def flag_residuals(
    residuals: np.ndarray | pd.Series,
    median: float,
    scale: float,
    z_threshold: float = Z_THRESHOLD,
) -> np.ndarray:
    """Flag every hour whose |robust z| exceeds the threshold."""
    z = robust_z(residuals, median, scale)
    return np.abs(z) > z_threshold


def rolling_robust_z(
    residuals: pd.Series | np.ndarray,
    window: int = ADAPTIVE_WINDOW_HOURS,
    min_periods: int = ADAPTIVE_MIN_PERIODS,
    lag: int = 24,
) -> pd.Series:
    """Robust z-score against a TRAILING median/MAD instead of a frozen one.

    Why this exists: a single median/MAD learned in one season becomes the wrong
    yardstick a few months later (occupancy, term dates and weather all move the
    residual distribution). Recalibrating on the last ``window`` hours fixes that
    without any future information: everything is shifted by ``lag`` = 24 h first,
    so hour t is only ever compared with residuals the system already observed.

    Trade-off to be honest about: a *sustained* anomaly slowly pulls the trailing
    median toward itself, so adaptive thresholds catch fewer persistent faults
    than a frozen threshold does. It buys precision, not recall.
    """
    series = pd.Series(np.asarray(residuals, dtype="float64"))
    past = series.shift(lag)
    centre = past.rolling(window, min_periods=min_periods).median()
    spread = (
        (past - centre).abs().rolling(window, min_periods=min_periods).median()
        * MAD_TO_SIGMA
    )
    # Warm-up hours reuse the earliest available estimate rather than flagging.
    centre = centre.ffill().bfill()
    spread = spread.replace(0, np.nan).ffill().bfill().fillna(1.0)
    return (series - centre) / spread


def apply_persistence(
    flags: np.ndarray,
    k: int = PERSIST_K,
    n: int = PERSIST_N,
) -> np.ndarray:
    """Keep a flag only if >= k-1 of the previous n-1 hours were also flagged.

    Single noisy hours are the main source of false alarms in hourly meters;
    real problems (plant left running, failing equipment) persist. With k=2, n=3
    an isolated one-hour spike is suppressed but a two-or-more hour event is not.
    """
    flags = np.asarray(flags, dtype=bool)
    if len(flags) == 0:
        return flags

    flag_series = pd.Series(flags.astype("int8"))
    previous = flag_series.shift(1).rolling(n - 1, min_periods=1).sum().to_numpy()
    return flags & (previous >= (k - 1))


def flatline_flags(values: np.ndarray | pd.Series, window: int = 4) -> np.ndarray:
    """Flag hours where the meter reading has not moved for ``window`` hours.

    A frozen reading is physically implausible for a real building and is the
    signature of a communications dropout. Residual-based detection cannot see
    it -- the expected value also falls toward the frozen level -- so it needs a
    variance test of its own.

    ``window=4`` is a compromise: long enough that genuine low-variation night
    hours do not trip it, short enough that a flatline is caught within 4 hours
    rather than only after 6. Detection necessarily lags the fault's first hours.
    """
    series = pd.Series(np.asarray(values, dtype="float64"))
    rolling_std = series.rolling(window, min_periods=window).std(ddof=0)
    return ((rolling_std == 0) & (series > 0)).to_numpy()


# --------------------------------------------------------------------------
# Isolation Forest
# --------------------------------------------------------------------------
IF_FEATURES = [
    "residual",
    "load_ratio",
    "residual_std_24",
    "hour_sin",
    "hour_cos",
    "is_weekend",
    "airTemperature",
]


def isolation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the multivariate matrix Isolation Forest sees.

    Deliberately includes *context* (hour, weekend, temperature) as well as the
    residual, so the forest can learn that, say, a 40 kWh residual at 3 a.m. in
    July is stranger than the same residual at 3 p.m. in term time.
    """
    frame = pd.DataFrame(index=df.index)
    frame["residual"] = df["residual"]
    # Ratio is scale-free, which lets one forest be fitted per building safely.
    frame["load_ratio"] = df["electricity"] / df["expected"].replace(0, np.nan)
    frame["residual_std_24"] = (
        pd.Series(df["residual"].to_numpy()).shift(1).rolling(24, min_periods=6).std()
    )
    frame["hour_sin"] = df["hour_sin"]
    frame["hour_cos"] = df["hour_cos"]
    frame["is_weekend"] = df["is_weekend"]
    frame["airTemperature"] = df["airTemperature"]
    return frame


@dataclass
class IsolationDetector:
    """Fitted scaler + forest + the validation-derived score threshold."""

    scaler: StandardScaler
    model: IsolationForest
    threshold: float

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Higher = more normal (sklearn's ``decision_function`` convention)."""
        matrix = isolation_features(df)
        # Missing inputs are filled with the training median rather than dropped,
        # so an hour is never silently skipped.
        matrix = matrix.fillna(matrix.median(numeric_only=True)).fillna(0.0)
        return self.model.decision_function(self.scaler.transform(matrix))


def fit_isolation_forest(
    val_df: pd.DataFrame,
    seed: int = SYNTHETIC_SEED,
    quantile: float = IF_VALID_QUANTILE,
) -> IsolationDetector:
    """Fit on validation residuals and learn the alert threshold there.

    ``quantile=99`` means "flag the most extreme 1% of validation hours": the
    contamination level is set by observed normal behaviour instead of the
    sklearn default, which keeps it comparable with the z-score method.
    """
    matrix = isolation_features(val_df)
    matrix = matrix.fillna(matrix.median(numeric_only=True)).fillna(0.0)

    scaler = StandardScaler().fit(matrix)
    model = IsolationForest(
        n_estimators=300,
        max_samples="auto",
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    ).fit(scaler.transform(matrix))

    val_scores = model.decision_function(scaler.transform(matrix))
    threshold = float(np.quantile(val_scores, (100.0 - quantile) / 100.0))
    return IsolationDetector(scaler=scaler, model=model, threshold=threshold)


def flag_isolation(detector: IsolationDetector, df: pd.DataFrame) -> np.ndarray:
    return detector.score(df) < detector.threshold


# --------------------------------------------------------------------------
# Synthetic anomaly injection
# --------------------------------------------------------------------------
def inject_anomalies(
    df: pd.DataFrame,
    building_id: str,
    n_events: int = N_EVENTS_PER_TYPE,
    settings: dict = SYNTHETIC_SETTINGS,
    seed: int = SYNTHETIC_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create known faults in a copy of one building's test series.

    Four types, chosen because they cover the realistic failure modes and they
    stress different detectors:

    * ``spike``       -- short multiplicative surge (faulty sensor, switching).
    * ``after_hours`` -- load added at night / weekend (plant or lights left on).
    * ``flatline``    -- frozen reading for hours (comms dropout, meter fault).
    * ``drift``       -- slow additive ramp over days (calibration / failing unit).

    Windows never overlap, so each label is unambiguous. Returns the modified
    frame and a label table (one row per event).
    """
    frame = df.copy().reset_index(drop=True)
    values = frame["electricity"].to_numpy(dtype="float64").copy()
    n = len(values)
    if n < 24 * 30:
        return frame, pd.DataFrame(columns=["building_id", "anomaly_type", "start", "end", "magnitude"])

    rng = np.random.default_rng(seed)
    median_load = float(np.nanmedian(values)) or 1.0
    occupied = np.zeros(n, dtype=bool)
    events: list[dict] = []

    def free(start: int, length: int) -> bool:
        end = start + length
        if end >= n:
            return False
        # 24 h guard band so one event's after-effects do not overlap another's.
        return not occupied[max(0, start - 24) : min(n, end + 24)].any()

    for anomaly_type, cfg in settings.items():
        for _ in range(n_events):
            length = int(rng.integers(cfg["hours"][0], cfg["hours"][1] + 1))
            for _attempt in range(300):
                start = int(rng.integers(0, max(1, n - length - 1)))

                if anomaly_type == "after_hours":
                    hour = int(frame.loc[start, "hour"])
                    weekend = bool(frame.loc[start, "is_weekend"])
                    if not (hour >= 22 or hour <= 5 or weekend):
                        continue

                if not free(start, length):
                    continue

                idx = np.arange(start, start + length)
                if anomaly_type == "spike":
                    magnitude = float(rng.uniform(*cfg["scale"]))
                    values[idx] *= magnitude
                elif anomaly_type == "after_hours":
                    magnitude = float(rng.uniform(*cfg["scale"]))
                    values[idx] += magnitude * median_load
                elif anomaly_type == "flatline":
                    magnitude = 1.0
                    values[idx] = values[start]   # frozen at the pre-fault reading
                elif anomaly_type == "drift":
                    magnitude = float(rng.uniform(*cfg["scale"]))
                    values[idx] += np.linspace(0, magnitude * median_load, length)
                else:  # pragma: no cover - configuration error
                    raise ValueError(f"unknown anomaly type: {anomaly_type}")

                occupied[idx] = True
                events.append(
                    {
                        "building_id": building_id,
                        "anomaly_type": anomaly_type,
                        "start": frame.loc[start, "timestamp"],
                        "end": frame.loc[start + length - 1, "timestamp"],
                        "magnitude": round(magnitude, 3),
                    }
                )
                break

    frame["electricity"] = values
    labels = pd.DataFrame(events)
    if not labels.empty:
        labels = labels.sort_values("start").reset_index(drop=True)
        labels["event_id"] = [f"{building_id}-{i:02d}" for i in range(len(labels))]
    return frame, labels


def label_hours(labels: pd.DataFrame, timestamps: pd.Series) -> np.ndarray:
    """Boolean hour-level ground truth derived from the event table."""
    truth = np.zeros(len(timestamps), dtype=bool)
    if labels.empty:
        return truth
    ts = pd.to_datetime(timestamps).to_numpy()
    for _, row in labels.iterrows():
        truth |= (ts >= np.datetime64(row["start"])) & (ts <= np.datetime64(row["end"]))
    return truth

"""
Interpretability: SHAP attributions, plain-English anomaly tagging, and the
money/emissions consequence of each flagged hour.

Two ideas:

1. **SHAP** turns one prediction into per-feature contributions that sum to it
   (base value + contributions = prediction). That answers both "what matters
   overall?" (mean |SHAP|) and "why was THIS hour flagged?" (one row of SHAP).

2. **Tagging** converts numbers into a sentence a facility manager can act on.
   The tag is decided per *episode* (consecutive flagged hours) because "3.0x
   load for 7 hours" is a different problem from "one 1-hour surge".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from src.anomaly import flatline_flags

# Severity bands on the robust z-score. Deliberately coarse: three labels are
# enough for a human to prioritise, and finer bands imply precision we do not
# have.
SEVERITY_HIGH_Z = 6.0
SEVERITY_MEDIUM_Z = 4.5


def shap_explanation(model, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Return (shap values [n_rows, n_features], base value) for a tree model.

    TreeExplainer gives exact Shapley values for tree ensembles and runs in
    milliseconds, so there is no need for sampling here.
    """
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)
    values = np.asarray(explanation.values, dtype="float64")
    base = float(np.ravel(explanation.base_values)[0])
    return values, base


def global_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Mean absolute SHAP per feature: the model's global ranking."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    frame = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    frame = frame.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    frame["rank"] = frame.index + 1
    return frame


def local_contributions(
    shap_row: np.ndarray,
    feature_values: pd.Series,
    feature_names: list[str],
    top_k: int = 6,
) -> pd.DataFrame:
    """Largest push-toward / push-away contributions for a single hour."""
    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "value": [feature_values[f] for f in feature_names],
            "shap": shap_row,
        }
    )
    frame["abs_shap"] = frame["shap"].abs()
    return (
        frame.sort_values("abs_shap", ascending=False)
        .head(top_k)
        .drop(columns="abs_shap")
        .reset_index(drop=True)
    )


def _severity(z_value: float) -> str:
    abs_z = abs(z_value)
    if abs_z >= SEVERITY_HIGH_Z:
        return "high"
    if abs_z >= SEVERITY_MEDIUM_Z:
        return "medium"
    return "low"


def _describe(
    anomaly_type: str,
    hours: int,
    actual: float,
    expected: float,
    hour: int,
    weekend: bool,
    ratio: float,
) -> str:
    """One plain-English sentence per episode, with the numbers that matter."""
    when = "weekend" if weekend else "weekday"
    excess = max(actual - expected, 0.0)
    if anomaly_type == "meter fault (flatline)":
        return (
            f"Reading frozen at {actual:,.1f} kWh for {hours} h - identical values "
            "hour after hour are physically implausible; likely a meter or "
            "communications fault, so treat the reading as missing."
        )
    if anomaly_type == "short spike":
        return (
            f"One-off surge to {ratio:.1f}x the {expected:,.1f} kWh expected at "
            f"{hour:02d}:00 ({when}); short enough to be a switching or sensor "
            "event rather than a behavioural change."
        )
    if anomaly_type == "slow drift":
        return (
            f"Consumption drifted {excess:,.1f} kWh above expectation over "
            f"{hours} h ({ratio:.1f}x); a gradual rise points to calibration "
            "drift or a degrading unit rather than a schedule change."
        )
    if anomaly_type == "after-hours load":
        return (
            f"Using {actual:,.1f} kWh at {hour:02d}:00 on a {when} versus "
            f"{expected:,.1f} kWh expected - load that should not be there "
            "after hours. Check plant schedules, lighting and HVAC setbacks."
        )
    if anomaly_type == "sustained high load":
        return (
            f"Runs {excess:,.1f} kWh/h above expectation for {hours} h "
            f"({ratio:.1f}x) - a sustained excess, so suspect a schedule "
            "override, a stuck damper/valve or extra equipment running."
        )
    if anomaly_type == "unexpected drop":
        return (
            f"Dropped to {actual:,.1f} kWh against {expected:,.1f} kWh expected "
            f"for {hours} h - possible outage, partial shutdown or meter issue; "
            "confirm whether the building was actually occupied."
        )
    return (
        f"{actual:,.1f} kWh against {expected:,.1f} kWh expected for {hours} h "
        f"({ratio:.1f}x) - unusual for this hour and weather."
    )


def tag_flagged_hours(
    df: pd.DataFrame,
    flags: np.ndarray,
    z_values: np.ndarray | None = None,
) -> pd.DataFrame:
    """Turn hour-level flags into a readable anomaly table.

    Episodes (consecutive flagged hours) are typed by their shape and context,
    then each hour inherits the episode's type, reason and severity. Returns one
    row per flagged hour so the dashboard can plot and filter it directly.
    """
    frame = df.reset_index(drop=True).copy()
    frame["flagged"] = np.asarray(flags, dtype=bool)
    frame["expected"] = frame["expected"].astype("float64")
    frame["residual"] = frame["electricity"] - frame["expected"]
    frame["excess_kwh"] = frame["residual"].clip(lower=0.0)
    frame["load_ratio"] = frame["electricity"] / frame["expected"].replace(0, np.nan)
    frame["z"] = (
        np.asarray(z_values, dtype="float64")
        if z_values is not None
        else np.zeros(len(frame))
    )
    frame["is_flat"] = flatline_flags(frame["electricity"])

    if not frame["flagged"].any():
        return frame.iloc[0:0].assign(
            anomaly_type=[], reason=[], severity=[], hours=[]
        )

    # Episode ids increment whenever the flag changes state.
    frame["episode"] = (frame["flagged"] != frame["flagged"].shift()).cumsum()

    tagged_episodes: list[tuple[dict, int]] = []
    for episode_id, group in frame[frame["flagged"]].groupby("episode"):
        hours = len(group)
        actual = float(group["electricity"].mean())
        expected = float(group["expected"].mean())
        ratio = float(np.nanmean(group["load_ratio"])) if hours else np.nan
        hour = int(group["timestamp"].iloc[0].hour)
        weekend = bool(group["is_weekend"].iloc[0])
        excess = actual - expected

        is_flat = bool(group["is_flat"].any())
        is_rising = False
        if hours >= 24:
            trend = np.polyfit(np.arange(hours), group["residual"].to_numpy(), 1)[0]
            is_rising = trend > 0

        if is_flat:
            anomaly_type = "meter fault (flatline)"
        elif hours >= 24 and is_rising:
            anomaly_type = "slow drift"
        elif hours <= 2 and excess > 0:
            anomaly_type = "short spike"
        elif (hour >= 22 or hour <= 5 or weekend) and excess > 0:
            anomaly_type = "after-hours load"
        elif hours >= 6 and excess > 0:
            anomaly_type = "sustained high load"
        elif excess < 0:
            anomaly_type = "unexpected drop"
        else:
            anomaly_type = "unusual load"

        reason = _describe(
            anomaly_type, hours, actual, expected, hour, weekend,
            ratio if np.isfinite(ratio) else 1.0,
        )
        peak_severity = _severity(float(group["z"].abs().max()))
        tagged_episodes.append(
            (
                {
                    "anomaly_type": anomaly_type,
                    "reason": reason,
                    "severity": peak_severity,
                    "hours": hours,
                },
                episode_id,
            )
        )

    tags = pd.DataFrame([t for t, _ in tagged_episodes])
    tags["episode"] = [e for _, e in tagged_episodes]

    table = frame[frame["flagged"]].merge(tags, on="episode", how="left")
    columns = [
        "timestamp",
        "building_id",
        "electricity",
        "expected",
        "residual",
        "excess_kwh",
        "z",
        "load_ratio",
        "severity",
        "anomaly_type",
        "reason",
        "hours",
        "hour",
        "is_weekend",
    ]
    return table[columns].reset_index(drop=True)


def estimate_impact(
    anomaly_table: pd.DataFrame,
    tariff_inr_per_kwh: float,
    co2_kg_per_kwh: float,
) -> dict[str, float]:
    """Energy, money and CO2 implied by the flagged hours.

    Only *positive* excess counts as waste: a negative residual (consumption
    below expectation) is not a saving we should bank -- it may itself be a
    fault -- so it is reported separately rather than netted off.
    """
    if anomaly_table.empty:
        return {
            "excess_kwh": 0.0,
            "deficit_kwh": 0.0,
            "cost_inr": 0.0,
            "co2_kg": 0.0,
            "flagged_hours": 0,
        }
    excess = float(anomaly_table["excess_kwh"].sum())
    deficit = float((-anomaly_table["residual"].clip(upper=0.0)).sum())
    return {
        "excess_kwh": round(excess, 1),
        "deficit_kwh": round(deficit, 1),
        "cost_inr": round(excess * tariff_inr_per_kwh, 0),
        "co2_kg": round(excess * co2_kg_per_kwh, 1),
        "flagged_hours": int(len(anomaly_table)),
    }

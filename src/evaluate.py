"""
Evaluation: forecast accuracy metrics and anomaly-detection precision/recall.

Forecast metrics (per building, per fold, per model):

* MAE  -- mean absolute error, kWh. Easy to explain to a facility manager.
* RMSE -- root mean squared error, kWh; punishes large misses, which is exactly
          what the anomaly detector consumes.
* CV(RMSE) -- RMSE / mean(actual), percent. Scale-free, so a 12 kWh auditorium
          and a 450 kWh laboratory can sit in one table (ASHRAE Guideline 14).
* NMBE -- mean(actual - predicted) / mean(actual), signed percent. Reveals the
          systematic bias that CV(RMSE) hides.
* sMAPE -- symmetric MAPE: percentage error that survives near-zero hours.

Detection metrics are hour-level: an hour is a positive if it was injected as an
anomaly or flagged by the detector.

Important subtlety: **precision is only meaningful overall, recall is what you
report per type.** If you isolate one anomaly type and keep normal hours, the
flags falling on *other* anomaly types count as false positives and precision
collapses for reasons that have nothing to do with that type. So we report:

* overall precision / recall / F1 per method, and
* per-type recall (of the hours of this type, how many were caught).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_denominator(values: np.ndarray) -> float:
    denom = float(np.mean(values))
    return denom if abs(denom) > 1e-9 else 1.0


def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the five headline metrics for one model on one fold."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.size == 0:
        return {k: np.nan for k in ("mae", "rmse", "cv_rmse_pct", "nmbe_pct", "smape_pct")}

    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    denom = _safe_denominator(y_true)
    cv_rmse = 100.0 * rmse / denom
    nmbe = 100.0 * float(np.mean(errors)) / denom

    # sMAPE guard: when both values are ~0 the percentage carries no information.
    scale = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    valid = scale > 1e-6
    smape = (
        100.0 * float(np.mean(np.abs(errors[valid]) / scale[valid]))
        if valid.any()
        else np.nan
    )

    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "cv_rmse_pct": round(cv_rmse, 2),
        "nmbe_pct": round(nmbe, 2),
        "smape_pct": round(smape, 2),
    }


def detection_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Hour-level confusion-matrix metrics for one detector over one set of hours."""
    predicted = np.asarray(predicted, dtype=bool)
    truth = np.asarray(truth, dtype=bool)

    tp = int(np.sum(predicted & truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    tn = int(np.sum(~predicted & ~truth))

    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
        else np.nan
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(float(precision), 3) if np.isfinite(precision) else np.nan,
        "recall": round(float(recall), 3) if np.isfinite(recall) else np.nan,
        "f1": round(float(f1), 3) if np.isfinite(f1) else np.nan,
        "false_alarm_rate_pct": round(100.0 * fp / max(fp + tn, 1), 2),
    }


def summarise_detection(
    detections: dict[str, np.ndarray],
    truth: np.ndarray,
    types: np.ndarray | None = None,
) -> pd.DataFrame:
    """Score every detector overall, and compute recall per anomaly type.

    ``types`` labels each hour with the injected anomaly type, using the string
    ``"normal"`` for untouched hours.
    """
    rows: list[dict] = []
    for method, flags in detections.items():
        flags = np.asarray(flags, dtype=bool)
        rows.append(
            {"method": method, "anomaly_type": "ALL", **detection_metrics(flags, truth)}
        )
        if types is None:
            continue
        types = np.asarray(types, dtype=object)
        for anomaly_type in pd.unique(types):
            if anomaly_type in (None, "", "normal"):
                continue
            mask = types == anomaly_type
            if not mask.any():
                continue
            # Recall only: of this type's hours, how many did we catch?
            tp = int(np.sum(flags & mask))
            fn = int(np.sum(~flags & mask))
            rows.append(
                {
                    "method": method,
                    "anomaly_type": anomaly_type,
                    "tp": tp,
                    "fp": np.nan,
                    "fn": fn,
                    "tn": np.nan,
                    "precision": np.nan,
                    "recall": round(tp / max(tp + fn, 1), 3),
                    "f1": np.nan,
                    "false_alarm_rate_pct": np.nan,
                }
            )
    return pd.DataFrame(rows)


def errors_by_hour(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    building_id: str,
    model: str,
) -> pd.DataFrame:
    """MAE and bias by hour of day -- where in the daily cycle does it fail?"""
    frame = pd.DataFrame(
        {
            "hour": df["hour"].to_numpy(),
            "error": df["electricity"].to_numpy() - np.asarray(y_pred, dtype="float64"),
        }
    )
    grouped = frame.groupby("hour")["error"].agg(
        mae=lambda s: float(np.mean(np.abs(s))),
        bias=lambda s: float(np.mean(s)),
        n="size",
    )
    grouped = grouped.reset_index()
    grouped.insert(0, "building_id", building_id)
    grouped.insert(1, "model", model)
    return grouped.round(3)

"""
End-to-end pipeline: load -> clean/features -> forecast -> detect -> explain.

Run from the project root:

    .venv/bin/python run_pipeline.py          # Windows: .venv\Scripts\python run_pipeline.py

Writes (all relative paths, nothing hard-coded):

    data/processed/*.parquet   small dashboard-ready tables
    models/*.joblib            fitted models and Isolation Forests
    reports/*.csv|json         metric tables used in PROJECT_GUIDE.md
    reports/figures/*.png      static EDA and SHAP figures

The pipeline is deterministic: fixed random seeds, fixed chronological splits.
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd

from src import eda
from src.anomaly import (
    apply_persistence,
    fit_isolation_forest,
    fit_residual_thresholds,
    flag_isolation,
    flag_residuals,
    flatline_flags,
    inject_anomalies,
    label_hours,
    robust_z,
    rolling_robust_z,
)
from src.config import (
    BUILDING_LABELS,
    BUILDINGS,
    CO2_KG_PER_KWH,
    MODELS_DIR,
    MODEL_LABELS,
    MODEL_NAMES,
    PERSIST_K,
    PERSIST_N,
    PRIMARY_MODEL,
    PROCESSED_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    SITE_ID,
    TARIFF_INR_PER_KWH,
    Z_THRESHOLD,
)
from src.data_loader import build_panel
from src.evaluate import (
    errors_by_hour,
    forecast_metrics,
    summarise_detection,
)
from src.explain import (
    estimate_impact,
    global_importance,
    shap_explanation,
    tag_flagged_hours,
)
from src.features import (
    FEATURE_COLUMNS,
    build_features,
    modelling_rows,
    rebuild_history_features,
)
from src.models import fit_and_predict_all, split_by_config

METHODS = [
    "residual",
    "residual+persistence",
    "isolation_forest",
    "hybrid",
    "adaptive_hybrid",
]


def ensure_directories() -> None:
    for path in (PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, REPORTS_DIR / "figures"):
        path.mkdir(parents=True, exist_ok=True)


def hourly_table(features: pd.DataFrame) -> pd.DataFrame:
    """Full two-year hourly table used by the dashboard's Patterns tab."""
    table = features[
        [
            "timestamp",
            "building_id",
            "electricity",
            "airTemperature",
            "hour",
            "dayofweek",
            "month",
            "is_weekend",
            "is_holiday",
        ]
    ].copy()
    table["building_label"] = table["building_id"].map(BUILDING_LABELS)
    return table


def detect(
    df: pd.DataFrame,
    thresholds: dict[str, float],
    detector,
    z_threshold: float = Z_THRESHOLD,
) -> dict[str, np.ndarray]:
    """Run every detector on one frame that has ``expected`` and ``residual``.

    ``adaptive_hybrid`` needs an ``adaptive_z`` column (see ``rolling_robust_z``)
    and falls back to the static hybrid when it is absent.
    """
    raw = flag_residuals(df["residual"], thresholds["median"], thresholds["scale"], z_threshold)
    persisted = apply_persistence(raw, PERSIST_K, PERSIST_N)
    flat = flatline_flags(df["electricity"])
    isolation = flag_isolation(detector, df)
    hybrid = persisted | flat

    if "adaptive_z" in df.columns:
        adaptive = apply_persistence(
            np.abs(df["adaptive_z"].to_numpy(dtype="float64")) > z_threshold, PERSIST_K, PERSIST_N
        ) | flat
    else:  # pragma: no cover - defensive, keeps the function total
        adaptive = hybrid

    return {
        "residual": raw,
        "residual+persistence": persisted,
        "isolation_forest": isolation,
        "hybrid": hybrid,
        "adaptive_hybrid": adaptive,
    }


def run_building(
    building_id: str,
    frame: pd.DataFrame,
    feature_rows: pd.DataFrame,
) -> dict:
    """Train, forecast, explain and stress-test one building."""
    print(f"  - {BUILDING_LABELS[building_id]}: fitting models ...", flush=True)
    folds = split_by_config(frame)

    fitted, predictions = fit_and_predict_all(
        folds["train"], {"val": folds["val"], "test": folds["test"]}
    )

    metrics_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    hour_error_rows: list[pd.DataFrame] = []
    for fold_name in ("val", "test"):
        fold = folds[fold_name]
        for model_name in MODEL_NAMES:
            pred = predictions[fold_name][model_name]
            metrics_rows.append(
                {
                    "building_id": building_id,
                    "building_label": BUILDING_LABELS[building_id],
                    "fold": fold_name,
                    "model": model_name,
                    **forecast_metrics(fold["electricity"], pred),
                }
            )
            if fold_name == "test":
                hour_error_rows.append(errors_by_hour(fold, pred, building_id, model_name))
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "timestamp": fold["timestamp"],
                        "building_id": building_id,
                        "fold": fold_name,
                        "model": model_name,
                        "actual": fold["electricity"].to_numpy(),
                        "expected": pred,
                    }
                )
            )

    # ---------------------------------------------------------------- residuals
    # The thresholds use the VALIDATION fold only, so the test period is scored
    # by a rule that never saw it.
    val = folds["val"].assign(expected=predictions["val"][PRIMARY_MODEL])
    val["residual"] = val["electricity"] - val["expected"]
    test = folds["test"].assign(expected=predictions["test"][PRIMARY_MODEL])
    test["residual"] = test["electricity"] - test["expected"]

    thresholds = fit_residual_thresholds(val["residual"])
    detector = fit_isolation_forest(val, seed=RANDOM_SEED)
    print(
        f"    residual centre={thresholds['median']:.2f} kWh, "
        f"scale={thresholds['scale']:.2f} kWh, IF cut={detector.threshold:.4f}"
    )

    # ------------------------------------------------------- SHAP (clean test)
    shap_values, shap_base = shap_explanation(
        fitted[PRIMARY_MODEL], test[FEATURE_COLUMNS]
    )
    importance = global_importance(shap_values, FEATURE_COLUMNS)
    importance.insert(0, "building_id", building_id)

    shap_long = pd.DataFrame(
        {
            "timestamp": np.repeat(test["timestamp"].to_numpy(), len(FEATURE_COLUMNS)),
            "building_id": building_id,
            "feature": np.tile(np.asarray(FEATURE_COLUMNS, dtype=object), len(test)),
            "shap": shap_values.reshape(-1),
            "value": test[FEATURE_COLUMNS].to_numpy().reshape(-1),
        }
    )
    eda.shap_summary(
        shap_values,
        test[FEATURE_COLUMNS],
        REPORTS_DIR / "figures",
        BUILDING_LABELS[building_id],
    )

    # Adaptive (trailing 30-day) z, warm-started from validation residuals so the
    # first test hours are not scored against an empty window.
    adaptive_sequence = pd.concat([val["residual"], test["residual"]], ignore_index=True)
    test["adaptive_z"] = rolling_robust_z(adaptive_sequence).iloc[len(val):].to_numpy()

    # ---------------------------------------------- anomalies on the REAL data
    clean_detections = detect(test, thresholds, detector)
    residual_z = robust_z(test["residual"], thresholds["median"], thresholds["scale"])
    clean_tags = tag_flagged_hours(
        test, clean_detections["hybrid"], residual_z
    )
    clean_tags["injected"] = False

    test_context = test[
        [
            "timestamp",
            "building_id",
            "electricity",
            "expected",
            "residual",
            "hour",
            "dayofweek",
            "is_weekend",
            "hour_sin",
            "hour_cos",
            "airTemperature",
            "adaptive_z",
        ]
    ].copy()
    test_context["z"] = residual_z
    test_context["if_score"] = detector.score(test)
    test_context["flat_flag"] = flatline_flags(test["electricity"])
    test_context["building_label"] = BUILDING_LABELS[building_id]

    # ------------------------------------ synthetic anomalies + honest scoring
    print("    injecting synthetic anomalies and scoring detectors ...", flush=True)
    injected_raw, labels = inject_anomalies(
        test.drop(columns=["expected", "residual"]), building_id
    )
    history = frame[frame["timestamp"] < test["timestamp"].min()].tail(200)
    rebuilt = rebuild_history_features(
        pd.concat([history, injected_raw], ignore_index=True)
    ).dropna(subset=[*FEATURE_COLUMNS, "electricity"])
    injected = rebuilt[rebuilt["timestamp"] >= test["timestamp"].min()].reset_index(drop=True)

    injected["expected"] = np.clip(
        fitted[PRIMARY_MODEL].predict(injected[FEATURE_COLUMNS]), 0.0, None
    )
    injected["residual"] = injected["electricity"] - injected["expected"]
    injected_sequence = pd.concat([val["residual"], injected["residual"]], ignore_index=True)
    injected["adaptive_z"] = rolling_robust_z(injected_sequence).iloc[len(val):].to_numpy()

    truth = label_hours(labels, injected["timestamp"])
    types = np.full(len(injected), "normal", dtype=object)
    for row in labels.itertuples():
        mask = ((injected["timestamp"] >= row.start) & (injected["timestamp"] <= row.end)).to_numpy()
        types[mask] = row.anomaly_type

    injected_detections = detect(injected, thresholds, detector)
    scored = summarise_detection(injected_detections, truth, types)
    scored.insert(0, "building_id", building_id)

    # Detector sensitivity sweep: how do precision/recall trade off with z?
    sensitivity_rows: list[dict] = []
    for z_value in (3.0, 3.5, 4.0, 4.5, 5.0, 6.0):
        swept = detect(injected, thresholds, detector, z_threshold=z_value)
        for method, flags in swept.items():
            scored_z = summarise_detection({method: flags}, truth)
            row = scored_z.iloc[0].to_dict()
            sensitivity_rows.append(
                {
                    "building_id": building_id,
                    "z_threshold": z_value,
                    **{k: row[k] for k in ("method", "precision", "recall", "f1", "false_alarm_rate_pct")},
                    "validation_false_alarm_pct": round(
                        100.0
                        * flag_residuals(
                            val["residual"], thresholds["median"], thresholds["scale"], z_value
                        ).mean(),
                        2,
                    ),
                }
            )

    joblib.dump(
        {name: fitted[name] for name in MODEL_NAMES}, MODELS_DIR / f"{building_id}_models.joblib"
    )
    joblib.dump(detector, MODELS_DIR / f"{building_id}_isolation.joblib")

    print(f"    done. events injected={len(labels)}, flagged clean hours={len(clean_tags)}")

    return {
        "building_id": building_id,
        "metrics": pd.DataFrame(metrics_rows),
        "predictions": pd.concat(prediction_rows, ignore_index=True),
        "hour_errors": pd.concat(hour_error_rows, ignore_index=True),
        "thresholds": {
            "building_id": building_id,
            "median": thresholds["median"],
            "scale": thresholds["scale"],
            "if_threshold": detector.threshold,
            "shap_base_value": shap_base,
            "mean_kwh": float(frame["electricity"].mean()),
            "sqm": float(frame["sqm"].iloc[0]),
            "usage": str(frame["sub_primaryspaceusage"].iloc[0]),
            "clean_flagged_hours": int(len(clean_tags)),
            "validation_false_alarm_pct": round(
                100.0
                * flag_residuals(
                    val["residual"], thresholds["median"], thresholds["scale"], Z_THRESHOLD
                ).mean(),
                2,
            ),
        },
        "shap_long": shap_long,
        "importance": importance,
        "test_context": test_context,
        "clean_tags": clean_tags,
        "events": labels,
        "scored": scored,
        "sensitivity": pd.DataFrame(sensitivity_rows),
    }


def main() -> None:
    started = time.time()
    ensure_directories()

    print("1/5 loading BDG2 data ...", flush=True)
    panel = build_panel(list(BUILDINGS), SITE_ID)
    print(f"    panel rows={len(panel):,} buildings={panel['building_id'].nunique()}")

    print("2/5 cleaning + feature engineering ...", flush=True)
    features, quality = build_features(panel)
    quality.to_csv(REPORTS_DIR / "building_quality.csv", index=False)
    frame = modelling_rows(features)
    print(f"    modelling rows={len(frame):,} features={len(FEATURE_COLUMNS)}")

    hourly = hourly_table(features)
    hourly.to_parquet(PROCESSED_DIR / "hourly.parquet", index=False)

    print("3/5 EDA figures ...", flush=True)
    for figure in eda.make_all(hourly, REPORTS_DIR / "figures"):
        print(f"    wrote {figure.relative_to(REPORTS_DIR.parent)}")

    print("4/5 per-building models, detectors, SHAP ...", flush=True)
    results = []
    for building_id in BUILDINGS:
        building_frame = frame[frame["building_id"] == building_id].reset_index(drop=True)
        results.append(run_building(building_id, building_frame, frame))

    print("5/5 aggregating results ...", flush=True)

    metrics = pd.concat([r["metrics"] for r in results], ignore_index=True)
    predictions = pd.concat([r["predictions"] for r in results], ignore_index=True)
    hour_errors = pd.concat([r["hour_errors"] for r in results], ignore_index=True)
    residuals = pd.concat([r["test_context"] for r in results], ignore_index=True)
    clean_tags = pd.concat([r["clean_tags"] for r in results], ignore_index=True)
    events = pd.concat([r["events"] for r in results], ignore_index=True)
    scored = pd.concat([r["scored"] for r in results], ignore_index=True)
    sensitivity = pd.concat([r["sensitivity"] for r in results], ignore_index=True)
    shap_long = pd.concat([r["shap_long"] for r in results], ignore_index=True)
    importance = pd.concat([r["importance"] for r in results], ignore_index=True)

    # Per-type rows carry recall only; aggregate those separately.
    overall = scored[scored["anomaly_type"] == "ALL"].groupby("method", as_index=False)[
        ["precision", "recall", "f1", "false_alarm_rate_pct"]
    ].mean(numeric_only=True).round(3)

    per_type = (
        scored[scored["anomaly_type"] != "ALL"]
        .groupby(["method", "anomaly_type"], as_index=False)["recall"]
        .mean()
        .round(3)
        .pivot(index="anomaly_type", columns="method", values="recall")
        .reset_index()
    )

    # ---------------- write dashboard inputs (small parquet) ----------------
    predictions.to_parquet(PROCESSED_DIR / "predictions.parquet", index=False)
    residuals.to_parquet(PROCESSED_DIR / "residuals.parquet", index=False)
    metrics.to_parquet(PROCESSED_DIR / "metrics.parquet", index=False)
    hour_errors.to_parquet(PROCESSED_DIR / "errors_by_hour.parquet", index=False)
    scored.to_parquet(PROCESSED_DIR / "detection.parquet", index=False)
    sensitivity.to_parquet(PROCESSED_DIR / "sensitivity.parquet", index=False)
    importance.to_parquet(PROCESSED_DIR / "importance.parquet", index=False)
    shap_long.to_parquet(PROCESSED_DIR / "shap.parquet", index=False)

    # ---------------- write report tables + summary -------------------------
    metrics.to_csv(REPORTS_DIR / "forecast_metrics.csv", index=False)
    scores_csv = scored.copy()
    scores_csv["method"] = scores_csv["method"].map(
        {
            "residual": "Residual (robust z)",
            "residual+persistence": "Residual + persistence",
            "isolation_forest": "Isolation Forest",
            "hybrid": "Hybrid (persistence + flatline)",
            "adaptive_hybrid": "Adaptive hybrid (30-day recalibration)",
        }
    )
    scores_csv.to_csv(REPORTS_DIR / "detection_scores.csv", index=False)
    sensitivity.to_csv(REPORTS_DIR / "detection_sensitivity.csv", index=False)
    importance.to_csv(REPORTS_DIR / "shap_importance.csv", index=False)
    clean_tags.to_csv(REPORTS_DIR / "anomalies_clean_test.csv", index=False)
    events.to_csv(REPORTS_DIR / "injected_events.csv", index=False)

    thresholds_table = pd.DataFrame([r["thresholds"] for r in results])
    thresholds_table.to_csv(REPORTS_DIR / "building_thresholds.csv", index=False)
    thresholds_table.to_parquet(PROCESSED_DIR / "buildings.parquet", index=False)

    impacts = []
    for building_id, group in clean_tags.groupby("building_id"):
        impact = estimate_impact(group, TARIFF_INR_PER_KWH, CO2_KG_PER_KWH)
        impact["building_id"] = building_id
        impact["building_label"] = BUILDING_LABELS[building_id]
        impacts.append(impact)
    impact_table = pd.DataFrame(impacts)[
        ["building_id", "building_label", "flagged_hours", "excess_kwh", "deficit_kwh", "cost_inr", "co2_kg"]
    ]
    impact_table.to_csv(REPORTS_DIR / "anomaly_impact.csv", index=False)

    summary = {
        "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
        "site": SITE_ID,
        "buildings": BUILDING_LABELS,
        "n_buildings": len(BUILDINGS),
        "modelling_rows": int(len(frame)),
        "features": FEATURE_COLUMNS,
        "z_threshold": Z_THRESHOLD,
        "persistence": {"k": PERSIST_K, "n": PERSIST_N},
        "tariff_inr_per_kwh": TARIFF_INR_PER_KWH,
        "co2_kg_per_kwh": CO2_KG_PER_KWH,
        "runtime_seconds": round(time.time() - started, 1),
    }
    (REPORTS_DIR / "pipeline_summary.json").write_text(json.dumps(summary, indent=2))

    # ---------------- console report (this is what PROJECT_GUIDE quotes) ----
    print("\n=== Forecast accuracy, TEST fold (mean over buildings) ===")
    test_metrics = metrics[metrics["fold"] == "test"]
    print(
        test_metrics.groupby("model")[["mae", "rmse", "cv_rmse_pct", "nmbe_pct", "smape_pct"]]
        .mean()
        .round(2)
        .rename(index=MODEL_LABELS)
        .to_string()
    )

    print("\n=== Detection performance on injected test anomalies (mean over buildings) ===")
    print(overall.to_string(index=False))

    print("\n=== Per-type recall (mean over buildings) ===")
    print(per_type.to_string(index=False))

    print("\n=== Estimated avoidable consumption on real test-period data "
          f"(tariff Rs{TARIFF_INR_PER_KWH}/kWh, {CO2_KG_PER_KWH} kg CO2/kWh) ===")
    print(impact_table.drop(columns=["building_id"]).to_string(index=False))

    print(f"\nDone in {time.time() - started:.1f}s. Dashboard inputs in data/processed/.")


if __name__ == "__main__":
    main()

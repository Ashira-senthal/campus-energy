"""
Forecasting models and the chronological split.

Five candidates, all trained per building on the same feature matrix:

* ``snaive_24``  / ``snaive_168`` -- seasonal naive baselines (no fitting).
* ``ridge``      -- regularised linear regression behind a median imputer and a
                    scaler (median imputation because weather has gaps; the
                    imputer and scaler are fitted on the TRAIN fold only).
* ``random_forest`` -- non-linear bagged trees.
* ``lightgbm``   -- gradient boosting; the strongest tabular learner here and the
                    model used for SHAP and residuals.

Prediction is one-step-ahead: every feature for hour t is knowable before hour t
(lags are >= 24 h). That is what makes the residual an honest "surprise" signal.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    MODEL_NAMES,
    RANDOM_SEED,
    SPLITS,
)
from src.features import FEATURE_COLUMNS

BASELINE_MODELS = ("snaive_24", "snaive_168")


def split_by_config(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Chronological train / validation / test slices from ``config.SPLITS``.

    Deliberately index-based slicing on timestamps, never ``train_test_split``:
    shuffling hourly data lets the model interpolate from neighbouring hours and
    puts future information into training, which inflates every metric.
    """
    slices: dict[str, pd.DataFrame] = {}
    for name, (start, end) in SPLITS.items():
        mask = (df["timestamp"] >= pd.Timestamp(start)) & (
            df["timestamp"] <= pd.Timestamp(end)
        )
        slices[name] = df.loc[mask].reset_index(drop=True)
    return slices


def make_estimator(name: str, seed: int = RANDOM_SEED) -> Any | None:
    """Build one unfitted estimator, or ``None`` for the rule-based baselines."""
    if name in BASELINE_MODELS:
        return None

    if name == "ridge":
        return Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        )

    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            max_features=0.5,
            n_jobs=-1,
            random_state=seed,
        )

    if name == "lightgbm":
        return LGBMRegressor(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )

    raise ValueError(f"unknown model: {name}")


def fit_models(
    train_df: pd.DataFrame,
    model_names: list[str] = MODEL_NAMES,
    feature_columns: list[str] = FEATURE_COLUMNS,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Fit every candidate on ONE training fold and return the fitted objects.

    Baselines are stored as ``None`` because they need no parameters -- they are
    defined purely by which lag column they read.
    """
    x_train = train_df[feature_columns]
    y_train = train_df["electricity"]

    fitted: dict[str, Any] = {}
    for name in model_names:
        estimator = make_estimator(name, seed=seed)
        if estimator is not None:
            estimator.fit(x_train, y_train)
        fitted[name] = estimator
    return fitted


def predict(
    name: str,
    estimator: Any,
    df: pd.DataFrame,
    feature_columns: list[str] = FEATURE_COLUMNS,
) -> np.ndarray:
    """Predict with one model for one fold. Negative kWh is clipped to zero."""
    if name == "snaive_24":
        raw = df["kwh_lag_24"].to_numpy(dtype="float64")
    elif name == "snaive_168":
        raw = df["kwh_lag_168"].to_numpy(dtype="float64")
    else:
        raw = np.asarray(estimator.predict(df[feature_columns]), dtype="float64")
    # Energy cannot be negative; clipping keeps the metrics physically sensible.
    return np.clip(raw, 0.0, None)


def fit_and_predict_all(
    train_df: pd.DataFrame,
    folds: dict[str, pd.DataFrame],
    model_names: list[str] = MODEL_NAMES,
    feature_columns: list[str] = FEATURE_COLUMNS,
    seed: int = RANDOM_SEED,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """Fit all models once, then predict on every requested fold.

    Returns ``(fitted, predictions)`` where ``predictions[fold][model]`` is a
    numpy array aligned with ``folds[fold]`` rows.
    """
    fitted = fit_models(train_df, model_names, feature_columns, seed)

    predictions: dict[str, dict[str, np.ndarray]] = {}
    for fold_name, fold_df in folds.items():
        predictions[fold_name] = {
            name: predict(name, fitted[name], fold_df, feature_columns)
            for name in model_names
        }
    return fitted, predictions

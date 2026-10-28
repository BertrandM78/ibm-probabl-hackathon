"""Shared feature helpers. Each function takes a visits frame and returns a new one.

Rules (see Bob.md):
- a feature may only use columns that exist in X_test (never `target`);
- per-patient features (lags, leads, patient means) are fine: valid/test folds
  contain all visits of their patients;
- anything fitted on the target (encodings, PK curves, ...) must be fitted
  inside fit_predict, on `train` only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GROUP = "patient_id"
CATEGORICALS = ["cohort", "gene", "rater_id"]
DROP = ["Index", "patient_id", "target"]


def add_basic(df: pd.DataFrame) -> pd.DataFrame:
    """Disease duration, visit position, missingness flags."""
    df = df.copy()
    df["time_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]
    g = df.groupby(GROUP, sort=False)
    df["visit_rank"] = g.cumcount()
    df["n_visits"] = g["age"].transform("size")
    df["years_since_first_visit"] = df["age"] - g["age"].transform("min")
    for col in ["on", "off", "ledd", "time_since_intake_on", "time_since_intake_off", "gene", "age_at_diagnosis"]:
        df[f"{col}_isna"] = df[col].isna().astype(int)
    df["on_off_gap"] = df["off"] - df["on"]
    return df


def add_patient_context(df: pd.DataFrame, cols=("on", "off", "ledd")) -> pd.DataFrame:
    """Previous/next observed value and patient-level mean of each column."""
    df = df.copy()
    g = df.groupby(GROUP, sort=False)
    for col in cols:
        df[f"{col}_prev"] = g[col].transform(lambda s: s.shift(1).ffill())
        df[f"{col}_next"] = g[col].transform(lambda s: s.shift(-1).bfill())
        df[f"{col}_pmean"] = g[col].transform("mean")
    return df


def to_model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ids/target, cast string columns to pandas category (for HGB/LightGBM)."""
    X = df.drop(columns=[c for c in DROP if c in df.columns])
    for col in CATEGORICALS:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


def align_categories(X_train: pd.DataFrame, X_other: pd.DataFrame) -> pd.DataFrame:
    """Give X_other the same category levels as X_train (unseen -> NaN)."""
    X_other = X_other.copy()
    for col in X_train.select_dtypes("category").columns:
        X_other[col] = pd.Categorical(X_other[col], categories=X_train[col].cat.categories)
    return X_other


def clip_target(pred: np.ndarray) -> np.ndarray:
    return np.clip(pred, 0, 132)

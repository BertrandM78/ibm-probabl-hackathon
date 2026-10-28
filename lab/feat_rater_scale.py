"""H04: common multiplicative rater factor on on/off, applied before patient context."""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import GROUP


def fit_rater_scale(train: pd.DataFrame, k: float = 20.0) -> tuple[pd.Series, float]:
    """s_r = median(off/target) per rater, shrunk towards the global median (k pseudo-visits)."""
    m = (train["off"] > 0) & (train["target"] > 0)
    ratio = (train.loc[m, "off"] / train.loc[m, "target"])
    s_global = float(ratio.median())
    g = ratio.groupby(train.loc[m, "rater_id"])
    med, n = g.median(), g.size()
    s = (n * med + k * s_global) / (n + k)
    return s, s_global


def check_order(df: pd.DataFrame) -> None:
    """Patients contiguous and age non-decreasing within patient (needed by shift-based context)."""
    pid = df[GROUP].to_numpy()
    starts = np.r_[True, pid[1:] != pid[:-1]]
    assert pd.Series(pid[starts]).is_unique, "patients not contiguous"
    assert (df.groupby(GROUP, sort=False)["age"].diff().fillna(0) >= 0).all(), "age not sorted"


def add_rater_scale(df: pd.DataFrame, s: pd.Series, s_global: float) -> pd.DataFrame:
    """Per-visit rater factor, normalised on/off and patient mean of s (index/order preserved)."""
    check_order(df)
    df = df.copy()
    df["s_rater"] = df["rater_id"].map(s).astype(float).fillna(s_global)
    df["on_n"] = df["on"] / df["s_rater"]
    df["off_n"] = df["off"] / df["s_rater"]
    df["s_pmean"] = df.groupby(GROUP, sort=False)["s_rater"].transform("mean")
    return df

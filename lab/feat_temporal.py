"""Generic temporal feature engineering (no target): lags/leads, rolling means, deltas, per-patient trends.

Everything is intra-patient and target-free, so it can be computed on train and valid/test alike.
Rows must be sorted by patient then age (file order; checked). All new columns are prefixed `t_`.

Delta-target helpers: `delta_target` (train only) and `reconstruct` (per-patient least squares
combining level predictions A_t and delta predictions D_t = L_t - L_{t-1}).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import GROUP

SERIES = ("on", "off", "ledd", "time_since_intake_on", "time_since_intake_off", "t_gap", "t_ratio")
SHORT = {"time_since_intake_on": "tsi_on", "time_since_intake_off": "tsi_off", "t_gap": "gap", "t_ratio": "ratio"}
LAGS = (1, 2)
LEADS = (1, 2)


def _check_order(df):
    a = df[GROUP].to_numpy()
    starts = np.r_[True, a[1:] != a[:-1]]
    assert pd.Series(a[starts]).is_unique, "patients not contiguous"
    d = df.groupby(GROUP, sort=False)["age"].diff()
    assert (d.dropna() >= 0).all(), "visits not sorted by age"


def _patient_ols(df, y, x="age"):
    """Per-patient OLS of column y on x over observed rows: slope, intercept, fitted value at each row."""
    m = df[y].notna()
    xv, yv = df[x].where(m), df[y]
    g = pd.DataFrame({GROUP: df[GROUP], "n": m.astype(float), "x": xv, "y": yv,
                      "xx": xv * xv, "xy": xv * yv}).groupby(GROUP, sort=False)
    s = g[["n", "x", "y", "xx", "xy"]].transform("sum")
    den = s["n"] * s["xx"] - s["x"] ** 2
    ok = (s["n"] >= 2) & (den > 1e-9)
    slope = ((s["n"] * s["xy"] - s["x"] * s["y"]) / den).where(ok)
    icpt = ((s["y"] - slope * s["x"]) / s["n"]).where(ok)
    return slope, icpt, icpt + slope * df[x]


def add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """Massive temporal block: lags t-1/t-2, leads t+1/t+2, deltas, rolling/expanding means, trends."""
    _check_order(df)
    df = df.copy()
    new = {}
    df["t_gap"] = df["off"] - df["on"]
    df["t_ratio"] = (df["on"] / df["off"]).where(df["off"] > 0)
    g = df.groupby(GROUP, sort=False)

    # time and dose steps between visits
    new["t_vis_idx"] = g.cumcount()
    new["t_vis_left"] = g["age"].transform("size") - 1 - new["t_vis_idx"]
    new["t_dt_prev"] = g["age"].diff()
    new["t_dt_next"] = -g["age"].diff(-1)
    new["t_tsd"] = df["age"] - df["age_at_diagnosis"]
    new["t_age_rel"] = df["age"] - g["age"].transform("mean")
    new["t_span"] = g["age"].transform("max") - g["age"].transform("min")
    new["t_rater_same_prev"] = (df["rater_id"] == g["rater_id"].shift(1)).astype(float).where(new["t_vis_idx"] > 0)
    new["t_rater_npat"] = g["rater_id"].transform("nunique")

    for col in SERIES:
        k = SHORT.get(col, col)
        s = df[col]
        gs = g[col]
        for L in LAGS:
            new[f"t_{k}_lag{L}"] = gs.shift(L)
        for L in LEADS:
            new[f"t_{k}_lead{L}"] = gs.shift(-L)
        new[f"t_{k}_d_prev"] = s - new[f"t_{k}_lag1"]
        new[f"t_{k}_d_next"] = new[f"t_{k}_lead1"] - s
        # last / next observed value and how far away it is
        prev_obs = gs.transform(lambda v: v.shift(1).ffill())
        next_obs = gs.transform(lambda v: v.shift(-1).bfill())
        age_obs = df["age"].where(s.notna())
        new[f"t_{k}_prevobs"] = prev_obs
        new[f"t_{k}_nextobs"] = next_obs
        new[f"t_{k}_dt_prevobs"] = df["age"] - age_obs.groupby(df[GROUP], sort=False).transform(lambda v: v.shift(1).ffill())
        new[f"t_{k}_dt_nextobs"] = age_obs.groupby(df[GROUP], sort=False).transform(lambda v: v.shift(-1).bfill()) - df["age"]
        new[f"t_{k}_interp"] = s.fillna(_interp(df, col))
        # rolling / expanding means (NaN-aware)
        for w in (3, 5):
            new[f"t_{k}_roll{w}"] = gs.transform(lambda v, w=w: v.rolling(w, center=True, min_periods=1).mean())
        new[f"t_{k}_past_mean"] = gs.transform(lambda v: v.shift(1).expanding().mean())
        new[f"t_{k}_fut_mean"] = gs.transform(lambda v: v[::-1].shift(1).expanding().mean()[::-1])
        # patient summary
        for stat in ("mean", "std", "min", "max", "median"):
            new[f"t_{k}_p{stat}"] = gs.transform(stat)
        new[f"t_{k}_pcount"] = gs.transform("count")
        new[f"t_{k}_dev"] = s - new[f"t_{k}_pmean"]
        new[f"t_{k}_prank"] = gs.rank(pct=True)
        # rates per year
        new[f"t_{k}_rate_prev"] = new[f"t_{k}_d_prev"] / new["t_dt_prev"].where(new["t_dt_prev"] > 0)
        if col in ("on", "off", "ledd", "t_gap"):
            slope, icpt, fit = _patient_ols(df, col)
            new[f"t_{k}_slope"] = slope
            new[f"t_{k}_trend"] = fit
            new[f"t_{k}_trend_res"] = s - fit

    new["t_ledd_cum_change"] = df["ledd"] - g["ledd"].transform("first")
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def _interp(df, col):
    """Linear interpolation in age within patient (inside the observed range; edges stay NaN)."""
    out = pd.Series(np.nan, index=df.index)
    for _, idx in df.groupby(GROUP, sort=False).indices.items():
        y = df[col].to_numpy()[idx]
        m = ~np.isnan(y)
        if m.sum() >= 2:
            x = df["age"].to_numpy()[idx]
            v = np.interp(x, x[m], y[m], left=np.nan, right=np.nan)
            out.iloc[idx] = v
    return out


# ---------------------------------------------------------------- delta target

def delta_target(df: pd.DataFrame, y: pd.Series) -> pd.Series:
    """target_t - target_{t-1} within patient (NaN on first visits). Train only."""
    return y - y.groupby(df[GROUP], sort=False).shift(1)


def reconstruct(df: pd.DataFrame, level: np.ndarray, delta: np.ndarray, lam: float) -> np.ndarray:
    """Per patient: argmin_L sum (L_t - A_t)^2 + lam * sum_{t>0} (L_t - L_{t-1} - D_t)^2 (tridiagonal solve)."""
    out = np.empty(len(df))
    for _, idx in df.groupby(GROUP, sort=False).indices.items():
        a, d = level[idx], delta[idx]
        n = len(idx)
        if n == 1 or lam == 0:
            out[idx] = a
            continue
        # difference operator rows t=1..n-1: L_t - L_{t-1}
        Dm = np.zeros((n - 1, n))
        Dm[np.arange(n - 1), np.arange(1, n)] = 1
        Dm[np.arange(n - 1), np.arange(n - 1)] = -1
        M = np.eye(n) + lam * Dm.T @ Dm
        out[idx] = np.linalg.solve(M, a + lam * Dm.T @ d[1:])
    return out

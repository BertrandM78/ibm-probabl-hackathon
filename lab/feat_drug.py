"""H01: common drug-effect curve g(tsi_on) (Emax of Bateman, ON side only) + patient amplitude A_p."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from feat_rater_scale import check_order
from features import GROUP

TSI = "time_since_intake_on"
T_GRID = np.linspace(0, 6.3, 631)
CLIP_ON_EQ = (0.0, 132.0)


def _emax_bateman(t, p):
    em, ka, ke, ec, h = p
    c = np.clip(np.exp(-ke * t) - np.exp(-ka * t), 1e-12, None)
    return em * c**h / (ec**h + c**h)


def _saturation(t, p):
    em, k = p
    return em * (1 - np.exp(-k * t))


def _on_side(df: pd.DataFrame) -> pd.Series:
    """Visits usable for the ON-side ratio: on, off, tsi_on present, off > 5, on > 0."""
    return df["on"].notna() & df["off"].notna() & df[TSI].notna() & (df["off"] > 5) & (df["on"] > 0)


def g_eval(t, g_params: dict) -> np.ndarray:
    """Normalised curve g = E / max E on [0, 6.3] (NaN where t is NaN)."""
    t = np.asarray(t, dtype=float)
    f = _emax_bateman if g_params["kind"] == "emax_bateman" else _saturation
    out = f(np.nan_to_num(t), g_params["p"]) / g_params["emax"]
    return np.where(np.isnan(t), np.nan, out)


def fit_drug_curve(train: pd.DataFrame, k: float = 0.5, thr: float = 0.5) -> tuple[dict, float, float]:
    """Fit g on train (ON side only, no target), then A0 = mean train a_v, gbar = mean train g(tsi_on)."""
    m = _on_side(train)
    t = train.loc[m, TSI].to_numpy(float)
    e = 1 - (train.loc[m, "on"] / train.loc[m, "off"]).to_numpy(float)
    try:
        o = least_squares(lambda p: e - _emax_bateman(t, p), [0.5, 1.5, 0.3, 0.3, 3],
                          bounds=([0, 0, 0, 1e-4, 0.5], [1, 20, 5, 1, 10]))
        g_params = {"kind": "emax_bateman", "p": o.x}
        emax = _emax_bateman(T_GRID, o.x).max()
        assert o.success and np.isfinite(emax) and emax > 0.05
    except Exception:
        o = least_squares(lambda p: e - _saturation(t, p), [0.5, 1.0], bounds=([0, 0], [1, 20]))
        g_params = {"kind": "saturation", "p": o.x}
        emax = _saturation(T_GRID, o.x).max()
    g_params["emax"] = float(emax)
    gv = g_eval(t, g_params)
    ok = gv > thr
    A0 = float(np.clip(e[ok] / gv[ok], 0, 1).mean())
    has_t = train["on"].notna() & train[TSI].notna()
    gbar = float(np.nanmean(g_eval(train.loc[has_t, TSI], g_params)))
    return g_params, A0, gbar


def add_drug_amplitude(df: pd.DataFrame, g_params: dict, A0: float, gbar: float,
                       k: float = 0.5, thr: float = 0.5) -> pd.DataFrame:
    """a_v, leave-one-out shrunk A_p, g_on, n_av, on_eq (needs s_rater for on_eq_n). Order preserved."""
    check_order(df)
    df = df.copy()
    g_on = g_eval(df[TSI], g_params)
    usable = _on_side(df).to_numpy() & (np.nan_to_num(g_on) > thr)
    with np.errstate(divide="ignore", invalid="ignore"):
        a_v = np.clip((1 - df["on"] / df["off"]).to_numpy(float) / g_on, 0, 1)
    a_v = np.where(usable, a_v, np.nan)
    tmp = pd.DataFrame({GROUP: df[GROUP].to_numpy(), "a": np.nan_to_num(a_v), "n": usable.astype(float)},
                       index=df.index)
    grp = tmp.groupby(GROUP, sort=False)
    s_oth = grp["a"].transform("sum") - tmp["a"]
    n_oth = grp["n"].transform("sum") - tmp["n"]
    df["A_p"] = ((s_oth + k * A0) / (n_oth + k)).clip(0.15, 0.85)
    df["n_av"] = n_oth
    df["g_on"] = g_on
    g_use = np.where(np.isnan(g_on), gbar, g_on)
    on_eq = df["on"] / (1 - df["A_p"] * g_use)
    df["on_eq"] = on_eq.where(df["on"] != 0).clip(*CLIP_ON_EQ)
    if "s_rater" in df.columns:
        df["on_eq_n"] = df["on_eq"] / df["s_rater"]
    return df


def add_on_eq_context(df: pd.DataFrame, col: str = "on_eq_n") -> pd.DataFrame:
    """prev/next/pmean of col, excluding visits with on == 0 or off == 0."""
    df = df.copy()
    src = df[col].where((df["on"] != 0) & (df["off"] != 0))
    g = src.groupby(df[GROUP], sort=False)
    df[f"{col}_prev"] = g.transform(lambda s: s.shift(1).ffill())
    df[f"{col}_next"] = g.transform(lambda s: s.shift(-1).bfill())
    df[f"{col}_pmean"] = g.transform("mean")
    return df

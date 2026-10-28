"""H42a (= H39): A_p also pools on/off pairs without tsi_on, using g = gbar (weight W, shrink k)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import feat_drug
from feat_rater_scale import check_order
from features import GROUP


def add_drug_amplitude_pairs(df: pd.DataFrame, g_params: dict, A0: float, gbar: float,
                             k: float = 0.5, thr: float = 0.5, W: float = 1.0) -> pd.DataFrame:
    """feat_drug.add_drug_amplitude, then A_p / n_av / on_eq(_n) recomputed with the no-tsi pairs (no target)."""
    out = feat_drug.add_drug_amplitude(df, g_params, A0, gbar, k, thr)
    check_order(df)
    g_on = feat_drug.g_eval(df[feat_drug.TSI], g_params)
    pair = (df["on"].notna() & df["off"].notna() & (df["off"] > 5) & (df["on"] > 0)).to_numpy()
    use_t = pair & (np.nan_to_num(g_on) > thr)
    use_nt = pair & np.isnan(g_on)
    e = (1 - df["on"] / df["off"]).to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        a_t = np.clip(e / g_on, 0, 1)
        a_nt = np.clip(e / gbar, 0, 1)
    w_nt = W * use_nt
    a = np.where(use_t, a_t, 0) + np.where(w_nt > 0, a_nt * w_nt, 0)
    w = use_t.astype(float) + w_nt
    tmp = pd.DataFrame({GROUP: df[GROUP].to_numpy(), "a": np.nan_to_num(a), "w": w}, index=df.index)
    grp = tmp.groupby(GROUP, sort=False)
    s_oth = grp["a"].transform("sum") - tmp["a"]
    n_oth = grp["w"].transform("sum") - tmp["w"]
    out["A_p"] = ((s_oth + k * A0) / (n_oth + k)).clip(0.15, 0.85)
    out["n_av"] = n_oth
    g_use = np.where(np.isnan(g_on), gbar, g_on)
    out["on_eq"] = (out["on"] / (1 - out["A_p"] * g_use)).where(out["on"] != 0).clip(*feat_drug.CLIP_ON_EQ)
    if "s_rater" in out.columns:
        out["on_eq_n"] = out["on_eq"] / out["s_rater"]
    return out

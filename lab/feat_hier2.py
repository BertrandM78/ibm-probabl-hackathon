"""H09: trajectory v2 on top of H06 - level-dependent noise weights and a population curvature.

Curvature: kappa = pooled within-patient quadratic coefficient of target on centred age (train only);
the hierarchical line is fitted on (y_v, target) - kappa * x^2 and kappa * x^2 is added back.
Heteroscedasticity: sd(y_v - target) ~ c0 + c1 * target (train only); visit weight w_v / f(mu_v) with
mu_v = first-pass posterior trajectory (no target), f normalised to mean 1 on train.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from feat_harmonized import harmonized_score
from feat_hier import add_hier, centred_age, fit_prior
from features import GROUP

HB_COLS = ("hb_prior", "hb_post_loo", "hb_post_full")  # hb_int_loo = level at x=0, no offset


def _within(v: pd.Series, x: pd.Series, pid: pd.Series) -> pd.Series:
    """Residual of v after a per-patient OLS on [1, x]."""
    t = pd.DataFrame({"v": v, "x": x, "xx": x * x, "xv": x * v})
    S = t.groupby(pid, sort=False).transform("mean")
    b = ((S["xv"] - S["x"] * S["v"]) / (S["xx"] - S["x"] ** 2)).fillna(0.0)
    return v - S["v"] - b * (x - S["x"])


def fit_curvature(train: pd.DataFrame) -> float:
    x, pid = centred_age(train), train[GROUP]
    q, t = _within(x * x, x, pid), _within(train["target"], x, pid)
    return float((q * t).sum() / (q * q).sum())


def fit_hetero(train: pd.DataFrame) -> dict:
    """Linear fit of the residual sd of y_v on target, over target deciles (train only)."""
    y, _ = harmonized_score(train)
    d = pd.DataFrame({"e": y - train["target"], "t": train["target"]}).dropna()
    b = d.groupby(pd.qcut(d["t"], 10), observed=True).agg(t=("t", "mean"), sd=("e", "std"))
    c1, c0 = np.polyfit(b["t"], b["sd"], 1)
    c0, c1 = max(c0, 1.0), max(c1, 0.0)
    f_mean = float(np.mean(((c0 + c1 * d["t"].clip(lower=0)) / (c0 + c1 * d["t"].mean())) ** 2))
    return {"c0": c0, "c1": c1, "t_mean": float(d["t"].mean()), "f_mean": f_mean}


def level_factor(mu: np.ndarray, het: dict) -> np.ndarray:
    m = np.clip(mu, 0, None)
    return ((het["c0"] + het["c1"] * m) / (het["c0"] + het["c1"] * het["t_mean"])) ** 2 / het["f_mean"]


def _shifted(df: pd.DataFrame, kappa: float) -> pd.DataFrame:
    q = kappa * centred_age(df) ** 2
    out = df.assign(y_v=df["y_v"] - q, _q=q)
    if "target" in out:
        out["target"] = out["target"] - q
    return out


def fit_prior2(train: pd.DataFrame, kappa: float) -> dict:
    return fit_prior(_shifted(train, kappa))


def add_hier2(df: pd.DataFrame, prior: dict, kappa: float, het: dict | None, is_train: bool) -> pd.DataFrame:
    """Two-pass hierarchical trajectory with curvature offset and level-dependent weights (same hb_* names)."""
    d = _shifted(df, kappa)
    p1 = add_hier(d, prior, is_train)
    if het is not None:
        f = level_factor(p1["hb_post_full"].to_numpy() + d["_q"].to_numpy(), het)
        p1 = add_hier(d.assign(w_v=d["w_v"] / f), prior, is_train)
        p1["hb_level_f"] = f
    for c in HB_COLS:
        p1[c] = p1[c] + d["_q"]
    p1["hb_prior_gap"] = p1["hb_post_loo"] - p1["hb_prior"]
    out = p1.drop(columns="_q")
    out["y_v"] = df["y_v"]
    if "target" in df:
        out["target"] = df["target"]
    return out

"""H05: harmonised per-visit score y_v (off/s, else on_eq/s) weighted by 1/var, pooled per patient."""
from __future__ import annotations

import numpy as np
import pandas as pd

from feat_rater_scale import check_order
from features import GROUP

TSI = "time_since_intake_on"


def harmonized_score(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """y_v = off_n if off > 0, else on_eq_n (NaN if on == 0). Returns (y_v, source in {off, on_t, on_nt, none})."""
    use_off = df["off_n"].notna() & (df["off"] > 0)
    use_on = ~use_off & df["on_eq_n"].notna() & (df["on"] > 0)
    y = df["off_n"].where(use_off, df["on_eq_n"].where(use_on))
    src = np.select([use_off, use_on & df[TSI].notna(), use_on], ["off", "on_t", "on_nt"], "none")
    return y, pd.Series(src, index=df.index)


def fit_noise_var(train: pd.DataFrame) -> dict:
    """Noise variance of y_v per source = mean squared residual vs target (train only)."""
    y, src = harmonized_score(train)
    r2 = (y - train["target"]) ** 2
    return {k: float(r2[src == k].mean()) for k in ("off", "on_t", "on_nt")}


def _wls_sums(x, y, w, pid):
    """Per-patient sums of the weighted least-squares problem, broadcast to visits."""
    t = pd.DataFrame({"w": w, "wx": w * x, "wy": w * y, "wxx": w * x * x, "wxy": w * x * y}, index=pid.index)
    return t.groupby(pid, sort=False).transform("sum"), t


def fit_slope_prior(train: pd.DataFrame, var: dict) -> tuple[float, float]:
    """b0: pooled within-patient WLS slope of y_v on age (no target); lam = 1/var(target slopes) (train)."""
    df = add_harmonized(train, var, b0=0.0, lam=1.0, context=False)
    w, y, x = df["w_v"].to_numpy(), np.nan_to_num(df["y_v"].to_numpy()), df["age"].to_numpy()
    S, _ = _wls_sums(x, y, w, df[GROUP])
    with np.errstate(divide="ignore", invalid="ignore"):
        xbar, ybar = S["wx"] / S["w"], S["wy"] / S["w"]
    m = S["w"] > 0
    xc, yc = (x - xbar)[m], (y - ybar)[m]
    b0 = float((w[m] * xc * yc).sum() / (w[m] * xc * xc).sum())

    def slope(d):
        xc = d["age"] - d["age"].mean()
        return (xc * (d["target"] - d["target"].mean())).sum() / (xc * xc).sum()

    tau2 = float(train.groupby(GROUP, sort=False)[["age", "target"]].apply(slope).var())
    return b0, 1.0 / tau2


def add_harmonized(df: pd.DataFrame, var: dict, b0: float, lam: float, context: bool = True) -> pd.DataFrame:
    """y_v, w_v, then LOO / full weighted patient mean, LOO ridge-WLS trajectory on age, prev/next. Needs off_n, on_eq_n."""
    check_order(df)
    df = df.copy()
    y, src = harmonized_score(df)
    df["y_v"] = y
    df["w_v"] = src.map({"off": 1 / var["off"], "on_t": 1 / var["on_t"], "on_nt": 1 / var["on_nt"]}).fillna(0.0).astype(float)
    if not context:
        return df
    pid = df[GROUP]
    w, yv, age = df["w_v"].to_numpy(), np.nan_to_num(y.to_numpy()), df["age"].to_numpy()
    S, own = _wls_sums(age, yv, w, pid)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["h_wmean_full"] = (S["wy"] / S["w"]).where(S["w"] > 0)
        L = S - own                                   # leave-one-out sums
        df["h_wsum_loo"] = L["w"]
        ok = L["w"] > 0
        xbar, ybar = L["wx"] / L["w"], L["wy"] / L["w"]
        df["h_wmean_loo"] = ybar.where(ok)
        sxx = (L["wxx"] - L["wx"] ** 2 / L["w"]).clip(lower=0)
        sxy = L["wxy"] - L["wx"] * L["wy"] / L["w"]
        b = (sxy + lam * b0) / (sxx + lam)
        dx = age - xbar
        df["h_slope_loo"] = b.where(ok)
        df["h_traj_loo"] = (ybar + b * dx).where(ok)
        df["h_traj_se"] = np.sqrt(1 / L["w"] + dx ** 2 / (sxx + lam)).where(ok)
        # full (own visit included): posterior-like trajectory value at the visit
        xf, yf = S["wx"] / S["w"], S["wy"] / S["w"]
        sxx_f = (S["wxx"] - S["wx"] ** 2 / S["w"]).clip(lower=0)
        bf = (S["wxy"] - S["wx"] * S["wy"] / S["w"] + lam * b0) / (sxx_f + lam)
        df["h_traj_full"] = (yf + bf * (age - xf)).where(S["w"] > 0)
    # previous / next available y_v and its weight
    for c in ("y_v", "w_v"):
        src_c = df[c].where(df["w_v"] > 0)
        gc = src_c.groupby(pid, sort=False)
        df[f"h_{c}_prev"] = gc.transform(lambda s: s.shift(1).ffill())
        df[f"h_{c}_next"] = gc.transform(lambda s: s.shift(-1).bfill())
    return df

"""H06: hierarchical linear trajectory a_p + b_p*(age - mean age) with a covariate-dependent Gaussian prior.

Prior (mu_a, mu_b) = small HGB on patient covariates (no y_v), trained on per-patient OLS (a, b) of target
(train only, cross-fitted for train rows). Prior covariance = residual cov minus OLS estimation noise.
Likelihood = harmonised y_v with variance var_src + sigma_line^2. Conjugate 2x2 posterior, LOO and full.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

import cv
from feat_rater_scale import check_order
from features import GROUP

PRIOR_PARAMS = dict(max_iter=200, learning_rate=0.05, max_depth=3, early_stopping=False,
                    categorical_features="from_dtype", random_state=cv.SEED)


def centred_age(df: pd.DataFrame) -> pd.Series:
    return df["age"] - df.groupby(GROUP, sort=False)["age"].transform("mean")


def patient_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """One row per patient (sort=False order), covariates without target nor y_v."""
    d = df.assign(tsd=df["age"] - df["age_at_diagnosis"], ledd_na=df["ledd"].isna().astype(float))
    g = d.groupby(GROUP, sort=False)
    cov = g.agg(cohort=("cohort", "first"), gene=("gene", "first"), sexM=("sexM", "first"),
                aad=("age_at_diagnosis", "first"), age_m=("age", "mean"), tsd_m=("tsd", "mean"),
                ledd_m=("ledd", "mean"), ledd_na=("ledd_na", "mean"), A_m=("A_p", "mean"),
                s_m=("s_rater", "mean"), n=("age", "size"))
    cov["span"] = g["age"].max() - g["age"].min()
    for c in ("cohort", "gene"):
        cov[c] = cov[c].astype("category")
    return cov


def patient_ols(train: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Per-patient OLS of target on centred age: a (level at mean age), b, n, sxx; pooled line sigma^2."""
    x, y = centred_age(train), train["target"]
    t = pd.DataFrame({"n": 1.0, "y": y, "xx": x * x, "xy": x * y, "yy": y * y}, index=train.index)
    S = t.groupby(train[GROUP], sort=False).sum()
    a = S["y"] / S["n"]
    b = S["xy"] / S["xx"]
    rss = (S["yy"] - S["n"] * a ** 2 - b * S["xy"]).clip(lower=0)
    sig2 = float(rss.sum() / (S["n"] - 2).sum())
    return pd.DataFrame({"a": a, "b": b, "n": S["n"], "sxx": S["xx"]}), sig2


def fit_prior(train: pd.DataFrame) -> dict:
    """Prior models for (a, b), cross-fitted prior means for train patients, prior cov, sigma_line^2."""
    cov = patient_covariates(train)
    P, sig2 = patient_ols(train)
    P = P.loc[cov.index]
    models, oof = {}, pd.DataFrame(index=cov.index, columns=["a", "b"], dtype=float)
    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=cv.SEED)
    for t in ("a", "b"):
        for itr, ite in gkf.split(cov, groups=cov.index):
            m = HistGradientBoostingRegressor(**PRIOR_PARAMS).fit(cov.iloc[itr], P[t].iloc[itr])
            oof.iloc[ite, oof.columns.get_loc(t)] = m.predict(cov.iloc[ite])
        models[t] = HistGradientBoostingRegressor(**PRIOR_PARAMS).fit(cov, P[t])
    R = (P[["a", "b"]] - oof).to_numpy()
    Sig = np.cov(R.T) - np.diag([np.mean(sig2 / P["n"]), np.mean(sig2 / P["sxx"])])
    Sig[0, 0], Sig[1, 1] = max(Sig[0, 0], 1.0), max(Sig[1, 1], 0.05)
    return {"models": models, "prior_train": oof, "Sigma": Sig, "sig2": sig2,
            "rmse_prior": np.sqrt((R ** 2).mean(axis=0)), "sd_ab": P[["a", "b"]].std().to_numpy()}


def _post(Q, mu_a, mu_b, Sw, Swx, Swxx, Swy, Swxy, x):
    """2x2 conjugate posterior; returns mean at x, its sd, posterior slope and intercept."""
    p11, p12, p22 = Q[0, 0] + Sw, Q[0, 1] + Swx, Q[1, 1] + Swxx
    r1 = Q[0, 0] * mu_a + Q[0, 1] * mu_b + Swy
    r2 = Q[1, 0] * mu_a + Q[1, 1] * mu_b + Swxy
    det = p11 * p22 - p12 ** 2
    ma, mb = (p22 * r1 - p12 * r2) / det, (p11 * r2 - p12 * r1) / det
    var = (p22 - 2 * x * p12 + x * x * p11) / det
    return ma + mb * x, np.sqrt(var), mb, ma


def add_hier(df: pd.DataFrame, prior: dict, is_train: bool) -> pd.DataFrame:
    """Prior and posterior trajectory features. Needs y_v / w_v (add_harmonized) and A_p, s_rater."""
    check_order(df)
    df = df.copy()
    pid = df[GROUP]
    if is_train:
        mu = prior["prior_train"]
    else:
        cov = patient_covariates(df)
        mu = pd.DataFrame({t: prior["models"][t].predict(cov) for t in ("a", "b")}, index=cov.index)
    mu_a, mu_b = pid.map(mu["a"]).to_numpy(float), pid.map(mu["b"]).to_numpy(float)
    x = centred_age(df).to_numpy(float)
    ok = df["w_v"].to_numpy() > 0
    w = np.where(ok, 1.0 / (1.0 / np.where(ok, df["w_v"], 1.0) + prior["sig2"]), 0.0)
    y = np.where(ok, np.nan_to_num(df["y_v"].to_numpy(float)), 0.0)
    own = pd.DataFrame({"w": w, "wx": w * x, "wxx": w * x * x, "wy": w * y, "wxy": w * x * y}, index=df.index)
    S = own.groupby(pid, sort=False).transform("sum")
    L = S - own
    Q = np.linalg.inv(prior["Sigma"])
    df["hb_prior"] = mu_a + mu_b * x
    df["hb_prior_slope"] = mu_b
    for tag, T in (("loo", L), ("full", S)):
        m, sd, b, a = _post(Q, mu_a, mu_b, *(T[c].to_numpy() for c in ("w", "wx", "wxx", "wy", "wxy")), x)
        df[f"hb_post_{tag}"], df[f"hb_se_{tag}"] = m, sd
        if tag == "loo":
            df["hb_slope_loo"], df["hb_int_loo"] = b, a
    df["hb_prior_gap"] = df["hb_post_loo"] - df["hb_prior"]
    return df

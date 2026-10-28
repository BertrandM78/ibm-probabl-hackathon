"""H58: visit-quality noise model - location and scale of the standardised noise depend on visit covariates.

z = (y_v - target) / (c0 + c1 * target) (e012 scale). Two small HGB on visit covariates (no target):
m(x) = E[z | x], s(x) ~ E[|z - m| | x] (normalised to mean 1 per source), cross-fitted by patient for
train rows. The skew-normal template is refitted per source on u = (z - m) / s; the MAP then uses
loc = sv * (m + s * xi), scale = sv * s * omega (feat_cubic.cubic_map / feat_skew.skew_map, vis=(m, s)).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

import cv
from feat_harmonized import harmonized_score
from feat_skew import SOURCES, sn_fit
from features import GROUP

COVS = ("time_since_intake_on", "time_since_intake_off", "ledd", "age", "age_at_diagnosis",
        "time_since_diagnosis", "sexM", "cohort", "gene", "rater_id", "visit_rank", "n_visits",
        "on_isna", "off_isna", "ledd_isna", "g_on")
Q_PARAMS = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
                early_stopping=False, categorical_features="from_dtype", random_state=cv.SEED)
S_CLIP = (0.5, 2.0)


def _cov(df: pd.DataFrame, src: pd.Series, levels: dict) -> pd.DataFrame:
    X = df[list(COVS)].copy()
    X["src"] = src.values
    for c, lv in levels.items():
        X[c] = pd.Categorical(X[c].astype(str), categories=lv)
    return X


def _fit(X, v):
    return HistGradientBoostingRegressor(**Q_PARAMS).fit(X, v)


def fit_quality(raw_tr: pd.DataFrame, target: pd.Series, het: dict, loc: bool = True, scale: bool = True):
    """Returns (vis_train, models, tmpl): cross-fitted (m, s) for train rows, full models, SN template on u."""
    y, src = harmonized_score(raw_tr)
    t = target.to_numpy(float)
    z = ((y - t) / (het["c0"] + het["c1"] * np.clip(t, 0, None))).to_numpy(float)
    levels = {c: sorted(set(raw_tr[c].astype(str)) | {"nan"}) for c in ("cohort", "gene", "rater_id")}
    levels["src"] = [*SOURCES, "none"]
    X = _cov(raw_tr, src, levels)
    ok = np.isfinite(z)
    m_tr, a_tr = np.zeros(len(z)), np.ones(len(z))
    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=cv.SEED)
    for itr, ite in gkf.split(X, groups=raw_tr[GROUP]):
        f = itr[ok[itr]]
        if loc:
            mm = _fit(X.iloc[f], z[f])
            m_tr[ite] = mm.predict(X.iloc[ite])
            r = np.abs(z[f] - mm.predict(X.iloc[f]))       # in-fold residual for the scale fit
        else:
            r = np.abs(z[f])
        if scale:
            a_tr[ite] = _fit(X.iloc[f], r).predict(X.iloc[ite])
    models = {"levels": levels}
    if loc:
        models["m"] = _fit(X[ok], z[ok])
        r = np.abs(z[ok] - models["m"].predict(X[ok]))
    else:
        r = np.abs(z[ok])
    if scale:
        models["s"] = _fit(X[ok], r)
    norm = {k: float(a_tr[ok & (src == k).to_numpy()].mean()) for k in SOURCES}
    models["norm"] = norm
    s_tr = _norm_scale(a_tr, src, norm) if scale else np.ones(len(z))
    u = (z - m_tr) / s_tr
    tmpl = {}
    for k in SOURCES:
        uk = u[ok & (src == k).to_numpy()]
        tmpl[k] = sn_fit(uk)
        print(f"  [quality] {k:5s} sd(z)={np.std(z[ok & (src == k).to_numpy()]):.3f} sd(u)={uk.std():.3f} "
              f"sd(m)={np.std(m_tr[ok & (src == k).to_numpy()]):.3f} "
              f"s q10/q90={np.quantile(s_tr[ok & (src == k).to_numpy()], [.1, .9]).round(2)} tmpl={np.round(tmpl[k], 3)}")
    return (m_tr, s_tr), models, tmpl


def _norm_scale(a, src, norm):
    n = src.map(norm).fillna(1.0).to_numpy(float)
    return np.clip(a / n, *S_CLIP)


def predict_quality(raw: pd.DataFrame, models: dict) -> tuple[np.ndarray, np.ndarray]:
    _, src = harmonized_score(raw)
    X = _cov(raw, src, models["levels"])
    m = models["m"].predict(X) if "m" in models else np.zeros(len(X))
    s = _norm_scale(models["s"].predict(X), src, models["norm"]) if "s" in models else np.ones(len(X))
    return m, s


def quality_cubic_hook(train_target: pd.Series, keep_quad: bool = True, **qkw):
    """e022b hook with the per-visit noise model in both the cubic and the quadratic skew MAP."""
    from feat_cubic import cubic_map, fit_cubic_prior
    from feat_skew import skew_map

    def hook(tr, va, ctx):
        raw_tr = ctx["tr_raw"].assign(target=train_target)
        vis_tr, models, tmpl = fit_quality(ctx["tr_raw"], train_target, ctx["het"], **qkw)
        vis_va = predict_quality(ctx["va_raw"], models)
        prior = fit_cubic_prior(raw_tr)
        mu1_tr, mu1_va = tr["hb_post_full"].to_numpy(), va["hb_post_full"].to_numpy()
        o_tr = cubic_map(tr, prior, ctx["het"], tmpl, mu1_tr, is_train=True, vis=vis_tr)
        o_va = cubic_map(va, prior, ctx["het"], tmpl, mu1_va, is_train=False, vis=vis_va)
        if keep_quad:
            qa = (ctx["prior"], ctx["kappa"], ctx["het"], tmpl)
            q_tr = skew_map(tr, *qa, mu1_tr, is_train=True, vis=vis_tr).add_prefix("q_")
            q_va = skew_map(va, *qa, mu1_va, is_train=False, vis=vis_va).add_prefix("q_")
            o_tr, o_va = o_tr.join(q_tr), o_va.join(q_va)
        return tr.assign(**o_tr), va.assign(**o_va)
    return hook

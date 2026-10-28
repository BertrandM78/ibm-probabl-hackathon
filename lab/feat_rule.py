"""H33 / H34 / H46: generator monotonicity rule, latent trajectory P vs target T, first-visit noise template.

Generator (dgp2.md, dgp4.md): measurements follow a latent cubic P(t); the target is
    T_0 = P_0 ; T_i = P_i if P_i >= T_{i-1} else c * T_{i-1}   (c ~ 1.02 per visit, any age gap),
floored at 0 and rounded to 0.1. c is re-estimated on the training targets of each fold (fit_rule_c).
H33: features E[rule(P)] - P_hat from the Laplace posterior of the cubic MAP (draws seeded with cv.SEED).
H34: prior OLS cubic and noise template fitted on uncensored visits only (T = P there): visits whose
     target follows the rule (T_i = round(c T_{i-1}, 1) or T_i = T_{i-1}) are dropped from both fits.
H46: noise template sources split at the first visit (<src>_v0), patched into feat_skew / feat_cubic.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pandas as pd

import cv
import feat_cubic
import feat_harmonized
import feat_skew
from feat_cubic import basis, cubic_map, fit_cubic_prior
from feat_skew import fit_skew_template, skew_map
from features import GROUP

N_DRAWS = 200
RULE_COLS = ("r_det", "r_exp", "r_ptrig", "r_sd")


def fit_rule_c(train: pd.DataFrame) -> float:
    """Per-visit multiplier: mode of T_i / T_{i-1} on (1, 1.1] (T_{i-1} > 20), refined by the median within +-0.002."""
    g = train.groupby(GROUP, sort=False)["target"]
    tp = g.shift(1)
    r = (train["target"] / tp)[(tp > 20) & (train["target"] > tp)]
    r = r[r <= 1.1].to_numpy()
    h, e = np.histogram(r, bins=np.arange(1.0, 1.1 + 1e-9, 0.0005))
    mode = e[np.argmax(h)] + 0.00025
    c = float(np.median(r[np.abs(r - mode) < 0.002]))
    print(f"  [rule] c = {c:.5f} (mode bin {mode:.4f}, n = {np.sum(np.abs(r - mode) < 0.002)})")
    return c


def censored(train: pd.DataFrame, c: float) -> np.ndarray:
    """Visits whose target is set by the rule (P unobserved there): T_i = round(c T_{i-1}, 1) or T_i = T_{i-1}."""
    t = train["target"].to_numpy(float)
    tp = train.groupby(GROUP, sort=False)["target"].shift(1).to_numpy(float)
    ok = np.isfinite(tp)
    g = ok & (tp > 2.5) & (np.abs(t - np.round(c * tp, 1)) < 0.051)
    z = ok & (t == tp)
    return g | z


def apply_rule(P: np.ndarray, rank: np.ndarray, c: float) -> np.ndarray:
    """rule along visits (last axis), rows sorted by patient then age; rank = visit index in patient."""
    T = np.maximum(np.round(P, 1), 0.0)
    for r in range(1, int(rank.max()) + 1):
        i = np.flatnonzero(rank == r)
        prev = T[..., i - 1]
        T[..., i] = np.where(P[..., i] >= prev, np.maximum(np.round(P[..., i], 1), 0.0), np.round(c * prev, 1))
    return T


def rule_features(df: pd.DataFrame, coef: np.ndarray, cov: np.ndarray, c: float) -> pd.DataFrame:
    """Deterministic and posterior-expected rule correction around the MAP trajectory P_hat."""
    pid = df[GROUP].to_numpy()
    starts = np.r_[True, pid[1:] != pid[:-1]]
    pat = np.cumsum(starts) - 1
    rank = df.groupby(GROUP, sort=False).cumcount().to_numpy()
    B = basis(df)
    p_hat = np.einsum("nd,nd->n", B, coef[pat])
    L = np.linalg.cholesky(cov + 1e-9 * np.eye(cov.shape[1]))
    rng = np.random.default_rng(cv.SEED)
    zs = rng.standard_normal((N_DRAWS, coef.shape[0], coef.shape[1]))
    cs = coef[None] + np.einsum("pde,spe->spd", L, zs)                  # S x P x D
    Ps = np.einsum("nd,snd->sn", B, cs[:, pat])                        # S x N
    Ts = apply_rule(Ps, rank, c)
    out = pd.DataFrame({"r_det": apply_rule(p_hat, rank, c) - p_hat,
                        "r_exp": Ts.mean(axis=0) - p_hat,
                        "r_ptrig": (Ts > np.maximum(np.round(Ps, 1), 0.0) + 0.05).mean(axis=0),
                        "r_sd": Ts.std(axis=0)}, index=df.index)
    assert np.isfinite(out.to_numpy()).all(), "NaN in rule features"
    return out


def rule_hook(train_target: pd.Series, keep_quad: bool = True, rule_feats: bool = True, latent_fit: bool = False):
    """feat_cubic.cubic_hook + H33 rule features (rule_feats) and/or H34 prior/template on uncensored visits."""

    def hook(tr, va, ctx):
        raw_tr = ctx["tr_raw"].assign(target=train_target)
        c = fit_rule_c(raw_tr)
        keep = ~censored(raw_tr, c) if latent_fit else None
        if latent_fit:
            print(f"  [rule] H34: {int((~keep).sum())} censored train visits ({(~keep).mean():.3%}) dropped from prior / template")
        tmpl = fit_skew_template(raw_tr if keep is None else raw_tr[keep], ctx["het"])
        prior = fit_cubic_prior(raw_tr, keep=keep)
        mu1_tr, mu1_va = tr["hb_post_full"].to_numpy(), va["hb_post_full"].to_numpy()
        o_tr, c_tr, v_tr = cubic_map(tr, prior, ctx["het"], tmpl, mu1_tr, is_train=True, return_post=True)
        o_va, c_va, v_va = cubic_map(va, prior, ctx["het"], tmpl, mu1_va, is_train=False, return_post=True)
        if rule_feats:
            o_tr = o_tr.join(rule_features(tr, c_tr, v_tr, c))
            o_va = o_va.join(rule_features(va, c_va, v_va, c))
            print(f"  [rule] P(trig) mean train {o_tr['r_ptrig'].mean():.4f} valid {o_va['r_ptrig'].mean():.4f}; "
                  f"r_exp mean train {o_tr['r_exp'].mean():.3f} valid {o_va['r_exp'].mean():.3f}")
            for k in ("r_exp", "r_ptrig", "r_sd"):
                ratio = o_va[k].std() / o_tr[k].std()
                print(f"  [rule] shift {k}: q50/q90 train {o_tr[k].quantile(.5):.3f}/{o_tr[k].quantile(.9):.3f} "
                      f"valid {o_va[k].quantile(.5):.3f}/{o_va[k].quantile(.9):.3f} sd ratio {ratio:.3f}"
                      + ("  ALERT" if abs(ratio - 1) > 0.05 else ""))
        if keep_quad:
            qa = (ctx["prior"], ctx["kappa"], ctx["het"], tmpl)
            q_tr = skew_map(tr, *qa, mu1_tr, is_train=True).add_prefix("q_")
            q_va = skew_map(va, *qa, mu1_va, is_train=False).add_prefix("q_")
            o_tr, o_va = o_tr.join(q_tr), o_va.join(q_va)
        return tr.assign(**o_tr), va.assign(**o_va)
    return hook


# ---------------------------------------------------------------- H46: first-visit template sources

def _split_v0(df):
    y, src = feat_harmonized.harmonized_score(df)
    v0 = (df.groupby(GROUP, sort=False).cumcount() == 0).to_numpy()
    src = src.astype(str)
    return y, pd.Series(np.where(v0 & (src != "none"), src + "_v0", src), index=src.index)


@contextmanager
def v0_template():
    """Template / MAP sources <src>_v0 at visit 0 (feat_skew, feat_cubic only; y_v weights unchanged)."""
    saved = feat_skew.harmonized_score, feat_cubic.harmonized_score, feat_skew.SOURCES
    base = ("off", "on_t", "on_nt")
    feat_skew.harmonized_score = feat_cubic.harmonized_score = _split_v0
    feat_skew.SOURCES = base + tuple(b + "_v0" for b in base)
    try:
        yield
    finally:
        feat_skew.harmonized_score, feat_cubic.harmonized_score, feat_skew.SOURCES = saved

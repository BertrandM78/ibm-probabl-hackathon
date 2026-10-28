"""H17: hierarchical cubic trajectory c0 + c1 x + c2 x^2 + c3 x^3 under the e012 skew-normal noise.

Basis: x = (age - patient mean age) / XS, shared by all patients (review blocker 1: only c3 is
translation-invariant, so coefficients and prior live in this per-patient centred basis; the patient's
position in the disease enters through the prior mean, via tsd_m in patient_covariates).
Prior: per-patient OLS cubic of target (train only) -> 4 small HGB on covariates (cross-fitted means for
train patients); Sigma = residual cov on patients with >= MIN_N_SIG visits minus mean OLS estimation
noise, robust (MinCovDet), projected PD (review blocker 2). sig2 = pooled cubic misfit of target (elbows), used as e_line.
MAP: vectorised damped Newton in 4D on stacked problems (full + one LOO per visit), as feat_skew.skew_map.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import log_ndtr
from sklearn.covariance import MinCovDet
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

import cv
from feat_harmonized import harmonized_score
from feat_hier import PRIOR_PARAMS, centred_age, patient_covariates
from feat_rater_scale import check_order
from feat_skew import LOG2PI, fit_skew_template
from features import GROUP

XS = 5.0          # age scale of the basis (years)
D = 4             # cubic
MIN_N_SIG = 8     # patients used for Sigma: short windows give unstable c2/c3 (rounding, elbows)
COEF = [f"c{i}" for i in range(D)]


def basis(df: pd.DataFrame) -> np.ndarray:
    x = centred_age(df).to_numpy(float) / XS
    return np.vander(x, D, increasing=True)


def patient_cubic(train: pd.DataFrame, keep: np.ndarray | None = None) -> tuple[pd.DataFrame, np.ndarray, float]:
    """Per-patient OLS cubic of target: coefficients (P x D), (X'X)^-1 (P x D x D), pooled misfit sig2.

    keep: optional row mask (H34: drop censored visits, basis unchanged); ignored for patients with <= D kept rows.
    """
    B, y = basis(train), train["target"].to_numpy(float)
    rows, inv, rss, dof = {}, [], 0.0, 0
    for pid, idx in train.groupby(GROUP, sort=False).indices.items():
        if keep is not None and keep[idx].sum() > D:
            idx = idx[keep[idx]]
        Xp, yp = B[idx], y[idx]
        XtX_inv = np.linalg.pinv(Xp.T @ Xp)
        c = XtX_inv @ Xp.T @ yp
        rows[pid] = c
        inv.append(XtX_inv)
        rss += float(((yp - Xp @ c) ** 2).sum())
        dof += len(idx) - D
    C = pd.DataFrame.from_dict(rows, orient="index", columns=COEF)
    return C, np.stack(inv), rss / dof


def fit_cubic_prior(train: pd.DataFrame, sigma_est: str = "mcd", weighted: bool = False,
                    keep: np.ndarray | None = None) -> dict:
    """Prior models for the 4 coefficients, cross-fitted means for train patients, Sigma, sig2.

    sigma_est: "mcd" (robust, drops elbow patients) or "cov" (plain, denoised). weighted: HGB sample weight
    1 / var(coef | n visits, capped at MIN_N_SIG) - short windows give very noisy OLS c2/c3.
    """
    cov = patient_covariates(train)
    C, inv, sig2 = patient_cubic(train, keep)
    assert list(C.index) == list(cov.index), "patient order mismatch (inv is positional)"
    nb = cov["n"].clip(upper=MIN_N_SIG)
    models, oof = {}, pd.DataFrame(index=cov.index, columns=COEF, dtype=float)
    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=cv.SEED)
    for t in COEF:
        w = (1 / nb.map(C[t].groupby(nb).var())).to_numpy() if weighted else None
        for itr, ite in gkf.split(cov, groups=cov.index):
            m = HistGradientBoostingRegressor(**PRIOR_PARAMS).fit(
                cov.iloc[itr], C[t].iloc[itr], sample_weight=None if w is None else w[itr])
            oof.iloc[ite, oof.columns.get_loc(t)] = m.predict(cov.iloc[ite])
        models[t] = HistGradientBoostingRegressor(**PRIOR_PARAMS).fit(cov, C[t], sample_weight=w)
    big = (cov["n"] >= MIN_N_SIG).to_numpy()
    R = (C - oof).to_numpy()[big]
    raw = MinCovDet(random_state=cv.SEED).fit(R).covariance_ if sigma_est == "mcd" else np.cov(R.T)
    Sig = raw - sig2 * inv[big].mean(axis=0)
    ev, V = np.linalg.eigh((Sig + Sig.T) / 2)
    floor = 1e-3 * np.diag(np.cov(R.T))
    Sig = V @ np.diag(np.maximum(ev, 1e-4)) @ V.T + np.diag(floor)
    print(f"  [cubic] {sigma_est} w={weighted} sig2={sig2:.3f} n_sigma={big.sum()} "
          f"diag(Sigma)={np.round(np.diag(Sig), 2)} rmse_prior={np.round(np.sqrt((R ** 2).mean(axis=0)), 3)}")
    return {"models": models, "prior_train": oof, "Sigma": Sig, "sig2": sig2}


def _prior_means(df, prior, is_train):
    if is_train:
        mu = prior["prior_train"]
    else:
        cov = patient_covariates(df)
        mu = pd.DataFrame({t: prior["models"][t].predict(cov) for t in COEF}, index=cov.index)
    return np.column_stack([df[GROUP].map(mu[t]).to_numpy(float) for t in COEF])


def cubic_map(df, prior, het, tmpl, mu1, is_train, tol=1e-8, max_iter=40, vis=None, return_post=False):
    """Skew-normal MAP of the 4 cubic coefficients, full and LOO (cf. feat_skew.skew_map).

    return_post: also return the full-problem MAP coefficients (P x D) and Laplace covariances (P x D x D).
    """
    check_order(df)
    pid = df[GROUP].to_numpy()
    starts = np.r_[True, pid[1:] != pid[:-1]]
    pat = np.cumsum(starts) - 1
    n_p, start_p = np.bincount(pat), np.flatnonzero(starts)
    N, P = len(df), n_p.size
    mu = _prior_means(df, prior, is_train)                     # N x D
    B = basis(df)
    yv = df["y_v"].to_numpy(float)
    ok = (df["w_v"].to_numpy() > 0) & np.isfinite(yv)
    y = np.where(ok, yv, 0.0)
    _, src = harmonized_score(df)
    sv = het["c0"] + het["c1"] * np.clip(mu1, 0, None)
    par = np.array([tmpl.get(k, (0.0, 1.0, 0.0)) for k in src])
    xi, om, al = par[:, 0], par[:, 1], par[:, 2]
    if vis is not None:                                         # per-visit noise (H58): z = m + s * u
        xi, om = vis[0] + vis[1] * xi, vis[1] * om
    de = al / np.sqrt(1 + al * al)
    om_e = np.sqrt(om * om + prior["sig2"] / sv ** 2)
    de_e = om * de / om_e
    al_e = de_e / np.sqrt(1 - de_e ** 2)
    sc, loc = sv * om_e, sv * xi
    w_g = np.where(ok, 1 / (1 / np.where(ok, df["w_v"], 1.0) + prior["sig2"]), 0.0)
    # stacked rows: problems 0..P-1 = full per patient, P+v = LOO of visit v
    reps = n_p[pat]
    lp = np.repeat(np.arange(N), reps)
    off = np.arange(reps.sum()) - np.repeat(np.cumsum(reps) - reps, reps)
    rows_loo = start_p[pat][lp] + off
    rows = np.r_[np.arange(N), rows_loo]
    prob = np.r_[pat, P + lp]
    m = ok[rows] & np.r_[np.ones(N, bool), rows_loo != lp]
    K = P + N
    pm = np.r_[mu[start_p], mu]                                 # K x D prior means
    Q = np.linalg.inv(prior["Sigma"])
    Br, yr, sr, lr, ar = B[rows], y[rows], sc[rows], loc[rows], al_e[rows]
    mw = m.astype(float)
    iu = np.triu_indices(D)

    def S(v):
        return np.bincount(prob, weights=v * mw, minlength=K)

    def sym(cols):
        H = np.empty((K, D, D))
        for (i, j), h in zip(zip(*iu), cols):
            H[:, i, j] = H[:, j, i] = h
        return H

    # Gaussian posterior as init
    wr = w_g[rows]
    H0 = sym([S(wr * Br[:, i] * Br[:, j]) for i, j in zip(*iu)]) + Q
    r0 = np.column_stack([S(wr * Br[:, i] * yr) for i in range(D)]) + pm @ Q
    c = np.linalg.solve(H0, r0[..., None])[..., 0]

    def terms(c, deriv):
        u = (yr - np.einsum("rd,rd->r", Br, c[prob]) - lr) / sr
        t = ar * u
        lc = log_ndtr(t)
        dc = c - pm
        lpost = S(-0.5 * u * u + lc) - 0.5 * np.einsum("kd,de,ke->k", dc, Q, dc)
        if not deriv:
            return lpost
        M = np.exp(-0.5 * t * t - LOG2PI - lc)
        d1 = (u - ar * M) / sr
        d2 = (-1 - ar * ar * np.clip(M * (t + M), 0, 1)) / sr ** 2
        g = np.column_stack([S(d1 * Br[:, i]) for i in range(D)]) - dc @ Q
        H = sym([S(d2 * Br[:, i] * Br[:, j]) for i, j in zip(*iu)]) - Q
        return lpost, g, H

    act, it = np.ones(K, bool), -1
    for it in range(max_iter):
        lp0, g, H = terms(c, True)
        st = -np.linalg.solve(H, g[..., None])[..., 0]
        st[~act] = 0.0
        for _ in range(30):
            bad = terms(c + st, False) < lp0 - 1e-12
            if not bad.any():
                break
            st[bad] *= 0.5
        c = c + st
        act &= np.abs(st).max(axis=1) > tol
        if not act.any():
            break
    _, _, H = terms(c, True)
    Cov = np.linalg.inv(-H)
    print(f"  [cubic] MAP {'train' if is_train else 'valid'}: {K} problems, {it + 1} iter, "
          f"non-converged={int(act.sum())}")

    def at(k):
        return np.einsum("nd,nd->n", B, c[k]), np.sqrt(np.einsum("nd,nde,ne->n", B, Cov[k], B))

    full, se_full = at(pat)
    loo, se_loo = at(P + np.arange(N))
    cl = c[P:]
    x = B[:, 1]
    out = pd.DataFrame({"hb_post_loo": loo, "hb_post_full": full, "hb_se_loo": se_loo, "hb_se_full": se_full,
                        "hb_slope_loo": (cl[:, 1] + 2 * cl[:, 2] * x + 3 * cl[:, 3] * x * x) / XS,
                        "hb_int_loo": cl[:, 0], "hb_prior": np.einsum("nd,nd->n", B, mu),
                        "hc_curv_loo": cl[:, 2], "hc_cub_loo": cl[:, 3],
                        "hc_curv_full": c[pat, 2], "hc_cub_full": c[pat, 3]}, index=df.index)
    out["hb_prior_gap"] = out["hb_post_loo"] - out["hb_prior"]
    assert np.isfinite(out.to_numpy()).all(), "NaN in cubic MAP output"
    return (out, c[:P], Cov[:P]) if return_post else out


def cubic_hook(train_target: pd.Series, keep_quad: bool = False, **prior_kw):
    """build_e011 hook: replace the hb_* trajectory columns by the cubic skew MAP (keep_quad: keep e012 as q_*)."""
    from feat_skew import skew_map

    def hook(tr, va, ctx):
        raw_tr = ctx["tr_raw"].assign(target=train_target)
        tmpl = fit_skew_template(raw_tr, ctx["het"])
        prior = fit_cubic_prior(raw_tr, **prior_kw)
        mu1_tr, mu1_va = tr["hb_post_full"].to_numpy(), va["hb_post_full"].to_numpy()
        o_tr = cubic_map(tr, prior, ctx["het"], tmpl, mu1_tr, is_train=True)
        o_va = cubic_map(va, prior, ctx["het"], tmpl, mu1_va, is_train=False)
        if keep_quad:
            qa = (ctx["prior"], ctx["kappa"], ctx["het"], tmpl)
            q_tr = skew_map(tr, *qa, mu1_tr, is_train=True).add_prefix("q_")
            q_va = skew_map(va, *qa, mu1_va, is_train=False).add_prefix("q_")
            o_tr, o_va = o_tr.join(q_tr), o_va.join(q_va)
        return tr.assign(**o_tr), va.assign(**o_va)
    return hook

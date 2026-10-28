"""H11: patient trajectory under a skew-normal (location-scale) measurement noise, plus its Gaussian control.

Noise model per source k (off / on_t / on_nt), fitted on train only:
    z = (y_v - target) / (c0 + c1 * target) ~ SN(xi_k, omega_k, alpha_k).
Trajectory: y_v = mu_v + sigma_v * z + e_line, mu_v = a_p + b_p x + kappa x^2, sigma_v = c0 + c1 * mu_pass1
(frozen, no target), e_line ~ N(0, sig2). SN convolved with a Gaussian is SN (exact): omega_eff^2 =
omega^2 + sig2 / sigma_v^2, delta_eff = omega * delta / omega_eff. MAP of (a_p, b_p) under the e007
Gaussian prior by a vectorised damped Newton on stacked problems (full + one LOO per visit).
Control (e012a): Gaussian trajectory on y_v recalibrated per source, y_cal = (y_v - d0) / (1 + d1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr
from sklearn.ensemble import HistGradientBoostingRegressor

import cv
from feat_drug import add_drug_amplitude, add_on_eq_context, fit_drug_curve
from feat_harmonized import add_harmonized, fit_noise_var, fit_slope_prior, harmonized_score
from feat_hier import centred_age, patient_covariates
from feat_hier2 import add_hier2, fit_curvature, fit_hetero, fit_prior2
from feat_rater_scale import add_rater_scale, check_order, fit_rater_scale
from features import GROUP, add_basic, add_patient_context, align_categories, to_model_matrix

SOURCES = ("off", "on_t", "on_nt")
HB_OUT = ("hb_post_loo", "hb_post_full", "hb_se_loo", "hb_se_full", "hb_slope_loo", "hb_int_loo", "hb_prior_gap")
LOG2PI = 0.5 * np.log(2 * np.pi)


# ---------------------------------------------------------------- e011 pipeline with a trajectory hook

def build_e011(train, valid, hook=None):
    """e009 features (e011 frame); hook(tr, va, ctx) -> (tr, va) may overwrite the HB_OUT columns."""
    s, s_global = fit_rater_scale(train)
    g_params, A0, gbar = fit_drug_curve(train)

    def base(df):
        df = add_rater_scale(add_basic(df), s, s_global)
        df = add_drug_amplitude(df, g_params, A0, gbar, k=0.5)
        df = add_patient_context(df, cols=("on", "off", "ledd", "on_n", "off_n"))
        return add_on_eq_context(df, "on_eq_n")

    tr, va = base(train), base(valid)
    var = fit_noise_var(tr)
    b0, lam = fit_slope_prior(tr, var)
    tr, va = add_harmonized(tr, var, b0, lam), add_harmonized(va, var, b0, lam)
    kappa, het = fit_curvature(tr), fit_hetero(tr)
    prior = fit_prior2(tr, kappa)
    ctx = dict(kappa=kappa, het=het, prior=prior, var=var, tr_raw=tr, va_raw=va)
    tr = add_hier2(tr, prior, kappa, het, is_train=True)
    va = add_hier2(va, prior, kappa, het, is_train=False)
    if hook is not None:
        tr, va = hook(tr, va, ctx)
    return tr, va, kappa


def hgb_predict(tr, va, y):
    X_tr = to_model_matrix(tr)
    X_va = align_categories(X_tr, to_model_matrix(va))
    model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, early_stopping=False,
                                          random_state=cv.SEED)
    return np.clip(model.fit(X_tr, y).predict(X_va), 0, 132)


# ---------------------------------------------------------------- e012a: per-source mean recalibration

def fit_calib(train: pd.DataFrame) -> dict:
    """(d0, d1) per source: OLS of y_v - target on target (train only)."""
    y, src = harmonized_score(train)
    out = {}
    for k in SOURCES:
        m = (src == k) & y.notna()
        d1, d0 = np.polyfit(train.loc[m, "target"], (y - train["target"])[m], 1)
        out[k] = (float(d0), float(d1))
    return out


def apply_calib(df: pd.DataFrame, cal: dict) -> pd.Series:
    y, src = harmonized_score(df)
    d0 = src.map({k: v[0] for k, v in cal.items()}).astype(float)
    d1 = src.map({k: v[1] for k, v in cal.items()}).astype(float)
    return (y - d0) / (1 + d1)


def fit_hetero_y(y: pd.Series, target: pd.Series) -> dict:
    """fit_hetero on an arbitrary y_v (same decile fit and bounds)."""
    d = pd.DataFrame({"e": y - target, "t": target}).dropna()
    b = d.groupby(pd.qcut(d["t"], 10), observed=True).agg(t=("t", "mean"), sd=("e", "std"))
    c1, c0 = np.polyfit(b["t"], b["sd"], 1)
    c0, c1 = max(c0, 1.0), max(c1, 0.0)
    f_mean = float(np.mean(((c0 + c1 * d["t"].clip(lower=0)) / (c0 + c1 * d["t"].mean())) ** 2))
    return {"c0": c0, "c1": c1, "t_mean": float(d["t"].mean()), "f_mean": f_mean}


def meancal_hook(train_target: pd.Series):
    def hook(tr, va, ctx):
        raw_tr, raw_va = ctx["tr_raw"], ctx["va_raw"]
        cal = fit_calib(raw_tr)
        yc_tr, yc_va = apply_calib(raw_tr, cal), apply_calib(raw_va, cal)
        _, src = harmonized_score(raw_tr)
        var = {k: float(((yc_tr - train_target) ** 2)[src == k].mean()) for k in SOURCES}
        het = fit_hetero_y(yc_tr, train_target)
        print(f"  calib (d0,d1)={ {k: tuple(round(x, 3) for x in v) for k, v in cal.items()} } "
              f"var={ {k: round(v, 1) for k, v in var.items()} } c0={het['c0']:.2f} c1={het['c1']:.3f}")

        def redo(df, raw, yc, is_train):
            _, s = harmonized_score(raw)
            w = s.map({k: 1 / var[k] for k in SOURCES}).fillna(0.0).astype(float)
            h = add_hier2(raw.assign(y_v=yc, w_v=w), ctx["prior"], ctx["kappa"], het, is_train)
            return df.assign(**{c: h[c] for c in HB_OUT})

        return redo(tr, raw_tr, yc_tr, True), redo(va, raw_va, yc_va, False)
    return hook


# ---------------------------------------------------------------- skew-normal template

def _sn_nll(p, z):
    xi, lw, a = p
    w = np.exp(lw)
    t = (z - xi) / w
    lc = log_ndtr(a * t)
    M = np.exp(-0.5 * (a * t) ** 2 - LOG2PI - lc)
    nll = -np.mean(-lw - 0.5 * t * t + lc)
    g = -np.array([np.mean((t - a * M) / w), np.mean(-1 + t * t - a * M * t), np.mean(M * t)])
    return nll, g


def sn_moments(z: np.ndarray) -> tuple[float, float, float]:
    m, s = z.mean(), z.std()
    g = np.clip(np.mean(((z - m) / s) ** 3), -0.99, 0.99)
    r = abs(g) ** (2 / 3)
    d = np.sign(g) * np.sqrt(np.pi / 2 * r / (r + ((4 - np.pi) / 2) ** (2 / 3)))
    w = s / np.sqrt(1 - 2 * d * d / np.pi)
    return float(m - w * d * np.sqrt(2 / np.pi)), float(w), float(np.clip(d / np.sqrt(1 - d * d), -10, 10))


def sn_fit(z: np.ndarray) -> tuple[float, float, float]:
    """MLE of SN(xi, omega, alpha), |alpha| <= 10; Gaussian fallback (alpha = 0) if it fails."""
    xi, w, a = sn_moments(z)
    r = minimize(_sn_nll, [xi, np.log(w), a], args=(z,), jac=True, method="L-BFGS-B",
                 bounds=[(None, None), (np.log(1e-3), None), (-10, 10)])
    if not r.success or not np.isfinite(r.fun):
        print(f"  [skew] SN fit failed ({r.message}) -> Gaussian fallback")
        return float(z.mean()), float(z.std()), 0.0
    return float(r.x[0]), float(np.exp(r.x[1])), float(r.x[2])


def fit_skew_template(train: pd.DataFrame, het: dict) -> dict:
    y, src = harmonized_score(train)
    t = train["target"]
    z = (y - t) / (het["c0"] + het["c1"] * t.clip(lower=0))
    out = {}
    for k in SOURCES:
        zk = z[(src == k) & z.notna()].to_numpy(float)
        out[k] = sn_fit(zk)
        sk = np.mean(((zk - zk.mean()) / zk.std()) ** 3)
        print(f"  [skew] {k:5s} n={zk.size} skew(z)={sk:+.2f} xi={out[k][0]:+.3f} "
              f"omega={out[k][1]:.3f} alpha={out[k][2]:+.2f}")
    return out


# ---------------------------------------------------------------- vectorised MAP

def _prior_means(df, prior, is_train):
    if is_train:
        mu = prior["prior_train"]
    else:
        cov = patient_covariates(df)
        mu = pd.DataFrame({t: prior["models"][t].predict(cov) for t in ("a", "b")}, index=cov.index)
    return df[GROUP].map(mu["a"]).to_numpy(float), df[GROUP].map(mu["b"]).to_numpy(float)


def skew_map(df, prior, kappa, het, tmpl, mu1, is_train, tol=1e-8, max_iter=25, vis=None):
    """Skew-normal MAP trajectory, full and LOO, for any frame (is_train only selects the prior).

    df: e011 frame (y_v, w_v = Gaussian pass-2 weight); mu1: Gaussian pass-2 hb_post_full (no target).
    Returns the HB_OUT columns as a DataFrame on df.index.
    """
    check_order(df)
    pid = df[GROUP].to_numpy()
    starts = np.r_[True, pid[1:] != pid[:-1]]
    pat = np.cumsum(starts) - 1
    n_p, start_p = np.bincount(pat), np.flatnonzero(starts)
    N, P = len(df), n_p.size
    mu_a, mu_b = _prior_means(df, prior, is_train)
    x = centred_age(df).to_numpy(float)
    yv = df["y_v"].to_numpy(float)
    ok = (df["w_v"].to_numpy() > 0) & np.isfinite(yv)
    y = np.where(ok, yv - kappa * x * x, 0.0)
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
    pa = np.r_[mu_a[start_p], mu_a]
    pb = np.r_[mu_b[start_p], mu_b]
    Q = np.linalg.inv(prior["Sigma"])
    xr, yr, sr, lr, ar = x[rows], y[rows], sc[rows], loc[rows], al_e[rows]
    mw = m.astype(float)

    def S(v):
        return np.bincount(prob, weights=v * mw, minlength=K)

    # Gaussian pass-2 posterior as init (same weights as e011)
    wr = w_g[rows]
    s0, s1, s2, t0, t1 = S(wr), S(wr * xr), S(wr * xr * xr), S(wr * yr), S(wr * xr * yr)
    h11, h12, h22 = Q[0, 0] + s0, Q[0, 1] + s1, Q[1, 1] + s2
    r1 = Q[0, 0] * pa + Q[0, 1] * pb + t0
    r2 = Q[0, 1] * pa + Q[1, 1] * pb + t1
    det = h11 * h22 - h12 ** 2
    a, b = (h22 * r1 - h12 * r2) / det, (h11 * r2 - h12 * r1) / det

    def terms(a, b, deriv):
        u = (yr - a[prob] - b[prob] * xr - lr) / sr
        t = ar * u
        lc = log_ndtr(t)
        da, db = a - pa, b - pb
        lpost = S(-0.5 * u * u + lc) - 0.5 * (Q[0, 0] * da * da + 2 * Q[0, 1] * da * db + Q[1, 1] * db * db)
        if not deriv:
            return lpost
        M = np.exp(-0.5 * t * t - LOG2PI - lc)
        d1 = (u - ar * M) / sr
        d2 = (-1 - ar * ar * np.clip(M * (t + M), 0, 1)) / sr ** 2
        g = (S(d1) - Q[0, 0] * da - Q[0, 1] * db, S(d1 * xr) - Q[0, 1] * da - Q[1, 1] * db)
        H = (S(d2) - Q[0, 0], S(d2 * xr) - Q[0, 1], S(d2 * xr * xr) - Q[1, 1])
        return lpost, g, H

    act, it = np.ones(K, bool), -1
    for it in range(max_iter):
        lp0, (ga, gb), (haa, hab, hbb) = terms(a, b, True)
        det = haa * hbb - hab ** 2
        sa, sb = -(hbb * ga - hab * gb) / det, -(haa * gb - hab * ga) / det
        sa, sb = np.where(act, sa, 0.0), np.where(act, sb, 0.0)
        step = 1.0
        for _ in range(30):
            na, nb = a + step * sa, b + step * sb
            bad = terms(na, nb, False) < lp0 - 1e-12
            if not bad.any():
                break
            sa, sb = np.where(bad, 0.5 * sa, sa), np.where(bad, 0.5 * sb, sb)
        a, b = a + sa, b + sb
        act &= np.maximum(np.abs(sa), np.abs(sb)) > tol
        if not act.any():
            break
    _, _, (haa, hab, hbb) = terms(a, b, True)
    det = haa * hbb - hab ** 2
    print(f"  [skew] MAP {'train' if is_train else 'valid'}: {K} problems, {it + 1} iter, "
          f"non-converged={int(act.sum())}")

    def at(k, xv):
        var = (-hbb[k] + 2 * xv * hab[k] - xv * xv * haa[k]) / det[k]
        return a[k] + b[k] * xv + kappa * xv * xv, np.sqrt(var)

    full, se_full = at(pat, x)
    loo, se_loo = at(P + np.arange(N), x)
    out = pd.DataFrame({"hb_post_loo": loo, "hb_post_full": full, "hb_se_loo": se_loo, "hb_se_full": se_full,
                        "hb_slope_loo": b[P:], "hb_int_loo": a[P:]}, index=df.index)
    out["hb_prior_gap"] = out["hb_post_loo"] - df["hb_prior"].to_numpy()
    assert np.isfinite(out.to_numpy()).all(), "NaN in skew MAP output"
    return out


def skew_hook(train_target: pd.Series, log: dict | None = None):
    def hook(tr, va, ctx):
        tmpl = fit_skew_template(ctx["tr_raw"].assign(target=train_target), ctx["het"])
        if log is not None:
            log["tmpl"] = tmpl
        args = (ctx["prior"], ctx["kappa"], ctx["het"], tmpl)
        o_tr = skew_map(tr, *args, tr["hb_post_full"].to_numpy(), is_train=True)
        o_va = skew_map(va, *args, va["hb_post_full"].to_numpy(), is_train=False)
        return tr.assign(**o_tr), va.assign(**o_va)
    return hook

"""H05: harmonised per-visit score weighted by noise variance, pooled into a patient trajectory (extends e005)."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import cv
from feat_drug import add_drug_amplitude, add_on_eq_context, fit_drug_curve
from feat_harmonized import add_harmonized, fit_noise_var, fit_slope_prior
from feat_rater_scale import add_rater_scale, fit_rater_scale
from features import add_basic, add_patient_context, align_categories, to_model_matrix

DESCRIPTION = "H05 : score harmonisé y_v (off/s sinon on_eq/s, off==0/on==0 exclus) pondéré 1/var (3 variances fittées sur train) → moyenne patient LOO + trajectoire WLS ridge sur l'âge"
FEATURES = "e005 + y_v, w_v, h_wmean_loo/full, h_wsum_loo, h_slope_loo, h_traj_loo/full, h_traj_se, y_v/w_v prev/next"


def build_base(df, s, s_global, g_params, A0, gbar):
    df = add_rater_scale(add_basic(df), s, s_global)
    df = add_drug_amplitude(df, g_params, A0, gbar, k=0.5)
    df = add_patient_context(df, cols=("on", "off", "ledd", "on_n", "off_n"))
    return add_on_eq_context(df, "on_eq_n")


def fit_predict(train, valid):
    s, s_global = fit_rater_scale(train)
    g_params, A0, gbar = fit_drug_curve(train)
    tr = build_base(train, s, s_global, g_params, A0, gbar)
    va = build_base(valid, s, s_global, g_params, A0, gbar)
    var = fit_noise_var(tr)
    b0, lam = fit_slope_prior(tr, var)
    print(f"  var={ {k: round(v, 1) for k, v in var.items()} } rmse={ {k: round(v ** .5, 2) for k, v in var.items()} } b0={b0:.3f} lam={lam:.3f}")
    X_tr = to_model_matrix(add_harmonized(tr, var, b0, lam))
    X_va = align_categories(X_tr, to_model_matrix(add_harmonized(va, var, b0, lam)))
    model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, early_stopping=False,
                                          random_state=cv.SEED)
    model.fit(X_tr, train["target"])
    return np.clip(model.predict(X_va), 0, 132)

"""H06: hierarchical linear trajectory (covariate prior + conjugate Gaussian posterior) on top of e006."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import cv
from feat_drug import add_drug_amplitude, add_on_eq_context, fit_drug_curve
from feat_harmonized import add_harmonized, fit_noise_var, fit_slope_prior
from feat_hier import add_hier, fit_prior
from feat_rater_scale import add_rater_scale, fit_rater_scale
from features import add_basic, add_patient_context, align_categories, to_model_matrix

OFFSET = False  # variant B: HGB on target - clip(hb_post_full, 0)

DESCRIPTION = "H06 : trajectoire hiérarchique a_p + b_p·(âge − âge moyen), prior (a, b) par HGB sur covariables patient (cross-fit), posterior gaussien 2×2 LOO/full sur y_v (var + σ²_ligne)"
FEATURES = "e006 + hb_prior, hb_prior_slope, hb_post_loo/full, hb_se_loo/full, hb_slope_loo, hb_int_loo, hb_prior_gap"


def build_base(df, s, s_global, g_params, A0, gbar):
    """Same as e006."""
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
    tr, va = add_harmonized(tr, var, b0, lam), add_harmonized(va, var, b0, lam)
    prior = fit_prior(tr)
    print(f"  sig_line={prior['sig2'] ** .5:.2f} Sigma={np.round(prior['Sigma'], 2).tolist()} "
          f"prior rmse a/b={np.round(prior['rmse_prior'], 2).tolist()} sd a/b={np.round(prior['sd_ab'], 2).tolist()}")
    tr, va = add_hier(tr, prior, is_train=True), add_hier(va, prior, is_train=False)
    X_tr = to_model_matrix(tr)
    X_va = align_categories(X_tr, to_model_matrix(va))
    off_tr = np.clip(tr["hb_post_full"].to_numpy(), 0, None) if OFFSET else 0.0
    off_va = np.clip(va["hb_post_full"].to_numpy(), 0, None) if OFFSET else 0.0
    model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, early_stopping=False,
                                          random_state=cv.SEED)
    model.fit(X_tr, train["target"] - off_tr)
    return np.clip(off_va + model.predict(X_va), 0, 132)

"""H09: e007 trajectory with population curvature and level-dependent noise weights (two-pass)."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import cv
from feat_drug import add_drug_amplitude, add_on_eq_context, fit_drug_curve
from feat_harmonized import add_harmonized, fit_noise_var, fit_slope_prior
from feat_hier2 import add_hier2, fit_curvature, fit_hetero, fit_prior2
from feat_rater_scale import add_rater_scale, fit_rater_scale
from features import add_basic, add_patient_context, align_categories, to_model_matrix

CURV, HETERO = True, True

DESCRIPTION = "H09 : trajectoire v2 = e007 + courbure de population κ·x² (fit train) + poids hétéroscédastiques sd(y_v) = c0 + c1·niveau (2 passes, μ sans target)"
FEATURES = "e007 (hb_* recalculés) + hb_level_f"


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
    kappa = fit_curvature(tr) if CURV else 0.0
    het = fit_hetero(tr) if HETERO else None
    prior = fit_prior2(tr, kappa)
    print(f"  kappa={kappa:.4f} het={het and {k: round(v, 3) for k, v in het.items()}} "
          f"sig_line={prior['sig2'] ** .5:.2f} prior rmse a/b={np.round(prior['rmse_prior'], 2).tolist()}")
    tr = add_hier2(tr, prior, kappa, het, is_train=True)
    va = add_hier2(va, prior, kappa, het, is_train=False)
    print(f"  traj-only train rmse hb_post_full={np.sqrt(np.mean((tr['hb_post_full'] - train['target']) ** 2)):.3f} "
          f"hb_post_loo={np.sqrt(np.mean((tr['hb_post_loo'] - train['target']) ** 2)):.3f}")
    X_tr = to_model_matrix(tr)
    X_va = align_categories(X_tr, to_model_matrix(va))
    model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, early_stopping=False,
                                          random_state=cv.SEED)
    model.fit(X_tr, train["target"])
    return np.clip(model.predict(X_va), 0, 132)

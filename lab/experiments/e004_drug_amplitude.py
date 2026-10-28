"""H01: e003 + patient drug amplitude A_p on a common Emax(Bateman) curve g(tsi_on), on_eq features."""
from sklearn.ensemble import HistGradientBoostingRegressor

import cv
from feat_drug import add_drug_amplitude, add_on_eq_context, fit_drug_curve
from feat_rater_scale import add_rater_scale, fit_rater_scale
from features import add_basic, add_patient_context, align_categories, to_model_matrix

DESCRIPTION = "H01 : amplitude médicament par patient A_p (LOO, shrink k=0.5) sur courbe Emax(Bateman) g(tsi_on) fittée côté ON, on_eq"
FEATURES = "e003 + A_p, g_on, n_av, on_eq, on_eq_n (=on_eq/s_rater) + prev/next/pmean de on_eq_n"

LOG_PARAMS = True


def build(df, s, s_global, g_params, A0, gbar):
    df = add_rater_scale(add_basic(df), s, s_global)
    df = add_drug_amplitude(df, g_params, A0, gbar, k=0.5)
    df = add_patient_context(df, cols=("on", "off", "ledd", "on_n", "off_n"))
    df = add_on_eq_context(df, "on_eq_n")
    return to_model_matrix(df)


def fit_predict(train, valid):
    s, s_global = fit_rater_scale(train)
    g_params, A0, gbar = fit_drug_curve(train)
    if LOG_PARAMS:
        print(f"  g[{g_params['kind']}] p={g_params['p'].round(4)} emax={g_params['emax']:.4f} A0={A0:.4f} gbar={gbar:.4f}")
    X_tr = build(train, s, s_global, g_params, A0, gbar)
    X_va = align_categories(X_tr, build(valid, s, s_global, g_params, A0, gbar))
    model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=cv.SEED)
    model.fit(X_tr, train["target"])
    return model.predict(X_va)

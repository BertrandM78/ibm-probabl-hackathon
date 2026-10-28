"""e004 with HGB early stopping disabled (no visit-level internal split), max_iter explicit."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import cv
from feat_drug import add_drug_amplitude, add_on_eq_context, fit_drug_curve
from feat_rater_scale import add_rater_scale, fit_rater_scale
from features import add_basic, add_patient_context, align_categories, to_model_matrix

DESCRIPTION = "e004 sans early stopping HGB (pas de split interne par visite), max_iter=500 explicite, prédictions clip [0, 132]"
FEATURES = "identiques à e004"


def build(df, s, s_global, g_params, A0, gbar):
    df = add_rater_scale(add_basic(df), s, s_global)
    df = add_drug_amplitude(df, g_params, A0, gbar, k=0.5)
    df = add_patient_context(df, cols=("on", "off", "ledd", "on_n", "off_n"))
    df = add_on_eq_context(df, "on_eq_n")
    return to_model_matrix(df)


def fit_predict(train, valid):
    s, s_global = fit_rater_scale(train)
    g_params, A0, gbar = fit_drug_curve(train)
    X_tr = build(train, s, s_global, g_params, A0, gbar)
    X_va = align_categories(X_tr, build(valid, s, s_global, g_params, A0, gbar))
    model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, early_stopping=False,
                                          random_state=cv.SEED)
    model.fit(X_tr, train["target"])
    return np.clip(model.predict(X_va), 0, 132)

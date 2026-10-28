"""H04: e002 + rater-normalised on/off (common factor s_r) and their patient context."""
from sklearn.ensemble import HistGradientBoostingRegressor

import cv
from feat_rater_scale import add_rater_scale, fit_rater_scale
from features import add_basic, add_patient_context, align_categories, to_model_matrix

DESCRIPTION = "H04 : facteur rater multiplicatif commun on/off (median off/target, shrink 20), normalisé avant contexte"
FEATURES = "e002 (bruts + contexte) + on_n/off_n prev/next/pmean + s_rater + s_pmean"


def build(df, s, s_global):
    df = add_rater_scale(add_basic(df), s, s_global)
    df = add_patient_context(df, cols=("on", "off", "ledd", "on_n", "off_n"))
    return to_model_matrix(df)


def fit_predict(train, valid):
    s, s_global = fit_rater_scale(train)
    X_tr = build(train, s, s_global)
    X_va = align_categories(X_tr, build(valid, s, s_global))
    model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=cv.SEED)
    model.fit(X_tr, train["target"])
    return model.predict(X_va)

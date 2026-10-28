"""HGB + disease duration, missingness flags and per-patient prev/next/mean context."""
from sklearn.ensemble import HistGradientBoostingRegressor

import cv
from features import add_basic, add_patient_context, align_categories, to_model_matrix

DESCRIPTION = "HGB + features de base + contexte patient (prev/next/mean)"
FEATURES = "brutes + durée maladie + flags NaN + on/off/ledd prev/next/pmean"


def build(df):
    return to_model_matrix(add_patient_context(add_basic(df)))


def fit_predict(train, valid):
    X_tr = build(train)
    X_va = align_categories(X_tr, build(valid))
    model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=cv.SEED)
    model.fit(X_tr, train["target"])
    return model.predict(X_va)

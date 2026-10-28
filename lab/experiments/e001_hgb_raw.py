"""HistGradientBoosting on raw columns, NaNs kept as signal."""
from sklearn.ensemble import HistGradientBoostingRegressor

import cv
from features import align_categories, to_model_matrix

DESCRIPTION = "HGB sur colonnes brutes, NaN natifs"
FEATURES = "brutes + cohort/gene/rater_id en category"


def fit_predict(train, valid):
    X_tr = to_model_matrix(train)
    X_va = align_categories(X_tr, to_model_matrix(valid))
    model = HistGradientBoostingRegressor(random_state=cv.SEED)
    model.fit(X_tr, train["target"])
    return model.predict(X_va)

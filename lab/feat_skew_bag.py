"""H14: seed-bagged HGB on top of the e012 skew-trajectory feature build (imports feat_skew, unchanged)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import cv
from features import align_categories, to_model_matrix


def bag_hgb_predict(tr, va, y, n_models: int) -> np.ndarray:
    """Average of n_models HistGradientBoostingRegressor with different random_state (same otherwise)."""
    X_tr = to_model_matrix(tr)
    X_va = align_categories(X_tr, to_model_matrix(va))
    preds = np.zeros(len(va))
    for i in range(n_models):
        model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, early_stopping=False,
                                              random_state=cv.SEED + i)
        preds += model.fit(X_tr, y).predict(X_va)
    return np.clip(preds / n_models, 0, 132)

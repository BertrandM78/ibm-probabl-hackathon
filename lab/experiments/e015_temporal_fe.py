"""Pure tabular pivot: HGB on raw columns + a massive generic temporal block (lags, leads, rolling, deltas, trends)."""
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).parent))
import cv  # noqa: E402
from feat_temporal import add_temporal  # noqa: E402
from features import add_basic, align_categories, to_model_matrix  # noqa: E402

DESCRIPTION = "Pivot tabulaire pur : HGB sur brutes + bloc temporel générique massif (lags t-1/t-2, leads t+1/t+2, deltas, rolling 3/5, moyennes passées/futures, stats et tendance OLS par patient), sans module pharmaco/rater/trajectoire"
FEATURES = "add_basic + add_temporal (≈ 220 colonnes t_*) sur on/off/ledd/tsi_on/tsi_off/gap/ratio"


def build(df):
    return to_model_matrix(add_temporal(add_basic(df)))


def hgb(X_tr, y, X_va):
    model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, early_stopping=False,
                                          random_state=cv.SEED)
    return model.fit(X_tr, y).predict(X_va)


def fit_predict(train, valid):
    X_tr = build(train)
    X_va = align_categories(X_tr, build(valid))
    return np.clip(hgb(X_tr, train["target"], X_va), 0, 132)

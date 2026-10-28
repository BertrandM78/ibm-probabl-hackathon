"""e017 with stronger HGB: bigger trees, column subsampling (max_features) and a 3-seed bag for level and delta models."""
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).parent))
import cv  # noqa: E402
from e012_traj_skew import build as build_e012  # noqa: E402
from feat_temporal import add_temporal, delta_target, reconstruct  # noqa: E402
from features import align_categories, to_model_matrix  # noqa: E402

LAM = 4.0
N_BAG = 3      # max_features makes random_state matter (unlike e014)
HGB = dict(max_iter=1500, learning_rate=0.03, max_leaf_nodes=127, min_samples_leaf=80,
           max_features=0.3, early_stopping=False)

DESCRIPTION = "e017 avec HGB v2 (127 feuilles, min_leaf 80, max_features 0.3, lr 0.03 × 1500) + bag 3 seeds, niveau et delta, reconstruction λ = 4"
FEATURES = "identiques à e017 ; 2 × 3 HGB (niveau / delta)"


def fit_predict(train, valid):
    train, valid = train.reset_index(drop=True), valid.reset_index(drop=True)
    tr, va, _ = build_e012(train, valid)
    tr, va = add_temporal(tr), add_temporal(va)
    X_tr = to_model_matrix(tr)
    X_va = align_categories(X_tr, to_model_matrix(va))
    y = train["target"]
    d = delta_target(train, y)
    m = d.notna().to_numpy()
    A = np.clip(_bag(X_tr, y, X_va), 0, 132)
    D = _bag(X_tr[m], d[m], X_va)
    return np.clip(reconstruct(valid, A, D, LAM), 0, 132)


def _bag(X, y, X_va):
    return np.mean([HistGradientBoostingRegressor(**HGB, random_state=cv.SEED + s).fit(X, y).predict(X_va)
                    for s in range(N_BAG)], axis=0)

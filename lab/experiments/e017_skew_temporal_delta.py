"""e012 + generic temporal block + delta target: level HGB and delta HGB, per-patient reconstruction, then projection."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).parent))
import cv  # noqa: E402
from e010_traj_v2_proj import project  # noqa: E402
from e012_traj_skew import build as build_e012  # noqa: E402
from feat_skew import hgb_predict  # noqa: E402
from feat_temporal import add_temporal, delta_target, reconstruct  # noqa: E402
from features import align_categories, to_model_matrix  # noqa: E402

LAM = 4.0      # e016 inner choice (fixed, no inner loop on the e012 build)
W_PROJ = 0.0   # fold-0 diagnostic: projection on top of the reconstruction hurts (2.986 -> 3.049)
DUMP = os.environ.get("E017_DUMP")  # optional dir: save components per call for post-hoc variant analysis

DESCRIPTION = "e012 + bloc temporel e015 + cible delta : HGB niveau + HGB delta (target_t − target_{t−1}), reconstruction par patient λ = 4 fixe (remplace la projection OLS : w = 0)"
FEATURES = "e012 + add_temporal (t_*) ; 2 HGB (niveau / delta)"


def fit_predict(train, valid):
    train, valid = train.reset_index(drop=True), valid.reset_index(drop=True)
    tr, va, kappa = build_e012(train, valid)
    tr, va = add_temporal(tr), add_temporal(va)
    y = train["target"]
    d = delta_target(train, y)
    m = d.notna().to_numpy()
    A = hgb_predict(tr, va, y)
    D = _hgb_delta(tr, va, d, m)
    L = reconstruct(valid, A, D, LAM)
    pred = np.clip(L + W_PROJ * (project(valid, L, kappa) - L), 0, 132)
    if DUMP:
        comp = pd.DataFrame({"Index": valid["Index"], "A": A, "D": D, "L": L,
                             "A_proj": A + W_PROJ * (project(valid, A, kappa) - A), "pred": pred})
        Path(DUMP).mkdir(parents=True, exist_ok=True)
        comp.to_csv(Path(DUMP) / f"comp_{len(os.listdir(DUMP))}.csv", index=False)
    return pred


def _hgb_delta(tr, va, d, m):
    """Delta HGB (no [0, 132] clip: deltas are signed), trained on non-first visits only."""
    X_tr = to_model_matrix(tr)
    X_va = align_categories(X_tr, to_model_matrix(va))
    model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, early_stopping=False,
                                          random_state=cv.SEED)
    return model.fit(X_tr[m], d[m]).predict(X_va)

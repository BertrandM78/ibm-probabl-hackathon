"""Delta target: HGB learns target_t - target_{t-1}; levels rebuilt per patient by least squares with e015 levels.

At test time target_{t-1} is unknown, so the delta model alone cannot give a level. We combine
A_t (e015 level model) and D_t (delta model) per patient:
    L = argmin sum (L_t - A_t)^2 + lam * sum_{t>0} (L_t - L_{t-1} - D_t)^2,
lam chosen on an inner patient-grouped cross-fit on train (lam -> inf = patient mean of A + cumulated D).
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
import cv  # noqa: E402
from e015_temporal_fe import build, hgb  # noqa: E402
from feat_temporal import delta_target, reconstruct  # noqa: E402
from features import GROUP, align_categories  # noqa: E402

LAMS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 1e4)
INNER = 3

DESCRIPTION = "Cible alternative target_delta (target_t − target_{t−1}) apprise par HGB sur le bloc temporel e015, niveaux reconstruits par patient (moindres carrés niveaux e015 + deltas, λ par cross-fit interne)"
FEATURES = "identiques à e015 ; 2 HGB (niveau / delta), reconstruction tridiagonale par patient"


def level_delta(X_tr, train, X_va):
    y = train["target"]
    d = delta_target(train, y)
    m = d.notna().to_numpy()
    return hgb(X_tr, y, X_va), hgb(X_tr[m], d[m], X_va)


def pick_lam(X_tr, train):
    A = np.empty(len(train))
    D = np.empty(len(train))
    gkf = GroupKFold(n_splits=INNER, shuffle=True, random_state=cv.SEED)
    for i_tr, i_va in gkf.split(X_tr, groups=train[GROUP]):
        A[i_va], D[i_va] = level_delta(X_tr.iloc[i_tr], train.iloc[i_tr], X_tr.iloc[i_va])
    y = train["target"].to_numpy()
    scores = {lam: cv.rmse(y, reconstruct(train, A, D, lam)) for lam in LAMS}
    dt = delta_target(train, train["target"]).to_numpy()
    m = ~np.isnan(dt)
    print(f"[e016] inner: delta RMSE {cv.rmse(dt[m], D[m]):.3f} (sd {dt[m].std():.3f}) ; "
          + " ".join(f"λ={k:g}:{v:.4f}" for k, v in scores.items()), file=sys.stderr)
    best = min(scores, key=scores.get)
    smooth = {lam: cv.rmse(y, reconstruct(train, A, np.zeros_like(D), lam)) for lam in LAMS}
    print("[e016] control D=0 (pure smoothing): " + " ".join(f"λ={k:g}:{v:.4f}" for k, v in smooth.items()),
          file=sys.stderr)
    return best


def fit_predict(train, valid):
    train, valid = train.reset_index(drop=True), valid.reset_index(drop=True)
    X_tr = build(train)
    X_va = align_categories(X_tr, build(valid))
    lam = pick_lam(X_tr, train)
    A, D = level_delta(X_tr, train, X_va)
    print(f"[e016] λ = {lam:g}", file=sys.stderr)
    return np.clip(reconstruct(valid, A, D, lam), 0, 132)

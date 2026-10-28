"""H23: bagged MLP (level + delta, reconstruction) on the e022b feature frame, as a blend partner for e022b."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, str(Path(__file__).parent))
import cv  # noqa: E402
from e017_skew_temporal_delta import LAM  # noqa: E402
from e019_long_surp_shape import BLOCKS  # noqa: E402
from feat_cubic import cubic_hook  # noqa: E402
from feat_long import add_long, shift_report  # noqa: E402
from feat_skew import build_e011  # noqa: E402
from feat_temporal import add_temporal, delta_target, reconstruct  # noqa: E402
from features import CATEGORICALS, to_model_matrix  # noqa: E402

N_SEEDS = 5
MLP_PARAMS = dict(hidden_layer_sizes=(256, 128), alpha=1e-2, batch_size=256, max_iter=80)
Z_CLIP = 6.0

DESCRIPTION = ("H23 : MLP (256-128, α 1e-2, 80 epochs) bagué 5 seeds niveau + 5 seeds delta sur les features e022b "
               "(one-hot, flags NaN, imputation médiane, standardisation, clip ±6, fit train), reconstruction λ = 4")
FEATURES = "e022b (dense) ; 2 × 5 MLPRegressor (niveau / delta)"


def dense(tr, va):
    """Dense standardised matrices; every statistic fitted on train."""
    X_tr, X_va = to_model_matrix(tr), to_model_matrix(va)
    cats = [c for c in CATEGORICALS if c in X_tr]
    oh = []
    for X in (X_tr, X_va):
        d = pd.concat([pd.get_dummies(pd.Categorical(X[c].astype(str), categories=sorted(X_tr[c].astype(str).unique())),
                                      prefix=c, dtype=float).set_index(X.index) for c in cats], axis=1)
        oh.append(d)
    num_tr = X_tr.drop(columns=cats).astype(float).replace([np.inf, -np.inf], np.nan)
    num_va = X_va.drop(columns=cats).astype(float).replace([np.inf, -np.inf], np.nan)[num_tr.columns]
    na_cols = num_tr.columns[num_tr.isna().any()]
    med, out = num_tr.median(), []
    for num, o in ((num_tr, oh[0]), (num_va, oh[1])):
        flags = num[na_cols].isna().astype(float).add_suffix("_na")
        out.append(pd.concat([num.fillna(med).fillna(0.0), flags, o], axis=1))
    mu, sd = out[0].mean(), out[0].std().replace(0, 1.0)
    return [np.clip(((o - mu) / sd).to_numpy(np.float32), -Z_CLIP, Z_CLIP) for o in out]


def mlp_bag(X_tr, y, X_va):
    m, s = y.mean(), y.std()
    preds = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for i in range(N_SEEDS):
            net = MLPRegressor(random_state=cv.SEED + i, **MLP_PARAMS).fit(X_tr, (y - m) / s)
            preds.append(net.predict(X_va) * s + m)
    return np.mean(preds, axis=0)


def fit_predict(train, valid):
    train, valid = train.reset_index(drop=True), valid.reset_index(drop=True)
    tr, va, _ = build_e011(train, valid, cubic_hook(train["target"], keep_quad=True))
    shift_report(tr, va)
    tr, va = add_long(add_temporal(tr), BLOCKS), add_long(add_temporal(va), BLOCKS)
    X_tr, X_va = dense(tr, va)
    y = train["target"].to_numpy(float)
    d = delta_target(train, train["target"]).to_numpy(float)
    m = np.isfinite(d)
    A = np.clip(mlp_bag(X_tr, y, X_va), 0, 132)
    D = mlp_bag(X_tr[m], d[m], X_va)
    return np.clip(reconstruct(valid, A, D, LAM), 0, 132)

"""H27/H28 probes on e019: drop the t_* block, FDR feature selection, categorical encoding variants."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import TargetEncoder
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).parent))
import cv  # noqa: E402
import feat_skew  # noqa: E402
import e017_skew_temporal_delta as e017  # noqa: E402
from e019_long_surp_shape import BLOCKS  # noqa: E402
from e012_traj_skew import build as build_e012  # noqa: E402
from feat_long import add_long  # noqa: E402
from feat_temporal import add_temporal, delta_target, reconstruct  # noqa: E402

MODE = os.environ.get("E021_MODE", "fdr")  # no_t | fdr | no_cat | te_cat
CATS = ["rater_id", "gene", "cohort"]

DESCRIPTION = f"H27/H28 : e019 avec sélection / encodage des features (mode {MODE})"
FEATURES = f"e019, mode {MODE}"


def _fdr_keep(tr, y):
    """Univariate BH-FDR 0.05 on numeric columns (NaN -> column median, train only)."""
    num = tr.drop(columns=[c for c in ["Index", "patient_id", "target", *CATS] if c in tr.columns])
    num = num.select_dtypes("number")
    X = num.fillna(num.median()).to_numpy()
    _, p = f_regression(X, y)
    rej = multipletests(np.nan_to_num(p, nan=1.0), alpha=0.05, method="fdr_bh")[0]
    drop = list(num.columns[~rej])
    print(f"  fdr: {rej.sum()}/{len(rej)} kept, dropped {drop[:15]}{'...' if len(drop) > 15 else ''}")
    return drop


def fit_predict(train, valid):
    train, valid = train.reset_index(drop=True), valid.reset_index(drop=True)
    tr, va, _ = build_e012(train, valid)
    tr, va = add_long(add_temporal(tr), BLOCKS), add_long(add_temporal(va), BLOCKS)
    y = train["target"]
    if MODE == "no_t":
        drop = [c for c in tr.columns if c.startswith("t_")]
    elif MODE == "fdr":
        drop = _fdr_keep(tr, y)
    elif MODE == "no_cat":
        drop = [c for c in CATS if c in tr.columns]
    else:
        drop = []
    if MODE == "te_cat":
        te = TargetEncoder(random_state=cv.SEED)  # internal cross-fitting on train
        cols = [c for c in CATS if c in tr.columns]
        tr[cols] = te.fit_transform(tr[cols].astype(str), y)
        va[cols] = te.transform(va[cols].astype(str))
    print(f"  {MODE}: dropping {len(drop)} / {tr.shape[1]} columns")
    tr, va = tr.drop(columns=drop), va.drop(columns=drop)
    d = delta_target(train, y)
    m = d.notna().to_numpy()
    A = feat_skew.hgb_predict(tr, va, y)
    D = e017._hgb_delta(tr, va, d, m)
    return np.clip(reconstruct(valid, A, D, e017.LAM), 0, 132)

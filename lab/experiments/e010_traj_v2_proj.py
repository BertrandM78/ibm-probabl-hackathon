"""H09-B: e009 + projection of HGB predictions onto each patient's curve (line + kappa x^2).

Shrink weight w fitted in-fold: inner GroupKFold cross-fit of the full e009 pipeline on train,
w = least-squares coefficient of target - pred on proj - pred (train OOF only).
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

import cv

sys.path.insert(0, str(Path(__file__).parent))
import e009_traj_v2 as base  # noqa: E402
from feat_hier import centred_age  # noqa: E402
from feat_hier2 import _within, fit_curvature  # noqa: E402
from features import GROUP  # noqa: E402

DESCRIPTION = "H09-B : e009 + projection des préd. HGB sur la courbe patient (droite + κ·x²), poids w fitté par cross-fit interne sur train"
FEATURES = "identiques à e009 ; post-traitement intra-patient sans target"


def project(df, pred, kappa):
    x = centred_age(df)
    return pred - _within(pred - kappa * x * x, x, df[GROUP]).to_numpy()


def fit_predict(train, valid):
    train = train.reset_index(drop=True)
    oof, d = np.zeros(len(train)), np.zeros(len(train))
    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=cv.SEED)
    for itr, ite in gkf.split(train, groups=train[GROUP]):
        tr_in = train.iloc[itr].reset_index(drop=True)
        va_in = train.iloc[ite].reset_index(drop=True)
        oof[ite] = base.fit_predict(tr_in, va_in.drop(columns="target"))
        d[ite] = project(va_in, oof[ite], fit_curvature(tr_in)) - oof[ite]
    w = float(np.clip((d * (train["target"].to_numpy() - oof)).sum() / (d * d).sum(), 0, 1))
    print(f"  proj w={w:.3f} inner oof rmse {np.sqrt(np.mean((oof - train['target']) ** 2)):.3f} -> "
          f"{np.sqrt(np.mean((oof + w * d - train['target']) ** 2)):.3f}")
    kappa = fit_curvature(train)
    pred = base.fit_predict(train, valid)
    return np.clip(pred + w * (project(valid, pred, kappa) - pred), 0, 132)

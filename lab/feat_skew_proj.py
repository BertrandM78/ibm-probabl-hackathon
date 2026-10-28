"""H13: combine HGB with the skew-normal MAP trajectory (e012) — blend weights and offset variant.

Components on a held-out frame (one e012 pipeline build):
    hgb  : HGB prediction (e012 features), offset=True -> hb_post_full + HGB on target - hb_post_full;
    ols  : projection of hgb on the patient's OLS line + kappa x^2 (e010/e012);
    traj : skew MAP trajectory hb_post_full (clipped [0, 132]);
    tshift: traj shifted to the patient mean of hgb (projection of hgb on {traj + c_p}).
Blend weights are fitted on an inner GroupKFold cross-fit of train (train target only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

import cv
from e010_traj_v2_proj import project
from feat_skew import build_e011, hgb_predict, skew_hook
from features import GROUP, align_categories, to_model_matrix


def hgb_raw(tr, va, y):
    """hgb_predict without the output clip (for residual targets)."""
    X_tr = to_model_matrix(tr)
    X_va = align_categories(X_tr, to_model_matrix(va))
    model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, early_stopping=False,
                                          random_state=cv.SEED)
    return model.fit(X_tr, y).predict(X_va)


def components(train: pd.DataFrame, valid: pd.DataFrame, offset: bool = False) -> pd.DataFrame:
    tr, va, kappa = build_e011(train, valid, skew_hook(train["target"]))
    base_tr = np.clip(tr["hb_post_full"].to_numpy(), 0, 132)
    traj = np.clip(va["hb_post_full"].to_numpy(), 0, 132)
    if offset:
        res = hgb_raw(tr, va, train["target"].to_numpy() - base_tr)  # residual: no [0, 132] clip
        hgb = np.clip(traj + res, 0, 132)
    else:
        hgb = hgb_predict(tr, va, train["target"])
    shift = traj + pd.Series(hgb - traj).groupby(valid[GROUP].to_numpy()).transform("mean").to_numpy()
    return pd.DataFrame({"hgb": hgb, "ols": project(valid, hgb, kappa), "traj": traj, "tshift": shift},
                        index=valid.index)


def inner_components(train: pd.DataFrame, offset: bool = False, n_splits: int = 5) -> pd.DataFrame:
    """Out-of-fold components on train (inner patient-grouped cross-fit)."""
    train = train.reset_index(drop=True)
    parts = []
    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=cv.SEED)
    for itr, ite in gkf.split(train, groups=train[GROUP]):
        tr_in = train.iloc[itr].reset_index(drop=True)
        va_in = train.iloc[ite].reset_index(drop=True)
        parts.append(components(tr_in, va_in.drop(columns="target"), offset).set_index(pd.Index(ite)))
    return pd.concat(parts).sort_index()


def fit_weights(comp: pd.DataFrame, y: np.ndarray, cols: tuple[str, ...]) -> np.ndarray:
    """LS weights of target - hgb on (c - hgb) for c in cols (pred = hgb + sum w_c (c - hgb))."""
    D = np.column_stack([comp[c] - comp["hgb"] for c in cols])
    return np.linalg.lstsq(D, y - comp["hgb"].to_numpy(), rcond=None)[0]


def blend(comp: pd.DataFrame, w: dict) -> np.ndarray:
    pred = comp["hgb"].to_numpy().copy()
    for c, wc in w.items():
        pred += wc * (comp[c].to_numpy() - comp["hgb"].to_numpy())
    return np.clip(pred, 0, 132)


def report(comp: pd.DataFrame, y: np.ndarray, label: str) -> None:
    r = lambda p: float(np.sqrt(np.mean((np.clip(p, 0, 132) - y) ** 2)))  # noqa: E731
    w1 = fit_weights(comp, y, ("traj",))
    w2 = fit_weights(comp, y, ("ols", "traj"))
    print(f"  [{label}] hgb {r(comp['hgb']):.3f} traj {r(comp['traj']):.3f} "
          f"ols.5 {r(blend(comp, {'ols': .5})):.3f} | w_traj={w1[0]:.3f} -> {r(blend(comp, {'traj': w1[0]})):.3f} "
          f"| w_ols={w2[0]:.3f} w_traj={w2[1]:.3f} -> {r(blend(comp, dict(zip(('ols', 'traj'), w2)))):.3f} "
          f"| w_tshift={fit_weights(comp, y, ('tshift',))[0]:.3f} -> "
          f"{r(blend(comp, {'tshift': fit_weights(comp, y, ('tshift',))[0]})):.3f} tshift.5 {r(blend(comp, {'tshift': .5})):.3f} "
          f"| corr(err hgb, traj)={np.corrcoef(comp['hgb'] - y, comp['traj'] - y)[0, 1]:.3f}")


def plain_and_offset(train: pd.DataFrame, valid: pd.DataFrame, w: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """e012 prediction and its offset twin (HGB on target - hb_post_full), both OLS-projected with w; one build."""
    tr, va, kappa = build_e011(train, valid, skew_hook(train["target"]))
    traj = np.clip(va["hb_post_full"].to_numpy(), 0, 132)
    plain = hgb_predict(tr, va, train["target"])
    off = np.clip(traj + hgb_raw(tr, va, train["target"].to_numpy() - np.clip(tr["hb_post_full"].to_numpy(), 0, 132)),
                  0, 132)
    proj = lambda p: np.clip(p + w * (project(valid, p, kappa) - p), 0, 132)  # noqa: E731
    return proj(plain), proj(off)

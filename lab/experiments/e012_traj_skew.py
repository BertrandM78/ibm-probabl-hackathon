"""H11: e011 with the pass-2 trajectory replaced by a skew-normal MAP (vectorised Newton, full + LOO)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from e010_traj_v2_proj import project  # noqa: E402
from feat_skew import build_e011, hgb_predict, skew_hook  # noqa: E402

W_PROJ = 0.5

DESCRIPTION = "H11 : trajectoire MAP sous bruit skew-normal location-scale par source (z = (y_v − target)/(c0 + c1·target) ~ SN, fit train ; σ_v = c0 + c1·μ_pass1 figé ; convolution exacte avec σ_ligne), prior gaussien e007, Newton vectorisé ; projection w = 0.5"
FEATURES = "identiques à e011 ; hb_post_*/hb_se_* (Laplace)/hb_slope_loo/hb_int_loo/hb_prior_gap = MAP skew"


def build(train, valid):
    return build_e011(train, valid, skew_hook(train["target"]))


def fit_predict(train, valid):
    tr, va, kappa = build(train, valid)
    pred = hgb_predict(tr, va, train["target"])
    return np.clip(pred + W_PROJ * (project(valid, pred, kappa) - pred), 0, 132)

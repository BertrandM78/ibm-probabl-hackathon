"""H11 control: e011 with y_v recalibrated in mean per source before the Gaussian trajectory (no skew)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from e010_traj_v2_proj import project  # noqa: E402
from feat_skew import build_e011, hgb_predict, meancal_hook  # noqa: E402

W_PROJ = 0.5

DESCRIPTION = "H11 contrôle : e011 + y_v recalé par source (y_cal = (y_v − d0)/(1 + d1), OLS de y_v − target sur target, train) avant la trajectoire gaussienne ; variances et c0/c1 refittés sur y_cal"
FEATURES = "identiques à e011 ; hb_post_*/hb_se_*/hb_slope_loo/hb_int_loo/hb_prior_gap recalculés sur y_cal"


def build(train, valid):
    return build_e011(train, valid, meancal_hook(train["target"]))


def fit_predict(train, valid):
    tr, va, kappa = build(train, valid)
    pred = hgb_predict(tr, va, train["target"])
    return np.clip(pred + W_PROJ * (project(valid, pred, kappa) - pred), 0, 132)

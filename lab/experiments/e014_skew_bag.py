"""H14: e012 pipeline with the single HGB replaced by a bag of N seeds (same hyperparameters), then
the usual w=0.5 projection on the patient curve. Tests whether e013c's gain was HGB variance reduction."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from e010_traj_v2_proj import project  # noqa: E402
from feat_skew import build_e011, skew_hook  # noqa: E402
from feat_skew_bag import bag_hgb_predict  # noqa: E402

N_MODELS = 5
W_PROJ = 0.5

DESCRIPTION = "H14 : e012 (trajectoire MAP skew) avec HGB remplacé par un bag de N seeds HGB (mêmes hyperparams), projection w = 0.5"
FEATURES = "identiques à e012 ; HGB -> moyenne de N HistGradientBoostingRegressor(random_state=SEED..SEED+N-1)"


def build(train, valid):
    return build_e011(train, valid, skew_hook(train["target"]))


def fit_predict(train, valid):
    tr, va, kappa = build(train, valid)
    pred = bag_hgb_predict(tr, va, train["target"], N_MODELS)
    return np.clip(pred + W_PROJ * (project(valid, pred, kappa) - pred), 0, 132)

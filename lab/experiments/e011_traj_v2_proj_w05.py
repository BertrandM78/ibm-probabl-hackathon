"""H09-B frozen: e009 + projection onto the patient curve with fixed w = 0.5 (submission version, no inner cross-fit)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import e009_traj_v2 as base  # noqa: E402
from e010_traj_v2_proj import project  # noqa: E402
from feat_hier2 import fit_curvature  # noqa: E402

W_PROJ = 0.5

DESCRIPTION = "H09-B figé : e009 + projection des préd. HGB sur la courbe patient (droite + κ·x²), w = 0.5 fixe (w in-fold e010 : 0.49–0.52)"
FEATURES = "identiques à e009 ; post-traitement intra-patient sans target"


def fit_predict(train, valid):
    kappa = fit_curvature(train.reset_index(drop=True))
    pred = base.fit_predict(train, valid)
    return np.clip(pred + W_PROJ * (project(valid, pred, kappa) - pred), 0, 132)

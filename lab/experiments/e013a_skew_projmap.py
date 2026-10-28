"""H13-a: e012 HGB blended with the skew MAP trajectory, pred = (1 - w) hgb + w hb_post_full, w by inner cross-fit."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from feat_skew_proj import blend, components, fit_weights, inner_components, report  # noqa: E402

DESCRIPTION = "H13-a : préd. HGB e012 mélangées à la trajectoire MAP skew (hb_post_full) au lieu de la droite OLS : (1−w)·hgb + w·hb_post_full, w fitté par cross-fit interne sur train"
FEATURES = "identiques à e012 ; post-traitement = hb_post_full skew (sans projection OLS)"


def fit_predict(train, valid):
    ci = inner_components(train)
    y = train["target"].to_numpy()
    report(ci, y, "inner")
    w = float(np.clip(fit_weights(ci, y, ("traj",))[0], 0, 1))
    print(f"  w_traj={w:.3f}")
    return blend(components(train, valid), {"traj": w})

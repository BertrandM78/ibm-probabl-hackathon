"""H20+H21: e017 + surprises of debiased scores around the skew MAP trajectory (visit + per-patient shape)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from e012_traj_skew import build as build_e012  # noqa: E402
from e017_skew_temporal_delta import LAM, _hgb_delta  # noqa: E402
from feat_long import add_long, shift_report  # noqa: E402
from feat_skew import hgb_predict  # noqa: E402
from feat_temporal import add_temporal, delta_target, reconstruct  # noqa: E402

BLOCKS = ("surp", "shape")

DESCRIPTION = "H20+H21 : e017 + surprises y_v − hb_post_loo (visite, voisines, 1re visite, infl) + forme patient (max/min/std/skew/médiane, zmax, résidus vs hb_post_full)"
FEATURES = "e017 + L_* (feat_long, blocs surp + shape)"


def make_fit_predict(blocks):
    def fit_predict(train, valid):
        train, valid = train.reset_index(drop=True), valid.reset_index(drop=True)
        tr, va, _ = build_e012(train, valid)
        shift_report(tr, va)
        tr, va = add_long(add_temporal(tr), blocks), add_long(add_temporal(va), blocks)
        y = train["target"]
        d = delta_target(train, y)
        m = d.notna().to_numpy()
        A = hgb_predict(tr, va, y)
        D = _hgb_delta(tr, va, d, m)
        return np.clip(reconstruct(valid, A, D, LAM), 0, 132)
    return fit_predict


fit_predict = make_fit_predict(BLOCKS)

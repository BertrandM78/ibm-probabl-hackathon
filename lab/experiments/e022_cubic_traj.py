"""H17: e019 with the quadratic skew MAP trajectory replaced by a hierarchical cubic skew MAP (4D)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from e017_skew_temporal_delta import LAM, _hgb_delta  # noqa: E402
from e019_long_surp_shape import BLOCKS  # noqa: E402
from feat_cubic import cubic_hook  # noqa: E402
from feat_long import add_long, shift_report  # noqa: E402
from feat_skew import build_e011, hgb_predict  # noqa: E402
from feat_temporal import add_temporal, delta_target, reconstruct  # noqa: E402

DESCRIPTION = ("H17 : e019 avec trajectoire cubique hiérarchique (base âge centré patient /5, prior 4D par HGB "
               "cross-fit, Σ robuste MinCovDet sur patients ≥ 8 visites débruitée, MAP skew-normal Newton 4D full + LOO)")
FEATURES = "e019 avec hb_* = MAP cubique + hc_curv/hc_cub (loo/full)"


def make_fit_predict(keep_quad, **prior_kw):
    def fit_predict(train, valid):
        return _run(train, valid, keep_quad, **prior_kw)
    return fit_predict


def _run(train, valid, keep_quad, **prior_kw):
    train, valid = train.reset_index(drop=True), valid.reset_index(drop=True)
    tr, va, _ = build_e011(train, valid, cubic_hook(train["target"], keep_quad=keep_quad, **prior_kw))
    shift_report(tr, va)
    tr, va = add_long(add_temporal(tr), BLOCKS), add_long(add_temporal(va), BLOCKS)
    y = train["target"]
    d = delta_target(train, y)
    m = d.notna().to_numpy()
    A = hgb_predict(tr, va, y)
    D = _hgb_delta(tr, va, d, m)
    return np.clip(reconstruct(valid, A, D, LAM), 0, 132)


fit_predict = make_fit_predict(False)

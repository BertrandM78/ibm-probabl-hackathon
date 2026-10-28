"""H58: e022b with a visit-quality noise model (location/scale of z by visit covariates) in both skew MAPs."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from e017_skew_temporal_delta import LAM, _hgb_delta  # noqa: E402
from e019_long_surp_shape import BLOCKS  # noqa: E402
from feat_long import add_long, shift_report  # noqa: E402
from feat_quality import quality_cubic_hook  # noqa: E402
from feat_skew import build_e011, hgb_predict  # noqa: E402
from feat_temporal import add_temporal, delta_target, reconstruct  # noqa: E402

DESCRIPTION = ("H58 : e022b avec biais de mesure par visite — localisation m(x) de z = (y_v − target)/(c0 + c1·target) "
               "par HGB sur covariables visite (tsi_on/off, ledd, âge, rater, source, NaN…), cross-fit patient ; template SN refitté sur z − m (échelle s(x) écartée : fold 0 sans effet)")
FEATURES = "identiques à e022b ; MAP cubique + quadratique avec loc = sv·(m + ξ)"


def make_fit_predict(**qkw):
    def fit_predict(train, valid):
        train, valid = train.reset_index(drop=True), valid.reset_index(drop=True)
        tr, va, _ = build_e011(train, valid, quality_cubic_hook(train["target"], keep_quad=True, **qkw))
        shift_report(tr, va)
        tr, va = add_long(add_temporal(tr), BLOCKS), add_long(add_temporal(va), BLOCKS)
        y = train["target"]
        d = delta_target(train, y)
        m = d.notna().to_numpy()
        A = hgb_predict(tr, va, y)
        D = _hgb_delta(tr, va, d, m)
        return np.clip(reconstruct(valid, A, D, LAM), 0, 132)
    return fit_predict


fit_predict = make_fit_predict(loc=True, scale=False)   # fold 0: loc 2.7211, scale 2.7540, both 2.7465 (e022b 2.7551)

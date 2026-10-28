"""H33: e022b + generator rule features E[rule(P)] - P_hat from the cubic MAP Laplace posterior (c refit in-fold)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from e017_skew_temporal_delta import LAM, _hgb_delta  # noqa: E402
from e019_long_surp_shape import BLOCKS  # noqa: E402
from feat_long import add_long, shift_report  # noqa: E402
from feat_rule import rule_hook  # noqa: E402
from feat_skew import build_e011, hgb_predict  # noqa: E402
from feat_temporal import add_temporal, delta_target, reconstruct  # noqa: E402

DESCRIPTION = ("H33 : e022b + règle générateur (T_i = c·T_{i−1} si P_i < T_{i−1}, c refitté in-fold) appliquée à 200 tirages "
               "de la postérieure Laplace de la MAP cubique → r_det, r_exp, r_ptrig, r_sd")
FEATURES = "e022b + r_det, r_exp, r_ptrig, r_sd"


def run(train, valid, **hook_kw):
    train, valid = train.reset_index(drop=True), valid.reset_index(drop=True)
    tr, va, _ = build_e011(train, valid, rule_hook(train["target"], **hook_kw))
    shift_report(tr, va)
    tr, va = add_long(add_temporal(tr), BLOCKS), add_long(add_temporal(va), BLOCKS)
    y = train["target"]
    d = delta_target(train, y)
    m = d.notna().to_numpy()
    A = hgb_predict(tr, va, y)
    D = _hgb_delta(tr, va, d, m)
    return np.clip(reconstruct(valid, A, D, LAM), 0, 132)


def fit_predict(train, valid):
    return run(train, valid, rule_feats=True)

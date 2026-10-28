"""H42a: e022b with A_p also pooling on/off pairs without tsi_on (g = gbar, W = 1, k = 0.5), config of p2_h39."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import feat_skew  # noqa: E402
from e022b_cubic_traj_quad import DESCRIPTION as _D, FEATURES as _F, fit_predict as _fp  # noqa: E402
from feat_ap_pairs import add_drug_amplitude_pairs  # noqa: E402

DESCRIPTION = _D + " + H42a : A_p LOO inclut les paires on/off sans tsi_on (g = gbar, poids 1, shrink 0.5)"
FEATURES = _F + " ; A_p/n_av/on_eq(_n) recalculés avec paires sans tsi (propagés à y_v, MAP, t_*, L_*)"


def fit_predict(train, valid):
    # gbar, A0, g fitted on train in build_e011 (fit_drug_curve: on/off/tsi only, no target).
    orig = feat_skew.add_drug_amplitude
    feat_skew.add_drug_amplitude = add_drug_amplitude_pairs
    try:
        return _fp(train, valid)
    finally:
        feat_skew.add_drug_amplitude = orig

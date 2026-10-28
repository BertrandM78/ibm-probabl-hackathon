"""H60 robustness/ablation of e025: ablation : encodage temporel réduit (dt_prev/next, first/last)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e025_gru_raw import DESCRIPTION as _D, GRU_PARAMS  # noqa: E402
from feat_seq_raw import fit_predict_seq  # noqa: E402

DESCRIPTION = _D + " — ablation : encodage temporel réduit (dt_prev/next, first/last)"
FEATURES = "e025 ; time_set=dt"


def fit_predict(train, valid):
    return fit_predict_seq(train, valid, verbose=False, threads=4, **{**GRU_PARAMS, "time_set": "dt"})

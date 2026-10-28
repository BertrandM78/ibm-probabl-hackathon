"""H60 robustness/ablation of e025: ablation : aucun encodage temporel (ordre seul + âge brut)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e025_gru_raw import DESCRIPTION as _D, GRU_PARAMS  # noqa: E402
from feat_seq_raw import fit_predict_seq  # noqa: E402

DESCRIPTION = _D + " — ablation : aucun encodage temporel (ordre seul + âge brut)"
FEATURES = "e025 ; time_set=none"


def fit_predict(train, valid):
    return fit_predict_seq(train, valid, verbose=False, threads=4, **{**GRU_PARAMS, "time_set": "none"})

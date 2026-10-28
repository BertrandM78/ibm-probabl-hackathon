"""H60 robustness/ablation of e025: robustesse : seeds 142–146 au lieu de 42–46."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e025_gru_raw import DESCRIPTION as _D, GRU_PARAMS  # noqa: E402
from feat_seq_raw import fit_predict_seq  # noqa: E402

DESCRIPTION = _D + " — robustesse : seeds 142–146 au lieu de 42–46"
FEATURES = "e025 ; seed_offset=100"


def fit_predict(train, valid):
    return fit_predict_seq(train, valid, verbose=False, threads=4, **{**GRU_PARAMS, "seed_offset": 100})

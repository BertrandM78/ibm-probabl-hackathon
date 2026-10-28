"""H60-B: small Transformer encoder (attention between a patient's visits), raw inputs; time only via T_* encodings."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e025_gru_raw import GRU_PARAMS  # noqa: E402
from feat_seq_raw import fit_predict_seq  # noqa: E402

TF_PARAMS = {**GRU_PARAMS, "arch": "transformer"}

DESCRIPTION = ("H60-B : Transformer 2 couches × 64 (4 têtes, pre-norm, sans embedding positionnel) sur la séquence de "
               "visites, mêmes entrées / entraînement que e025 (dropout 0.4, wd 5e-2, bag 5 seeds)")
FEATURES = "identiques à e025 ; Transformer ×5"


def fit_predict(train, valid):
    return fit_predict_seq(train, valid, verbose=False, threads=4, **TF_PARAMS)

"""H60: bidirectional GRU over each patient's visit sequence, raw inputs only (no pharmaco/rater/trajectory module)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from feat_seq_raw import fit_predict_seq  # noqa: E402

GRU_PARAMS = dict(arch="gru", time_set="full", hidden=64, layers=2, dropout=0.4, wd=5e-2, epochs=60, seeds=5)

DESCRIPTION = ("H60 : BiGRU 2×64 sur la séquence de visites du patient, entrées brutes (num. standardisées + masques NaN "
               "+ embeddings rater/gene/cohort + encodage temporel full), MSE visite, 60 epochs OneCycle, "
               "dropout 0.4, wd 5e-2, bag 5 seeds")
FEATURES = "brutes + masques + T_* (dt_prev/next, since_first, to_last, age_rel, pos, n_vis, tsd) ; BiGRU ×5"


def fit_predict(train, valid):
    return fit_predict_seq(train, valid, verbose=False, **GRU_PARAMS)

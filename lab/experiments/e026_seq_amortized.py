"""H44b (H31): 'amortized' Transformer bag on harmonized columns only (no MAP / temporal / long blocks)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1]))
from e017_skew_temporal_delta import LAM  # noqa: E402
from e019_long_surp_shape import BLOCKS  # noqa: E402
from feat_cubic import cubic_hook  # noqa: E402
from feat_long import add_long  # noqa: E402
from feat_seq import check_order, setup_torch  # noqa: E402
from feat_seq_h44 import RAW, seq_bag  # noqa: E402
from feat_skew import build_e011  # noqa: E402
from feat_temporal import add_temporal  # noqa: E402
from features import align_categories, to_model_matrix  # noqa: E402

DESCRIPTION = ("H44b (H31) : Transformer « amortized » (3 seeds, H 128, 2 couches, AdamW 2e-3, OneCycle 100 ep, epoch "
               "sur 10 % patients train) sur colonnes harmonisées seules, têtes niveau + delta, reconstruction λ = 4 "
               "(config sonde m3_seq.py tf raw lvl 100)")
FEATURES = "brutes + y_v, w_v, on_n, off_n, s_rater, g_on, on_eq_n, A_p (build e022b, sous-ensemble) -> dense"

ARCHS, N_SEEDS, EPOCHS = ("tf",), 3, 100


def fit_predict(train, valid):
    setup_torch()
    train, valid = train.reset_index(drop=True), valid.reset_index(drop=True)
    check_order(train)
    check_order(valid)
    # same build as the probe (f*.pkl dump), then column subset
    tr, va, _ = build_e011(train, valid, cubic_hook(train["target"], keep_quad=True))
    tr, va = add_long(add_temporal(tr), BLOCKS), add_long(add_temporal(va), BLOCKS)
    Xtr = to_model_matrix(tr)
    Xva = align_categories(Xtr, to_model_matrix(va))
    cols = [c for c in Xtr.columns if c in RAW]
    print(f"  [e026] cols {cols}", flush=True)
    return seq_bag(train, valid, Xtr[cols], Xva[cols], ARCHS, N_SEEDS, EPOCHS, lam=LAM, tag="e026")

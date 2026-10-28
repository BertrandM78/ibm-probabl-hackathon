"""H44a: e023 Transformer bag with offset target (target - hb_post_full of the e022b build), level + delta heads."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1]))
from e017_skew_temporal_delta import LAM  # noqa: E402
from e019_long_surp_shape import BLOCKS  # noqa: E402
from feat_cubic import cubic_hook  # noqa: E402
from feat_long import add_long  # noqa: E402
from feat_seq import check_order, setup_torch  # noqa: E402
from feat_seq_h44 import seq_bag  # noqa: E402
from feat_skew import build_e011  # noqa: E402
from feat_temporal import add_temporal  # noqa: E402
from features import align_categories, to_model_matrix  # noqa: E402

DESCRIPTION = ("H44a : réseaux séquentiels e023 (5 Transformer, H 128, 2 couches, AdamW 2e-3, OneCycle 25 ep, epoch sur "
               "10 % patients train) en cible offset target − hb_post_full (build e022b), têtes niveau + delta, "
               "préd. = base + sortie, reconstruction λ = 4 (config sonde m3_seq.py tf all off)")
FEATURES = "e022b (build_e011 + cubic_hook(keep_quad) + add_temporal + add_long) -> dense ; base hb_post_full"

ARCHS, N_SEEDS, EPOCHS = ("tf",), 5, 25  # probe s0_tf_all_off.pkl = EP 25 (bit-identical, scratch/h44/repro_off.py)


def fit_predict(train, valid):
    setup_torch()
    train, valid = train.reset_index(drop=True), valid.reset_index(drop=True)
    check_order(train)
    check_order(valid)
    tr, va, _ = build_e011(train, valid, cubic_hook(train["target"], keep_quad=True))
    tr, va = add_long(add_temporal(tr), BLOCKS), add_long(add_temporal(va), BLOCKS)
    Xtr = to_model_matrix(tr)
    Xva = align_categories(Xtr, to_model_matrix(va))
    return seq_bag(train, valid, Xtr, Xva, ARCHS, N_SEEDS, EPOCHS,
                   base_tr=Xtr["hb_post_full"], base_va=Xva["hb_post_full"], lam=LAM, tag="e025")

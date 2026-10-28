"""H30: per-patient sequence nets (5 BiGRU + 5 Transformer seeds) on the e022b feature build, level + delta heads."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1]))
import cv  # noqa: E402
from e017_skew_temporal_delta import LAM  # noqa: E402
from e019_long_surp_shape import BLOCKS  # noqa: E402
from feat_cubic import cubic_hook  # noqa: E402
from feat_long import add_long  # noqa: E402
from feat_seq import check_order, dense, pack, setup_torch, train_predict, unpack  # noqa: E402
from feat_skew import build_e011  # noqa: E402
from feat_temporal import add_temporal, delta_target, reconstruct  # noqa: E402
from features import align_categories, to_model_matrix  # noqa: E402

DESCRIPTION = ("H30 : réseaux séquentiels par patient sur la matrice e022b densifiée (fit train), 5 BiGRU + 5 Transformer "
               "(H 128, 2 couches, AdamW 2e-3, OneCycle 25 ep, epoch sur 10 % patients train), têtes niveau + delta, "
               "moyenne des 10, reconstruction λ = 4")
FEATURES = "e022b (build_e011 + cubic_hook(keep_quad) + add_temporal + add_long) -> dense (one-hot, flags NaN, médiane, z, clip ±6)"

ARCHS, N_SEEDS = ("gru", "tf"), 5


def fit_predict(train, valid):
    setup_torch()
    train, valid = train.reset_index(drop=True), valid.reset_index(drop=True)
    check_order(train)
    check_order(valid)
    tr, va, _ = build_e011(train, valid, cubic_hook(train["target"], keep_quad=True))
    tr, va = add_long(add_temporal(tr), BLOCKS), add_long(add_temporal(va), BLOCKS)
    Xtr = to_model_matrix(tr)
    Xva = align_categories(Xtr, to_model_matrix(va))
    Dtr, Dva = dense(Xtr, Xva)
    F = Dtr.shape[1]
    y = train["target"].to_numpy(float)
    d = delta_target(train, train["target"]).to_numpy(float)
    mu, sd = y.mean(), y.std()
    dmu, dsd = np.nanmean(d), np.nanstd(d)
    tr_pack = pack(train, Dtr, (y - mu) / sd, (d - dmu) / dsd)
    va_pack = pack(valid, Dva)
    As, Ds = [], []
    for arch in ARCHS:
        for s in range(N_SEEDS):
            t0 = time.time()
            out, ep, inner = train_predict(arch, cv.SEED + s, tr_pack, va_pack, F)
            A, D = unpack(va_pack[0], out, len(valid))
            As.append(A * sd + mu)
            Ds.append(D * dsd + dmu)
            print(f"  [e023] {arch} seed {cv.SEED + s}: best ep {ep}, inner rmse {inner * sd:.3f}, "
                  f"{time.time() - t0:.0f}s, F={F}", flush=True)
    A, D = np.mean(As, 0), np.mean(Ds, 0)
    pred = np.clip(reconstruct(valid, A, D, LAM), 0, 132)
    print(f"  [e023] checksum: mean {pred.mean():.6f}, sum|p| {np.abs(pred).sum():.4f}, p[:3] {np.round(pred[:3], 5)}", flush=True)
    return pred

"""H44 helpers: e023 sequence nets with a fixed epoch budget, optional offset base and column subset.

Same machinery as feat_seq (dense fitted on train, epoch picked on 10 % inner train patients),
mirroring scratch/meth/m3_seq.py (argv EP, feats raw, target off).
"""

from __future__ import annotations

import time

import numpy as np

import cv
import feat_seq
from feat_seq import dense, pack, train_predict, unpack
from feat_temporal import delta_target, reconstruct

# m3_seq.py feats=raw: harmonized columns + covariates, no MAP / temporal / long blocks
RAW = ["on", "off", "ledd", "time_since_intake_on", "time_since_intake_off", "age", "age_at_diagnosis", "sexM",
       "cohort", "gene", "y_v", "w_v", "on_n", "off_n", "s_rater", "g_on", "on_eq_n", "A_p"]


def seq_bag(train, valid, Xtr, Xva, archs, n_seeds, epochs, base_tr=None, base_va=None, lam=4.0, tag="seq"):
    """Bag of nets on (Xtr, Xva); level head learns target - base, delta head the target delta."""
    Dtr, Dva = dense(Xtr, Xva)
    F = Dtr.shape[1]
    y = train["target"].to_numpy(float)
    base_tr = np.zeros(len(train)) if base_tr is None else np.asarray(base_tr, float)
    base_va = np.zeros(len(valid)) if base_va is None else np.asarray(base_va, float)
    t_lvl = y - base_tr
    d = delta_target(train, train["target"]).to_numpy(float)
    mu, sd = t_lvl.mean(), t_lvl.std()
    dmu, dsd = np.nanmean(d), np.nanstd(d)
    tr_pack = pack(train, Dtr, (t_lvl - mu) / sd, (d - dmu) / dsd)
    va_pack = pack(valid, Dva)
    old, feat_seq.EPOCHS = feat_seq.EPOCHS, epochs
    As, Ds = [], []
    try:
        for arch in archs:
            for s in range(n_seeds):
                t0 = time.time()
                out, ep, inner = train_predict(arch, cv.SEED + s, tr_pack, va_pack, F)
                A, D = unpack(va_pack[0], out, len(valid))
                As.append(A * sd + mu + base_va)
                Ds.append(D * dsd + dmu)
                print(f"  [{tag}] {arch} seed {cv.SEED + s}: best ep {ep}/{epochs}, inner rmse {inner * sd:.3f}, "
                      f"{time.time() - t0:.0f}s, F={F}", flush=True)
    finally:
        feat_seq.EPOCHS = old
    A, D = np.mean(As, 0), np.mean(Ds, 0)
    pred = np.clip(reconstruct(valid, A, D, lam), 0, 132)
    print(f"  [{tag}] checksum: mean {pred.mean():.6f}, sum|p| {np.abs(pred).sum():.4f}, p[:3] {np.round(pred[:3], 5)}",
          flush=True)
    return pred

"""H30 helpers: per-patient sequence nets (BiGRU / Transformer) on a dense e022b matrix.

Port of scratch/meth/m3_seq.py: dense() is fitted on train only, the epoch is chosen
on 10 % of the train patients (per-seed patient split), nothing is chosen on valid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

GROUP = "patient_id"
N_THREADS = 8
H, EPOCHS, LR, BATCH, W_DELTA = 128, 25, 2e-3, 64, 0.5


def setup_torch() -> None:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(N_THREADS)


def check_order(df: pd.DataFrame) -> None:
    """Patients contiguous and ages strictly increasing within patient (packing relies on row order)."""
    pid = df[GROUP].to_numpy()
    starts = pid[1:] != pid[:-1]
    assert pd.Series(pid[np.r_[True, starts]]).is_unique, "patients not contiguous"
    same = ~starts
    assert (np.diff(df["age"].to_numpy())[same] > 0).all(), "ages not increasing within patient"


def dense(Xa: pd.DataFrame, Xb: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """One-hot categories, NaN flags, median fill, standardise, clip ±6 — all fitted on Xa."""
    outa, outb = [], []
    for c in Xa.columns:
        a, b = Xa[c], Xb[c]
        if str(a.dtype) == "category":
            for lv in a.cat.categories:
                outa.append((a == lv).astype(float).to_numpy())
                outb.append((b == lv).astype(float).to_numpy())
            continue
        a, b = a.astype(float).to_numpy(), b.astype(float).to_numpy()
        na, nb = np.isnan(a), np.isnan(b)
        if na.any():
            outa.append(na.astype(float))
            outb.append(nb.astype(float))
        med = np.nanmedian(a)
        sd = np.nanstd(a) + 1e-9
        if not np.isfinite(med) or sd < 1e-6:
            continue
        outa.append(np.clip((np.where(na, med, a) - med) / sd, -6, 6))
        outb.append(np.clip((np.where(nb, med, b) - med) / sd, -6, 6))
    return np.stack(outa, 1).astype(np.float32), np.stack(outb, 1).astype(np.float32)


def pack(df, D, y_lvl=None, y_d=None):
    """(P, T, F) padded patient tensors; targets already standardised."""
    idx = list(df.groupby(GROUP, sort=False).indices.values())
    P, T, F = len(idx), max(map(len, idx)), D.shape[1]
    X = np.zeros((P, T, F), np.float32)
    M = np.zeros((P, T), bool)
    Y = np.zeros((P, T, 2), np.float32)
    MD = np.zeros((P, T), bool)
    for i, ix in enumerate(idx):
        n = len(ix)
        X[i, :n] = D[ix]
        M[i, :n] = True
        if y_lvl is not None:
            Y[i, :n, 0] = y_lvl[ix]
            ok = ~np.isnan(y_d[ix])
            Y[i, :n, 1] = np.where(ok, y_d[ix], 0)
            MD[i, :n] = ok
    return idx, torch.tensor(X), torch.tensor(M), torch.tensor(Y), torch.tensor(MD)


class Net(nn.Module):
    def __init__(self, F, arch, h=H):
        super().__init__()
        self.arch = arch
        self.inp = nn.Sequential(nn.Linear(F, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, h), nn.GELU())
        if arch == "gru":
            self.seq = nn.GRU(h, h, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
            ho = 2 * h
        else:  # transformer, no positional encoding
            layer = nn.TransformerEncoderLayer(h, 4, 2 * h, 0.1, batch_first=True, norm_first=True)
            self.seq = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            ho = h
        self.head = nn.Sequential(nn.Linear(ho + h, h), nn.GELU(), nn.Linear(h, 2))

    def forward(self, X, M):
        h = self.inp(X)
        if self.arch == "gru":
            pk = nn.utils.rnn.pack_padded_sequence(h, M.sum(1).cpu(), batch_first=True, enforce_sorted=False)
            z, _ = self.seq(pk)
            z, _ = nn.utils.rnn.pad_packed_sequence(z, batch_first=True, total_length=X.shape[1])
        else:
            z = self.seq(h, src_key_padding_mask=~M)
        return self.head(torch.cat([z, h], -1))


def train_predict(arch, seed, tr_pack, va_pack, F):
    """Train one net (seed), epoch picked on 10 % inner train patients; returns standardised (level, delta) on valid."""
    _, Xt, Mt, Yt, MDt = tr_pack
    _, Xv, Mv, _, _ = va_pack
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    P = Xt.shape[0]
    perm = rng.permutation(P)
    inner, fit = perm[: P // 10], perm[P // 10:]
    net = Net(F, arch)
    opt = torch.optim.AdamW(net.parameters(), LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * (len(fit) // BATCH + 1))

    def losses(ix):
        out = net(Xt[ix], Mt[ix])
        m, md = Mt[ix], MDt[ix]
        l1 = ((out[..., 0] - Yt[ix][..., 0]) ** 2)[m].mean()
        l2 = ((out[..., 1] - Yt[ix][..., 1]) ** 2)[md].mean()
        return l1, l2

    best, best_state, best_ep = np.inf, None, -1
    for ep in range(EPOCHS):
        net.train()
        rng.shuffle(fit)
        for b in range(0, len(fit), BATCH):
            l1, l2 = losses(fit[b:b + BATCH])
            opt.zero_grad()
            (l1 + W_DELTA * l2).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            l1, _ = losses(inner)
        if l1.item() < best:
            best, best_ep = l1.item(), ep
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        out = net(Xv, Mv).numpy()
    return out, best_ep, float(np.sqrt(best))


def unpack(idx, out, n_rows):
    A, D = np.empty(n_rows), np.empty(n_rows)
    for i, ix in enumerate(idx):
        A[ix] = out[i, :len(ix), 0]
        D[ix] = out[i, :len(ix), 1]
    return A, D

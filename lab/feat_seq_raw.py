"""Sequence models over a patient's visits (bidirectional GRU / small Transformer), trained with visit-level MSE.

Each patient = one sequence of visits (file order: patient then age). Per visit:
- numeric raw columns, standardised on train, NaN -> 0 plus a missingness mask;
- categorical embeddings (rater_id, gene with a NaN level, cohort);
- explicit irregular-time encodings (TIME_SETS): gaps to previous/next visit, time since first visit,
  age centred on the patient, visit position.
Optional `extra` columns (e.g. e022b structural features) are appended to the numeric block.
All statistics (scalers, category levels) are fitted on train only; the target is never an input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

import cv
from features import GROUP

RAW = ["on", "off", "ledd", "time_since_intake_on", "time_since_intake_off", "age", "age_at_diagnosis", "sexM"]
MASKED = ["on", "off", "ledd", "time_since_intake_on", "time_since_intake_off", "age_at_diagnosis", "gene"]
TIME_SETS = {
    "none": [],
    "dt": ["dt_prev", "dt_next", "first", "last"],
    "full": ["dt_prev", "dt_next", "first", "last", "since_first", "to_last", "age_rel", "pos", "n_vis", "tsd"],
}
CATS = {"rater_id": 8, "gene": 4, "cohort": 2}
MAX_LEN = 12


def visit_frame(df: pd.DataFrame, time_set: str) -> pd.DataFrame:
    """Per-visit numeric inputs (no target)."""
    g = df.groupby(GROUP, sort=False)["age"]
    out = pd.DataFrame(index=df.index)
    for c in RAW:
        out[c] = df[c].astype(float)
    out["gap"] = df["off"] - df["on"]
    for c in MASKED:
        out[f"{c}_na"] = df[c].isna().astype(float)
    t = {
        "dt_prev": g.diff().fillna(0.0),
        "dt_next": (-g.diff(-1)).fillna(0.0),
        "first": (g.cumcount() == 0).astype(float),
        "last": (g.cumcount(ascending=False) == 0).astype(float),
        "since_first": df["age"] - g.transform("min"),
        "to_last": g.transform("max") - df["age"],
        "age_rel": df["age"] - g.transform("mean"),
        "pos": g.cumcount() / (g.transform("size") - 1).clip(lower=1),
        "n_vis": g.transform("size").astype(float),
        "tsd": df["age"] - df["age_at_diagnosis"],
    }
    for c in TIME_SETS[time_set]:
        out[f"T_{c}"] = t[c]
    return out


class Encoder:
    """Fits scalers / category levels on train, turns a frame into padded tensors."""

    def __init__(self, train: pd.DataFrame, time_set: str, extra: pd.DataFrame | None):
        num = self._num(train, time_set, extra)
        self.time_set = time_set
        self.cols = num.columns
        self.mu, self.sd = num.mean(), num.std().replace(0, 1.0).fillna(1.0)
        self.levels = {c: sorted(train[c].dropna().astype(str).unique()) for c in CATS}

    @staticmethod
    def _num(df, time_set, extra):
        num = visit_frame(df, time_set)
        if extra is not None:
            ex = extra.astype(float).replace([np.inf, -np.inf], np.nan)
            num = pd.concat([num, ex, ex.isna().astype(float).add_suffix("_xna")], axis=1)
        return num

    def tensors(self, df: pd.DataFrame, extra: pd.DataFrame | None):
        num = self._num(df, self.time_set, extra)[self.cols]
        X = np.clip(((num - self.mu) / self.sd).fillna(0.0).to_numpy(np.float32), -6, 6)
        C = np.column_stack([
            pd.Categorical(df[c].astype(str).where(df[c].notna(), "__nan__"), categories=self.levels[c]).codes + 1
            for c in CATS])  # 0 = NaN / unseen
        pid = df[GROUP].to_numpy()
        starts = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]])
        lens = np.diff(np.r_[starts, len(df)])
        assert lens.max() <= MAX_LEN
        P, L = len(starts), MAX_LEN
        rows = np.full((P, L), -1)
        for i, (s, n) in enumerate(zip(starts, lens)):
            rows[i, :n] = np.arange(s, s + n)
        mask = rows >= 0
        Xp = np.zeros((P, L, X.shape[1]), np.float32)
        Cp = np.zeros((P, L, C.shape[1]), np.int64)
        Xp[mask], Cp[mask] = X[rows[mask]], C[rows[mask]]
        return torch.from_numpy(Xp), torch.from_numpy(Cp), torch.from_numpy(mask), rows, lens


class SeqNet(nn.Module):
    def __init__(self, n_num, levels, arch="gru", hidden=64, layers=2, dropout=0.1, heads=4):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(len(levels[c]) + 1, d) for c, d in CATS.items()])
        d_in = n_num + sum(CATS.values())
        self.inp = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.arch = arch
        if arch == "gru":
            self.seq = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                              dropout=dropout if layers > 1 else 0.0)
            d_out = 2 * hidden
        else:  # transformer: no positional embedding, order/time comes from the T_* inputs
            layer = nn.TransformerEncoderLayer(hidden, heads, 2 * hidden, dropout, batch_first=True,
                                               norm_first=True, activation="gelu")
            self.seq = nn.TransformerEncoder(layer, layers)
            d_out = hidden
        self.head = nn.Sequential(nn.Linear(d_out + hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, X, C, mask):
        e = torch.cat([X] + [emb(C[..., i]) for i, emb in enumerate(self.emb)], dim=-1)
        h0 = self.inp(e)
        if self.arch == "gru":
            lens = mask.sum(1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(h0, lens, batch_first=True, enforce_sorted=False)
            h, _ = self.seq(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=X.shape[1])
        else:
            h = self.seq(h0, src_key_padding_mask=~mask)
        return self.head(torch.cat([h, h0], dim=-1)).squeeze(-1)


def fit_predict_seq(train, valid, *, arch="gru", time_set="full", extra_tr=None, extra_va=None,
                    seeds=1, epochs=60, batch=64, lr=2e-3, wd=1e-4, hidden=64, layers=2, dropout=0.1,
                    seed_offset=0, threads=8, verbose=True):
    """Train `seeds` sequence nets on train (visit-level MSE on the standardised target), average on valid."""
    torch.set_num_threads(threads)
    enc = Encoder(train, time_set, extra_tr)
    Xt, Ct, Mt, rows_t, _ = enc.tensors(train, extra_tr)
    Xv, Cv, Mv, rows_v, _ = enc.tensors(valid, extra_va)
    y = train["target"].to_numpy(np.float32)
    m, s = y.mean(), y.std()
    Yt = torch.zeros(Mt.shape)
    Yt[Mt] = torch.from_numpy((y[rows_t[Mt.numpy()]] - m) / s)
    preds = []
    for k in range(seeds):
        torch.manual_seed(cv.SEED + seed_offset + k)
        rng = np.random.default_rng(cv.SEED + seed_offset + k)
        net = SeqNet(Xt.shape[-1], enc.levels, arch, hidden, layers, dropout)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        steps = epochs * int(np.ceil(len(Xt) / batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
        for ep in range(epochs):
            net.train()
            perm, tot = rng.permutation(len(Xt)), 0.0
            for i in range(0, len(perm), batch):
                b = torch.from_numpy(perm[i:i + batch])
                out = net(Xt[b], Ct[b], Mt[b])
                loss = ((out - Yt[b]) ** 2)[Mt[b]].mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += loss.item() * len(b)
            if verbose and (ep % 10 == 9 or ep == epochs - 1):
                print(f"  [seq] seed {k} epoch {ep + 1}: train RMSE {np.sqrt(tot / len(Xt)) * s:.3f}", flush=True)
        net.eval()
        with torch.no_grad():
            out = torch.cat([net(Xv[i:i + 512], Cv[i:i + 512], Mv[i:i + 512]) for i in range(0, len(Xv), 512)])
        p = np.empty(len(valid))
        p[rows_v[Mv.numpy()]] = out[Mv].numpy() * s + m
        preds.append(p)
    return np.clip(np.mean(preds, axis=0), 0, 132)

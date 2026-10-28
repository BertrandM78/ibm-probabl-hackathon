"""Blend several runs through their OOF predictions, with the same locked folds.

    .venv/bin/python lab/blend.py e002_hgb_basic_context e005_pk_debias --name b001_hgb_pk

Weights are non-negative least squares (+ intercept-free), learned on 4 folds of
OOF predictions and scored on the 5th, so the reported RMSE is honest. The test
blend uses weights fitted on all OOF rows. All runs need lab/runs/<exp>/oof.csv,
and test.csv for the test blend. Logged like any cv.py run.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import nnls

import cv


def load_oof(exp: str) -> pd.DataFrame:
    return pd.read_csv(cv.RUNS_DIR / exp / "oof.csv").set_index(cv.ID)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--name", required=True, help="blend id, e.g. b001_hgb_pk")
    args = ap.parse_args()

    train = cv.load_train()
    oofs = [load_oof(r) for r in args.runs]
    base = oofs[0]
    for r, o in zip(args.runs, oofs):
        if not o.index.equals(base.index) or not (o["fold"] == base["fold"]).all():
            raise SystemExit(f"{r} was not run on the same rows/folds")
    P = np.column_stack([o["pred"].to_numpy() for o in oofs])
    y = base[cv.TARGET].to_numpy()
    fold = base["fold"].to_numpy()

    t0 = time.time()
    pred = np.empty(len(y))
    for k in np.unique(fold):
        w, _ = nnls(P[fold != k], y[fold != k])
        pred[fold == k] = P[fold == k] @ w
    w_all, _ = nnls(P, y)

    oof = base[[cv.GROUP, "fold", cv.TARGET]].assign(pred=pred).reset_index()
    oof = oof.set_index(cv.ID).loc[train[cv.ID]].reset_index()
    out_dir = cv.RUNS_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(out_dir / "oof.csv", index=False)

    metrics = cv.compute_metrics(oof, train)
    ref = cv.best_run(exclude=args.name)
    cmp = cv.paired_comparison(oof, ref) if ref else None
    metrics["comparison"] = cmp
    metrics["weights"] = dict(zip(args.runs, map(float, w_all)))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    class Meta:  # what append_results reads from an experiment module
        DESCRIPTION = "Blend NNLS : " + " + ".join(f"{r}×{w:.2f}" for r, w in zip(args.runs, w_all))
        FEATURES = "OOF des runs listés"

    cv.append_results(args.name, Meta, metrics, cmp, time.time() - t0)
    print(f"[blend] {args.name}: RMSE OOF {metrics['rmse_oof']:.4f} | weights {metrics['weights']}")
    if cmp:
        print(f"[blend] vs {cmp['ref']}: {cmp['delta_mean']:+.4f} ± {cmp['delta_std']:.4f} -> {cmp['verdict']}")

    tests = [cv.RUNS_DIR / r / "test.csv" for r in args.runs]
    if all(p.exists() for p in tests):
        T = [pd.read_csv(p).set_index(cv.ID)["target"] for p in tests]
        blend = sum(w * t for w, t in zip(w_all, T))
        blend.rename("target").reset_index().to_csv(out_dir / "test.csv", index=False)
        print(f"[blend] test predictions -> {out_dir / 'test.csv'}")


if __name__ == "__main__":
    main()

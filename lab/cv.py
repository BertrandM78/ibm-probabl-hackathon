"""Locked cross-validation protocol for the lab.

Every model is evaluated here, with the same patient-grouped folds, so every
number in lab/results.csv is comparable.

An experiment is a Python file in lab/experiments/ that defines:

    def fit_predict(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray

- `train`: visits of the training patients, WITH the `target` column.
- `valid`: visits of the held-out patients, WITHOUT `target` (all their
  visits, like X_test: per-patient features over `valid` are legitimate).
- returns one prediction per row of `valid`, in the same order.

Optional module attributes: `DESCRIPTION` (str, one line), `FEATURES` (str).

Usage (from the repo root):

    .venv/bin/python lab/cv.py lab/experiments/e001_hgb_raw.py
    .venv/bin/python lab/cv.py lab/experiments/e001_hgb_raw.py --test     # + test preds
    .venv/bin/python lab/cv.py lab/experiments/e001_hgb_raw.py --folds 0  # quick, not logged
    .venv/bin/python lab/cv.py lab/experiments/e001_hgb_raw.py --ref e000_dummy

Outputs, per experiment `<exp>` (= file stem):
    lab/runs/<exp>/oof.csv        Index, patient_id, fold, target, pred
    lab/runs/<exp>/metrics.json   global, per-fold, subgroup and per-rater RMSE
    lab/runs/<exp>/test.csv       Index,target  (with --test)
    lab/results.csv               one row appended per full run
    lab/LOG.md                    one draft row appended per full run

Do not change N_SPLITS, SEED, the fold file or the metric without the user's
approval: it would make every past result incomparable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
LAB_DIR = ROOT / "lab"
FOLDS_PATH = LAB_DIR / "folds.csv"
RESULTS_PATH = LAB_DIR / "results.csv"
LOG_PATH = LAB_DIR / "LOG.md"
RUNS_DIR = LAB_DIR / "runs"

N_SPLITS = 5
SEED = 42
TARGET = "target"
ID = "Index"
GROUP = "patient_id"


# ---------------------------------------------------------------- data

def load_train() -> pd.DataFrame:
    """X_train merged with y_train, in file order (patients contiguous, age sorted)."""
    X = pd.read_csv(DATA_DIR / "X_train.csv")
    y = pd.read_csv(DATA_DIR / "y_train.csv")
    df = X.merge(y, on=ID, how="inner", validate="one_to_one")
    assert len(df) == len(X) == len(y), "X_train / y_train misaligned"
    return df


def load_test() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "X_test.csv")


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---------------------------------------------------------------- folds

def get_folds(train: pd.DataFrame) -> pd.Series:
    """Fold id per row, from the saved patient -> fold map (created once)."""
    if FOLDS_PATH.exists():
        folds = pd.read_csv(FOLDS_PATH)
    else:
        patients = train[GROUP].drop_duplicates().to_numpy()
        gkf = GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        fold_of = np.empty(len(patients), dtype=int)
        for k, (_, idx) in enumerate(gkf.split(patients, groups=patients)):
            fold_of[idx] = k
        folds = pd.DataFrame({GROUP: patients, "fold": fold_of})
        folds.to_csv(FOLDS_PATH, index=False)
        print(f"[cv] created {FOLDS_PATH.relative_to(ROOT)}")
    fold = train[GROUP].map(folds.set_index(GROUP)["fold"])
    if fold.isna().any():
        raise RuntimeError("Some train patients are missing from lab/folds.csv")
    return fold.astype(int)


def folds_hash() -> str:
    return hashlib.md5(FOLDS_PATH.read_bytes()).hexdigest()[:8]


# ---------------------------------------------------------------- metrics

def subgroup_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Slices where the true-OFF estimation is expected to behave differently."""
    visit_rank = df.groupby(GROUP).cumcount()
    return {
        "off_measured": df["off"].notna(),
        "off_missing": df["off"].isna(),
        "on_missing": df["on"].isna(),
        "ledd_present": df["ledd"].notna(),
        "ledd_missing": df["ledd"].isna(),
        "tsi_on_present": df["time_since_intake_on"].notna(),
        "tsi_off_present": df["time_since_intake_off"].notna(),
        "cohort_B": df["cohort"] == "B",
        "first_visit": visit_rank == 0,
    }


def compute_metrics(oof: pd.DataFrame, train: pd.DataFrame) -> dict:
    y, p = oof[TARGET], oof["pred"]
    per_fold = [rmse(g[TARGET], g["pred"]) for _, g in oof.groupby("fold")]
    subgroups = {
        name: {"rmse": rmse(y[m], p[m]), "n": int(m.sum())}
        for name, m in subgroup_masks(train).items()
        if m.sum() > 0
    }
    per_rater = (
        oof.assign(rater_id=train["rater_id"].to_numpy(), se=(y - p) ** 2)
        .groupby("rater_id")
        .agg(rmse=("se", lambda s: float(np.sqrt(s.mean()))), n=("se", "size"))
        .sort_values("rmse", ascending=False)
    )
    return {
        "rmse_oof": rmse(y, p),
        "fold_mean": float(np.mean(per_fold)),
        "fold_std": float(np.std(per_fold, ddof=1)) if len(per_fold) > 1 else 0.0,
        "per_fold": per_fold,
        "subgroups": subgroups,
        "per_rater": per_rater.round(3).reset_index().to_dict(orient="records"),
    }


def paired_comparison(oof: pd.DataFrame, ref: str) -> dict | None:
    """Per-fold RMSE difference against a reference run (same folds => paired)."""
    ref_path = RUNS_DIR / ref / "oof.csv"
    if not ref_path.exists():
        return None
    ref_oof = pd.read_csv(ref_path).set_index(ID)
    cur = oof.set_index(ID)
    if not cur.index.equals(ref_oof.index) or not (cur["fold"] == ref_oof["fold"]).all():
        print(f"[cv] WARNING: {ref} was run with different rows/folds, no paired comparison")
        return None
    deltas = [
        rmse(g[TARGET], g["pred"]) - rmse(g[TARGET], ref_oof.loc[g.index, "pred"])
        for _, g in cur.groupby("fold")
    ]
    mean, std = float(np.mean(deltas)), float(np.std(deltas, ddof=1))
    if mean < 0 and abs(mean) > std:
        verdict = "GAIN"
    elif mean > 0 and abs(mean) > std:
        verdict = "WORSE"
    else:
        verdict = "NOISE"
    return {
        "ref": ref,
        "delta_mean": mean,
        "delta_std": std,
        "folds_improved": int(sum(d < 0 for d in deltas)),
        "verdict": verdict,
    }


def best_run(exclude: str) -> str | None:
    if not RESULTS_PATH.exists():
        return None
    res = pd.read_csv(RESULTS_PATH)
    res = res[(res["exp"] != exclude) & (res["folds_hash"] == folds_hash())]
    if res.empty:
        return None
    return str(res.sort_values("rmse_oof").iloc[0]["exp"])


# ---------------------------------------------------------------- running

def load_experiment(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(LAB_DIR))  # lets experiments `import cv` / `import features`
    spec.loader.exec_module(module)
    if not hasattr(module, "fit_predict"):
        raise AttributeError(f"{path} must define fit_predict(train, valid)")
    return module


def check_preds(pred, n: int, where: str) -> np.ndarray:
    pred = np.asarray(pred, dtype=float).ravel()
    if pred.shape != (n,):
        raise ValueError(f"{where}: expected {n} predictions, got {pred.shape}")
    if not np.isfinite(pred).all():
        raise ValueError(f"{where}: predictions contain NaN or inf")
    return pred


def run_cv(fit_predict, train: pd.DataFrame, fold: pd.Series, only_folds=None) -> pd.DataFrame:
    parts = []
    for k in sorted(fold.unique()):
        if only_folds is not None and k not in only_folds:
            continue
        tr_mask, va_mask = fold != k, fold == k
        tr = train[tr_mask].reset_index(drop=True)
        va = train[va_mask].reset_index(drop=True)
        np.random.seed(SEED)
        pred = check_preds(fit_predict(tr.copy(), va.drop(columns=[TARGET]).copy()), len(va), f"fold {k}")
        parts.append(pd.DataFrame({
            ID: va[ID], GROUP: va[GROUP], "fold": k, TARGET: va[TARGET], "pred": pred,
        }))
        print(f"[cv] fold {k}: RMSE {rmse(va[TARGET], pred):.4f}  (n={len(va)})")
    return pd.concat(parts, ignore_index=True)


def append_results(exp: str, module, metrics: dict, cmp: dict | None, seconds: float) -> None:
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "exp": exp,
        "rmse_oof": round(metrics["rmse_oof"], 4),
        "fold_mean": round(metrics["fold_mean"], 4),
        "fold_std": round(metrics["fold_std"], 4),
        **{f"rmse_{k}": round(v["rmse"], 4) for k, v in metrics["subgroups"].items()},
        "ref": cmp["ref"] if cmp else "",
        "delta_vs_ref": round(cmp["delta_mean"], 4) if cmp else "",
        "delta_std": round(cmp["delta_std"], 4) if cmp else "",
        "verdict": cmp["verdict"] if cmp else "",
        "seconds": round(seconds, 1),
        "folds_hash": folds_hash(),
        "description": getattr(module, "DESCRIPTION", ""),
    }
    df = pd.DataFrame([row])
    if RESULTS_PATH.exists():
        df = pd.concat([pd.read_csv(RESULTS_PATH), df], ignore_index=True)
    df.to_csv(RESULTS_PATH, index=False)

    # Draft line in LOG.md; the verdict/notes are then reviewed by hand.
    delta = f"{cmp['delta_mean']:+.3f} ± {cmp['delta_std']:.3f} vs {cmp['ref']}" if cmp else "-"
    line = (
        f"| {exp} | {row['timestamp'][:10]} | {getattr(module, 'DESCRIPTION', '')} "
        f"| {getattr(module, 'FEATURES', '')} | {metrics['rmse_oof']:.3f} "
        f"| {metrics['fold_mean']:.3f} ± {metrics['fold_std']:.3f} | {delta} "
        f"| {cmp['verdict'] if cmp else '-'} | _à compléter_ |\n"
    )
    with LOG_PATH.open("a") as f:
        f.write(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", type=Path, help="lab/experiments/eNNN_name.py")
    ap.add_argument("--test", action="store_true", help="also fit on all train and predict X_test")
    ap.add_argument("--folds", type=int, nargs="+", help="run only these folds (quick check, not logged)")
    ap.add_argument("--ref", help="reference run for the paired comparison (default: best so far)")
    args = ap.parse_args()

    exp = args.experiment.stem
    module = load_experiment(args.experiment.resolve())
    train = load_train()
    fold = get_folds(train)

    t0 = time.time()
    oof = run_cv(module.fit_predict, train, fold, only_folds=args.folds)
    seconds = time.time() - t0

    if args.folds is not None:
        print(f"[cv] quick run on folds {args.folds}: RMSE {rmse(oof[TARGET], oof['pred']):.4f} (not logged)")
        return

    oof = oof.set_index(ID).loc[train[ID]].reset_index()  # back to train row order
    out_dir = RUNS_DIR / exp
    out_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(out_dir / "oof.csv", index=False)

    metrics = compute_metrics(oof, train)
    ref = args.ref or best_run(exclude=exp)
    cmp = paired_comparison(oof, ref) if ref else None
    metrics["comparison"] = cmp
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    append_results(exp, module, metrics, cmp, seconds)

    print(f"\n[cv] {exp}: RMSE OOF {metrics['rmse_oof']:.4f} | folds {metrics['fold_mean']:.4f} ± {metrics['fold_std']:.4f}")
    for name, v in metrics["subgroups"].items():
        print(f"     {name:<16} {v['rmse']:.4f}  (n={v['n']})")
    worst = metrics["per_rater"][:3]
    print("     worst raters   " + ", ".join(f"{r['rater_id']}={r['rmse']:.2f}" for r in worst))
    if cmp:
        print(f"[cv] vs {cmp['ref']}: {cmp['delta_mean']:+.4f} ± {cmp['delta_std']:.4f} "
              f"({cmp['folds_improved']}/{N_SPLITS} folds better) -> {cmp['verdict']}")

    if args.test:
        test = load_test()
        np.random.seed(SEED)
        pred = check_preds(module.fit_predict(train.copy(), test.copy()), len(test), "test")
        pd.DataFrame({ID: test[ID], TARGET: pred}).to_csv(out_dir / "test.csv", index=False)
        print(f"[cv] test predictions -> {(out_dir / 'test.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

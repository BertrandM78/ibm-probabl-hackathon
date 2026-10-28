"""Turn a lab run into an official Kaggle submission: skore report on Skore Hub + CSV.

    .venv/bin/python lab/submit.py e022b_cubic_traj_quad
    .venv/bin/python lab/submit.py b003_seq4 --key b003_seq4_v2
    .venv/bin/python lab/submit.py e022b_cubic_traj_quad --cv        # + CrossValidationReport (5 fits)
    .venv/bin/python lab/submit.py e022b_cubic_traj_quad --no-hub    # dry run, nothing pushed

Steps:
1. Wrap the experiment (or the blend: components of lab/runs/<bNNN>/metrics.json with
   their NNLS weights) in a sklearn estimator: fit() stores train, predict() calls
   fit_predict(train, X). Everything learned on `target` is therefore refitted by skore.
2. EstimatorReport on the patient-grouped holdout = lab fold 0 (lab/folds.csv). Its RMSE
   must equal per_fold[0] of lab/runs/<exp>/metrics.json (same code, same rows).
3. project.put(<key>, report) on Skore Hub -> URL to paste in the Kaggle Submission
   Description (one new key per submission; rule of the competition).
4. submissions/<key>.csv = lab/runs/<exp>/test.csv (fit on all train, written by
   `cv.py --test` / `blend.py`), refitted here if missing, then validated.
5. One row appended to the "Leaderboard Kaggle" table of lab/LOG.md (LB to fill).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

import cv

SUBMISSIONS_DIR = cv.ROOT / "submissions"
SKORE_PROJECT = "bobathon-esilv"


class LabModel(RegressorMixin, BaseEstimator):
    """sklearn view of a lab run: weighted sum of experiments' fit_predict.

    X must hold the raw columns (incl. Index, patient_id), complete patients, file order.
    """

    def __init__(self, components: tuple[tuple[str, float], ...] = ()):
        self.components = components

    def fit(self, X, y):
        self.train_ = X.reset_index(drop=True).assign(**{cv.TARGET: np.asarray(y, float)})
        return self

    def predict(self, X):
        valid = X.reset_index(drop=True).drop(columns=[cv.TARGET], errors="ignore")
        pred = np.zeros(len(valid))
        for exp, w in self.components:
            module = cv.load_experiment(cv.LAB_DIR / "experiments" / f"{exp}.py")
            np.random.seed(cv.SEED)
            pred += w * cv.check_preds(module.fit_predict(self.train_.copy(), valid.copy()), len(valid), exp)
        return pred


def components_of(run: str) -> tuple[tuple[str, float], ...]:
    if (cv.LAB_DIR / "experiments" / f"{run}.py").exists():
        return ((run, 1.0),)
    metrics = cv.RUNS_DIR / run / "metrics.json"
    weights = json.loads(metrics.read_text()).get("weights") if metrics.exists() else None
    if not weights:
        raise SystemExit(f"{run}: no lab/experiments/{run}.py and no blend weights in {metrics}")
    return tuple((exp, float(w)) for exp, w in weights.items() if w > 0)


def load_skore_credentials() -> dict:
    """Export hub URI + API key from .skore (written by scripts/skore-agent); never print the key."""
    path = cv.ROOT / ".skore"
    if not path.is_file():
        raise SystemExit("Missing .skore: run scripts/skore-agent (docs/SETUP.md)")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    os.environ["SKORE_HUB_URI"] = cfg["hub_url"]
    os.environ["SKORE_HUB_API_KEY"] = cfg["api_key"]
    return {k: v for k, v in cfg.items() if k != "api_key"}


def append_leaderboard(run: str, key: str, cv_rmse: float | None, holdout: float) -> None:
    lines = cv.LOG_PATH.read_text().splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines) if l.startswith("| Date | Exp."))
    end = start
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    cv_txt = f"{cv_rmse:.3f}" if cv_rmse is not None else "-"
    row = f"| {date.today()} | {run} | {key} | {cv_txt} (holdout skore {holdout:.3f}) | _à remplir_ | |\n"
    cv.LOG_PATH.write_text("".join(lines[:end] + [row] + lines[end:]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="experiment (eNNN_name) or blend (bNNN_name) with a full lab run")
    ap.add_argument("--key", help="Skore Hub report key = CSV name (default: run name); new key per submission")
    ap.add_argument("--cv", action="store_true", help="also put a CrossValidationReport on the lab folds")
    ap.add_argument("--no-hub", action="store_true", help="dry run: build the reports and the CSV, push nothing")
    args = ap.parse_args()

    from skore import EstimatorReport, Project, evaluate, login

    run, key = args.run, args.key or args.run
    out = SUBMISSIONS_DIR / f"{key}.csv"
    if out.exists() and not args.no_hub:
        raise SystemExit(f"{out.relative_to(cv.ROOT)} exists: pick a new --key (one key per submission)")
    model = LabModel(components_of(run))
    print(f"[submit] {run} -> key {key} | components {dict(model.components)}")

    train = cv.load_train()
    fold = cv.get_folds(train).to_numpy()
    X, y = train.drop(columns=[cv.TARGET]), train[cv.TARGET]

    tr, te = np.flatnonzero(fold != 0), np.flatnonzero(fold == 0)
    report = EstimatorReport(clone(model), X_train=X.iloc[tr], y_train=y.iloc[tr], X_test=X.iloc[te], y_test=y.iloc[te])
    holdout = float(report.metrics.rmse())
    metrics_path = cv.RUNS_DIR / run / "metrics.json"
    lab = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    lab_fold0 = lab.get("per_fold", [None])[0]
    print(f"[submit] EstimatorReport holdout (lab fold 0): RMSE {holdout:.4f}"
          + (f" | lab fold 0 {lab_fold0:.4f}" if lab_fold0 is not None else ""))
    if lab_fold0 is not None and abs(holdout - lab_fold0) > 0.02:
        print("[submit] WARNING: holdout differs from the lab fold 0 -> code changed since the lab run?")

    cv_report = None
    if args.cv:
        splits = [(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(cv.N_SPLITS)]
        cv_report = evaluate(clone(model), X, y, splitter=splits)
        print(cv_report.metrics.summarize())

    if not args.no_hub:
        cfg = load_skore_credentials()
        login(mode="hub")
        project = Project(name=SKORE_PROJECT, mode="hub", workspace=cfg["workspace"])
        project.put(key, report)  # URL printed here -> Kaggle Submission Description
        if cv_report is not None:
            project.put(f"{key}_cv", cv_report)

    test_path = cv.RUNS_DIR / run / "test.csv"
    if test_path.exists():
        sub = pd.read_csv(test_path)
    else:
        test = cv.load_test()
        final = clone(model).fit(X, y)
        sub = pd.DataFrame({cv.ID: test[cv.ID], cv.TARGET: final.predict(test)})
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    out = out if not args.no_hub else SUBMISSIONS_DIR / f"{key}_dryrun.csv"
    sub.to_csv(out, index=False)
    check = subprocess.run([sys.executable, str(cv.LAB_DIR / "check_submission.py"), str(out)])
    if check.returncode != 0:
        raise SystemExit(f"[submit] {out.relative_to(cv.ROOT)} failed the Kaggle format check")
    print(f"[submit] {out.relative_to(cv.ROOT)} ready")

    if not args.no_hub:
        append_leaderboard(run, key, lab.get("rmse_oof"), holdout)
        print("[submit] upload the CSV on Kaggle with the report URL above in the Submission Description,"
              " then fill the public LB in lab/LOG.md")


if __name__ == "__main__":
    main()

"""Validate a prediction file against the Kaggle format (stdlib only, usable from hooks).

    python3 lab/check_submission.py lab/runs/e002_hgb_basic_context/test.csv
    python3 lab/check_submission.py submissions/<key>.csv --compare lab/runs/e002_hgb_basic_context/test.csv

Checks: header `Index,target`, one row per X_test visit, same Index set (and order)
as sample_submission.csv, no duplicate, every target finite, range [0, 132].
--compare: RMSE / max abs diff between two prediction files (e.g. a
submission vs the lab run it comes from) - they should be near identical.
Exit code 0 if valid, 1 otherwise.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample_submission.csv"


def read(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    return rows[0] if rows else [], rows[1:]


def check(path: Path) -> list[str]:
    errors: list[str] = []
    header, rows = read(path)
    if header != ["Index", "target"]:
        errors.append(f"header is {header}, expected ['Index', 'target']")
        return errors
    _, sample_rows = read(SAMPLE)
    expected = [r[0] for r in sample_rows]
    got = [r[0] for r in rows]
    if len(got) != len(expected):
        errors.append(f"{len(got)} rows, expected {len(expected)}")
    if len(set(got)) != len(got):
        errors.append(f"{len(got) - len(set(got))} duplicated Index values")
    if set(got) != set(expected):
        missing, extra = set(expected) - set(got), set(got) - set(expected)
        errors.append(f"Index mismatch: {len(missing)} missing, {len(extra)} unexpected")
    elif got != expected:
        errors.append("Index order differs from sample_submission.csv (accepted by Kaggle, but check alignment)")
    bad, out_of_range = 0, 0
    for _, value in rows:
        try:
            v = float(value)
        except ValueError:
            bad += 1
            continue
        if not math.isfinite(v):
            bad += 1
        elif not 0 <= v <= 132:
            out_of_range += 1
    if bad:
        errors.append(f"{bad} target values are empty, NaN or inf")
    if out_of_range:
        errors.append(f"{out_of_range} target values outside [0, 132]")
    return errors


def compare(a: Path, b: Path) -> str:
    pa = {r[0]: float(r[1]) for r in read(a)[1]}
    pb = {r[0]: float(r[1]) for r in read(b)[1]}
    common = pa.keys() & pb.keys()
    diffs = [pa[k] - pb[k] for k in common]
    rmse = math.sqrt(sum(d * d for d in diffs) / len(diffs))
    return f"compare: {len(common)} common rows, RMSE between files {rmse:.4f}, max |diff| {max(map(abs, diffs)):.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", type=Path, nargs="+")
    ap.add_argument("--compare", type=Path)
    args = ap.parse_args()
    ok = True
    for path in args.files:
        errors = check(path)
        if errors:
            ok = False
            print(f"INVALID {path}:", *[f"  - {e}" for e in errors], sep="\n")
        else:
            print(f"OK {path}")
        if args.compare:
            print(compare(path, args.compare))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""PostToolUse hook: validate any lab/runs/*/test.csv written in the last 2 minutes.

Exit code 2 = invalid file, errors on stderr: fix it before calling it ready (see .bob/rules/lab.md).
"""

import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[2]
recent = [p for p in (root / "lab" / "runs").glob("*/test.csv") if time.time() - p.stat().st_mtime < 120]
if not recent:
    sys.exit(0)

res = subprocess.run(
    [sys.executable, str(root / "lab" / "check_submission.py"), *map(str, recent)],
    capture_output=True,
    text=True,
)
if res.returncode != 0:
    print(res.stdout, file=sys.stderr)
    sys.exit(2)

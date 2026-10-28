"""Guard: block edits of the raw data and the locked fold file (see .bob/rules/lab.md).

Input: JSON {tool_input: {file_path}, cwd} on stdin. The user can still edit these files by hand.
Exit code 2 = block, reason on stderr.
"""

import json
import sys
from pathlib import Path

PROTECTED = ["data/", "lab/folds.csv"]

event = json.load(sys.stdin)
tool_input = event.get("tool_input", {})
target = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
root = Path(event.get("cwd", ".")).resolve()

try:
    rel = Path(target).resolve().relative_to(root).as_posix()
except ValueError:
    sys.exit(0)

for p in PROTECTED:
    if rel == p or rel.startswith(p):
        print(
            f"Blocked: {rel} is protected (raw data / locked CV folds). "
            "Changing it would make every result in lab/results.csv incomparable. "
            "Ask the user if this is really intended.",
            file=sys.stderr,
        )
        sys.exit(2)

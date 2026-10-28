"""H17 variant: e022 + the e012 quadratic skew MAP kept as q_* columns."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e022_cubic_traj import DESCRIPTION as _D, make_fit_predict  # noqa: E402

DESCRIPTION = _D + " + MAP quadratique e012 conservée (q_*)"
FEATURES = "e022 + q_* (MAP skew quadratique e012)"

fit_predict = make_fit_predict(True)

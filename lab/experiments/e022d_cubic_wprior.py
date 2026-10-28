"""H17 ablation (leak-review): e022b with prior HGB pondéré 1/var(coef | n visites)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e022_cubic_traj import make_fit_predict  # noqa: E402
from e022b_cubic_traj_quad import DESCRIPTION as _D  # noqa: E402

DESCRIPTION = _D + " — ablation : prior HGB pondéré 1/var(coef | n visites)"
FEATURES = "e022b"

fit_predict = make_fit_predict(True, weighted=True)

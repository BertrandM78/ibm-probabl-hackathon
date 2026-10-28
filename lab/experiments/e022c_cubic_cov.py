"""H17 ablation (leak-review): e022b with Σ = np.cov débruitée au lieu de MinCovDet."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e022_cubic_traj import make_fit_predict  # noqa: E402
from e022b_cubic_traj_quad import DESCRIPTION as _D  # noqa: E402

DESCRIPTION = _D + " — ablation : Σ = np.cov débruitée au lieu de MinCovDet"
FEATURES = "e022b"

fit_predict = make_fit_predict(True, sigma_est="cov")

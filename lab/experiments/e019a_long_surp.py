"""H21 ablation: e017 + visit-level surprises only (no per-patient shape)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e019_long_surp_shape import make_fit_predict  # noqa: E402

DESCRIPTION = "H21 (ablation) : e017 + surprises visite seules (surp, infl, voisines, 1re visite)"
FEATURES = "e017 + L_* (feat_long, bloc surp)"
fit_predict = make_fit_predict(("surp",))

"""H20 ablation: e017 + per-patient shape of surprises only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e019_long_surp_shape import make_fit_predict  # noqa: E402

DESCRIPTION = "H20 (ablation) : e017 + forme patient des surprises seule (max/min/std/skew/médiane, zmax, résidus vs hb_post_full)"
FEATURES = "e017 + L_p_* (feat_long, bloc shape)"
fit_predict = make_fit_predict(("shape",))

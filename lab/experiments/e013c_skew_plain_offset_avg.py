"""H13-c: fixed 50/50 average of e012 (HGB on target) and e013b (HGB offset on hb_post_full), one shared build."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from feat_skew_proj import plain_and_offset  # noqa: E402

DESCRIPTION = "H13-c : moyenne fixe 50/50 de e012 (HGB sur target) et e013b (HGB offset sur hb_post_full skew), chacun projeté OLS w = 0.5, build commun"
FEATURES = "identiques à e012 ; 2 HGB (target / target − hb_post_full)"


def fit_predict(train, valid):
    plain, off = plain_and_offset(train, valid)
    return 0.5 * (plain + off)

"""H13-b: e012 with HGB as offset on the skew MAP trajectory (target - hb_post_full), then OLS projection w = 0.5."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from feat_skew_proj import blend, components  # noqa: E402

DESCRIPTION = "H13-b : HGB en offset sur la trajectoire MAP skew (cible target − clip(hb_post_full)), préd. = hb_post_full + résidu, puis projection OLS w = 0.5 comme e012"
FEATURES = "identiques à e012 ; offset hb_post_full skew"


def fit_predict(train, valid):
    return blend(components(train, valid, offset=True), {"ols": 0.5})

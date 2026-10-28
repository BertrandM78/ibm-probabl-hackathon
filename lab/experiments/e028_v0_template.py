"""H46: e022b with first-visit noise template sources (<src>_v0) in the skew template, cubic and quadratic MAPs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e027_rule_post import run  # noqa: E402
from feat_rule import v0_template  # noqa: E402

DESCRIPTION = "H46 : e022b avec template de bruit SN séparé à la 1re visite (off_v0, on_t_v0, on_nt_v0) dans template + MAP cubique + MAP quadratique"
FEATURES = "e022b (hb_*, q_* recalculés avec le template v0)"


def fit_predict(train, valid):
    with v0_template():
        return run(train, valid, rule_feats=False)

"""H34: e022b with cubic prior OLS and noise template fitted on uncensored train visits (T = P), c refit in-fold."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e027_rule_post import run  # noqa: E402

DESCRIPTION = ("H34 : e022b avec prior cubique (OLS patient) et template SN fittés hors visites censurées par la règle "
               "(T_i = round(c·T_{i−1}) ou T_i = T_{i−1}, c in-fold) : P latent au lieu du target monotonisé")
FEATURES = "e022b (hb_*, q_* recalculés)"


def fit_predict(train, valid):
    return run(train, valid, rule_feats=False, latent_fit=True)

"""User hypothesis (MNAR): OFF missingness frequency per patient (overall, cumulative, run length) on top of e019."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from e017_skew_temporal_delta import LAM, _hgb_delta  # noqa: E402
from e012_traj_skew import build as build_e012  # noqa: E402
from e019_long_surp_shape import BLOCKS  # noqa: E402
from feat_long import add_long  # noqa: E402
from feat_skew import hgb_predict  # noqa: E402
from feat_temporal import add_temporal, delta_target, reconstruct  # noqa: E402

DESCRIPTION = "MNAR OFF : e019 + fréquence de OFF manquant par patient (globale, cumulée, cumulée avant visite, série en cours, parmi visites traitées)"
FEATURES = "e019 + M_p_miss, M_cum_miss, M_cum_miss_prev, M_run_miss, M_p_miss_treated"


def add_miss(df: pd.DataFrame) -> pd.DataFrame:
    """Patient-level OFF missingness rates (no target). Order preserved."""
    df = df.copy()
    miss = df["off"].isna().astype(float)
    g = miss.groupby(df["patient_id"], sort=False)
    df["M_p_miss"] = g.transform("mean")
    df["M_cum_miss"] = g.transform(lambda s: s.expanding().mean())
    df["M_cum_miss_prev"] = g.transform(lambda s: s.shift(1).expanding().mean())
    df["M_run_miss"] = g.transform(lambda s: s.groupby((s != 1).cumsum()).cumsum())
    df["M_p_miss_treated"] = miss.where(df["on"].notna()).groupby(df["patient_id"], sort=False).transform("mean")
    return df


def fit_predict(train, valid):
    train, valid = train.reset_index(drop=True), valid.reset_index(drop=True)
    tr, va, _ = build_e012(train, valid)
    tr, va = (add_miss(add_long(add_temporal(d), BLOCKS)) for d in (tr, va))
    y = train["target"]
    d = delta_target(train, y)
    m = d.notna().to_numpy()
    A = hgb_predict(tr, va, y)
    D = _hgb_delta(tr, va, d, m)
    return np.clip(reconstruct(valid, A, D, LAM), 0, 132)

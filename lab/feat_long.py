"""H20/H21: surprises of debiased scores around the skew MAP trajectory (no target used).

surp = y_v - hb_post_loo per visit; per-patient shape of surp (max/min/std/skew/median) and of
y_v - hb_post_full. The noise of y_v has a sharp right edge, so max(surp) lower-bounds the patient level.
"""
import numpy as np
import pandas as pd

from feat_rater_scale import check_order

GROUP = "patient_id"


def add_long(df: pd.DataFrame, blocks=("surp", "shape")) -> pd.DataFrame:
    """Add L_* columns; NaN kept where y_v is NaN (skew needs >= 3 values). infl = 0 when y_v is NaN (by design)."""
    check_order(df)
    df = df.copy()
    surp = df["y_v"] - df["hb_post_loo"]
    g = surp.groupby(df[GROUP], sort=False)
    if "surp" in blocks:
        df["L_surp"] = surp
        df["L_infl"] = df["hb_post_full"] - df["hb_post_loo"]
        df["L_surp_prev"] = g.shift(1)
        df["L_surp_next"] = g.shift(-1)
        df["L_first_surp"] = g.transform("first")
    if "shape" in blocks:
        for s in ("max", "min", "std", "skew", "median"):
            df[f"L_p_surp_{s}"] = g.transform(s)
        se = df["hb_se_loo"].where(np.isfinite(df["hb_se_loo"]) & (df["hb_se_loo"] > 0))
        df["L_p_zmax"] = (surp / se).groupby(df[GROUP], sort=False).transform("max")
        gr = (df["y_v"] - df["hb_post_full"]).groupby(df[GROUP], sort=False)
        for s in ("max", "std", "median"):
            df[f"L_p_res_{s}"] = gr.transform(s)
    return df


def shift_report(tr: pd.DataFrame, va: pd.DataFrame) -> None:
    """Print train/valid distribution of surp and per-patient max(surp); warn if std differs by > 5 %."""
    q = [0.01, 0.1, 0.5, 0.9, 0.99]
    for name, f in (("surp", lambda d: d["y_v"] - d["hb_post_loo"]),
                    ("p_surp_max", lambda d: (d["y_v"] - d["hb_post_loo"]).groupby(d[GROUP]).transform("max"))):
        a, b = f(tr).dropna(), f(va).dropna()
        ratio = b.std() / a.std()
        print(f"  [long] {name:10s} train q={np.round(a.quantile(q).to_numpy(), 2)} sd={a.std():.2f} | "
              f"valid q={np.round(b.quantile(q).to_numpy(), 2)} sd={b.std():.2f}"
              + ("  WARNING sd shift > 5%" if abs(ratio - 1) > 0.05 else ""), flush=True)

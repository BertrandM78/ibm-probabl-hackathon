# EDA — Feature engineering longitudinal / asynchrone (vs e017)

Scripts : `lab/scratch/long_build.py` (features e012 construites par fold, côté valid → OOF cohérent),
`long_eda1.py` (espacement, structure du résidu), `long_eda2.py` (candidats × résidu),
`long_eda3.py` / `long_eda4.py` (HGB sur le résidu OOF, GroupKFold = folds du lab, par blocs),
`long_eda5.py` (enveloppe haute), `long_fold0.py k` (pipeline e017 complet ± nouveaux blocs, folds 0 et 1, non loggé).
Résidu r = target − pred e017 (RMSE 3.072).

## 1. Ce que le bloc temporel e015 couvre déjà
`feat_temporal.add_temporal` opère sur les colonnes **brutes** (on, off, ledd, tsi, gap, ratio) et contient déjà
Δâge prev/next (`t_dt_prev/next`), rang/visites restantes, dernière/prochaine valeur observée et son ancienneté,
interpolation linéaire en âge, pente OLS patient, stats patient (mean/std/min/max), `t_off_pcount`.
**Manque** : tout traitement temporel des scores débiaisés (`off_n`, `on_eq_n`, `y_v`) au-delà de prev/next/pmean,
et surtout tout ce qui décrit **la position des mesures par rapport à la trajectoire MAP** (`y_v − hb_post_loo`).

## 2. Espacement des visites : aucune information
- Âges arrondis à 0.1 an. Δâge : médiane 0.9 an, moyenne 1.05, sd 0.48, q1–q99 = 0.4–2.9, min 0.2.
- Irrégulier (CV intra-patient médian 0.31), à peine plus long en début de suivi (1.11 au rang 1 → 0.97–1.00 au-delà).
- Pas informatif : corr(Spearman) espacement moyen patient vs pente du target 0.037, vs niveau 0.010 ;
  corr Δâge vs off manquant 0.04. corr Δâge vs Δtarget 0.49 = simple progression (≈ 2.5/an).
- RMSE e017 plat selon le quintile de Δâge (3.01–3.15).

## 3. Structure du résidu e017
- Autocorrélation lag-1 du résidu intra-patient **0.905** ; 60 % de la variance est au niveau patient
  (RMSE moyenne patient 2.39, intra 1.94) ; corr(r, moyenne LOO des autres visites) 0.70.
  → l'erreur restante est une erreur de **niveau patient**, pas un bruit visite.
- Par position × source : first/off 2.83, first/on_t 4.44, last/on_t 4.04, last/on_nt 4.50, mid/off 2.50.
- Nombre de OFF mesurés : RMSE 5.12 (0), 4.62 (1), 3.87 (2), ~3.0–3.2 (3–7), 2.11 (8).

## 4. Candidats × résidu (corrélation de Spearman avec r)
| candidat | all | off_missing | first |
|---|---|---|---|
| Δâge prev/next, rang relatif, n_off | ≤ 0.025 | ≤ 0.015 | ≤ 0.03 |
| interp y_v en âge (LOO) − hb_post_loo | 0.041 | 0.025 | 0.094 |
| interp y_v par **rang** − hb_post_loo | 0.039 | 0.023 | 0.094 (âge ≈ rang) |
| noyau gaussien y_v (bw 1 an) − hb_post_loo | 0.050 | 0.030 | 0.097 |
| surp = y_v − hb_post_loo | 0.023 | 0.022 | **0.219** |
| infl = hb_post_full − hb_post_loo | 0.029 | 0.025 | **0.214** |
| off_n dernière/prochaine valide − hb, ancienneté | ≤ 0.03 | ≤ 0.02 | 0.06 |
| asynchrone : 1 − on_n_t / interp(off_n autres visites) | −0.010 | −0.021 | −0.205 |
| z-max patient de surp/hb_se_loo | **0.134** | | |

Corrélations marginales faibles hors première visite, mais effets non linéaires : voir §5.

## 5. HGB sur le résidu OOF (GroupKFold, mêmes folds) — Δ RMSE vs 3.072
| bloc ajouté aux colonnes de base (hb_*, y_v, off_n, pred…) | all | Δ ± std (fold) | off_missing | first |
|---|---|---|---|---|
| base seule (contrôle = bruit de réajustement) | 3.087 | +0.015 ± 0.015 | 3.583 | 3.085 |
| Δâge / position / n_off | 3.088 | +0.016 ± 0.017 | 3.585 | 3.094 |
| interp/noyaux de y_v en âge | 3.054 | −0.018 ± 0.019 | 3.559 | 3.044 |
| surprises voisines (surp, surp_prev/next, infl) | 3.034 | −0.038 ± 0.019 | 3.529 | 3.009 |
| off_n dernière/prochaine valide | 3.069 | −0.003 ± 0.014 | 3.565 | 3.085 |
| asynchrone on_t × off interpolé | 3.087 | +0.015 ± 0.014 | 3.582 | 3.088 |
| **forme patient des surprises** (max, min, sd, skew, n on_nt, 1re surp) | **2.962** | **−0.110 ± 0.028** | 3.434 | 3.040 |
| tout | 2.929 | −0.143 ± 0.027 | 3.405 | 2.993 |

Ablation forme patient : `max(surp)` seul −0.078 ± 0.034, `skew` −0.042, `first_surp` −0.029, `min` / `mean` / `on_nt` ≈ 0.
Lecture : le bruit de y_v a un bord droit abrupt (y_v/target q99 = 1.49 OFF, 1.54 on_t ; queue gauche longue) ;
la plus grande surprise positive du patient borne son niveau par le bas. Patients du quintile haut de max(surp) :
résidu moyen +0.56 (vs −0.17 à −0.39 aux quintiles bas) → e017 sous-estime ces patients.

## 6. Contrôle dans le vrai pipeline e017 (folds 0 et 1, `long_fold0.py`)
| variante | fold 0 L | off_missing | first | fold 1 L | off_missing | first |
|---|---|---|---|---|---|---|
| e017 | 2.986 | 3.482 | 3.159 | 3.003 | 3.504 | 3.079 |
| + surprises (surp, infl, prev/next, 1re) | 2.867 | 3.379 | 2.971 | 2.900 | 3.424 | 2.931 |
| + forme patient (stats de surp et de y_v − hb_post_full) | 2.824 | 3.286 | 3.067 | 2.779 | 3.242 | 2.995 |
| + les deux | **2.747** | 3.218 | 2.947 | **2.725** | 3.213 | 2.870 |

Gain −0.24 / −0.28 sur 2 folds, bien au-delà de l'estimation par empilement : HGB n'arrive pas à reconstruire
lui-même les différences y_v − trajectoire ni leurs agrégats patient. À confirmer sur 5 folds + revue fuite.

## 7. Ce qui ne marche pas (ne pas tester)
Espacement des visites, interpolation pondérée en âge vs par rang (identiques à 0.002 près), ancienneté du dernier OFF,
features asynchrones on(t) × off(t'), rang relatif : tous ≤ bruit de contrôle.

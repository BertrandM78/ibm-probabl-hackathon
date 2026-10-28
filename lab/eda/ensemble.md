# EDA — Diversité par ensembling (référence e017_skew_temporal_delta, CV 3.072)

Scripts : `lab/scratch/ens/ens1_corr.py` (corrélations + blends par paire), `ens2_theory.py` (frontière
théorique), `ens3_dump.py` (features e017 fold 0/3, fit train du fold seulement), `ens4/5_proto.py`
(familles, fold 0), `ens6_stack.py` (trajectoire seule + stacking résidu), `ens7_mlp.py` (MLP fold 0),
`ens8_mlp_f3.py` (MLP fold 3).

## 1. Runs existants : résidus OOF

Corrélation des résidus (extrait) :

| | e002 | e005 | e007 | e012 | e013b | e015 | e016 | e017 |
|---|---|---|---|---|---|---|---|---|
| e017 | 0.667 | 0.747 | 0.857 | 0.947 | 0.932 | 0.751 | 0.796 | 1 |

- Blend NNLS hors-fold par paire avec e017 : **aucun gain > 0.006** (meilleur e013b −0.0055, w 0.19).
  NNLS sur les 11 runs : 3.0675 (−0.004 ; poids e017 0.80, e013b 0.19).
- Les paires anciennes gagnent beaucoup quand rho ≈ 0.74 et RMSE proches : e005+e015 4.06/4.07 → 3.79
  (−0.27) ; e007+e016 −0.09. e017 a déjà absorbé les deux familles (structurelle + temporelle).
- Stacking niveau 2 (HGB sur résidu e017 ~ agrégats patient, cross-fit par fold) : **+0.031 ± 0.005** (pire).

## 2. Frontière théorique (blend optimal 2 modèles, σ1 = 3.072)

RMSE_blend² = σ1²σ2²(1−ρ²)/(σ1²+σ2²−2ρσ1σ2) ; gain nul si ρ ≥ σ1/σ2.

| σ2 | ρ max (gain > 0) | ρ pour −0.02 | −0.05 | −0.10 |
|---|---|---|---|---|
| 3.1 | 0.991 | 0.958 | 0.919 | 0.855 |
| 3.2 | 0.960 | 0.913 | 0.870 | 0.804 |
| 3.5 | 0.878 | 0.810 | 0.759 | 0.688 |
| 4.0 | 0.768 | 0.684 | 0.625 | 0.549 |
| 4.5 | 0.683 | 0.590 | **0.527** | 0.449 |

Un modèle à 4.5 doit avoir ρ ≤ 0.53 pour −0.05 : aucun candidat observé ne s'en approche
(kNN 5.44 ρ 0.565 → 0 ; hb_post_loo 4.64 ρ 0.727 → 0). Le levier réaliste = σ2 ≈ 3.0–3.3 avec ρ ≈ 0.85–0.90.

Structure de l'erreur e017 : var 9.44 = **5.69 niveau patient** + 3.74 intra ; autocorr. lag-1 intra-patient
0.905 ; dernière visite 3.79 vs milieu 2.93, première 3.12.

## 3. Prototypes fold 0 (ref e017 = 2.986), mêmes features e017

Gain = RMSE blend − RMSE e017 (NNLS 2 poids fitté sur le fold : optimisme négligeable, mais indicatif).

| Modèle | RMSE seul | ρ résidus | 50/50 | NNLS (w e017) |
|---|---|---|---|---|
| HGB niveau A seul (composante e017) | 3.078 | 0.974 | +0.027 | 0 |
| Ridge α=10 (dense standardisé) | 3.251 | 0.896 | +0.050 | −0.004 |
| Ridge niveau+delta, reconstruction λ=4 | 3.170 | 0.913 | +0.024 | −0.008 |
| ExtraTrees 300 | 3.185 | 0.929 | +0.044 | −0.001 |
| RandomForest 300 | 3.143 | 0.943 | +0.035 | −0.001 |
| kNN 30 | 5.439 | 0.565 | +0.78 | 0 |
| HGB abs_error [rec D] | 3.012 | 0.981 | −0.002 | −0.007 |
| HGB depth 3 × 2000 [rec D] | 2.980 | 0.981 | −0.017 | −0.017 |
| HGB max_features 0.3 [rec D] | 2.963 | 0.986 | −0.022 | −0.025 |
| HGB 127 feuilles lr .02 × 1500 [rec D] | 2.957 | 0.986 | −0.025 | −0.031 (w 0.15) |
| HGB bag 3×(70 % patients, 60 % cols) [rec D] | 2.993 | 0.981 | −0.011 | −0.012 |
| HGB log(target/hb_post_full) [rec D] | 2.980 | 0.978 | −0.020 | −0.021 |
| HGB sqrt(target) [rec D] | 2.985 | 0.990 | −0.008 | −0.008 |
| MLP 256-128, 1 seed, 80 ep | 3.526 | 0.769 | +0.078 | −0.023 |
| MLP niveau+delta 1 seed, rec λ=4 | 3.227 | 0.836 | −0.010 | −0.041 |
| MLP 5 seeds niveau | 3.157 | 0.852 | −0.031 | −0.049 |
| **MLP 5 seeds niveau + delta 1 seed, rec λ=4** | **3.048** | **0.879** | **−0.063** | **−0.068 (w 0.58)** |
| MLP (512,256,128) α 0.1 | 3.365 | 0.795 | +0.021 | −0.036 |

Contrôle fold 3 (ref 3.228, le pire fold) : MLP 5 seeds niveau + delta rec 3.331, ρ 0.897, 50/50 −0.034,
NNLS −0.042 ; avec delta bagué 5 seeds 3.307, ρ 0.904, 50/50 −0.040, NNLS −0.046.

Constats :
- LightGBM inutilisable (libomp absent), XGBoost/CatBoost/torch non installés → sklearn seulement.
- Arbres (RF/ET/HGB variantes, autres cibles) : ρ 0.93–0.99, gains ≤ 0.03 ; le meilleur (127 feuilles)
  est un *meilleur modèle* plutôt qu'un modèle divers, déjà couvert par e018 (HGB v2).
- **Le MLP est la seule famille réellement décorrélée à qualité comparable** ; ses seeds sont très
  diverses (ρ inter-seeds 0.77–0.81 vs 1.000 pour HGB) → le bag de seeds est efficace ici
  (3.53 → 3.16 pour 5 seeds), et la reconstruction delta s'applique aussi (3.16 → 3.05).
- La moyenne 50/50 fixe donne déjà ≈ tout le gain NNLS (modèle simple, w ≈ 0.6/0.4).

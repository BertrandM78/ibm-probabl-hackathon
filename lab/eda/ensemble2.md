# EDA — Ensemble 2 : composantes complémentaires à e023 / b002 (réf. b002 CV 2.579)

Scripts : `lab/scratch/ens2/` (`load.py`, `p1_cond` poids conditionnels + stackers, `p2_post` post-traitements,
`p3_meta` colonnes MAP valid des 5 folds (dump `scratch/resid/feat_f*.pkl`, fit train seul), `p4_corr` corrélations
+ NNLS gloutonne, `p5_oracle` oracles + sélecteur, `p6_cand` / `p7_joint` candidats déjà sondés folds 0/3,
`p8_theory`). Logs `*.log` à côté. Tous les poids sont fittés hors-fold (NNLS imbriquée sur les 5 folds) ou, pour
les sondes à un seul fold, sur une moitié des patients du fold puis évalués sur l'autre (80 tirages).

## 1. Les runs existants sont épuisés

| run | RMSE | ρ résidu vs e023 | ρ vs b002 | MSE niveau patient | MSE intra |
|---|---|---|---|---|---|
| e023_seq_nn | 2.610 | 1 | 0.988 | 4.55 | 2.26 |
| e022b | 2.778 | 0.859 | 0.928 | 4.88 | 2.84 |
| b002 | 2.579 | 0.988 | 1 | 4.42 | 2.23 |
| MAP pure `hb_post_full` | 3.588 | 0.669 | 0.732 | 7.47 | 5.39 |
| HGB niveau seul (`A` e022b) | 2.877 | 0.830 | 0.899 | 4.88 | 3.40 |
| e017 / e012 / e007 | 3.07 / 3.23 / 3.55 | 0.79 / 0.74 / 0.68 | | | |

- ρ niveau patient e023/e022b **0.887**, intra 0.821 : les deux familles se trompent sur les mêmes patients.
- NNLS imbriquée (e023, e022b) + 1 run parmi les 21 runs + MAP pure : meilleur e022d **−0.002 ± 0.002** ; MAP pure,
  `A`, e012, e007 : poids 0. NNLS sur tout : −0.001 ± 0.004. Glouton : rien après e022d.
- Le MAP « pure » n'est pas une composante : poids NNLS 0 sur les 5 folds (σ 3.59, ρ 0.73 ; il faudrait ρ ≤ 0.6).

## 2. Sélecteur / stacking par patient : le potentiel « oracle » est du bruit

| sonde | RMSE |
|---|---|
| b002 | 2.579 |
| oracle meilleur des 2 par patient (e023 vs e022b) | 2.343 |
| oracle poids optimal par patient w_p ∈ [0, 1] | 2.328 (w moyen 0.57, sd 0.43) |
| oracle meilleur des 3 (+ MAP) par patient | 2.206 |
| **NUL** : B ± \|e023 − e022b\|/2, signe tiré au hasard par patient | **2.334** |
| oracle décalage constant par patient de b002 | 1.494 |

L'oracle réel (2.343) n'est pas meilleur que l'oracle nul sans information (2.334) : le gain « meilleur des deux »
vient du choix a posteriori, pas d'une information exploitable. Confirmé par les sondes apprises (Δ vs b002, 5 folds) :

| stacker (niveau 2, cross-fit) | Δ |
|---|---|
| NNLS par sous-groupe (n visites, cohorte, 1re/dernière, off manquant, quintile \|diff\|, quintile niveau) | −0.001 à +0.003 |
| ridge diff × méta visite (p1) | −0.000 ± 0.005 |
| ridge poids variable, méta patient (n visites, p_off_miss, hb_se_full, q_hb_se, \|diff\|, écarts au MAP, surp_std, zmax, prior gap, A_p NaN, niveau) | +0.002 ± 0.013 |
| décalage patient appris (ridge / HGB d2, pondéré n visites) | +0.010 / +0.011 (0/5) |
| HGB petit sur le résidu | +0.006 à +0.017 |
| post-traitements b002 (projection poly 1–3, noyau, cummax, règle ×1.02, recalage affine) | −0.002 à +0.015 |

L'incertitude postérieure MAP localise l'erreur (décile haut de `hb_se_full` : RMSE 4.02, 24 % de la SSE) mais pas
son signe : elle ne sert ni à pondérer, ni à décaler.

## 3. Où b002 se trompe (RMSE e023 / e022b / b002)

| sous-groupe | visites | e023 | e022b | b002 | part SSE b002 | part niveau patient |
|---|---|---|---|---|---|---|
| n visites 4–5 | 22 % | 2.87 | 2.90 | 2.81 | 26 % | 73 % |
| n visites 9+ | 67 % | 2.55 | 2.76 | 2.53 | 64 % | 64 % |
| OFF manquant > 60 % | 19 % | 3.55 | 3.75 | 3.53 | 36 % | 73 % |
| dernière visite | 12.5 % | 3.30 | 3.46 | 3.25 | 20 % | 45 % |
| cohorte B | 11 % | 2.95 | 2.98 | 2.89 | 14 % | 71 % |
| target > 50 | 24 % | 3.44 | 3.70 | 3.40 | 41 % | 65 % |
| décile haut \|e023 − e022b\| | 10 % | 3.61 | 4.34 | 3.52 | 19 % | 61 % |

e023 bat e022b dans **tous** les sous-groupes ; l'écart se resserre seulement sur 4–5 visites (2.87/2.90, poids
NNLS 0.55/0.45 contre 0.76/0.24 sur 6+ visites, mais NNLS par n visites ne donne que −0.001). Les zones dures
(peu de OFF, peu de visites, cohorte B) sont à ~70 % de niveau patient : il faut une information nouvelle, pas
un meilleur mélange.

## 4. Candidats déjà entraînés (folds 0/3, dumps `scratch/meth`, `scratch/nn2`) en 3e composante

Gain honnête = NNLS (e023, e022b, cand) fittée sur la moitié des patients du fold, évaluée sur l'autre, moins la
même chose sans le candidat.

| candidat | fold | RMSE seul | ρ vs e023 | ρ vs b002 | + 3e comp. | poids |
|---|---|---|---|---|---|---|
| LightGBM l2 63 f. | 0 / 3 | 2.698 / 2.844 | 0.873 / 0.888 | 0.933 / 0.937 | −0.002 / −0.015 | 0.24 / 0.37 (remplace e022b) |
| LightGBM Huber(4) | 0 / 3 | 2.681 / 2.861 | 0.874 / 0.890 | 0.933 / 0.938 | −0.007 / −0.006 | 0.29 / 0.33 |
| HGB v2 bag 3 | 0 | 2.688 | 0.872 | 0.934 | −0.007 | 0.28 |
| GRU / TF même cible (= sous-modèles e023) | 0 / 3 | 2.58 / 2.78 | 0.986–0.988 | 0.97–0.98 | +0.001 | 0 |
| MLP par visite | 0 | 2.704 | 0.930 | 0.952 | +0.000 | 0 |
| **TF cible target − hb_post_full** | 0 | **2.561** | **0.933** | 0.940 | **−0.020 ± 0.008** | 0.38 |
| **TF amortized 18 col. (H31)** | 0 | 2.774 | **0.807** | 0.836 | **−0.024 ± 0.010** | 0.24 |
| TF positionnel raw 1 seed (`nn2`) | 0 | 3.057 | 0.720 | 0.740 | −0.018 ± 0.011 | 0.16 |

Combinaisons (fold 0, même protocole) :

| ajout à (e023, e022b) | Δ | poids (e023, e022b, …) |
|---|---|---|
| offset + amortized | **−0.038 ± 0.012** | 0.43, 0.05, 0.32, 0.21 |
| offset + amortized + nn2 | −0.044 ± 0.015 | 0.39, 0.03, 0.30, 0.17, 0.11 |
| + LightGBM Huber | −0.045 ± 0.015 | e022b 0.00, LGB 0.07 |
| fixe 0.8·b002 + 0.2·moyenne(offset, amortized) | −0.026 | — |

- ρ résidus offset/amortized 0.814 : les deux gains sont presque additifs (−0.020 et −0.024 → −0.038).
- La diversité vient de la **cible** (offset MAP) et des **entrées** (colonnes harmonisées seules), pas des seeds
  ni de l'architecture : GRU et TF même cible ont ρ 0.986 et un poids nul une fois e023 présent.
- Avec ces deux composantes, e022b tombe à un poids ≈ 0.03–0.05 : le HGB n'est plus utile qu'au build de features.
- LightGBM n'apporte que −0.002 à −0.015 au-dessus de H30 (cohérent avec la fiche H32 « faible avec H30 »).

Biais : le fold 0 a servi au développement des réseaux (revue H30) ; ces chiffres sont à confirmer sur 1/2/4.

## 5. Frontière théorique autour de σ1 = 2.579 (ρ maximal admis)

| σ2 | −0.02 | −0.05 | −0.10 |
|---|---|---|---|
| 2.6 | 0.955 | 0.908 | 0.833 |
| 2.7 | 0.901 | 0.850 | 0.773 |
| 2.8 | 0.856 | 0.802 | 0.723 |
| 3.0 | 0.782 | 0.721 | 0.639 |

Les candidats tombent exactement là où la théorie les place (offset 2.56 / ρ 0.94 ; amortized 2.77 / ρ 0.84 :
≈ −0.02 chacun). Un gain −0.10 demanderait un modèle à 2.7 avec ρ ≤ 0.77 : aucune famille observée n'en est proche.

## 6. Seeds de e023

Fold 0, par seed (A seul) : TF 2.75–2.82, GRU 2.72–2.85 ; bag 5 : 2.612 / 2.605. Modèle MSE(N) = B + V/N :
V ≈ 1.27, B ≈ 6.57 par architecture. Passer de 10 à 40 seeds retire ≈ 0.09 de MSE à e023 (≈ −0.02 seul,
≈ −0.01 dans le blend) pour un coût ×4 (~2.7 h CPU) : moins rentable que deux composantes diverses à 5 seeds.

## Gain réaliste

1–2 nouvelles composantes (TF offset + TF amortized, 5 seeds chacune) : **−0.02 à −0.04** sur b002
(→ ≈ 2.54–2.56), avec une 3e (positionnel raw) ≈ −0.005 de plus. Sélecteur, stacking niveau 2, MAP pure, runs
existants : 0. Rien ici ne comble l'écart à 2.2 : 63–73 % de l'erreur des zones dures est du niveau patient, que
l'ensembling ne corrige pas.

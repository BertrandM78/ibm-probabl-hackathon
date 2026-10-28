# EDA — Loi de progression symbolique (2026-10-13)

Scripts : `lab/scratch/sym/s1_families.py` (familles par patient), `s2_sparse.py` (recherche parcimonieuse),
`s3_rule.py` (coude contraint), `s4_complement.py` + `s5_eval.py` (MAP sur y_v bruité par fold, stacking).
Familles et sélection sur le target des patients train du fold ; scores sur patients valid ; formes sur n ≥ 8 visites.

## 1. Ajustement du target (sans bruit), 2 621 patients n ≥ 8

| Famille | k | exact (< 0.045) | LOO interne | extrapolation 2 dern. | BIC gagne |
|---|---|---|---|---|---|
| P3 + coude libre | 6 | 0.85–0.87 | 0.21 | 1.49–1.74 | 0.27–0.33 |
| P3 cubique | 4 | 0.66 | 0.24 | 1.64 | 0.58–0.66 |
| P3 + coude à pente ≈ 0 (règle, avec option sans coude) | 5 | 0.73 | 0.27 | 1.49 | — |
| SIG + linéaire | 5 | 0.61 | 0.17 | **13.7** (instable) | 0.01 |
| SIG a + b·σ((d−m)/w) | 4 | 0.16 | 0.27 | **1.23** | 0 |
| EXP + linéaire | 4 | 0.42 | 0.27 | 2.23 | 0.01 |
| P2 / EXP / LOG+lin / POW / LOG / P1 | 2–4 | ≤ 0.10 | 0.39–1.56 | 1.6–4.8 | ≤ 0.05 |

Recherche parcimonieuse (20 termes : polynômes, log, √, exp(±d/τ), σ, (d−k)+ fixes ; pénalité 2 %/terme) :
termes retenus instables entre folds mais quasi colinéaires sur [0, 20] ans (e^(d/10), e^(d/20), d³, d⁴) → aucune
signature exp/log/sigmoïde au-delà d'un polynôme. LOO baisse avec la taille (P4 0.170) mais l'extrapolation
se dégrade (P4 2.27) ; meilleure extrapolation `1 + d + √(d+1) + log(d+1)` (1.46).
PySR non installé (Julia absent) ; inutile ici : la loi est polynomiale exacte à l'arrondi.

## 2. Récupérable depuis y_v bruité ? (MAP gaussien, prior sans covariables appris sur le target du fold-train)

| Trajectoire (valid, OOF) | all | cubic | hinge | other |
|---|---|---|---|---|
| e022b hb_post_full (prior covariables + skew) | 3.60 | 3.12 | 3.55 | 5.48 |
| E1 cubique | 4.84 | 4.31 | 5.23 | 7.52 |
| E2 cubique + coude (BMA sur k, β ~ 1.8 ± 1.9) | 4.81 | 4.29 | 5.23 | 7.52 |
| E3 concave | 5.61 | 4.77 | 5.74 | 8.39 |

AUC de P(coude) (cubique vs coude/other) = **0.44** : coude non détectable (bruit visite sd ≈ 10 vs écart cubique/coude ≈ 0.3).
Stacking du résidu OOF e022b (HGB cross-fit sur les folds) : contrôle +0.014 vs e022b ; E1 −0.009 ± 0.007,
E2 +0.004 ± 0.007, E3 −0.007 ± 0.007 vs contrôle ; **aucun ne bat e022b seul (2.779)**.

## 3. Conclusion

- La base cubique est adéquate : misfit oracle 0.02 / 0.27 / 0.36 (cubic / hinge / other) → **< 1 % du MSE** e022b.
- L'écart d'erreur entre classes (2.56 / 2.80 / 3.88) est en partie confondu (niveau 38 / 42 / 49, moins de OFF),
  mais persiste à niveau égal pour « other » (quintile haut 4.96 vs 3.50) → piste = **prior** trop rétrécissant pour les
  trajectoires atypiques, pas la famille de fonctions.
- Corrige `dgp` / rapport du 2026-10-13 : le plafond « −0.15 si coudes modélisés » était confondu ; H31 abandonnée (renumérotée H59 au merge avec main, où H31 = Transformer amortized).

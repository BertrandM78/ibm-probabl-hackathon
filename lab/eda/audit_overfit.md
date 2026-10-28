# Audit d'overfitting — e019 et e022b (2026-10-12)

Scripts : `lab/scratch/audit/` (fresh_folds.py, oof_stats.py, tails.py, curve.py, adv_test.py).

## Verdict

Pas de surapprentissage préoccupant. Le CV du lab est quasi non biaisé (optimisme ≤ 0.01) et le gain
e022b vs e019 se reproduit hors des folds de sélection. Seule réserve : un seul point LB (e012, +0.12).

## 1. Biais de sélection sur les folds figés (test principal)

~25 runs sélectionnés sur les mêmes 5 folds → relance sur 2 partitions patients neuves (seeds 7, 2026).

| Modèle | Folds lab | Seed 7 | Seed 2026 |
|--------|-----------|--------|-----------|
| e019   | 2.812 | 2.810 | 2.821 |
| e022b  | 2.778 | 2.782 | 2.785 |
| Δ apparié | −0.033 ± 0.027 (5/5) | −0.029 ± 0.033 (4/5) | −0.036 ± 0.030 (5/5) |

Optimisme de e022b ≈ +0.005 (sélection de la variante b parmi 022/b/c/d) : négligeable.

## 2. Robustesse du gain e022b vs e019 (OOF lab)

- Bootstrap patient : Δ −0.033, IC95 [−0.051, −0.015], P(Δ<0) = 1.00. e022b vs e022 : IC95 [−0.020, +0.004] → b ≈ e022.
- Queues symétriques (top 1 % gagnants −9.1k MSE, perdants +7.7k) ; sans 1 % extrêmes de chaque côté, 84 % du gain reste.
- Gain localisé où la cubique a du sens : n ≥ 9 visites (−0.25 à −0.61 MSE/visite) ; n = 4–8 neutre ou légère perte.
- Corrélation des erreurs e019/e022b 0.961.

## 3. Surapprentissage interne des HGB (e022b, folds 0 et 3, 290 features)

| itérations | train niveau | valid niveau | valid final f0 | valid final f3 |
|-----------|-------------|-------------|----------------|----------------|
| 200  | 2.12 | 2.89 | 2.787 | 2.940 |
| 500 (actuel) | 1.58 | 2.85 | 2.745 | 2.905 |
| 1000 | 1.10 | 2.83 | 2.729 | 2.892 |
| 1500 | 0.84 | 2.82 | 2.725 | 2.892 |

Gros écart train/valid (mémorisation), mais le valid baisse jusqu'à 1000–1500 itérations : pas de
surapprentissage nuisible ; 500 est même légèrement sous-ajusté (−0.013 à −0.020 à 1000 it., à vérifier sur 5 folds).

## 4. Asymétrie train/valid des features construites et dérive train ↔ test

- Classifieur train-fold vs valid-fold sur les 290 features : AUC 0.529 (f0), 0.506 (f3) ; aucune feature dominante
  (rater_id = composition du fold). Le cross-fit des priors fait bien son travail.
- Train vs test sur brutes + contexte patient : AUC 0.493 (StratifiedGroupKFold ; GroupKFold simple donne un AUC
  artificiel de 0.0, piège à retenir).
- Prédictions test moyenne 36.77 vs 37.46 en OOF, stable depuis e012 ; `off` brut test 26.4 vs 27.0
  → population (≈1.5 σ d'échantillonnage), pas le modèle.

## Limites

- CV ↔ LB : un seul point (e012 +0.12). Soumettre e019 et/ou e022b pour savoir si l'écart reste constant.
- Les constantes (XS, MIN_N_SIG, LAM, k…) ont été choisies sur les folds du lab, mais les partitions neuves les incluent
  sans dégradation.

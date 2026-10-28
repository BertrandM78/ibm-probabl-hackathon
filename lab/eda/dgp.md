# EDA — axe DGP (reverse-engineering du générateur)

Scripts : `lab/scratch/dgp/d1.py` … `d33.py` (train seul ; résidus = `lab/runs/e017_skew_temporal_delta/oof.csv`).
Tout ce qui utilise `target` ici est de l'EDA : aucune constante ne doit sortir figée dans une expérience
sans refit in-fold.

## Résumé

1. **Le true OFF est un polynôme cubique de la durée de maladie, par patient**, à l'arrondi 0.1 près
   (target toujours à 1 décimale). Patients ≥ 8 visites, target > 0 : cubique exacte (RMSE < 0.045,
   arrondi seul = 0.029) pour **67 %** (1 709 / 2 546). Résidu poolé par degré (≥ deg+3 visites) :
   droite 1.86, quadratique 0.41, cubique 0.22.
2. **Les 33 % restants = cubique + un coude** (hinge `(dur − k)+`) : 64 % d'entre eux deviennent exacts
   avec ce terme. Coude à dur ≈ 11.0 ± 2.9 ans, quand la pente de la cubique atteint ≈ 0 (médiane −0.25),
   pente après coude 2.18 ± 0.91. **Le target ne décroît jamais** (0 diff. négative / 39 014).
3. **Coefficients indépendants des covariables** : R² (adx, adx², cohort, sexM, gene, dmin) = 0.047 (c0),
   0.007 (c1), 0.041 (c2), 0.011 (c3). Cubique en dur (non centrée), patients exacts : c0 23.0 ± 13.4,
   c1 2.30 ± 1.09, c2 0.009 ± 0.095, c3 0.001 ± 0.011 ; corr(c1, c3) −0.81, corr(c0, c2) −0.88.
   Pente pop. selon dur : 2.1 (0–1 an) → 2.7 (4–10) → 1.4 (10–14) → 0.5–1.1 (> 14).
4. **Le target est latent, indépendant des mesures** : les écarts du target à sa quadratique patient ne
   corrèlent ni au bruit off (−0.027), ni à on, ledd, tsi (|r| < 0.02).
5. **Bruit de mesure** : facteur rater multiplicatif (intercept de régression quantile médiane −0.58 ± 0.73,
   pente 0.89 ± 0.09) ; 50 facteurs ≈ log-normaux (Shapiro log p = 0.98, uniforme rejeté KS p = 0.001),
   aucune valeur « ronde ». Choc visite asymétrique (mode 1.10–1.15, bord droit net, queue gauche), sd
   relative décroissante avec le niveau (0.36 à target 5–15 → 0.17 à > 60), présent aussi chez les non traités.
   **Imprévisible** : HGB GroupKFold sur toutes les covariables visite → R² 0.013.
   **Loi SN déjà efficace** : niveau patient (≥ 4 off) erreur relative 0.083 (SN) vs 0.088 (KDE non
   paramétrique) vs 0.119 (moyenne) → rien à gagner sur la forme de la loi.
6. **Facteur rater mal estimé par la médiane** : estimer s_r par MLE SN (forme commune) au lieu de la
   médiane de off/target réduit l'erreur du niveau patient de 0.0834 à 0.0800 (−4 % sd, −8 % MSE ;
   ΔMSE 5.8e-4 ± 0.9e-4, z = 6.5, demi-train → autre demi). Échelle relative par rater : non fiable
   (split-half 0.17) → c'est la **localisation** qui compte, pas une variance par rater.
7. **Manquants** : `off` manquant ≈ MCAR sachant le traitement (parmi on présent : 0.594–0.609 sur
   les 6 sextiles de target). `on`/`ledd` manquants = non traité, **et l'initiation du traitement est
   déclenchée par le target** : hazard logit = −4.63 + 0.113·target + 0.216·dur (vs −2.93 + 0.074·off_prev :
   le target courant explique mieux que le off observé). Aucune visite traitée avec target < 15.8 ;
   1re visite traitée ≥ 18.4. Le traitement est absorbant (0 retour).
   Zéros : off == 0 à 7.6 % pour target ≤ 20.9, ~0 au-delà ; on == 0 0.13 %.

## Résidus e017 (RMSE 3.072)

| Groupe | n | biais | RMSE |
|---|---|---|---|
| source off | 25 059 | +0.01 | 2.68 |
| source on_t | 14 437 | −0.05 | 3.45 |
| source on_nt | 4 494 | −0.02 | 3.90 |
| 1re visite / milieu / dernière | 5 576 / 33 438 / 5 576 | +0.06 / −0.01 / −0.05 | 3.12 / 2.93 / **3.79** |
| jamais traités / toujours traités | 3 283 / 9 893 | +0.02 / −0.08 | 1.46 / **3.94** |
| n_off patient 0 / 1 / 4 / 8 / 12 | | | 5.03 / 4.52 / 3.11 / 2.35 / 1.41 |
| dur 0–1 / 5–8 / 11–15 / > 15 | | | 2.59 / 3.13 / 4.00 / 4.89 |

- Biais par niveau de target (+0.5 bas, −1.8 à 60–80) = régression vers la moyenne normale ;
  **niveau patient parfaitement calibré** (régression target moyen ~ préd. moyenne : pente 1.001).
- Biais autour de l'initiation du traitement ≤ 0.09 : HGB exploite déjà le statut traité.
- Rater : biais moyen par rater sd 0.17, corr avec s 0.39, amplitude ±0.2 → négligeable en direct.
- Monotonie : 63 décroissances préd. sur 39 014 (moy. −0.16) ; isotonique par patient : −0.0001.
- **Décomposition intra-patient du MSE 9.44** : niveau 5.69 · pente 2.37 · courbure (quad) 1.22 · reste 0.15.
- **Courbure sous-dispersée** : corr(courbure préd., courbure target) par patient = 0.73, mais
  régression target ~ préd. de pente **1.69** (sd 0.125 vs 0.288). Pente : calibrée (1.05).
  Post-traitement `pred + α·Q + β·C3` (Q = composante x² centrée de la quadratique patient des préd.,
  C3 = composante cubique orthogonalisée), α, β fittés sur les 4 autres folds : α 0.40–0.43,
  β −0.49 ; Δ par fold −0.024, −0.028, −0.018, −0.016, −0.027 (5/5). Projection simple sur la
  quadratique patient (w 0.5–0.75) : −0.009 sur chaque fold.

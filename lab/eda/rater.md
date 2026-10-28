# EDA — axe Rater, 2e passe (réf. e023_seq_nn, CV 2.610 ; b002 2.579)

Scripts : `lab/scratch/rater2/r1.py` … `r6.py` (`r4b.py` = r4 appliqué à d'autres runs). Le target n'est utilisé
que sur le train ou sur l'OOF. Complète `measurement.md` §1, qui reste valable (rater multiplicatif pur commun
on/off, pas d'additif, pas de dérive, pas d'effet cohorte ni de dépendance au niveau).

## Verdict

**L'axe rater est épuisé pour e023 / b002.** Toutes les sondes sont au niveau du bruit, sauf une : un meilleur
estimateur de localisation de s, qui plafonne à −0.002 sur b002.

| Piste | Sonde | Constat |
|---|---|---|
| Biais OOF e023 par rater | sd du biais moyen par rater | 0.088 contre 0.086 ± 0.008 sous permutation : **nul** ; corr avec s −0.07 |
| RMSE OOF e023 par rater | sd 0.181 contre 0.110 sous permutation par visite | effet de grappe patient : fiabilité split-half par patient **0.053** |
| Résidu e023 intra-patient par rater | sd des moyennes par rater de (res − moyenne patient) | 0.038 contre 0.042 ± 0.003 (permutation intra-patient) : nul |
| Variance / forme du bruit propre au rater | split-half par patient (20 tirages) de stats de log(off/(s·target)) | IQR 0.05, MAD 0.08, q10 −0.13, taux bas −0.18, sd ON 0.11 : **aucune** forme propre |
| Manquants / valeurs erronées par rater | split-half des taux | off 0.02, on −0.08, ledd 0.00, tsi_on 0.00, off == 0 −0.08 : aucun |
| Interaction rater × patient | cov. du bruit off entre 2 visites d'un patient, même rater vs autre | 0.0023 ± 0.0017 (var. totale 0.158) ; ON 0.0007 ± 0.0012 : nulle |
| Affectation (rater attitré, sites) | pool de raters par patient | 2.74 raters/patient (contre 7.35 si iid), rater principal 53 % des visites, même rater qu'à la visite précédente 33.6 % (train) / 33.9 % (test). La stickiness ne dépend ni de la cohorte (0.338 / 0.318), ni du traitement, ni de l'écart de temps |
| Sites / clusters de raters | lift de co-occurrence des paires de raters | quantiles identiques à des pools tirés au hasard (médiane 0.67 / 0.67, q95 1.02 / 1.01) : **pas de site** |
| Le pool de raters prédit-il quelque chose ? | ridge sur les parts de visites par rater (CV patient) | R² target patient −0.001, résidu e023 patient −0.001, cohorte B −0.000 (chi² rater × cohorte p = 1.5e-5 au niveau visite = artefact de grappe) |
| Facteur on ≠ off, additif, dérive | — | déjà réfuté (`measurement.md` §1) |

## Le seul signal : localisation de s par un quantile haut

La loi de log(off/target) a un bord droit net (centré sur la médiane rater : q80 0.18, q90 0.24, q99 0.38). Un
quantile haut est donc un estimateur de localisation **bien plus précis** que la médiane (split-half par patient,
40 tirages, sd d'échantillonnage pour un demi-train ; dispersion entre raters 0.098 pour tous) :

| estimateur | q50 (actuel) | q70 | **q80** | q85 | q90 | q95 | q99 |
|---|---|---|---|---|---|---|---|
| sd d'échantillonnage (½ train) | 0.0270 | 0.0151 | **0.0103** | 0.0125 | 0.0141 | 0.0169 | 0.0294 |

→ variance de l'erreur sur s divisée par **6.9** (0.027² / 0.0103²). En in-fold (80 % du train), l'erreur de s
passe d'environ 2.1 % à 0.8 %. La médiane est en plus sensible au mélange de patients d'un rater : le corps de la
loi glisse chez les traités (q50 −0.014 contre +0.023) et en cohorte B, mais pas le bord (corr(q50 − q95, part de
traités du rater) 0.29).

**Ce que ça rapporte en aval** (D = log(s_méd / s_q80) in-fold, sd 0.021 ; correction cross-fold
pred·(1 + b·D), b fitté sur les 4 autres folds) :

| run | corr(rés. rel. (fold, rater), D) | pente b | RMSE OOF | Δ par fold |
|---|---|---|---|---|
| e023_seq_nn | 0.13 | 0.06–0.09 | 2.6096 → 2.6090 | −0.002 … +0.001 |
| e022b_cubic_traj_quad | 0.41 | 0.19–0.21 | 2.7781 → 2.7730 | −0.003 … −0.007 (5/5) |
| b002_seq_e022b | 0.22 | 0.11–0.13 | 2.5790 → 2.5771 | −0.004 … +0.001 (4/5) |

e023 n'est presque pas sensible à l'erreur de s (pente 0.07, alors qu'une propagation complète au niveau patient
donnerait ≈ 0.35). Plafond pour e023 : b²·E[pred²]·σ²_s ≈ 0.07² × 1 678 × 0.021² = 0.0036 de MSE, soit
**−0.0007 RMSE** même avec un s parfait. La seule piste qui reste est de mettre s_q80 dans e022b (sa MAP est plus
sensible) : environ −0.005 à −0.01 sur e022b, −0.002 attendu sur b002. C'est négligeable selon la règle de
décision.

Note sur la sonde `r1.py` : la pente 0.31 de résidu ~ log(s_in-fold / s_train-complet) est un **artefact** (s
complet contient les visites de valid, donc leur propre bruit). Seule la comparaison r4 (deux estimateurs in-fold)
est propre.

## Transductif (s estimé avec les patients valid/test, sans target)

C'est légitime (aucun target valid), mais inutile : l'erreur de s ne coûte au plus que −0.0007 sur e023, et un
estimateur sans target (effets fixes patient + rater) ne fait que corr 0.948 avec s (`measurement.md`), donc moins
bien que s_q80 sur le train seul.

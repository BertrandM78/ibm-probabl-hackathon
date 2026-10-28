# EDA — Erreur de niveau patient (réf. b002 CV 2.579, e023 CV 2.610)

Scripts : `lab/scratch/level/l1.py` … `l26.py` (frames `d.pkl` / `P.pkl`). Résidus OOF :
`lab/runs/{b002_seq_e022b,e023_seq_nn,e022b_cubic_traj_quad}/oof.csv`. r = préd. − target ;
niveau = moyenne patient de r ; rel = niveau / target moyen du patient. Tout ce qui touche `target`
relève de l'EDA (train / OOF), à refitter in-fold.

## Résumé

1. **Le niveau reste 66 % de la MSE et il est bien calibré.** b002 : MSE 6.65 = niveau 4.42 · pente 1.56
   · reste 0.67 (e023 : 6.81 = 4.55 · 1.61 · 0.65 ; e022b : 7.72 = 4.88 · 1.85 · 1.00). Avec un niveau
   parfait, b002 serait à **1.49**. Calibration : pente target moyen ~ préd. moyenne 0.999 (0.989 à 1.001
   dans tous les sous-groupes testés), biais par décile de préd. ≤ 0.15.
2. **Le signe de l'erreur de niveau n'est pas prévisible ; seule son ampleur l'est.** Un HGB cross-fit
   (folds du lab) sur 22 covariables patient (n, n_off, n_on, paires, raters, s moyen, cohorte, durée,
   préd. moyenne et pente, ratio on/off…) **dégrade** le score : α 0.25 → 2.5847, α 1 → 2.622 (base 2.579).
   |niveau| est prévisible (corr 0.386), mais la part d'erreurs positives reste 54–56 % dans chaque quintile.
3. **Les deux modèles partagent la même erreur de niveau** : corr. niveau e022b/e023 **0.887**, contre
   0.821 pour la partie intra-patient. Un blend séparé niveau/écart fait 2.5761 (−0.003). Des poids NNLS
   par sous-groupe observable (n_off, toujours traité, position, n, désaccord, off manquant) font 2.578 à
   2.585 : rien.
4. **L'ampleur est fixée par la quantité d'information de mesure**, et non par le rater ou les covariables :

| n_off patient | 0 | 1 | 2 | 3 | 4 | 5–6 | 7–8 | 9–12 |
|---|---|---|---|---|---|---|---|---|
| rms niveau | 4.67 | 3.87 | 2.86 | 2.04 | 1.93 | 1.93 | 1.48 | 0.90 |

| groupe | % patients | part du SSE niveau | rms | rel-rms |
|---|---|---|---|---|
| toujours traités | 22.5 % | **43 %** | 3.06 | 0.067 |
| n_off ≤ 2 | 19 % | 31 % | 3.32 | 0.082 |
| visites ON mais aucune paire on/off/tsi utilisable | 28 % | **37 %** | 2.62 | 0.069 |
| n_off ≥ 6 | 32 % | 23 % | 1.50 | 0.045 |
| jamais traités | 11 % | 1.4 % | 1.01 | 0.109 |

   Top 5 % des patients = 47.6 % du SSE niveau (top 10 % = 63.5 %). Ils sont sous-estimés à 66 %
   (niveau moyen −2.44), ont un target plus élevé (49 contre 35), 62 % de visites sans OFF (contre 39 %),
   3.1 visites on_nt (contre 1.2), et sont toujours traités dans 49 % des cas (contre 21 %). Chez les
   sous-estimés, les mesures sont basses de façon groupée : moyenne log(off/(s·T)) −0.22 et log(on/(s·T))
   −0.76, contre +0.005 et −0.48 chez les sur-estimés.
5. **Amplitude A_p : c'est le principal levier qui reste identifiable.** L'estimateur cible du target
   (sans paire, A_p = A0) ; sd inter-patient vraie de A ≈ **0.097** (variance inter 0.0095, intra-patient
   0.0036 pour a_v, train + test sans target → k* = 0.38, contre k = 0.5 actuellement : réglage correct).
   39 % des patients n'ont aucune paire utilisable. rel-rms du niveau selon le nombre de paires :
   0 → 0.082, 1 → 0.064, 2 → 0.054, 3 → 0.047, ≥ 4 → 0.041. corr(erreur relative du niveau, A_oracle −
   A_estimé) : **−0.50** sans paire (n = 132 patients avec ≥ 3 ON au plateau), −0.33 avec 1–2 paires,
   −0.23 avec ≥ 3 paires (A_oracle = médiane de (1 − on/(s·T))/g ; cette valeur partage le bruit ON,
   donc les corrélations sont optimistes). La variance de a_v dépend de g : 0.0123 (g 0.5–0.7) contre
   0.0040 (g > 0.85) ; l'arrondi entier ne pèse que 6 % de cette variance (2.3e-4 sur 3.9e-3).
6. **Post-pic (règle ×1.02, H33/H34)** : patients avec ≥ 3 visites G : biais niveau −0.59 / −1.39
   (nG 3–4 / ≥ 5), mais ce n'est que 13 % du SSE niveau. Les patients sans G (54 % du SSE) sont non biaisés
   (+0.06). Ce n'est pas le gros du problème.

## Pistes fermées (avec chiffres)

| piste | constat |
|---|---|
| Nombre de raters, rater rare, s moyen/dispersion des raters du patient | rms niveau plat : 2.33 / 2.14 (2 / 3 raters), 2.15–2.29 par bin de s moyen |
| Interaction rater × patient (bruit partagé si même rater) | produit des écarts centrés intra-patient, OFF non traités : même rater −0.0354 contre raters différents −0.0359 ; ON plateau −0.0132 contre −0.0140 → aucune |
| Cohorte, gène, sexe, âge, âge au diagnostic manquant | biais niveau ≤ 0.04 (sauf l'âge > 80 : +0.36, n = 103) ; cohorte B rms 2.47 contre 2.11, mais B = 4.35 visites et 2.7 OFF par patient contre 8.9 et 5.1 (info) |
| patient_id, Index | Spearman avec target moyen / niveau / durée < 0.01 ; pas de préfixe porteur |
| Seuil de traitement (aucune visite traitée à T < 15.8 ; P(traité) 0 si T ≤ 15, 0.03 à 15–20, 0.39 à 20–25) | déjà respecté : 67 visites traitées seulement ont une préd. < 20 |
| Blend par composantes / poids NNLS par sous-groupe | −0.003 / +0.000 à +0.006 |
| Template OFF spécifique à la 1re visite | **constat réel** : à k = 0, log(off/(s·T)) a iqr 0.27 contre 0.45 et q25 −0.10 contre −0.27 (moins de queue basse) ; régression quantile médiane : k0 +0.104, log T +0.090 par unité, traité −0.048, cohorte B +0.037 (± 0.007) ; au q75 tout ≈ 0 (k0 +0.000, log T −0.009). C'est le **taux de contamination basse** qui dépend de k, T, traité et cohorte, pas le cœur. Sonde (forme de T oracle, MLE KDE du log-niveau, demi-échantillons) : template commun 0.1467 contre template scindé k0 / k>0 0.1476 → **aucun gain** : l'estimation du niveau repose sur le cœur droit, que la contamination ne touche pas. Même plafond attendu pour H38 (template traité / non traité). |

## Lecture

L'erreur de niveau correspond à l'incertitude a posteriori : elle est calibrée, sans signe prévisible,
commune aux deux familles de modèles, et proportionnelle au manque d'OFF et à l'incertitude de A_p.
Il y a deux façons de la réduire : (a) **mieux identifier A_p** chez les patients ON-lourds (fusion H39/H40/H41 :
paires sans tsi, effet aléatoire, écart ON / trajectoire OFF), qui portent 37–43 % du SSE niveau ;
(b) **apprendre le modèle de mesure sur plus de patients** (le modèle est limité par les données,
−0.08 à −0.10 par doublement, `residual.md`), par exemple par pré-entraînement auto-supervisé sur X sans
target (train + valid + test).

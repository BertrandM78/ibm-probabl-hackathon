# EDA — Manquants + structure des visites (vs e022b, CV 2.778)

Scripts : `lab/scratch/miss/m1.py` … `m18.py` (+ `base.py`, `dump_AD.py`). Résidu r = target − pred OOF
`lab/runs/e022b_cubic_traj_quad/oof.csv`. Usage de `target` = EDA seulement.

## Résumé

1. **off_missing (3.25) n'est pas un biais, c'est un déficit d'information localisé hors de l'enveloppe des OFF.**
   Biais ≤ 0.17 partout. Parmi les visites traitées : off mesuré 2.82 vs manquant 3.25 ; une visite off-manquante
   **encadrée** par deux OFF mesurés fait 2.79 (= off mesuré). Le surcoût vient des visites hors enveloppe :
   après le dernier OFF 3.59 (n 6 322), avant le 1er OFF 4.21 (n 1 570), patient sans OFF 5.14 (n 290).
2. L'erreur croît avec la distance au OFF le plus proche (2.9 à < 1.2 an → 3.5 à 2–3 ans → 4.6 à > 5 ans) et
   est à 90 % un **décalage de niveau patient** (zone après : MSE 12.9 = 11.6 patient + 1.3 intra ;
   corr(r à la dernière visite OFF, r moyen après) 0.79). Calibrage parfait (Δtarget ~ Δpred : pente 1.02 après,
   1.005 avant), aucune décroissance prédite → pas de recalibrage possible.
3. **Variance théorique** : à niveau égal le bruit relatif off/on est proche ; ce qui manque est l'ancrage du
   facteur médicament A_p. RMSE des visites off-manquantes selon le nb de paires on/off utilisables du patient :
   3.80 (0) · 3.17 (1) · 3.17 (2) · 2.83 (3) · 3.01 (4) · 2.66 (≥ 5). corr(r patient, A_vrai − A_utilisé)
   0.33 / 0.28 / 0.29 / 0.24 (0 / 1 / 2–3 / ≥ 4 paires). Borne oracle (A_vrai bruité, in-sample) : 2.778 → 2.728.
4. **A_p** : constant dans le temps (a_v par tranche de durée 0.462–0.477, intra-patient vs dur p = 0.2 une fois
   target/ledd contrôlés). Variance intra 0.063 / inter 0.099 → shrink optimal k ≈ 0.40 (code : 0.5, OK).
   **2 962 paires on/off (24 %) sont jetées** car `tsi_on` manquant (g NaN → non utilisable) ; 910 des 1 646
   patients traités sans paire utilisable en ont ≥ 1. 1 − on/off sur ces paires : corr 0.39 avec A_vrai
   (0.63 pour une paire au plateau). Le résidu e022b reste corrélé à ce A (corr 0.15, 851 patients ;
   correction linéaire in-sample 3.712 → 3.663 sur 3 532 visites).
   Estimation asynchrone de A (on_t / off interpolé d'autres visites) : corr −0.035 avec A_vrai → inutile
   (cohérent avec `longitudinal.md` §7).
5. **Cohort B (n 4 954, 2.98)** = suivi court, pas une autre population. 4.3 visites (A 8.9), espacement
   1.52 an (A 0.99), n_off 2.7 (5.1). À n égal : n = 4 B 3.14 / A 2.91, mais niveau 32.7 / 29.9 → erreur
   relative identique (0.096 / 0.097). Patient-mean RMSE à n = 4–5 : 2.50 / 2.43. Calibré (pente 0.995).
   PK identique (on/off par tsi à ±0.01, A_p 0.548 / 0.544), raters communs (50/50).
   **Seule vraie différence : la forme du bruit OFF** — off/(s·target) médiane 1.066 vs 1.000 (stable sur
   5 folds 1.047–1.085, 90 % des raters), q99 1.40 vs 1.48, SN α −6.1 vs −3.5 ; régression quantile médiane
   off/s ~ target pente 1.12 vs 1.02 (p < 1e-4), q90 identique (1.206). **Mais** un template SN par cohorte
   n'améliore pas l'estimation du niveau patient (cross-fit, MLE d'échelle : B 0.1455 → 0.1464 débiaisé ;
   A 0.1142 → 0.1136). → pas de modèle séparé ni de template par cohorte.
6. **Peu de visites** : ce n'est pas n qui coûte mais n_off. Traités, RMSE par n_off patient : 5.14 (0),
   4.58 (1), 3.74 (2), 2.96 (3), ~3.1 (4–6), 2.15 (≥ 8) ; à n = 4, n_off 3–4 → 2.5. Covariables : rien
   (coefficients de trajectoire R² ≤ 0.05, `dgp.md`) ; age_at_diagnosis manquant 2.57 vs 2.79, gene
   manquant 2.84 vs 2.75 → pas de sous-groupe orphelin.
7. **Schéma de visites** : test = train (n moyen 7.9/8.0, span 7.2/7.3, dur0 1.49/1.50, off manquant 0.38/0.40,
   espacement 1.10/1.10). Abandon (n < 12 en cohorte A) non informatif : pente target 2.3–2.8 selon n,
   biais dernière visite −0.23 à +0.27 sans tendance. `tsi_on` : ICC 0.004 (aucune habitude patient),
   manquant tout-ou-rien pour 20 % des patients → pas d'imputation possible.
8. **LEDD** : ledd = 447 + 43·dur − 2.3·age_dx (R² 0.52, + target : coef 0.22, R² inchangé). Intra-patient :
   corr(ΔLEDD, Δrésidu) 0.0004, corr(ledd_c, r_c) 0.018, ledd_c ~ target_c −0.007. Résidu plat par quintile
   de LEDD (biais −0.02…+0.05). **Mort**, confirme H12. ledd_present 3.10 = simple effet « traité + niveau haut ».
9. Première visite déjà traitée et OFF manquant : 4.39 (n 750) vs non traitée 2.47 → même phénomène (zone avant).

## Tableaux clés

| zone (visites off manquant) | n | biais | RMSE |
|---|---|---|---|
| OFF mesuré (toutes) | 25 677 | −0.01 | 2.37 |
| encadrée par 2 OFF | 10 731 | +0.06 | 2.79 |
| après le dernier OFF | 6 322 | −0.01 | 3.59 |
| avant le 1er OFF | 1 570 | +0.17 | 4.21 |
| aucun OFF chez le patient | 290 | +0.01 | 5.14 |

| k-ième visite après le dernier OFF | 1 | 2 | 3 | 4 | ≥ 5 |
|---|---|---|---|---|---|
| RMSE | 3.20 | 3.60 | 3.99 | 3.92 | 4.69 |

## Reconstruction hétéroscédastique (fold 0, `dump_AD.py` + `m18.py`)

Composantes e022b fold 0 : niveau A seul 2.845, reconstruction λ = 4 2.745. Poids du terme niveau réduit
pour les visites hors enveloppe OFF (w_ext) et encadrées (w_inside) : meilleur 2.7416 (w 0.5 / 0.5),
soit **−0.003** → négligeable, non poursuivi (fold 3 non lancé, inutile).

## Pistes fermées par cette EDA (ne pas re-proposer)

- Template de bruit / modèle séparé par cohorte (MLE cross-fit sans gain, B calibré).
- Prior trajectoire plus informatif par covariables pour les patients à 4–5 visites (R² ≤ 0.05, cf. dgp.md).
- Espacement, abandon, structure test vs train, imputation de `tsi_on` (ICC 0.004).
- LEDD : niveau, changements, paliers (corr ≤ 0.02 avec le résidu, intra-patient).
- Recalibrage des incréments en zone d'extrapolation (pente 1.02 / 1.005), poids λ par zone (−0.003).
- A_p asynchrone (on_t / off interpolé) : corr −0.035 avec A vrai.

## Seuls leviers restants (faibles)

A_p : (a) récupérer les 2 962 paires sans `tsi_on` ; (b) propager l'incertitude de A_p (partagée par
toutes les visites ON d'un patient) dans la MAP. Borne oracle totale −0.05, attendu réaliste −0.005 à −0.02.

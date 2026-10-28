# EDA — axe Rater + Pharmaco (modèle de mesure)

Scripts : `lab/scratch/meas/m1.py` … `m13.py`, sondes fold 0 `probe_split.py`, `probe_onbias*.py`,
`probe_gls.py`, `probe_rater.py` (frames e022b fold 0 : `build0.py` → `tr0.pkl`/`va0.pkl`, prior cubique
`prior0.pkl`). Résidus : `lab/runs/e022b_cubic_traj_quad/oof.csv` (RMSE 2.778). Tout ce qui utilise
`target` est de l'EDA (train) ou fitté sur les folds train ; sondes = trajectoire MAP cubique seule
(`hb_post_full` valid fold 0, base 3.569 avec mu1 = sortie cubique), pas le HGB final.

## Résumé

**Aucune erreur de spécification coûteuse trouvée.** Les trois écarts réels au modèle actuel valent
chacun ≈ −0.01 à −0.02 RMSE au niveau trajectoire, donc probablement moins après HGB.

| Écart testé | Constat | Sonde fold 0 (traj. valid) |
|---|---|---|
| Rater additif / dépendant du niveau, du temps, de la cohorte | non (tests de permutation) | — |
| Rater différent on vs off | non : pente log s_on ~ log s_off 1.017, corr 0.963 | — |
| Erreur d'estimation de s (in-fold) | ≈ 2 % relatif ; s « oracle » (train complet) | 3.569 → 3.557 (−0.012) |
| Estimateurs de s alternatifs (log tronqué + ON + pseudo-target valid) | pires en pratique | 3.580 / 3.587 |
| OFF des patients traités contaminé (mélange) | π ≈ 0.25 de visites × 0.74 ; marche −0.09 (log) à l'initiation | template off scindé : 3.569 → 3.554 (−0.015) |
| Erreur persistante patient des sources ON (erreur de A_p) | sd 0.12 log (0.15–0.17 si n_av = 0, 30 % des visites ON) | GLS gaussien + effet aléatoire : 4.171 → 4.152 (−0.020) ; correction naïve : pire (+0.03) |
| tsi_on imputable (habitude patient) | non : corr. avec moyenne LOO patient −0.005 | — |
| A_p qui dérive (durée, LEDD) | non : pente intra-patient 0.00007/an, corr LEDD −0.008 | — |
| OFF trop précoce (tsi_off) | réel mais rare : 119 visites < 9 h ; biais OOF par bin ≤ 0.2 | — |
| Valeurs erronées isolables | peu nombreuses, non groupées par patient | — |

## 1. Rater

- **Affectation** : 2.74 raters différents par patient (moyenne), même rater qu'à la visite précédente
  dans 29 % des cas ; train et test identiques (2.74 / 0.296).
- **Purement multiplicatif** : par rater, OLS `off ~ a + b·target` : b/s = 1.017 ± 0.030 ; intercept
  sd 0.77 entre raters pour une erreur bootstrap de 0.68 → excès sd ≈ 0.34 point (négligeable).
- **Interactions rater × X** (sd du log-ratio médian, observé vs permutation intra-rater, 30 tirages) :

| X | observé | permutation |
|---|---|---|
| cohorte | 0.0285 | 0.0294 ± 0.0025 |
| durée (4 bins) | 0.0399 | 0.0446 ± 0.0031 |
| rang de visite | 0.0290 | 0.0331 ± 0.0020 |
| âge | 0.0350 | 0.0354 ± 0.0023 |
| tercile de target | 0.0287 | 0.0262 ± 0.0016 (+1.6σ, marginal) |

  → pas de dérive dans le temps, pas d'effet cohorte, pas de dépendance au niveau.
- **On vs off** : s_on (on/target, tsi 1.5–5 h) vs s_off : corr 0.963, pente log 1.017, résidu sd
  0.028 ≈ bruit (fiabilité split-half s_on 0.84 contre 0.93 pour s_off).
- **Précision de s** : écart s in-fold vs s train complet 0.9 % → erreur in-fold vraie ≈ 2 %. Les
  résidus OOF portent ce bruit : corr(résidu relatif moyen par (fold, rater), erreur de s) = 0.41 ;
  retirer linéairement la partie visible fait 2.778 → 2.770. Mais même le s « oracle » ne gagne que
  0.012 en trajectoire. Comparaison d'estimateurs par demi-échantillons (A 60 % / C 15 % / R 25 %,
  40 tirages) : moyenne log tronquée −32 % MSE, + visites ON seules −25 %, + visites non étiquetées
  (pseudo-target = préd. OOF) −20 % ; mais intégrés à la MAP, ils dégradent (3.580 contre 3.569) : la
  convention « médiane » est cohérente avec le template SN, les autres fonctionnelles de localisation
  ne le sont pas. **→ H19 (MLE de s) a un plafond très bas.**
- Variance ou asymétrie propre à un rater : déjà réfutée (`dgp.md`, split-half 0.17).

## 2. OFF des patients traités : mélange contaminé

- À target égal, log(off/(s·target)) médian : non traités −0.014 → +0.14 (15–20 → 55–60),
  traités −0.07 → −0.02. Écart apparié −0.12 (médiane).
- **Ce n'est pas un décalage pur**, c'est un mélange. Quantiles à target 25–60 (non traités appariés vs
  traités) : q05 0.526 / 0.473, q50 1.112 / 0.979, q95 1.313 / 1.308 : le bord droit est inchangé et
  c'est le corps de la loi qui glisse. Ajustement par quantiles : **π ≈ 0.25–0.30 des OFF traités
  multipliés par ≈ 0.74**, le reste inchangé. Perte 3e-4, contre 2.5e-3 pour le meilleur décalage pur.
- Intra-patient (1 392 patients avec ≥ 2 OFF non traités et ≥ 2 traités), ajusté du niveau : −0.096 ±
  0.010. Étude d'événement (fold 0 train) : médiane log off/target +0.042 (k = −1) → −0.047 (k = +1),
  ensuite stable ; erreur moyenne de y_v −0.34 → −2.07 points.
- Le phénomène ne dépend ni de A_p (quintiles −0.119 à −0.129), ni de la LEDD ; il dépend un peu de
  tsi_off (fraction sous la médiane non traitée : 0.74 avant 10 h, 0.66 au-delà de 14 h). Il est faiblement
  persistant par patient (corr. indicateur bas avec taux LOO patient +0.13, contre −0.08 chez les non traités ;
  dispersion du saut patient ≈ 0) et partagé avec ON à la même visite (corr 0.84, on/off constant à 0.60).
- Le template actuel mélange traités et non traités dans la source `off`. Le scinder (off / off_tr) donne
  ξ 0.98 / 1.04, ω 1.39 / 1.53, α −4.03 / −4.23, et une trajectoire valid 3.569 → 3.554 (−0.015),
  LOO 4.432 → 4.425. Le biais de trajectoire autour de l'initiation reste faible (≤ 0.4).

## 3. Sources ON : erreur persistante par patient (A_p)

- log(y_v/target) sur les visites de source on_t/on_nt (target > 15) : composante patient (cov.
  own/LOO) **0.0136 (sd 0.117)** sur une variance totale de 0.142 ; pour la source off, ≈ 0 (−0.0003).
  Selon n_av : 0 → sd 0.15–0.17 (29–30 % des visites ON, trajectoire RMSE 4.75), ≥ 1 → 0.03–0.11.
  Le modèle traite ces erreurs comme i.i.d., donc elles ne se moyennent pas.
- 98.6 % des patients avec une visite ON ont aussi des OFF : A_p est identifiable à travers la trajectoire,
  pas seulement par les paires de la même visite.
- La correction plug-in (rapport aux OFF seuls, shrinké) **dégrade** la trajectoire (3.569 → 3.601), parce
  qu'elle retire l'information propre des visites ON. La version correcte, un effet aléatoire dans la
  vraisemblance (GLS gaussien, cov. τ²·μ·μᵀ sur les visites ON, τ² par bin de n_av fitté sur train : 0.017 /
  0.005 / 0.006), donne valid 4.171 → 4.152 (−0.020) et train 4.337 → 4.296.
- Dans la préd. finale e022b, le proxy de cette erreur n'explique presque rien (corr. résidu −0.08 pour
  n_av = 0) ; les visites ON avec n_av = 0 restent pourtant dures (RMSE 3.93 contre 2.71 si n_av > 2).

## 4. Pharmaco (courbe, délais)

- tsi_on et tsi_off ne sont pas des habitudes patient (corr. avec la moyenne LOO patient −0.005 et 0.012)
  et ne sont pas liés entre eux (0.007). Impossible d'imputer tsi_on pour on_nt.
- A_p constant dans le temps : pente intra-patient de a_v avec la durée 0.00007/an (corr 0.007), avec la
  LEDD corr −0.008. Pas de wearing-off modélisable.
- Résidus OOF e022b par bin de tsi_on : biais compris entre −0.16 et +0.15, sans forme. La courbe
  Emax(Bateman) suffit.
- tsi_off : biais OOF +0.20 (< 9 h, n = 119), +0.08 (9–11 h), |biais| ≤ 0.05 ailleurs. C'est déjà capté.

## 5. Valeurs erronées / contamination grossière

| Cas | n | biais OOF | RMSE OOF |
|---|---|---|---|
| off == 0 | 618 | −0.31 | 1.35 |
| on == 0 | 59 | −0.60 | 2.26 |
| on > off | 54 | −0.21 | 2.56 |
| off/s > 1.6·target | 140 | −1.64 | 2.13 |
| off/s < 0.3·target, target > 20 | 340 | +0.77 | 2.89 |

Les visites « basses » (< 0.4, target > 20, taux 3.8 %) ne se regroupent pas par patient : 57 patients
en ont ≥ 2, contre 78.6 ± 8.6 attendus si indépendant. Il n'y a pas d'outlier franc à isoler, et une
vraisemblance de contamination générique n'a pas de cible. La seule contamination structurée est celle
des OFF traités (§2).

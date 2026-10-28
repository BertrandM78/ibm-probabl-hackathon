# EDA — axe Pharmaco (validation de H01)

Données train uniquement. Scripts : `lab/scratch/pk1.py` … `pk16.py`. Les paramètres rater `s` et la
courbe sont ajustés sur tout le train (EDA) : en expérience, tout refitter dans `fit_predict`.
Notations : `r_on = on/target`, `r_off = off/target`, `ro = on/off` (visites avec les deux), `s` =
facteur multiplicatif par rater.

## Résumé (5 constats)

1. **La courbe PK existe, mais côté ON uniquement.** `1 − on/off` vs `tsi_on` : 0.03 (0–0.25 h) →
   0.27 (0.75–1 h) → plateau 0.44–0.45 de 1.5 à 5 h → 0.42 (5–6.5 h). Pas de pic marqué : plateau.
2. **Côté OFF, effet résiduel faible et rare** : effet ≈ 0.12–0.13 à 6–9 h, 0.06 à 9–11 h, 0.03 à 11–12 h,
   ≈ 0 au-delà de 12 h. Seules ~1 100 visites (4 % des `off` mesurés) ont tsi_off < 11 h.
   **`off/(s·(1−E(tsi_off)))` ne fait pas mieux que `off/s`** (RMSE 9.44 vs 9.37). → **contradiction avec H01** :
   le débiais du OFF par `tsi_off` est quasi inutile.
3. **Le biais principal du OFF est un facteur rater multiplicatif**, commun à `on` et `off` :
   médiane `r_off` par rater 0.70 → 1.14, `r_on` 0.39 → 0.65 ; corr(rater r_on, rater r_off) = 0.974 ;
   `on/off` par rater quasi constant (0.575 ± 0.014). `off/s` : RMSE 10.20 → 9.37, biais −5.38 → −0.30.
   → **contradiction avec H03** (biais additif) : c'est multiplicatif.
4. **L'amplitude de l'effet médicament est propre au patient** : au plateau, corr(`on/off` visite,
   moyenne LOO des autres visites du patient) = **0.918**. Elle ne dépend ni de `ledd` (ro 0.545–0.562
   par quintile), ni de l'âge (0.548–0.567), ni de la durée (0.545–0.562), ni de cohort (0.556/0.560),
   ni du gène (0.551–0.571), ni du rater (std 0.011). Cohérent avec « courbe constante avec l'âge ».
5. **Bruit visite partagé on/off** : corr(off/(target·s), on/(target·s·(1−E))) = 0.859 à la même visite,
   mais aucune persistance patient (corr LOO −0.06, lag-1 −0.09). → bruit i.i.d. par visite, que seule
   la mise en commun des visites (trajectoire) peut réduire.

## 1. Forme de la courbe

Multiplicatif, pas additif : `r_on` au plateau varie peu avec target (0.425 → 0.469 du 1er au
6e sextile) alors que `on − target` passe de −18 à −26.

`1 − ro` (n = 12 101 visites avec on, off > 10, tsi_on) :

| tsi_on (h) | 0–.25 | .25–.5 | .5–.75 | .75–1 | 1–1.25 | 1.25–1.5 | 1.5–2 | 2–3 | 3–4 | 4–5 | 5–6.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 − ro (méd.) | 0.034 | 0.118 | 0.195 | 0.275 | 0.360 | 0.412 | 0.439 | 0.444 | 0.447 | 0.446 | 0.416 |
| n | 102 | 305 | 401 | 1163 | 984 | 1313 | 1590 | 1738 | 918 | 511 | 164 |

Ajustements (`curve_fit` / `least_squares`) :

| Modèle | Données | Paramètres | R² individuel |
|---|---|---|---|
| Bateman `a(e^{−ke t} − e^{−ka t})` | 1−ro seul | a 0.863, ka 0.691, ke 0.128 ; t_max 2.99 h, E(t_max) 0.479, t½ abs 1.0 h, t½ élim 5.4 h | 0.509 |
| Saturation `Em(1−e^{−kt})` | 1−ro seul | Em 0.483, k 1.17 | 0.492 |
| Bateman | joint on + off | dégénère (ka ≈ ke = 0.36, a → borne) ≈ `0.47·t·e^{−0.36t}`, pic 2.8 h | on 0.502 / off ≈ 0 |
| **Emax(Bateman)** `Em·C^h/(EC50^h + C^h)`, C = Bateman | joint on + off | **Em 0.571, ka 0.341, ke 0.339, EC50 0.0012, h 2.90** | **on 0.521** / off ≈ 0 |

E(t) retenue (Emax-Bateman jointe) : 0.25 h 0.03 · 0.5 h 0.13 · 1 h 0.33 · 1.5 h 0.42 · 2 h 0.45 ·
3 h 0.47 · 4 h 0.46 · 5 h 0.42 · 6 h 0.37 · 7 h 0.29 · 8 h 0.21 · 9 h 0.13 · 10 h 0.08 · 12 h 0.02 · ≥14 h 0.
Le Bateman pur (fit sur on seul) prédit E(10 h) = 0.24, incompatible avec l'effet observé côté OFF
(0.06) : il faut une forme plateau puis chute (Emax sur la concentration). R² ≈ 0.5 seulement car la
variance restante est surtout inter-patient (constat 4).

## 2. `time_since_intake_*` manquant

- `tsi_on` NaN ⇔ souvent `on` NaN (on NaN ⇒ tsi_on NaN toujours ; 7 458 visites ont on sans tsi_on).
  Pour ces visites `r_on` médian 0.498 (vs 0.50 si présent) mais `ro` plus dispersé (sd 0.131 vs 0.09) :
  temps inconnu tiré de la même loi → valeur implicite = E moyen ≈ 0.40 (pas « temps 0 »).
- `tsi_off` NaN : `off/(target·s)` médian 1.02 vs 1.032 si tsi_off > 14 h → effet ≈ 0, valeur implicite
  « délai long ». tsi_on NaN et tsi_off NaN fortement couplés (2 501 / 2 968 visites avec on+off).
- Visites `on` NaN (9 074, « non traités », ledd NaN, visites précoces) : `off` non biaisé par le médicament
  mais queue gauche plus lourde (x < 0.5 : 9.7 % vs 5.2 %).

## 3. Valeurs erronées

- `on == 0` : 59 visites, target médian 28.6 → clairement erronées. `off == 0` : 618, dont 98 avec target > 15.
- `on > off` : 54 / 12 457, quasi tous à tsi_on < 0.5 h (9.8 % à 0–0.25 h) → bruit, pas erreur.
- `off/(target·s)` : asymétrique, mode 1.2–1.3, coupure nette > 1.4 (q99 1.49), longue queue gauche
  (7.1 % < 0.5 ; 12.3 % au tiers bas de target, 2.4 % au tiers haut) → bruit additif + plancher, pas
  d'outliers isolables (|z robuste| > 3 : 0.5 %).

## 4. Estimateurs ponctuels vs target (Q6)

| Estimateur | sous-ensemble | n | RMSE | biais | corr |
|---|---|---|---|---|---|
| off brut | tous | 25 677 | 10.20 | −5.38 | 0.871 |
| off / s | tous | 25 677 | **9.37** | −0.30 | 0.886 |
| off / (s(1 − E(tsi_off))) | tous | 25 677 | 9.44 | −0.05 | 0.887 |
| on brut | tous | 31 370 | 23.92 | −21.79 | 0.669 |
| on / (s(1 − E_pop(tsi_on))) | tous | 31 370 | 13.67 | −0.44 | 0.721 |
| on / (s(1 − A_p·g(tsi_on))) | tous | 31 370 | **12.92** | −0.38 | 0.744 |
| idem, off manquant, A_p informé, tsi_on présent | | 12 729 | 13.30 → **12.18** | | |

`A_p` = amplitude patient estimée en leave-one-visit-out sur `(1 − on/off)/g(tsi_on)` des autres
visites (g > 0.5), shrinkée vers 0.47 (poids 0.5 visite) ; sd inter-patient 0.064. Aucun target utilisé.

## 5. Résidus OOF de e002

- Pas de structure PK résiduelle en population : biais par bin de tsi_on ∈ [−0.23, +0.14], par bin de
  tsi_off ∈ [−0.23, +0.07]. e002 a déjà appris E(t) moyen.
- **Amplitude patient non captée** : biais OOF +0.70 au quintile bas de A_p, −0.79 au quintile haut ;
  off_missing : corr(res/target, A_p) = −0.19.
- **Rater non capté proprement** : biais moyen par rater sd 0.152, corr avec s = 0.817.
- Correction linéaire cross-fittée (GroupKFold 5) du résidu e002 (indicatif, s in-sample) :

| Features de correction | RMSE | off_missing |
|---|---|---|
| (e002) | 4.581 | 5.209 |
| (s − .85)·pred | 4.579 | 5.211 |
| (mean_patient(s) − .85)·pred | 4.473 | 5.090 |
| s + mean_patient(s) | 4.430 | 5.021 |
| A_p (×g×pred, ×off_missing) | 4.547 | 5.147 |
| les 4 | **4.397** | **4.959** |

→ Le gros levier est le **mélange des raters dans les moyennes patient** (`pmean`/prev/next de on/off
bruts) : il faut normaliser on/off par `s` *avant* de construire le contexte patient.

## 6. H15 — visites on+off mesurés : y_v bivarié vs y_v actuel (une seule source)

Script : `lab/scratch/h15_bivariate.py`. Train only, `off_n = off/s`, `on_eq_n = on_eq/s` (features
H01/H04 déjà en place). Sous-ensemble : visites avec `off > 0` ET `on > 0` mesurés.

- **Fraction du sous-groupe** : train 12 433 / 44 590 (**27.9 %**), test 3 071 / 11 013 (**27.9 %**,
  cohérent) → sous-groupe non négligeable en taille, mais `harmonized_score` (H05) choisit déjà
  systématiquement `off_n` sur ce sous-groupe (jamais `on_eq_n`), donc la question est : perd-on de
  l'info en ignorant `on_eq_n` ici ?
- **Corrélation du choc à la même visite** (`e_off = off_n − target`, `e_on = on_eq_n − target`) :
  **0.862** globalement (cohérent avec le constat 5, 0.859). Par bin `tsi_on` : 0.973 (0–0.5 h) → 0.760
  (NaN) ; donc **pas constant**, plus faible quand le délai est inconnu ou le target élevé (tercile
  haut 0.826 vs bas 0.898) — mais jamais assez bas pour que `on_eq_n` apporte de l'info indépendante
  utile (voir plus bas).
- **sd du bruit par source, toujours on_eq_n > off_n** : tercile bas sd_off 8.88 / sd_on 9.96 ; milieu
  10.96 / 12.48 ; haut 11.76 / 14.25. RMSE globale : off_n seul **10.748**, on_eq_n seul **12.475**.
- **Combinaison optimale (grid alpha·off_n + (1−alpha)·on_eq_n), in-sample ET en CV 5-fold (poids/alpha
  fittés hors-fold)** : alpha* = **1.00 dans tous les cas** (global, par tercile de target, par bin
  tsi_on, y compris le sous-groupe tsi_on NaN où la corrélation est la plus faible 0.76). Moyenne
  simple 50/50 : RMSE 11.214 (pire que off seul). Moyenne pondérée inverse-variance : RMSE 11.091
  (toujours pire). **Aucune pondération, globale ou par sous-groupe, ne bat `off_n` seul.**
- **Pourquoi** : la corrélation (0.76–0.97) est partout ≥ au ratio sd_off/sd_on (0.82–0.89) nécessaire
  pour que `on_eq_n` apporte une réduction de variance ; le bruit résiduel de `on_eq_n` au-delà de ce
  qui est déjà partagé avec `off_n` est trop faible en relatif pour compenser sa plus grande variance
  propre (incertitude de la courbe PK + de `A_p`, cf. constat 1/4).
- **Conclusion** : `harmonized_score` (choix exclusif de `off_n` quand disponible) est déjà
  quasi-optimal sur ce sous-groupe ; aucune combinaison bivariée (simple, inverse-variance, ou alpha
  optimisé par sous-groupe) ne réduit le RMSE résiduel observable. H15 non retenue (voir IDEAS.md,
  statut ABANDONNÉ).

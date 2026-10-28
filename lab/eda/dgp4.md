# EDA — axe DGP, 4e passe (règle ×1.02, latent P, visite 0, générateur candidat)

Scripts : `lab/scratch/dgp4/` (`base.py` = train + folds + G/Z ; G = `T_i = round(1.02·T_{i−1}, 1)`, Z = `T_i = T_{i−1}`).
Les constats sur `target` sont de l'EDA ; tout ce qui entre dans un modèle est refitté par fold (`lab/feat_rule.py`).

## Résumé

| Question | Constat | Script |
|---|---|---|
| Le target décroît-il ? | **Jamais** : 0 baisse sur 39 014 incréments | a1 |
| Constante c | 1.0200–1.0201 par fold et par cohorte, indépendante de Δâge (méd. 1.0200 dans les 4 quartiles de Δâge, sd 0.0007) ; réestimée in-fold : 1.02004–1.02007 | a1, feat_rule.fit_rule_c |
| Déclenchement | `P_i < T_{i−1}` (seuil strict sur la valeur précédente du *target*) : ratios dans [1, 1.02) présents (~400 / bin de 0.002), donc pas `max(P, c·T)` | a1 |
| Stabilité | part G 8.95–9.51 % par fold ; A 9.1 % / B 10.0 % ; sexe 8.9 / 9.4 % ; gène 6.9–9.5 % | a1 |
| Reconstruction rule(cubique) vs T | P = OLS sur T : 74 % des visites exactes (0.05) ; OLS hors G/Z : 85 % ; fit non linéaire rule(P(θ)) : **88 %** (folds 0.864–0.894), patients exacts 79 % ; les « non exacts » sans déclenchement ont un résidu max médian 0.055 (tolérance d'arrondi) | b1, b2 |
| Mesures ↔ P ou T | u = off/(s·T) médiane 0.875 aux visites G contre 1.01 ailleurs (P < T) : les mesures suivent P | gen |
| Départ plat (Z1) | 17.9 % de T0 entiers (≈ 10 % attendus) ; 1er déclenchement à v1 **plat** dans 77 % des cas si T0 entier, contre 4–9 % sinon ; chez ces patients T0 = T1 ≈ round/ceil(P(t1)), pas P(t0) ; chez les autres T0 entiers T0 = round(P0) (85 %) | a2–a4, b3 |
| Identifiabilité | c : ≈ 3 400 visites exactes, erreur < 1e-4. P post-déclenchement : **censuré** (seule info du target : P_i < T_{i−1}) ; identifiable seulement via la cubique (≥ 4 visites non censurées) et via les mesures, qui suivent P | b1 |

## 1. Où perd le meilleur modèle (b003_seq4, e = target − pred)

| groupe | n | RMSE | biais | part SSE |
|---|---|---|---|---|
| visites G | 3 585 | 2.94 | +0.22 | 10.7 % |
| visites Z | 393 | 2.93 | −0.66 | 1.2 % |
| patients à queue (≥ 2 G/Z) | 8 814 | 2.81 | +0.31 | 24.2 % |
| autres patients | 35 776 | 2.47 | −0.09 | 75.8 % |

Annuler seulement le biais des patients à queue ≈ −0.004 RMSE : l'enjeu est la variance (détecter les patients post-pic).

## 2. Postérieure de P (fold 0, MAP cubique e022b, c in-fold) — `d1.py`

| reconstruction | RMSE trajectoire | visites G (biais) | patients à queue (biais) |
|---|---|---|---|
| P̂ (MAP) | 3.5688 | 4.25 (−0.92) | 4.32 (−1.72) |
| rule(P̂) | 3.5371 | 4.15 (−0.78) | 4.27 (−1.66) |
| **E[rule(P)]**, 200 tirages Laplace | **3.5118** | 4.00 (−0.45) | 4.13 (−1.45) |

Part déclenchée : vraie 8.8 % (dont ~2–3 % de G fortuits), rule(P̂) 1.35 %, postérieure 3.0 %. P(trig) **sous-calibrée**
(bin ≤ 0.01 : 0.2 % prédit contre 4.8 % observé ; 0.05–0.1 : 7 % contre 20 %) → la postérieure gaussienne est trop
étroite sur la courbure (cf. § 4).

## 3. Visite 0 (H46)

- IQR de log(off/(s·T)) hors G/Z, s refitté hors fold : v0 0.295–0.345 contre v1 0.436–0.476 et v2–3 0.46–0.49, **5/5 folds** ;
  médiane +0.08 / +0.10 à v0 contre +0.02 à v1. Vaut en A et B, T0 entier ou non, n = 4…12.
- Contrôlé par le niveau de T (log-bruit plus large à T bas) : T > 24, première OFF 0.242 contre 0.411 à la 2e.
- Ce n'est pas seulement le rang : **première OFF mesurée** à v > 0 (patients traités) IQR 0.313 contre 0.363 à la 2e.
  Chez les naïfs stricts : IQR 0.33 → 0.48 → 0.54 → 0.59 avec le nombre d'OFF antérieures.
- Côté ON (log on/T, tsi > 1.5 h) : v0 0.437 contre 0.52–0.57 ; corrélation ON inter-visites ≈ +0.2 partout (A_p).
- Sonde fold 0 (`c5.py`, trajectoire valid) : sources `_v0` 3.5688 → **3.5387** (v0 4.04 → 3.69, **v>0 3.495 → 3.516**) ;
  split « première OFF » 3.5536 ; « première OFF seule » 3.5446. Le gain est concentré sur v0 et dégrade un peu les autres visites.
- Bruit OFF : **autocorrélation de lag 1 négative**, ρ(v0, v1) −0.128 (folds −0.05 à −0.18), puis −0.03 à −0.08 entre
  visites consécutives, ≈ 0 au-delà. Pas d'ancrage du target sur la mesure de v0 (ρ(v0, v3) = +0.06).

## 4. Générateur candidat (`gen.py`) et écarts

Composantes : plan de visites réel ; P cubique (coefficients = régression sur durée moyenne, âge au diagnostic, cohorte
+ résidu) ; T = rule(P) c = 1.02, plancher 0, arrondi 0.1, sous-groupe π = 9 % T0 entier + 1er déclenchement plat ;
traitement absorbant (hazard logistique T × durée) ; ON mesuré ⇔ traité ; OFF toujours si non traité, sinon p = 0.40 ;
off = round(s_r · P · u), u tiré du gabarit empirique (v0 / reste × traité / non) ; ON partage le choc u, × (1 − A_p E(tsi)).

| statistique | réel | sim. MVN | sim. bootstrap résidus |
|---|---|---|---|
| target moyenne / sd / q95 | 37.5 / 16.5 / 64.8 | 37.2 / 16.8 / 65.1 | — |
| part G | 9.19 % | **5.37 %** | 10.66 % |
| excès du pic 1.02 | 5.69 | **2.91** | 5.63 |
| part Z | 1.01 % | 1.10 % | 1.06 % |
| T0 entier / Z1 si T0 entier | 17.9 % / 20.7 % | 21.4 % / **13.2 %** | 20.5 % / 12.6 % |
| corr(off, T) / RMSE off | 0.871 / 10.20 | 0.827 / 11.52 | — |
| off / on manquants | 42.4 / 29.7 % | 43.2 / 28.4 % | — |
| IQR log u v0 / v1 | 0.314 / 0.470 | 0.304 / 0.487 (par construction) | — |
| corr(u_v0, u_v1) | **−0.237** | −0.004 | — |
| u médian aux visites G / ailleurs | 0.875 / 1.012 | 0.903 / 1.004 | — |

Résidus des coefficients (patients ≥ 8 visites) : excès de kurtosis c0 −0.14, c1 0.56, **c2 1.92, c3 4.88**.

Mécanismes encore inexpliqués :
1. **Queues lourdes de la courbure** : un prior gaussien sous-génère les P décroissants (G 5.4 % contre 9.2 %) ; le
   bootstrap des résidus les retrouve. C'est la même cause que P̂ « trop monotone » (dgp2) et que la sous-calibration de P(trig).
2. **Départ plat** : chez les patients Z1, T0 = T1 ≈ entier proche de P(t1) (et non P(t0)) ; règle exacte non identifiée.
3. **Bruit OFF non i.i.d.** : dispersion qui croît avec le nombre d'OFF antérieures, et autocorrélation de lag 1 négative.
4. corr(off, T) plus forte en réel (0.871 contre 0.827) : gabarit u trop dispersé en simulation (u tiré sans dépendance au niveau de T).

## 5. Expériences (5 folds, c et tout paramètre lié au target refittés par fold, leak-review GO)

| run | modification (une seule) | vs e022b (2.778) | par fold | apport à b003_seq4 (2.541) |
|---|---|---|---|---|
| e027_rule_post | H33 : r_det, r_exp, r_ptrig, r_sd (postérieure Laplace × règle) | **−0.017 ± 0.015** (4/5) | −0.017, −0.008, −0.024, +0.002, −0.039 | −0.002 ± 0.002 (b004, e027 remplace e022b) |
| e028_v0_template | H46 : sources `_v0` dans template + MAP | +0.007 ± 0.007 (0/5) | +0.011, +0.002, +0.003, +0.016, +0.002 | ± 0 (b004c) |
| e029_latent_prior | H34 : prior + template hors visites censurées | +0.007 ± 0.017 (3/5) | −0.001, −0.003, −0.009, +0.020, +0.029 | poids 0 (b004d) |

Lecture : la règle est **exacte** (c identifié à 1e-4, stable entre folds et cohortes) mais l'information qu'elle apporte
est déjà captée à ~90 % par les modèles séquentiels (e023/e025/e026 voient toute la trajectoire de mesures). Le seul
levier restant côté règle est la **calibration de la postérieure de P** (H61). H46 est un vrai mécanisme de mesure mais
pas un gain : il faut modéliser la dépendance entre OFF successives (H62), pas seulement élargir v0.

Intégration minimale retenue pour e022b : `rule_hook(train_target, rule_feats=True)` à la place de `cubic_hook(…, keep_quad=True)`
(e027, une ligne). Pour les réseaux de b003 (e023/e025/e026, construits sur la même matrice), même substitution — non
lancée (coût ≈ 10 réseaux × 5 folds) vu l'apport blend de −0.002.

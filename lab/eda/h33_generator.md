# H33 — Reconstruction du générateur : la règle ×1.02 peut-elle dépasser b003_seq4 ? (2026-10-13)

**Décision : ABANDONNER la ligne H33 pour b003_seq4.** H61 (postérieure à queues lourdes) est
**dépriorisée, pas réfutée**. e027_rule_post reste documenté, hors modèle soumis.

Sources :
- travaux du coéquipier sur `origin/main` : `lab/eda/dgp4.md`, `lab/feat_rule.py`, e027 / e028 / e029 / b004
  dans LOG.md. Ils ne sont pas encore fusionnés dans la branche `bertrand` ;
- `lab/eda/dgp2.md`, `dgp3.md`, `methods2.md`, `ensemble2.md`, `level.md` ;
- vérifications de cette mission : `lab/scratch/h33/verify.py` et `oracle_rule.py`, avec leurs `.log`.

Aucun réseau réentraîné, aucun nouveau run cv.py. Revue leak-reviewer : GO pour l'abandon, sous les
reformulations reprises ci-dessous.

## 1. Ce que nous savons du générateur (faits observés)

| Fait | Preuve | Source |
|---|---|---|
| Le target ne décroît jamais | 0 baisse sur 39 014 incréments | verify.py |
| Pic du ratio T_i/T_{i−1} à 1.02 | bins 0.018–0.022 : 4 692 visites, contre ~450–650 par bin ailleurs | verify.py, dgp2 |
| c = 1.0200, constant | médiane par fold 1.01997–1.02003 (T_{i−1} ≥ 20) ; refitté in-fold 1.02004–1.02007 ; identique dans les 4 quartiles de Δâge | verify.py, dgp4 |
| La règle est indépendante du temps entre visites | même c pour Δâge 0.3–0.7 an et 1.1–5.4 ans | verify.py, dgp4 |
| Déclenchement ≠ max(P, c·T) | 3 615 ratios dans (1, 1.018) ; 4 144 incréments < round(1.02·T_{i−1}) − 0.05, dont 4 080 à T_{i−1} ≥ 20 (l'arrondi ne l'explique pas) | verify.py, revue |
| Part des visites G (×1.02) | 7.8–8.3 % par fold (ratio arrondi) ; ≈ 8.8 % selon dgp4, dont 2–3 % de G fortuits (P croît d'environ 2 %) | verify.py, dgp4 |
| Les mesures suivent P, pas T | off/(s·T) médian 0.875 aux visites G contre 1.01 ailleurs ; off/(s·P) plat | dgp2 § 2, dgp4 |
| La forme est reproduite à ~88–95 % | fit non linéaire rule(P(θ)) : 88 % des visites exactes à 0.05 (folds 0.864–0.894) | dgp2, dgp4 |

## 2. Ce qui reste hypothétique

- **La forme cubique de P.** Elle est exacte à l'arrondi pour 79–85 % des patients ; le reste est non
  identifié.
- **Le départ plat (Z1).** 17.9 % de T0 entiers, contre ~10 % attendus. Chez ces patients, le premier
  déclenchement est plat dans 77 % des cas, et T0 = T1 ≈ round(P(t1)) au lieu de P(t0). La règle exacte
  n'est pas trouvée.
- **Les queues lourdes des coefficients de P.** Excès de kurtosis : c2 1.9, c3 4.9. Un prior gaussien
  sous-génère les P décroissants (G 5.4 % simulés contre 9.2 % réels).
- **Les visites Z (T_i = T_{i−1}) mélangent deux mécanismes.** 64 des 393 sont des G arrondis à petit T.
- **Indépendance de la découverte.** La règle, la tolérance 0.05 et les définitions G/Z ont été choisies
  en regardant les targets des **5 folds**. La stabilité par fold est un contrôle de cohérence, pas une
  confirmation indépendante. Aucune confirmation hors échantillon n'est possible sur le train. Le risque
  est négligeable pour c (un paramètre, effet déterministe massif), mais réel pour les choix analytiques :
  ce biais gonfle les gains « prometteurs », donc il fragilise e027 et renforce l'abandon.

## 3. Équations

Pour un patient de visites t_0 < … < t_n :

```
P(t)  = a + b·x + c2·x² + c3·x³,          x = (t − t̄)/5         (latent, 4 ddl ; prior N(μ(cov), Σ))
T_0   = P(t_0)                                                     (sauf sous-groupe Z1 : T_0 ≈ round(P(t_1)))
T_i   = P(t_i)              si P(t_i) ≥ T_{i−1}
      = c · T_{i−1}         sinon,  c = 1.02  (indépendant de t_i − t_{i−1})
T_i  ← round(max(T_i, 0), 1)
off_i ≈ s_rater · P(t_i) · u_i,   u_i ~ SN (dispersion propre à v0, autocorrélation lag 1 négative)
on_i  ≈ s_rater · P(t_i) · u_i · (1 − A_p · g(tsi_i))
```

**La règle agit sur le score final, récursivement sur T_{i−1} ; les mesures dérivent de P.**
P est **censuré** après déclenchement : la seule information du target est P_i < T_{i−1}.
Après le pic, P n'est identifiable que par la cubique (≥ 4 visites non censurées) et par les mesures.

## 4. Limites des tentatives précédentes

| Essai | Résultat | Limite |
|---|---|---|
| rule(P̂) sur la trajectoire MAP (dgp2) | trajectoire fold 0 3.552 → 3.524 | P̂ trop monotone : déclenche 1.35 % des visites contre ~8.8 % |
| Post-traitement pred + a·(rule(P̂) − P̂) sur e022b | −0.0005 / −0.009 (folds 0/1), incohérent | même cause |
| Post-traitements cummax / ×1.02 sur b002 et e023 | ±0.0002 (22 baisses sur 44 590) | les réseaux produisent déjà une forme juste |
| H35 sonde (pipeline sur P latent + règle) | seul +0.023 / −0.013 ; 50/50 −0.012 / −0.024 (folds 0/1) | deux folds seulement, jamais lancé sur 5 |
| **e027** (H33 : r_det, r_exp, r_ptrig, r_sd sur 200 tirages Laplace) | **−0.017 ± 0.015 vs e022b (4/5)** ; dans b003 **−0.002 ± 0.002** | gain < 0.02 (négligeable), P(trig) postérieure 3–4 % (sous-calibrée), dans la zone de faux positifs de la règle à 5 folds |
| e029 (H34 : prior + template hors visites censurées) | +0.007 ± 0.017, NOISE, poids 0 dans b003 | le prior reste gaussien |
| e028 (H46 : template v0) | +0.007 ± 0.007, 0/5 | mécanisme réel (dispersion v0) mais pas de gain |

## 5. Déterministe (A) contre probabiliste (B)

- **A** : T̂ = R(E[P | obs]).
- **B** : T̂ = E[R(P) | obs] ≈ (1/N) Σ R(P_n), avec P_n = tirages **de coefficients** (a, b, c2, c3).
  Chaque tirage est une courbe entière : les corrélations temporelles sont préservées par construction.
- R est convexe vers le haut (elle remplace une baisse par une hausse) : par Jensen,
  E[R(P)] ≥ R(E[P]). L'écart vaut ≈ la probabilité de déclenchement × l'ampleur de la baisse évitée.
  Il n'est significatif que si la postérieure est bien calibrée sur la courbure.

| Trajectoire seule, fold 0 (dgp4 § 2) | RMSE | visites G (biais) | patients à queue (biais) |
|---|---|---|---|
| P̂ (MAP) | 3.5688 | 4.25 (−0.92) | 4.32 (−1.72) |
| A : rule(P̂) | 3.5371 | 4.15 (−0.78) | 4.27 (−1.66) |
| B : E[rule(P)], 200 tirages Laplace | **3.5118** | 4.00 (−0.45) | 4.13 (−1.45) |

A et B diffèrent réellement au niveau de la trajectoire. **Mais la postérieure Laplace de e022b n'est
pas celle de P** : prior gaussien trop étroit sur c2/c3, P(trig) prédite 3 % contre ~8.8 % observés
(bin ≤ 0.01 : 0.2 % prédit, 4.8 % observé). B est donc lui-même sous-estimé ; c'est l'objet de H61.

## 6. Test discriminant (phase 3)

| Question | Réponse |
|---|---|
| A et B diffèrent-ils ? | Oui sur la trajectoire (3.537 contre 3.512) ; dans le modèle e022b, l'écart entier vaut −0.017 |
| Fréquence de déclenchement | ~8–9 % des visites ; patients à queue (≥ 2 G/Z) = 20 % des visites |
| Part de la MSE de b003 concernée | visites déclenchées 11.9 %, visites post-déclenchement 20.0 %, patients à queue **24.2 %** ; RMSE aux déclenchements 2.93 contre 2.47 ailleurs |
| Information absente des réseaux ? | **Non pour l'essentiel** (voir les bornes oracle ci-dessous) |

**Bornes oracle sur l'OOF de b003_seq4 (2.5408).** Drapeaux G/Z dérivés du target : **non prédictifs**.

| Oracle | Résultat | Lecture |
|---|---|---|
| Correcteur HGB a posteriori, témoin sans drapeaux | +0.021 ± 0.023 | la machinerie dégrade b003 à elle seule (stacking de 2e niveau) |
| Même correcteur + drapeaux oracle | +0.002 ± 0.018 | — |
| **Valeur des drapeaux (oracle − témoin)** | **−0.019 ± 0.008, 5/5** (−0.012, −0.014, −0.026, −0.030, −0.014) | connaître parfaitement *quand* la règle se déclenche vaut ~0.02 **avec ce correcteur** |
| (a) récursion T_i = 1.02·T̂_{i−1} aux visites G/Z oracle | **+0.004 ± 0.009 (pire)** | appliquer la règle à partir du niveau prédit propage l'erreur de niveau |
| (b) cummax des prédictions (sans oracle) | 0.000 | b003 est déjà monotone |
| (c) visites G = round(1.02·T_{i−1} **vrai**) | −0.140 | exige le vrai niveau à t−1 : mesure la valeur du **niveau**, pas de la détection |

Ces bornes **ne sont pas** des bornes supérieures universelles. Elles valent pour un correcteur a
posteriori de b003. Un modèle de base réentraîné avec une postérieure calibrée pourrait aussi modifier
le **niveau** à toutes les visites, puisque les mesures post-pic suivent P.

**Conclusion du test.** Le critère d'arrêt de la mission est rempli : le mécanisme n'apporte presque
pas d'information absente de b003. Sa valeur restante passe par le niveau patient T_{i−1}, déjà
identifié comme goulot (66 % de l'erreur, `ensemble3.md`).

## 7. Gain RMSE sur cinq folds

Pas de nouvelle expérience : le diagnostic arrête la mission. Le seul résultat 5 folds existant est
celui du coéquipier : e027 −0.017 ± 0.015 vs e022b (4/5), soit −0.017 / −0.008 / −0.024 / +0.002 / −0.039.

## 8. Impact sur b003_seq4

- e027 à la place de e022b dans b003 (b004_seq4_rule) : 2.539, **−0.002 ± 0.002**.
- Atténuation observée de la règle : trajectoire −0.057 (fold 0) → modèle −0.017 → blend −0.002.
- Pour atteindre −0.01 dans le blend, il faudrait multiplier par ~5 le gain trajectoire en ne corrigeant
  que la calibration du déclenchement. Improbable, mais pas démontré impossible.

## 9. Risques méthodologiques

1. **Chemins multiples** : règle et définitions trouvées sur les 5 folds (§ 2). Avec 5 folds, |Δ| > std
   correspond à p ≈ 0.09 ; sur des dizaines de comparaisons, des « GAIN » de 0.01–0.02 sont attendus par
   hasard. e027 et b004 sont dans cette zone : à classer « négligeable / non confirmé ».
2. **Stacking de second niveau** dans les correcteurs sur OOF : symétrique entre les deux arms (la
   différence oracle − témoin n'est pas biaisée), mais le +0.021 du témoin n'est pas un vrai hors-échantillon.
3. **Postérieure ≠ postérieure de P** (§ 5) : tout E[R(P)] fondé sur la Laplace e022b est sous-calibré.
4. Les écarts entre sources sur la part déclenchée (8.8 % dans dgp4 contre « ~6 % » dans la ligne LOG
   d'e027) viennent de l'inclusion ou non des G fortuits. À unifier dans la documentation.
5. Aucun usage du leaderboard ; aucun oracle présenté comme prédictif ; mêmes folds et mêmes patients
   partout (`lab/folds.csv`, hash 815b8eaa).

## 10. Décision

**ABANDONNER H33 (et H35) pour b003.** La règle est réelle et exacte, mais b003 la reproduit déjà :
connaître le moment du déclenchement ne vaut que ~0.02 via un correcteur, rien en récursion, et e027
n'apporte que −0.002 au blend.

**H61 : dépriorisée, non réfutée.** Borne propre et peu coûteuse si l'on veut la fermer : e027 en mode
oracle (dans `valid`, `r_ptrig` remplacé par le drapeau G/Z), `--folds 0`, puis re-blend dans b003.
- Gain dans le blend < 0.01 : H61 est fermée.
- Gain au-dessus : la calibration de la postérieure mérite un run.

Coût : un fit e022b, ~1–2 min, aucun réseau. Nécessite de fusionner `origin/main` (feat_rule.py, e027).

## Deux mécanismes alternatifs pour l'erreur restante

### M1 — Bruit de mesure OFF autocorrélé (dépendance entre OFF successives)
- **Mécanisme** : le choc u_i de la mesure OFF dépend du précédent (ancrage ou correction du rater) :
  corr(u_v0, u_v1) = −0.24 en réel contre −0.004 dans le générateur candidat, et dispersion croissante
  avec le nombre de OFF antérieures (IQR 0.33 → 0.59, dgp4 § 3). La MAP suppose des chocs i.i.d. : elle
  sur-pondère les paires consécutives, ce qui fausse le **niveau** estimé chez les patients à 2–4 OFF.
- **Indice** : ρ lag 1 négatif sur 5/5 folds ; l'erreur de niveau est maximale à 1–2 OFF (RMS 3.8–2.9,
  `ensemble3.md`).
- **Test minimal** : vraisemblance bandée (AR(1) sur u) dans `cubic_map`, ρ fitté in-fold ; sonde de
  trajectoire au fold 0 avant tout run 5 folds (= H62 du coéquipier).

### M2 — Ancrage initial du générateur (T0 / départ plat)
- **Mécanisme** : pour un sous-groupe (~18 % de T0 entiers, contre ~10 % attendus), T0 n'est pas P(t0)
  mais un entier proche de P(t1), suivi d'un incrément nul. Le niveau de départ est mal ancré, ce qui
  décale toute la trajectoire des patients courts.
- **Indice** : patients n = 4 avec T0 entier, RMSE 3.12 (biais +0.46, +0.85 à v0) ; patients « tout
  entiers » 3.78 (biais +1.47) ; T0 entier = 16 % de la SSE (dgp3 § 3).
- **Test minimal** : sur train (EDA), comparer T0 à round(P̂(t0)), round(P̂(t1)) et ceil(P̂(t1)) par
  vraisemblance, puis indicatrice « T0 entier » et terme Z1 dans la MAP ; sonde au fold 0.

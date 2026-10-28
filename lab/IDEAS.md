# Backlog d'hypothèses

Ajouter une hypothèse avec `/idea`. Une fiche par idée, format imposé.
Priorité = gain attendu / effort. Statuts : `TODO` → `EN COURS` → `FAIT (eNNN)` / `ABANDONNÉ`.

## Tableau de priorisation

| ID | Axe | Hypothèse (courte) | Gain attendu | Effort | Priorité | Statut | Exp |
|----|-----|--------------------|--------------|--------|----------|--------|-----|
| H42 | Mesure | A_p a posteriori complet (H39 paires sans tsi + 2e passe LOO sur ON-only via T̂ in-fold, variance inverse, sd_A_p) → matrice e023 | moyen | moyen | 1 | H42a FAIT (e024 : −0.011 ± 0.048 vs e022b, NOISE ; folds 1–4 −0.004 ± 0.052 ; sans tsi −0.064, autres +0.007) ; H42b TODO — review GO sous conditions : découper H42a (config sondée W=1, gbar, k=0.5, e022b 5 folds) / H42b (a_traj : T̂ sans A_m ni mu1 ON, var incluant hb_se, rebuild 2 passes). H39 seul e022b fold 0 2.7448 → 2.7043 (−0.040) ; 20 % patients traités sans aucun tsi : e023 3.13 vs 2.50 | scratch/miss2/p2_h39.py, scratch/level/ |
| H43 | Méthode | Ensemble séquentiel élargi : e023 + 5 GRU fdrop 0.2 (50 ep) + 5 TF bdrop 0.3 (+ seeds neuves) | moyen | moyen | 2 | TODO — review GO sous conditions, à lancer en dernier (après H44) : contrôle apparié e023 + 10 seeds neuves obligatoire, blocs issus de dense(). fold 0 blend 2.5211 → 2.4927 (20 réseaux) ; ρ 0.93–0.95 vs 0.97 seeds | scratch/meth2/p1.py |
| H44 | Ensemble | TF cible offset `target − hb_post_full` + TF amortized (H31) en composantes 3–4 du blend b002 | moyen | moyen | 3 | CONFIRMÉE (partiel) — b003_seq4 CV 2.541, −0.038 ± 0.011 vs b002 (5/5, folds 1–4 −0.038) ; offset (e025) −0.022 ± 0.006, amortized (e026) −0.010 ± 0.005 (seuil 0.01 limite) ; poids stables ±0.015. Offset = EP 25 (sonde), pas 60 | scratch/ens2/ |
| H45 | Manquants | Augmentation : copies de 30 % des patients avec tsi_on/tsi_off masqués (MCAR tout-ou-rien) | faible-moyen | moyen | 4 | TODO — e022b fold 0 −0.019 ; redondance probable avec H42 | scratch/miss2/p3_aug.py |
| H46 | Structure | Template de bruit SN propre à la 1re visite (sources `<src>_v0`, + off_tr de H38) | faible-moyen | faible | 5 | RÉFUTÉ en final (e028) — +0.007 ± 0.007, 0/5 ; first_visit −0.018 mais v>0 dégradé ; blend b003 ±0. Mécanisme réel → H62 | e028_v0_template |
| H47 | Méthode | Réseaux finaux sans holdout interne 10 % (epoch fixe 25 ou moyenne snapshots 12–24) | faible | faible | 6 | TODO — attendu ≈ −0.009 | |
| H48 | Pharmaco | Courbe PK g_p(t) = g(t·(1 + β(A_p − A0))) : montée plus rapide si A haut | faible | faible | 7 | TODO — e022b fold 0 −0.008 | scratch/miss2/p5_pkA.py |
| H49 | Méthode | Pré-entraînement auto-supervisé masqué (on/off) sur train + valid + test, puis fine-tune | faible-moyen (incertain) | fort | 8 | TODO — non sondé ; pseudo-labeling ρ 0.99 laisse peu d'espoir | |
| H50 | Structure | Vraisemblance logistique de l'initiation du traitement dans la MAP cubique | faible | faible | 9 | TODO — full −0.0015, LOO 4.432 → 4.406 ; risque NOISE | scratch/dgp3/e8.py |
| H51 | Structure | Échelle du bruit OFF non traité selon min(n_off_avant, 3) | faible | faible | 10 | ABSORBÉ dans H62 | dgp4.md |
| H52 | Rater | Facteur rater localisé par le quantile 80 (recentré) au lieu de la médiane | négligeable | faible | 11 | TODO (seulement si e022b est rerun) — b002 −0.002 | scratch/rater2/r4.py |
| H53 | Ensemble | e023 à 40 seeds au lieu de 10 | faible | fort | 12 | TODO (après H43, redondant) — ≈ −0.01 dans le blend | |
| H54 | Ensemble | Sélecteur / stacking niveau 2 par patient (e023, e022b, MAP) | nul | faible | - | ABANDONNÉ (ensemble2.md) : oracle réel 2.343 = oracle nul 2.334 ; appris −0.001 à +0.011 | scratch/ens2/ |
| H55 | Méthode | Tête structurée cubique + rule(P), pseudo-labeling, perte niveau, SWA en ensemble, weight decay, dropout de visites | nul | - | - | ABANDONNÉ (methods2.md) : écarts ≤ 0.009 ; pseudo-labels = distillation (ρ 0.99) | scratch/meth2/ |
| H56 | Rater | Biais/variance par rater, rater × patient, sites, transductif | nul | - | - | ABANDONNÉ (rater.md) : tout au niveau du bruit ; axe épuisé | scratch/rater2/ |
| H57 | Manquants | PK par ledd/durée/âge, tsi_off résiduel, ledd, valeurs erronées, recalage OOF par manquants, loi en puissance, Σ par n | nul | - | - | ABANDONNÉ (missing2.md, dgp3.md, level.md) | |
| H30 | Méthode | Réseau séquentiel par patient (BiGRU + Transformer, 5 seeds chacun) sur features e022b, blend avec e022b | fort | moyen | 1 | CONFIRMÉE — e023 seul 2.610 (−0.169 ± 0.042) ; b002 blend NNLS 0.72/0.28 2.579 (−0.200 ± 0.033 vs e022b, 5/5, folds 1/2/4 −0.200) | e023_seq_nn, b002_seq_e022b |
| H32 | Méthode | LightGBM à la place de HGB (niveau + delta) dans e022b | moyen (faible si H30) | faible | 13 | TODO — folds 0/3 −0.047 / −0.061 ; libomp via torch | scratch/meth/m2_gbm.py |
| H33 | Structure | Règle générateur ×1.02 post-pic appliquée à la cubique latente P̂ → features `rule(P̂)` (+ variante incrément nul à v1, patients courts, dgp3.md) | faible-moyen | faible | 14 | FAIT (e027) — −0.017 ± 0.015 vs e022b (4/5) : net mais < 0.02, négligeable ; blend b003 −0.002 ± 0.002 (b004). Postérieure E[rule(P)] trajectoire fold 0 3.569 → 3.512 ; P(trig) sous-calibré → H61 | e027_rule_post, dgp4.md |
| H31 | Ensemble | Transformer « amortized » sur colonnes harmonisées, 3e composante du blend H30 | faible-moyen | faible | - | ABSORBÉ dans H44 | |
| H34 | Structure | Template SN + prior cubique fittés sur P latent (pas sur target monotonisé) | faible-moyen | moyen | 15 | ABANDONNÉ (e029) — prior + template hors visites censurées : +0.007 ± 0.017 NOISE, poids blend 0 | e029_latent_prior |
| H36 | Méthode | Générateur de bruit conditionnel (pente, position, échelle patient) → augmentation de données | moyen (incertain) | fort | 16 | TODO — data-limited (−0.08/doublement) mais augmentation naïve +0.09 à +0.18 | scratch/resid/p4.py |
| H37 | Structure | Localisation du bruit SN dépendante de la pente locale dans la MAP cubique | faible | faible | - | RÉFUTÉ (dgp3.md) : effet pente = règle ×1.02, couvert par H33 | |
| H39 | Manquants | A_p : récupérer les paires on/off sans tsi_on (g = gbar, poids réduit) | faible | faible | - | ABSORBÉ dans H42 | scratch/miss/ |
| H40 | Mesure | Effet aléatoire patient u_p sur les sources ON dans la MAP (incertitude A_p), Newton 5D | faible | moyen | 17 | TODO — GLS fold 0 4.171 → 4.152 (traj.) | scratch/meas/ |
| H38 | Mesure | Template OFF scindé traités / non traités (mélange contaminé ×0.74) | faible | faible | - | ABSORBÉ dans H46 (off_tr) — plafond ≈ 0 en final d'après level.md | scratch/meas/ |
| H35 | Ensemble | Pipeline e022b entraîné sur P latent + règle, blend 50/50 avec e022b | faible | moyen | 18 | TODO — −0.012 / −0.024 (folds 0/1) | scratch/dgp2/b4.py |
| H41 | Mesure | Features HGB : marche log y_v à l'initiation, écart ON vs trajectoire OFF seule | faible | faible | - | ABSORBÉ dans H42 (2e passe ON-only) | |
| H20 | Trajectoire | Stats patient des surprises `y_v − hb_post_loo` (max, skew, std…) dans e017 | fort | faible | - | FAIT : e019 (H20+H21) −0.260 ± 0.017 vs e017, 5/5, CV 2.812 (nouveau meilleur) ; e019b seul −0.200 | e019 / e019b |
| H21 | Trajectoire | Surprises visite + voisines + 1re visite (`surp`, `infl`) dans e017 | moyen | faible | - | FAIT : e019a seul −0.109 ; +0.06 au-delà de H20 dans e019 | e019a |
| H17 | Trajectoire | Trajectoire cubique hiérarchique (prior 4×4) dans la MAP skew | moyen | moyen | - | FAIT : e022b −0.033 ± 0.027 vs e019, 5/5, CV 2.778 (nouveau meilleur) ; e022 seul −0.025 | e022 / e022b |
| H23 | Ensemble | MLP bagué (5 seeds) niveau+delta, moyenné avec e022b | moyen | moyen | - | FAIT : b002_e022b_mlp (e022b×0.59 + e024_mlp_bag×0.41) −0.053 ± 0.036 vs e022b, 5/5, CV 2.727 ; MLP seul 2.841 | e024_mlp_bag / b002_e022b_mlp |
| H19 | Rater | Facteur rater par MLE skew-normal (off + on) au lieu de la médiane | très faible | faible | - | ABANDONNÉ (EDA measurement.md : s oracle −0.012 sur la trajectoire ; estimateurs alternatifs pires) | |
| H24 | Ensemble | Réglage grossier du MLP de H23 (epochs, α, largeur, 10 seeds) | moyen | moyen | 20 | TODO (après H23) | |
| H18 | Ensemble | Recalibrage courbure patient des préd. (α·Q + β·C3, cross-fit) | faible | faible-moyen | 21 | TODO (redondant si H17) | EDA −0.016 à −0.028, 5/5 |
| H22 | Trajectoire | Noyau en âge de y_v en écart à la trajectoire | faible | faible | 22 | TODO (si H20+H21) | |
| H25 | Ensemble | Modèle faible décorrélé / stacking / RF-ET / variantes HGB | faible | - | - | ABANDONNÉ (EDA ensemble.md) | |
| H26 | Manquants | Fréquence patient / cumulée de OFF manquant (MNAR) | - | faible | - | ABANDONNÉ (e020 NOISE) | Δ −0.002 ± 0.008 vs e019 |
| H27 | Méthode | Génération massive de features + sélection FDR / SHAP | - | moyen | - | ABANDONNÉ (sonde e021, fold 0) | sans t_* +0.063 ; FDR −0.003 |
| H28 | Méthode | Pipeline skrub (tabular_pipeline / DataOps), encodage des catégorielles | - | faible | - | ABANDONNÉ (sonde e021, fold 0) | sans catégorielles ±0.000 ; TargetEncoder +0.072 |
| H29 | Structure | Harmoniser les covariables fixes incohérentes intra-patient (vote majoritaire) | - | faible | - | RÉFUTÉ (EDA) : 0 incohérence | |
| H58 | Structure | Confiance par visite selon sa qualité : biais m(x) et échelle s(x) du bruit z par covariables visite dans la MAP skew | faible-moyen | faible | - | RÉFUTÉ : e023_quality_noise +0.004 ± 0.023 vs e022b, 1/5 (NOISE) | e023 |
| H59 | Trajectoire | Loi symbolique : cubique + coude (pente ≈ 0), familles exp/log/sigmoïde/puissance | - | moyen | - | ABANDONNÉ (EDA `eda/symbolic.md`) : coude non détectable sur y_v (AUC 0.44), aucune loi complémentaire à e022b | |
| H60 | Trajectoire | Modèle séquentiel : BiGRU sur la séquence de visites (entrées brutes + masques + Δt), blend avec b002_e022b_mlp | fort | moyen | - | FAIT : e025_gru_raw seul 2.803 (≈ e022b) mais ρ 0.77 ; b003_e022b_mlp_gru −0.109 ± 0.038 vs b002_e022b_mlp, 5/5, CV 2.618 (meilleur de la branche bertrand ; < b003_seq4 2.541 de main) ; Transformer e026_transformer_raw 2.968, non retenu. À faire : complémentarité avec b003_seq4 | e025_gru_raw / e026_transformer_raw / b003_e022b_mlp_gru |
| H63 | Trajectoire | Extrapolation de la progression hors fenêtre OFF à partir des observations ON | - | faible | - | ABANDONNÉ : R² correcteur HGB hors fenêtre −0.109 ; correction principale 2.5408 → 2.5562, pire 5/5 ; aucune information résiduelle (`eda/h63_on_progression.md`) | sonde OOF (scratch/h63) |
| H04 | Rater | Facteur rater multiplicatif commun on/off, normaliser avant le contexte patient | fort | faible | 1 | FAIT (e003_rater_scale) | CV 4.271, Δ -0.310 ± 0.024 |
| H01 | Pharmaco | Amplitude médicament par patient A_p sur courbe Emax(Bateman) commune ; `on_eq` | moyen | faible | 2 | FAIT (e004_drug_amplitude) | CV 4.069, Δ -0.202 ± 0.042 vs e003 |
| H06 | Trajectoire | Trajectoire linéaire hiérarchique `a_p + b_p·âge` (prior de pente par covariables), HGB en correcteur du résidu | fort | moyen | - | FAIT (e007, GAIN −0.034 ; offset e008 NOISE) | |
| H09 | Trajectoire | Trajectoire v2 : bruit hétéroscédastique (sd ∝ niveau) + courbure de population + projection finale sur la droite patient | moyen | faible | - | FAIT (e009 −0.022 ; e010 −0.063 vs e009) | e010 CV 3.465 |
| H10 | Trajectoire | Poids de projection par visite (1re visite w≈0, sinon selon source off/on_t/on_nt) | faible | faible | 23 | TODO (négligeable a priori) | |
| H16 | Méthode | Pivot tabulaire : bloc temporel générique (lags/leads/rolling/deltas/tendances) + cible delta target_t − target_{t−1} avec reconstruction par patient | fort | moyen | - | FAIT : e015 −0.507 vs e002 ; e016 −0.207 vs e015 ; e017 −0.155 ± 0.026 vs e012 (nouveau meilleur, CV 3.072) | e015 / e016 / e017 |
| H11 | Structure | Bruit de mesure asymétrique (skew ≈ −0.6, partagé on/off) → trajectoire à vraisemblance skew-normale au lieu de WLS gaussien | fort | moyen | 1 | FAIT : e012 GAIN −0.238 ± 0.046 vs e011 (asymétrie seule −0.231 vs e012a) | e012 / e012a |
| H13 | Ensemble | Mieux combiner HGB et la trajectoire MAP skew : mélange direct, HGB en offset, moyenne cible/offset | faible | faible | - | FAIT : e013a pire (fold 0), e013b NOISE, e013c −0.032 ± 0.034 (5/5, NOISE strict) | e013b / e013c |
| H12 | Structure | LEDD fonction explicite du true OFF (audit externe) | - | faible | - | RÉFUTÉ (EDA) | |
| H07 | Méthode | Variances et λ estimés itérativement (EM) ; prior de pente par cohorte/gène | faible | faible | 24 | EN ATTENTE (après H11, qui change le modèle de bruit) | |
| H15 | Pharmaco | Score bivarié on+off à la même visite (combinaison pondérée) | nul | faible | - | ABANDONNÉ (EDA) | |
| H08 | Structure | Plancher à 0 : off == 0 et droite négative en début de maladie (modèle censuré) | faible | moyen | 25 | TODO (clip [0,132] déjà en place depuis e005) | |
| H02 | Trajectoire | Lissage par patient (pente / effets aléatoires) des scores débiaisés | fort | moyen | - | ABSORBÉ (e006 → H06) | |
| H05 | Trajectoire | Score harmonisé par visite (off/s ou on_eq/s) pondéré par variance → trajectoire | moyen | moyen | 4 | FAIT (e006, GAIN −0.475) | |
| H03 | Rater | Biais additif par `rater_id` | moyen | faible | - | ABANDONNÉ (→ H04) | |
| H61 | Structure | Prior cubique à queues lourdes (Student-t / mélange d'échelles sur c2, c3) dans la MAP + E[rule(P)] | faible-moyen | moyen | 6 | TODO — excès de kurtosis c2 1.9, c3 4.9 ; générateur : prior gaussien → G 5.4 % vs 9.2 %, bootstrap résidus → 10.7 % ; sonde Σ×1.5 fold 0 E[rule(P)] 3.5118 → 3.5048 | scratch/dgp4/gen.py, d2.py |
| H62 | Structure | Bruit OFF non i.i.d. : échelle croissante avec le nb d'OFF antérieures + autocorr. lag 1 négative (vraisemblance bandée dans la MAP) | faible | moyen | 12 | TODO — IQR 0.33 → 0.48 → 0.54 → 0.59 (naïfs) ; ρ(v0,v1) −0.13 (5/5 folds), lag 1 −0.03 à −0.08 ensuite | scratch/dgp4/c2.py, c6.py |

Axes : `Pharmaco` · `Trajectoire` · `Manquants` · `Rater` · `Structure` · `Méthode` · `Ensemble`

---

## Modèle de fiche

```markdown
### HNN — <titre court>
- **Axe** : Pharmaco | Trajectoire | Manquants | Rater | Structure | Méthode | Ensemble
- **Hypothèse** : ce qu'on croit vrai sur le processus générateur.
- **Pourquoi ça devrait marcher** : indice EDA, argument du CONTEXT, résultat précédent (eNNN).
- **Test (≤ 10 lignes)** : ce que fait `fit_predict`, la référence à battre (eNNN).
- **Critère de succès** : Δ apparié < -std(Δ), et quels sous-groupes doivent s'améliorer.
- **Risques / fuites** : features indisponibles au test, estimation hors-fold, etc.
- **Coût** : temps d'implémentation, temps de run.
- **Statut** : TODO
```

---

## Fiches

### H01 — Amplitude médicament par patient sur courbe Emax(Bateman) commune (affinée, cf. `lab/eda/pharmaco.md`)
- **Axe** : Pharmaco
- **Hypothèse** : `on = off_éq × (1 − A_p·g(tsi_on))`, g commune de forme Emax(Bateman) (Em 0.571, ka 0.341, ke 0.339, EC50 0.0012, h 2.90 : plateau 1.5–5 h, ≈ 0 après 12 h). A_p propre au patient (moy. 0.47, std 0.064), indépendante de ledd/âge/cohorte/gène. **Le débiaisage de `off` par `tsi_off` est abandonné** (RMSE 9.37 → 9.44 : 4 % seulement des `off` ont tsi_off < 11 h).
- **Pourquoi ça devrait marcher** : `on/off` au plateau corrèle à 0.918 avec la moyenne des autres visites du patient ; biais OOF e002 de +0.70 à −0.79 selon le quintile de A_p ; correction cross-fittée du résidu 4.581 → 4.547 (off_missing 5.21 → 5.15).
- **Test** : étend e002. Fit de g par `curve_fit` sur `1 − on/off` (sans target → train + valid autorisé). `a_v = (1 − on/off)/g(tsi_on)` si g > 0.5 ; A_p = moyenne des autres visites du patient shrinkée vers 0.47 (poids 0.5). Features : `A_p`, `g_on`, `on_eq = on/(1 − A_p·g)` (g moyen si tsi_on NaN), prev/next/pmean de `on_eq`.
- **Critère de succès** : Δ ≤ −0.03 et |Δ| > std(Δ) ; off_missing ≤ −0.05, tsi_on_present en baisse.
- **Risques / fuites** : pas de target. 40 % des patients sans visite on+off+tsi_on → A_p au prior.
- **Coût** : faible · **Gain attendu** : moyen
- **Revue (leak-reviewer, 2026-10-07) : GO sous conditions**
  1. BLOQUANT : ne pas coder en dur les paramètres de la fiche (issus de `pk12.py`, fit joint utilisant `target`). Refitter g dans `fit_predict`, sur `train` seul, côté ON seul (`1 − on/off`), `least_squares` borné (Em ≤ 1, EC50 ≤ 1) + try/except ; fallback saturation `Em(1−e^{−kt})` ou médiane par bin interpolée. Normaliser `g = E / max E` sur [0, 6.3]. Ne pas réutiliser `lab/scratch/*.npy`.
  2. `g`, `A0` (= moyenne des `a_v` du train) et `gbar` fittés sur train une fois, passés tels quels à `build(valid)`.
  3. Une seule fonction `add_drug_amplitude(df, g_params, A0, gbar, k=0.5)` pour train et valid (LOO identique) ; ajouter `n_av` en feature.
  4. Robustesse : `a_v` clip [0, 1], `A_p` clip [0.15, 0.85], `on_eq = NaN` si `on == 0`, `on_eq` clip [0, 132] ; exclure `on==0`/`off==0` des prev/next/pmean de `on_eq`.
  5. Ne pas tuner k / seuil g > 0.5 sur les folds. Le gain 4.547 n'est qu'indicatif (folds différents, sans std).
- **Statut** : FAIT (e004_drug_amplitude) — CV 4.069, Δ -0.202 ± 0.042 vs e003 (5/5 folds) ; off_missing 4.87→4.58.

### H02 — Trajectoire par patient
- **Axe** : Trajectoire
- **Hypothèse** : le true OFF suit une progression lisse (≈ linéaire en durée de maladie) par patient ; les mesures sont du bruit autour.
- **Pourquoi ça devrait marcher** : 4 à 12 visites par patient, âges triés ; toutes les visites test d'un patient sont disponibles.
- **Test** : régression robuste par patient de `off_corrigé` (H01) sur `age`, shrinkage vers une pente population (modèle mixte `statsmodels` ou shrinkage empirique) ; prédiction lissée en feature.
- **Critère de succès** : gain surtout sur `off_missing`.
- **Risques / fuites** : aucun si on n'utilise pas `target` des patients de valid.
- **Coût** : moyen.
- **Statut** : ABSORBÉ — la régression ridge pondérée par patient de e006 (H05) en est une version ; la suite est H06.

### H03 — Biais rater
- **Axe** : Rater
- **Hypothèse** : chaque clinicien a un biais additif (ou multiplicatif) sur `on` / `off`.
- **Pourquoi ça devrait marcher** : « human subjectivity in scoring » est le biais n°1 listé ; 50 raters partagés train/test.
- **Test** : sur `train`, résidu moyen (`off − target`) par rater, shrinké ; soustraire avant modélisation.
- **Critère de succès** : la dispersion du RMSE par rater (`metrics.json → per_rater`) diminue.
- **Risques / fuites** : encodage calculé sur le fold d'entraînement uniquement (sinon fuite).
- **Coût** : faible.
- **Statut** : ABANDONNÉ — l'EDA Pharmaco montre un biais multiplicatif commun on/off (corr 0.974) → remplacé par H04.

### H04 — Facteur rater multiplicatif commun on/off, appliqué avant le contexte patient
- **Axe** : Rater
- **Hypothèse** : `on` et `off` mesurés sont multipliés par un même facteur `s_rater` ∈ [0.66, 1.10]. Les prev/next/pmean de e002 mélangent des raters et propagent ce biais.
- **Pourquoi ça devrait marcher** : corr(facteur rater on, off) = 0.974 ; `off/s` : RMSE 10.20 → 9.37, biais −5.38 → −0.30 ; biais OOF e002 par rater corrélé à 0.817 avec `s`. Correction de résidu indicative 4.581 → 4.430 (⚠ `s` fitté sur tout le train : chiffre optimiste).
- **Test** : étend e002. Dans `fit_predict`, `s_r` = médiane de `off/target` par rater (et `on/(target·(1−E))`), sur train, shrinkée. Diviser on/off par `s_r`, recalculer prev/next/pmean sur les valeurs normalisées ; ajouter `s_r` et la moyenne patient de `s`.
- **Critère de succès** : Δ ≤ −0.08 ; off_missing, off_measured et la dispersion per_rater baissent.
- **Risques / fuites** : `s` appris sur target → strictement dans `fit_predict`. ~900 visites/rater : sur-ajustement faible.
- **Coût** : faible · **Gain attendu** : fort
- **Revue (leak-reviewer, 2026-10-07) : GO sous conditions**
  1. BLOQUANT : diviser chaque visite par le `s` de **son** rater avant le shift (`off_n = off / rater_id.map(s)`), puis `add_patient_context` sur `on_n`, `off_n`. Utiliser `.map(s).fillna(s_global)`, pas de merge ni de tri sans restauration.
  2. Estimer `s_r = median(off/target)` sur `off > 0 & target > 0` ; shrinkage léger (~20 visites) vers la médiane globale ; pas de cross-fitting interne (Δs entre folds ≤ 0.01 vs sd inter-rater 0.086).
  3. Un seul `s_r` (estimé sur off) appliqué à on et off : pas de dépendance à E(tsi_on)/H01.
  4. `s_r` seul en colonne est redondant avec `rater_id` ; le gain doit venir de la normalisation + `mean_patient(s)`.
  5. Le 4.430 est indicatif (correction de résidu) : juger uniquement sur `--ref e002_hgb_basic_context`.
  6. Variante possible : `s` estimable sans target (effets fixes patient + rater sur `log(off)`, corr 0.948).
- **Statut** : FAIT (e003_rater_scale) — CV 4.271, Δ -0.310 ± 0.024 vs e002 (5/5 folds).

### H05 — Score harmonisé par visite pondéré par sa variance, entrée de la trajectoire
- **Axe** : Trajectoire
- **Hypothèse** : après correction rater (H04) et médicament (H01), chaque visite est un estimateur non biaisé du true OFF avec un bruit i.i.d. commun à on/off ; seule la mise en commun des visites le réduit.
- **Pourquoi ça devrait marcher** : bruit visite non persistant (−0.06 avec la moyenne des autres visites, −0.09 lag-1) mais commun on/off (0.859). RMSE ponctuels 9.4 (`off/s`) et 12.9 (`on_eq/s`) contre 4.58 pour e002 : le pooling est le levier.
- **Test** : étend H01 + H04. `y_v = off/s` sinon `on/(s(1−A_p g))`, poids 1/9.4², 1/12.9², 1/15² si tsi_on NaN (ne pas moyenner off et on_eq à la même visite). Features : moyenne patient pondérée leave-one-out, régression locale pondérée de `y_v` sur l'âge, `y_v`. Exclure `on==0` et `off==0` aberrants.
- **Critère de succès** : Δ ≤ −0.05 au-delà de H04 ; off_missing et first_visit.
- **Risques / fuites** : `s` et la courbe dans le fold ; poids fixés sur train.
- **Coût** : moyen · **Gain attendu** : moyen
- **Statut** : FAIT — e006, CV 3.583, −0.475 ± 0.049 vs e005 (5/5). Le gain vient surtout de la trajectoire WLS ridge sur l'âge.

### H06 — Trajectoire linéaire hiérarchique, HGB en correcteur
- **Axe** : Trajectoire
- **Hypothèse** : target ≈ a_p + b_p·âge par patient (résidu d'une droite par patient de 1.19 ; pente 2.5 ± 1.35 /an).
- **Pourquoi ça devrait marcher** : dans e006, `h_traj_loo` seul a un RMSE de 5.86 contre 9.71 pour la moyenne patient ; le gain −0.475 vient surtout de la pente.
- **Test** : régression bayésienne pondérée (poids 1/var de y_v) par patient avec prior (a, b) dépendant des covariables (cohort, gene, age_at_diagnosis, ledd), fitté sur train ; prédiction a_p + b_p·âge en feature ou en offset, HGB sur le résidu. Étend e006.
- **Critère de succès** : Δ < −std(Δ) vs e006 ; first_visit et patients à peu de visites.
- **Risques / fuites** : prior et hyperparamètres fittés sur train uniquement.
- **Coût** : moyen · **Gain attendu** : fort
- **Statut** : FAIT — e007 (features) CV 3.550, Δ −0.034 ± 0.012 vs e006, 5/5 folds : gain faible. La pente n'est pas prédictible par les covariables (prior b RMSE 1.31 vs sd 1.35) ; le gain vient du posterior 2D (intercept + σ²_ligne + pente prior 2.5 au lieu de b0 y_v 2.1) : trajectoire seule 4.71 OOF vs 5.25 pour e006, mais HGB en récupère peu. Variante offset e008 : NOISE (+0.001 ± 0.061), instable, mais moyenne 50/50 e007+e008 = 3.508 (−0.075 vs e006) → piste variance/blend. first_visit quasi inchangé (3.34→3.33).

### H07 — Variances et λ par EM, prior de pente par cohorte/gène
- **Axe** : Méthode
- **Hypothèse** : les variances de bruit (9.1 / 11.9 / 14.5) et le λ ridge (0.55), pris sur les résidus bruts, sont sous-optimaux.
- **Test** : itérer estimation des trajectoires ↔ variances ; prior de pente par cohorte/gène. Étend e006 (ou H06).
- **Coût** : faible · **Gain attendu** : moyen
- **Statut** : TODO

### H08 — Plancher à 0
- **Axe** : Structure
- **Hypothèse** : off == 0 (618 visites, target médian 1.15) traduit un plancher ; une droite par patient peut passer sous 0 en début de maladie.
- **Test** : traiter off == 0 comme censuré (≤ seuil) dans la trajectoire ; vérifier les résidus e006 près de 0.
- **Coût** : moyen · **Gain attendu** : faible
- **Statut** : TODO

### H09 — Trajectoire v2 : poids hétéroscédastiques, courbure de population, projection finale
- **Axe** : Trajectoire
- **Hypothèse** : (1) le bruit de mesure de y_v croît avec le niveau du patient (sd de `off/s − target` : 5.5 → 11.7 du quintile bas au haut), alors que e006/e007 pondèrent à variance constante ; (2) les vraies trajectoires sont légèrement concaves (courbure de pop. −0.03 à −0.05/an², résidu de la droite + au milieu, − aux bords, autocorr. lag-1 0.55) ; la courbure par patient est du bruit (sd 0.38).
- **Pourquoi ça devrait marcher** : décomposition de l'erreur OOF de e007 (`lab/scratch/traj_err_decomp.py`) : niveau patient 2.68, pente 1.74, intra-droite 1.55 → l'erreur est entre patients, donc dans l'estimation (a_p, b_p), qu'une pondération correcte améliore. Projeter les préd. e007 sur la droite patient (w 0.5) : 3.550 → 3.492. Le bruit ne dépend pas du rater (sd 8.1–9.9, split-half 0.11) → pas de poids par rater.
- **Test** : étend e007. Var. de y_v = var_src · f(μ_v), f(μ) = ((c0 + c1 μ)/(c0 + c1 μ̄))² fitté sur train (résidu vs target) ; μ_v = 1re passe hb_post_full (sans target), puis 2e passe. Courbure κ (quadratique poolée intra-patient du target, train) : trajectoire sur y_v − κ·x², x = âge centré patient, κ·x² rajouté. (e010) projection des préd. HGB sur la droite + κ x² du patient, poids fixé.
- **Critère de succès** : Δ < −std(Δ) vs e007 ; hb_post_full seul < 4.70 ; off_missing, first_visit.
- **Risques / fuites** : c0, c1, κ fittés sur train dans fit_predict ; μ_v sans target ; poids de projection fixé a priori (0.5, vu sur l'OOF e007 → légère sélection, à noter).
- **Coût** : faible · **Gain attendu** : moyen
- **Revue (leak-reviewer, 2026-10-07) : GO sous conditions** — aucun bloquant ; corrigés : `hb_int_loo` ne reçoit plus κx², c1 ≥ 0, κ refitté par sous-fold dans la boucle interne de e010. À la soumission : figer w = 0.5 (pas de cross-fit interne), tri patient/âge, sklearn ≥ 1.6, random_state 42.
- **Statut** : FAIT — e009 CV 3.527 (−0.022 ± 0.014 vs e007, 5/5, ~négligeable seul) ; e010 CV **3.465** (−0.063 ± 0.009 vs e009, 5/5). w fitté in-fold 0.49–0.52. first_visit se dégrade (3.26 → 3.32) → H10.

### H10 — Poids de projection par visite
- **Axe** : Trajectoire
- **Hypothèse** : la projection e010 n'aide pas la 1re visite (souvent off mesuré, non traitée, bruit sd 6.6) mais aide d'autant plus que la mesure de la visite est bruitée.
- **Pourquoi ça devrait marcher** : w* sur OOF e009 (`lab/scratch/proj_by_src.py`) : 1re visite ≈ 0 quelle que soit la source ; sinon off 0.49, on_t 0.62, on_nt 0.76 ; dernière visite 0.73 (RMSE 4.46, la pire).
- **Test** : e010 avec w par groupe (first × source), fitté sur l'OOF interne.
- **Critère de succès** : Δ < −std(Δ) vs e010 ; first_visit ≤ 3.26.
- **Risques / fuites** : aucun (w sur OOF interne) ; 8 poids au lieu d'un.
- **Coût** : faible · **Gain attendu** : faible (in-sample 3.465 → 3.454, ≈ −0.011, sous le seuil ~0.02)
- **Statut** : TODO

### H11 — Trajectoire à vraisemblance asymétrique (bruit skew-normal)
- **Axe** : Structure
- **Hypothèse** : le bruit de mesure visite n'est pas gaussien : `off/(s·target)` est fortement asymétrique à gauche (quantiles 1 % 0.19, 10 % 0.54, 50 % 1.00, 90 % 1.28, 99 % 1.49 ; skew −0.4 → −0.7 selon le niveau), bord droit abrupt, longue queue basse. Côté ON au plateau, skew du log −1.6. Le choc est partagé on/off à la même visite (log(on/target) médian −0.99 si off « bas » vs −0.54 sinon). Il n'est prédit par aucune covariable (rater, tsi_off, ledd, gène, cohorte, âge : taux plats) et n'est pas persistant par patient (corr LOO 0.013). On ne peut donc pas identifier la visite contaminée, mais l'estimateur de niveau patient doit tenir compte de l'asymétrie.
- **Pourquoi ça devrait marcher** : premier poste d'erreur de e010 = niveau patient (2.68). Patients ≥ 4 off mesurés (`lab/scratch/noise_estimators.py`) : sd relatif du niveau estimé 0.121 (moyenne), 0.154 (médiane), 0.109 (q75), **0.079 (MLE skew-normal, a −3.8, loc 1.30, scale 0.45)**, soit −35 % d'écart-type. Le WLS gaussien de e006–e010 est un estimateur de type moyenne.
- **Test (≤ 10 lignes)** : étend e011 (forme figée). Dans `fit_predict`, sur train : template skew-normal de `y_v/target` par source (off, on_t, on_nt). Par patient : MAP de (a_p, b_p) (+ κx² de pop.) sous vraisemblance multiplicative skew-normale + prior gaussien e007 (`scipy.optimize` par patient, ~7k petits problèmes 2D, ou Newton vectorisé). Remplacer hb_post_full/loo par leur version skew ; garder la projection w = 0.5.
- **Critère de succès** : Δ < −std(Δ) vs e011 ; trajectoire seule < 4.62 (hb_post_full e009) ; off_missing et off_measured en baisse.
- **Risques / fuites** : template fitté sur target train seulement ; templates par source requis (le skew ON est plus fort) ; coût d'optimisation (vectoriser, pas de boucle Python lente) ; vérifier que le gain survit à l'interaction avec l'hétéroscédasticité de e009 (template en ratio = déjà multiplicatif → c0/c1 probablement superflus).
- **Coût** : moyen · **Gain attendu** : fort (revu : Δ final attendu −0.02 à −0.08)
- **Revue (leak-reviewer, 2026-10-07) : GO sous conditions**
  1. BLOQUANT : template fitté dans `fit_predict` sur train seul (aucune constante des scripts scratch). Gain hors-échantillon vérifié (fold 0 : moyenne 0.121, MLE 0.083).
  2. BLOQUANT : une seule fonction MAP (full + LOO par masque) pour train et valid ; pas de raccourci LOO.
  3. BLOQUANT : critère « < 4.62 » invalide (hb_post_full biaisé −1.53 : y_v a une moyenne < 0 car s = médiane). Contrôle e012a = gaussien recalé en moyenne par source ; attribuer à l'asymétrie seulement e012 − e012a ; décision sur Δ vs e011.
  4. Modèle location-scale additif, pas en ratio : z = (y_v − μ)/σ, σ = c0 + c1·μ_pass1 figé, z ~ SN(ξ, ω, α) par source (fit sur (y_v − target)/(c0 + c1·target)).
  5. Newton 2×2 vectorisé (posterior concave), init gaussienne, demi-pas, log_ndtr ; MLE template maison borné |α| ≤ 10 ; se Laplace ; mêmes colonnes hb_* que e011.
- **Statut** : FAIT. e012_traj_skew CV 3.226 (−0.238 ± 0.046 vs e011, 5/5 folds) ; contrôle e012a (gaussien recalé) 3.458, NOISE (−0.007 ± 0.014) → effet de l'asymétrie = e012 − e012a = −0.231 ± 0.041. Trajectoire seule valid 4.62 → 4.32 (e012a) → 3.81 (e012). Template stable : off α −4.0, on_t α −1.5, on_nt α +1.4 (skew inverse). Nouveau meilleur run.

### H12 — LEDD fonction explicite du true OFF (audit externe)
- **Axe** : Structure
- **Hypothèse** : le LEDD prescrit serait un multiplicateur / une fonction du true OFF.
- **Résultat EDA** (`lab/scratch/ledd_struct.py`) : LEDD ≈ fonction de la **durée de maladie** (intra-patient +35/an ; inter-patient +52/an de durée moyenne, −2/an d'âge au diagnostic). À durée donnée, quasi aucun lien avec le target : corr des résidus de droites par patient −0.014, corr des pentes 0.022, +0.9 LEDD par point de target moyen. LEDD n'apporte pas de mesure supplémentaire du true OFF ; sa missingness (80 % à la 1re visite → 12 %) est celle de « traité ou non ».
- **Statut** : RÉFUTÉ

### H13 — Combiner HGB et la trajectoire MAP skew (suite de e012)
- **Axe** : Ensemble
- **Hypothèse** : la trajectoire MAP skew seule (hb_post_full, 3.81 valid) est devenue assez bonne pour améliorer le modèle final (3.23) autrement que comme feature : (a) mélange direct `(1−w)·hgb + w·hb_post_full` au lieu de la projection sur la droite OLS des préd. HGB ; (b) HGB en offset sur hb_post_full (H06-B, NOISE en e008 avec une trajectoire moins bonne) ; (c) moyenne fixe 50/50 de e012 et de la variante offset.
- **Pourquoi ça devrait marcher** : en e012 HGB n'exploite qu'une partie de la trajectoire ; b001 (e007+e008) avait montré un gain de variance HGB.
- **Test (≤ 10 lignes)** : helpers `lab/feat_skew_proj.py` (importe feat_skew, sans le modifier). e013a : w fitté par cross-fit interne GroupKFold sur train (LS de target − hgb sur hb_post_full − hgb). e013b : HGB sur target − clip(hb_post_full, 0, 132) (pas de clip du résidu), préd. = hb_post_full + résidu, projection OLS w = 0.5. e013c : 0.5·e012 + 0.5·e013b, build commun. Référence e012_traj_skew.
- **Critère de succès** : Δ < −std(Δ) vs e012 ; off_missing, first_visit.
- **Risques / fuites** : w de e013a et template skew fittés sur train seul ; poids 50/50 fixés a priori.
- **Coût** : faible (e013a ×6 à cause du cross-fit interne ; e013c +25 %).
- **Statut** : FAIT. (a) mort : w interne 0.09, fold 0 3.157 vs 3.141 ; erreurs HGB/trajectoire corr. 0.84–0.85, poids LS joint (OLS 0.38, traj −0.01) → la trajectoire n'ajoute rien à HGB au-delà de la feature, la projection OLS reste le bon post-traitement (w optimal ≈ 0.4 plutôt que 0.5, gain ≈ 0.002, négligeable). (b) e013b CV 3.207, −0.020 ± 0.052 (2/5) NOISE, instable comme e008. (c) e013c CV **3.195**, −0.032 ± 0.034, 5/5 folds meilleurs mais |Δ| < std (fold 4 −0.088) → NOISE strict ; erreurs e012/e013b corr. 0.973 = réduction de variance HGB. À confirmer par un bag de seeds / lr plus bas (H14 ?), plus simple à soumettre.

### H14 — Le gain de e013c est-il juste du bagging de seeds HGB ?
- **Axe** : Méthode
- **Hypothèse** : e013c (moyenne 50/50 e012 + e013b, −0.032 ± 0.034, erreurs corr. 0.973) réduit simplement la variance de `HistGradientBoostingRegressor`, sans vraie diversité de modèle ; un bag de plusieurs graines HGB sur le pipeline e012 (mêmes features, même w=0.5) devrait retrouver ce gain plus simplement (1 seule cible).
- **Pourquoi ça devrait marcher** : b001 (e007+e008) avait le même profil (corr. 0.953) ; si c'est bien de la variance, bagger les graines d'un seul modèle suffit.
- **Test (≤ 10 lignes)** : `lab/feat_skew_bag.py` (importe `feat_skew`, sans le modifier) : `bag_hgb_predict` moyenne N `HistGradientBoostingRegressor(random_state=SEED..SEED+N-1)` (mêmes hyperparams que e012). `e014_skew_bag.py` = build e012 (`build_e011` + `skew_hook`) + bag (N=5) + projection w=0.5, comme e012.
- **Critère de succès** : Δ < −std(Δ) vs e012 ; retrouver tout ou partie du gain de e013c.
- **Risques / fuites** : aucun (hyperparamètres et graines fixes, rien appris sur target au-delà de e012).
- **Coût** : faible (×N sur le seul `HGB.fit`, pas sur le build)
- **Statut** : RÉFUTÉ. e014 CV 3.226, Δ **+0.000 ± 0.000 vs e012** (0/5 folds, NOISE) : prédictions bit-identiques à e012 sur chaque fold (vérifié dès `--folds 0`, confirmé par le run complet). Cause : `HistGradientBoostingRegressor.random_state` ne pilote que le sous-échantillonnage du binning (`_BinMapper`, seuil `subsample=2e5`) — jamais déclenché avec ~35–36k lignes par fold train, donc **inerte** sur ce jeu de données (vérifié indépendamment sur un jeu toy : `random_state=1` vs `2` → prédictions identiques). Conséquence : bagger des graines HGB ne crée aucune diversité ici, donc aucun gain ; le gain de e013c vient nécessairement de la diversité des *cibles* (target brut vs résidu offset sur la trajectoire skew), pas d'une réduction de variance via les graines. Coût ×2.1 (90.8 s vs 43.4 s) pour rien → à ne pas mettre dans le modèle soumis. Piste si on veut retenter une diversité légère : varier `learning_rate`/`max_leaf_nodes` ou sous-échantillonner les lignes à la main plutôt que `random_state` (non testé, hors périmètre de H14).

### H15 — Score bivarié on+off à la même visite
- **Axe** : Pharmaco
- **Hypothèse** : quand on ET off sont mesurés à la même visite, combiner off_n et on_eq_n (moyenne pondérée par variance inverse, ou alpha appris) réduirait l'erreur par rapport à l'estimateur actuel y_v = off_n seul (H05/H11).
- **Pourquoi ça devrait marcher** : rejetée par l'EDA — corr(e_off, e_on) = 0.862 sur 12 433 visites train (28 % du train), mais sd_on (ex. 12.48 au tercile milieu) toujours > sd_off (10.96) ; le ratio sd_off/sd_on (0.82–0.89) est partout ≤ la corrélation observée, donc aucune pondération ne bat off_n seul.
- **Test (≤ 10 lignes)** : non applicable — pas de gain à tester. (testé en EDA : grid-search alpha·off_n+(1−alpha)·on_eq_n en CV 5-fold hors-fold, alpha*=1.00 dans tous les sous-groupes : global, tercile de target, bin tsi_on).
- **Critère de succès** : non atteint — RMSE combinaison (50/50 : 11.214 ; inv-var : 11.091) > RMSE off_n seul (10.748) sur les 12 433 visites concernées, dans tous les sous-groupes testés.
- **Risques / fuites** : n/a (idée non implémentée).
- **Coût** : faible (EDA seule, pas d'expérience lancée) · **Gain attendu** : nul (mesuré, pas estimé)
- **Statut** : ABANDONNÉ (EDA `lab/scratch/h15_bivariate.py`, section 6 de `lab/eda/pharmaco.md` : alpha*=1.00 partout, aucune combinaison ne bat off_n seul)

### H16 — Pivot tabulaire : bloc temporel massif + cible delta reconstruite
- **Axe** : Méthode
- **Hypothèse** : un HGB peut découvrir seul les interactions temporelles si on lui donne lags t−1/t−2, leads t+1/t+2, deltas d'âge et de LEDD, moyennes glissantes et tendances par patient ; prédire la variation `target_t − target_{t−1}` est plus facile que le niveau.
- **Problème de fond** : au test, `target_{t−1}` est inconnu → un delta seul ne donne pas de niveau, et un chaînage cumule les erreurs. Solution : moindres carrés par patient, `argmin Σ(L_t − A_t)² + λ Σ(L_t − L_{t−1} − D_t)²` (A = HGB niveau, D = HGB delta), système tridiagonal, sans target.
- **Test** : `lab/feat_temporal.py` ; e015 (FE seul), e016 (delta + reconstruction, λ par cross-fit interne), e017 (e012 + FE + delta, λ = 4, projection retirée).
- **Résultat** : e015 4.074 (−0.507 vs e002) ; e016 3.867 (−0.207 vs e015 ; contrôle lissage seul nettement moins bon → l'information vient du delta) ; e017 3.072 (−0.155 ± 0.026 vs e012, 5/5 folds).
- **Risques / fuites** : `delta_target` uniquement sur train dans fit_predict ; features `t_*` sans target, intra-patient. W_PROJ = 0 choisi sur un diagnostic fold 0 → /review avant soumission.
- **Statut** : FAIT (e017)


### H17 — Trajectoire cubique hiérarchique en durée dans la MAP skew
- **Axe** : Trajectoire (piste DGP)
- **Hypothèse** : le true OFF est, par patient, une cubique de la durée de maladie `c0 + c1·d + c2·d² + c3·d³` à coefficients aléatoires indépendants des covariables ; la forme actuelle `a_p + b_p·x + κ·x²` (κ commun) rate la courbure propre au patient.
- **Pourquoi ça devrait marcher** : EDA `lab/eda/dgp.md` : cubique exacte à l'arrondi 0.1 (RMSE < 0.045) pour 67 % des patients ≥ 8 visites ; résidu poolé droite 1.86 / quadratique 0.41 / cubique 0.22 ; 33 % ont un coude à dur ≈ 11 ans ; target jamais décroissant (0/39 014). Erreur e017 : MSE 9.44 = niveau 5.69 + pente 2.37 + courbure 1.22 ; courbure sous-dispersée (pente de calibration 1.69). Covariables → coefficients R² ≤ 0.05.
- **Test (≤ 10 lignes)** : sur e017 ; dans `fit_predict`, cubique en durée ajustée par patient sur le target train → moyenne + covariance 4×4 du prior (+ terme d'erreur de modèle pour les coudes). MAP Newton 4D sous le bruit skew-normal e012 (full + LOO), qui remplace les colonnes `hb_*` en entrée des HGB niveau/delta.
- **Critère de succès** : Δ < −std(Δ) vs e017 ; baisse attendue sur dernière visite (3.79), dur > 11, patients toujours traités.
- **Risques / fuites** : prior fitté sur train uniquement ; `age_at_diagnosis` manquant (5 %) → utiliser l'âge (cubique invariante par translation) ; avec 4 visites le prior domine (OK).
- **Coût** : moyen · **Gain attendu** : moyen (−0.02 à −0.05)
- **Statut** : FAIT (e022 / e022b). e022b (cubique + quadratique `q_*`) CV 2.778, Δ −0.033 ± 0.027 vs e019 (5/5) ; e022 cubique seule −0.025 ± 0.024. Trajectoire brute bien meilleure (fold 0 3.72 → 3.55) mais HGB en récupérait déjà la majeure partie. Σ : OLS c2/c3 instables sur fenêtres courtes → MinCovDet sur n ≥ 8. Ablations NOISE : np.cov (e022c), prior pondéré (e022d).
- **Revue leak-reviewer (2026-10-08)** : GO après correction de 2 BLOQUANTS. (1) Base : seul c3 est invariant par translation → fitter coefficients et prior dans une base commune, âge centré par patient /5 (durée moyenne en covariable du prior, cross-fit si la moyenne dépend des covariables). (2) Covariance : cubique OLS à 4 visites = 0 ddl → Σ estimée sur patients ≥ 6 visites, bruit d'estimation soustrait, projection définie positive, terme d'erreur de modèle. Contrôles : RMSE hb_post_loo valid vs e017, valeurs propres de Σ, non-convergences.

### H18 — Recalibrage de la courbure patient des prédictions finales
- **Axe** : Ensemble (post-traitement)
- **Hypothèse** : la composante quadratique patient des prédictions e017 est trop rétrécie et sa composante cubique est surtout du bruit.
- **Pourquoi ça devrait marcher** : EDA sur l'OOF e017 : `pred + α·Q + β·C3` (Q = composante x² centrée de la quadratique patient des préd., C3 = composante cubique orthogonalisée), α,β fittés sur les 4 autres folds → α ≈ 0.41, β ≈ −0.49 ; gain −0.024/−0.028/−0.018/−0.016/−0.027 (5/5).
- **Test (≤ 10 lignes)** : e017 + cross-fit interne GroupKFold sur train pour obtenir des préd. OOF, α,β par moindres carrés de (target − pred) sur [Q, C3], appliqués au valid. Forme figée α = 0.4, β = −0.5 si stable.
- **Critère de succès** : Δ ≈ −0.02, 5/5 folds ; ne pas dégrader la dernière visite (borner α aux bords si besoin).
- **Risques / fuites** : α,β de l'EDA viennent de l'OOF complet → refit in-fold obligatoire. Coût ×5 du cross-fit interne. Probablement redondant si H17 marche.
- **Coût** : faible à moyen · **Gain attendu** : faible (≈ −0.02, seuil négligeable)
- **Statut** : TODO

### H19 — Facteur rater par MLE skew-normal (off + on)
- **Axe** : Rater
- **Hypothèse** : avec un bruit asymétrique, la médiane de off/target est un estimateur sous-efficace du facteur rater ; une MLE skew-normale cohérente avec e012, utilisant aussi les visites ON via `on/(target·(1 − A_p·g))`, réduit l'erreur de niveau patient.
- **Pourquoi ça devrait marcher** : EDA (demi-train → demi-train) : erreur relative du niveau patient 0.0834 → 0.0800 (−8 % MSE, z = 6.5). Niveau = 5.69 des 9.44 de MSE e017 ; biais rater résiduel e017 sd 0.17, corr 0.39 avec s. Facteurs ≈ log-normaux sd 0.10, pas de valeurs rondes.
- **Test (≤ 10 lignes)** : e017 avec `s_rater` (médiane, shrink 20) remplacé par MLE SN à forme commune `z = off/(s_r·target) ~ SN(ξ, ω, α)`, optim 1D par rater + shrink vers 1 ; variante B jointe off + on_t. Aval inchangé.
- **Critère de succès** : Δ < −std(Δ) vs e017 ; baisse off_measured / off_missing ; sd du biais par rater < 0.12.
- **Risques / fuites** : fit sur target train uniquement, mêmes s_r appliqués au valid ; gain peut-être réduit (4/5 du train vs 1/2 en EDA).
- **Coût** : faible · **Gain attendu** : faible à moyen (−0.01 à −0.04)
- **Statut** : TODO

### H20 — Forme patient des surprises autour de la trajectoire MAP
- **Axe** : Trajectoire (piste FE longitudinal)
- **Hypothèse** : la distribution par patient des écarts `surp = y_v − hb_post_loo` renseigne sur l'erreur de niveau de la MAP skew. Le bruit de y_v a un bord droit abrupt (y_v/target q99 = 1.49), donc max/skew des surprises bornent le niveau par le bas. HGB ne calcule pas ces agrégats de différences seul (e015 ne travaille que sur les brutes).
- **Pourquoi ça devrait marcher** : EDA `lab/eda/longitudinal.md` : résidu e017 à 60 % niveau patient (autocorr. lag-1 0.905). HGB sur résidu OOF −0.110 ± 0.028 (contrôle +0.015) ; max(surp) seul −0.078 ; résidu moyen +0.56 au quintile haut du max. Dans le pipeline e017 (`scratch/long_fold0.py`), H20 + H21 : fold 0 2.986 → 2.747, fold 1 3.003 → 2.725 ; off_missing −0.26/−0.29, first_visit −0.21.
- **Test (≤ 10 lignes)** : e017 + après `build_e012`, par patient sur `surp` : max, min, std, skew, médiane ; max de `surp/hb_se_loo` ; max/std/médiane de `y_v − hb_post_full`. Dans les 2 HGB (niveau/delta), λ = 4 inchangé.
- **Critère de succès** : Δ < −std(Δ) vs e017, attendu −0.10 à −0.25 ; off_missing −0.15 au moins ; first_visit et last baissent.
- **Risques / fuites** : `hb_post_*` côté train dépend de templates fittés sur le target train → vérifier que la distribution de `surp` est la même train/valid (sinon optimisme in-fold). /review obligatoire avant run.
- **Coût** : faible · **Gain attendu** : fort
- **Statut** : TODO

### H21 — Surprises visite et voisines (dont première visite)
- **Axe** : Trajectoire (piste FE longitudinal)
- **Hypothèse** : HGB exploite mal l'écart visite ↔ trajectoire, surtout à la 1re visite où e017 lisse trop.
- **Pourquoi ça devrait marcher** : 1re visite : corr(résidu, surp) 0.22, corr(résidu, infl = hb_post_full − hb_post_loo) 0.21 ; HGB résidu −0.038 ± 0.019 ; dans e017 −0.12 / −0.10 (folds 0/1), first_visit −0.19 / −0.15.
- **Test (≤ 10 lignes)** : e017 + `surp`, `infl`, surp t−1/t+1, surp de la 1re visite du patient. Run commun avec H20 + ablation.
- **Critère de succès** : Δ < −std(Δ) vs e017 ; first_visit ≤ 3.00.
- **Risques / fuites** : idem H20 ; `infl` = 0 quand y_v NaN.
- **Coût** : faible · **Gain attendu** : moyen
- **Statut** : TODO
- **Revue leak-reviewer (2026-10-08)** : GO sous conditions pour H20+H21. Pas de fuite (prior (a,b) cross-fit côté train, `feat_hier.py:57-63` ; templates globaux ; distribution de surp train/valid identique sur fold 0 : std 11.89/11.80). Conditions : Δ rapporté sur folds 2/3/4 (seuls neufs, sélection faite sur 0/1) ; ablation surp / shape / les deux ; `check_order` en tête ; NaN natifs ; impression par fold des quantiles train/valid de `L_surp`, `L_p_surp_max` ; ne pas combiner avec H17.

### H22 — Noyau en âge des scores débiaisés, en écart à la trajectoire
- **Axe** : Trajectoire
- **Hypothèse** : une moyenne locale LOO de y_v (noyau gaussien 1 an, poids w_v) capte une déviation locale que la trajectoire globale ne voit pas.
- **Pourquoi ça devrait marcher** : HGB résidu −0.018 ± 0.019 (≈ −0.03 net du contrôle) — faible.
- **Test (≤ 10 lignes)** : seulement si H20+H21 gagnent : `yv_k1 − hb_post_loo`, `yv_k2 − hb_post_loo`, masse effective du noyau.
- **Critère de succès** : Δ < −std(Δ) vs e017+H20+H21, sinon jeter (< 0.02).
- **Risques / fuites** : doublon probable avec H17 (cubique).
- **Coût** : faible · **Gain attendu** : faible
- **Statut** : TODO

### H23 — MLP bagué niveau+delta, moyenné avec e017
- **Axe** : Ensemble
- **Hypothèse** : un MLP (256-128) sur les features e017 standardisées a des erreurs nettement moins corrélées que HGB ; ses seeds sont réellement diverses (corr 0.77–0.81 entre seeds, vs 1.000 pour HGB, cf. e014). Bag 5 seeds + MLP delta + reconstruction λ = 4 ≈ e017 seul, et leur moyenne gagne.
- **Pourquoi ça devrait marcher** : EDA `lab/eda/ensemble.md` (scratch/ens/) : fold 0 MLP 3.048 seul, ρ résidus 0.879, 50/50 −0.063 ; fold 3 3.307, ρ 0.904, 50/50 −0.040. Frontière théorique : à 3.05, ρ ≤ ~0.93 suffit pour −0.05.
- **Test (≤ 10 lignes)** : eNNN = build e012 + add_temporal → matrice dense (one-hot, flags NaN, imputation médiane, standardisation, clip ±6, tout fitté sur train) ; 5 `MLPRegressor((256,128), alpha=1e-2, batch_size=256, max_iter=80, random_state=SEED..SEED+4)` sur cible standardisée + 5 MLP delta ; reconstruction λ = 4. Sortie 0.6·e017 + 0.4·MLP (ou MLP seul en eNNN puis `blend.py`).
- **Critère de succès** : Δ ≤ −0.04, |Δ| > std(Δ), 5/5 folds.
- **Risques / fuites** : préprocessing fitté sur train uniquement ; seeds fixées pour la soumission ; warnings de non-convergence.
- **Coût** : moyen (~2.5 min/fold) · **Gain attendu** : moyen
- **Statut** : FAIT (2026-10-13) sur les features e022b. e024 MLP seul CV 2.841 (+0.061 vs e022b), ρ résidus 0.885 ; blend NNLS hors-fold b002 CV 2.727, **−0.053 ± 0.036, 5/5**, poids e022b 0.55–0.64 par fold ; 0.6/0.4 fixe −0.056. À faire avant soumission : reproductibilité fold 0 ×2, runs `--test`. H24 (réglage MLP) devient la suite logique.
- **Revue leak-reviewer (2026-10-08)** : GO sous conditions. MLP en expérience autonome (`--test`) puis `blend.py e017 eMLP` (NNLS hors-fold) ; Δ confirmé sur folds 1/2/4 ; figer 0.6/0.4 seulement si NNLS donne 0.55–0.65 sur les 5 folds ; reproductibilité `--folds 0` ×2 ≤ 1e-3 ; gain −0.02 à −0.04 → noté mais hors modèle soumis.

### H24 — Réglage grossier du MLP (sous ~3.0 seul)
- **Axe** : Ensemble
- **Hypothèse** : le MLP de H23 est sous-réglé ; meilleure qualité seule à ρ ≤ 0.90 amplifie le gain de blend.
- **Pourquoi ça devrait marcher** : fold 0 : epochs 20/40/80 → 3.75/3.60/3.53 (1 seed) ; 5 seeds 3.16 ; (512,256,128) α 0.1 : 3.365 vs 3.526. σ2 = 3.0, ρ = 0.88 → ≈ −0.10 théorique.
- **Test (≤ 10 lignes)** : grille courte sur folds 0/3 (`scratch/ens/f0.pkl`, `f3.pkl`) : epochs 80/150, α 1e-2/1e-1, (512,256,128), 10 seeds, retrait des t_* redondants ; choisir sur le blend 50/50, injecter dans H23.
- **Critère de succès** : MLP seul ≤ 3.0 fold 0 avec ρ ≤ 0.90, 50/50 ≤ −0.08 sur 2 folds.
- **Risques / fuites** : suroptimisation sur 2 folds → réglages grossiers seulement ; coût ×2.
- **Coût** : moyen · **Gain attendu** : moyen
- **Statut** : TODO (après H23)

### H25 — Modèle faible décorrélé / stacking / arbres alternatifs
- **Axe** : Ensemble
- **Hypothèse** : un modèle à ~4.5 décorrélé, un stacking ou une autre famille d'arbres améliorerait e017.
- **Pourquoi ça devrait marcher** : réfuté par l'EDA. À 4.5, il faut ρ ≤ 0.527 pour −0.05 ; meilleur observé kNN 5.44 / ρ 0.565 → 0. Blends des runs existants ≤ −0.006 (NNLS 11 runs −0.004). Stacking résidu +0.031. RF/ExtraTrees ρ 0.93–0.94, ≤ −0.001. Variantes HGB ρ ≥ 0.978, ≤ −0.025 (≈ réglage, couvert par e018). Ridge 3.17–3.25, ≤ −0.008. LightGBM indisponible (libomp), XGBoost/CatBoost/torch non installés.
- **Statut** : ABANDONNÉ (EDA `lab/eda/ensemble.md`) — exception : blend e018 + H23 si e018 retenu.

### H26 — Missingness OFF comme capteur de sévérité (MNAR)
- **Axe** : Manquants
- **Hypothèse** : un OFF non mesuré signale un patient trop sévère pour supporter le sevrage ; un patient qui rate de plus en plus ses OFF voit son true OFF augmenter.
- **Pourquoi ça devrait marcher** : CONTEXT (« OFF rarement mesuré ») + littérature (examens OFF pénibles). EDA : corr(target, fréquence patient de OFF manquant) = 0.54, cumulée 0.58.
- **Test** : e019 + `M_p_miss`, `M_cum_miss`, `M_cum_miss_prev`, `M_run_miss`, `M_p_miss_treated` (sans target).
- **Critère de succès** : Δ < −std(Δ) vs e019 ; off_missing en baisse.
- **Risques / fuites** : aucune (intra-patient, sans target).
- **Coût** : 15 min, run ~2 min.
- **Statut** : FAIT (e020_off_miss_freq) — CV 2.810, Δ −0.002 ± 0.008 vs e019 (2/5 folds) → NOISE. Le signal brut est confondu avec le statut traité : parmi les visites avec ON mesuré, target 44.13 (OFF mesuré) vs 44.40 (OFF manquant). Pas de MNAR exploitable au-delà de ce que le modèle sait déjà (`lab/scratch/mnar_off.py`).

### H27 — Génération massive de features + sélection FDR
- **Axe** : Méthode
- **Hypothèse** : les ~190 colonnes `t_*` diluent les variables structurelles de e019 (surprises, asymétrie) ; un filtre FDR (BH 0.05) ou une élimination récursive par SHAP réduirait le bruit, après avoir généré encore plus de features (ratios, pentes).
- **Pourquoi ça devrait marcher** : approches « brute-force FE + sélection » des concours AMP Parkinson / Amex (proposition externe).
- **Test** : sonde `e021_feat_select` sur le fold 0, depuis e019 : `E021_MODE=no_t` (retire tout le bloc `t_*`) et `E021_MODE=fdr` (f_regression + BH 0.05, fitté sur le train du fold).
- **Critère de succès** : Δ < −0.02 sur le fold 0 avant tout run complet.
- **Risques / fuites** : aucune si la sélection est fittée dans `fit_predict`.
- **Coût** : 20 min.
- **Statut** : ABANDONNÉ. Fold 0 (e019 = 2.7473) : sans `t_*` 2.8100 (+0.063) → le bloc temporel **apporte** du signal, l'hypothèse de dilution est contredite. FDR : garde 242/276 colonnes (avec n = 35 k, presque tout passe un test univarié ; il retire même `s_rater`), 2.7440 (−0.003) → négligeable. HGB ignore déjà bien les colonnes inutiles ; générer davantage de ratios/pentes pour les refiltrer ensuite n'a pas de raison d'aider. SHAP-RFE non tenté : coûteux, et la borne haute est faible vu ces deux sondes.

### H28 — Pipeline skrub (tabular_pipeline / DataOps) pour les catégorielles
- **Axe** : Méthode
- **Hypothèse** : un pipeline skrub (jointure des métadonnées, encodeurs qui lissent les catégories rares) traiterait mieux `rater_id` / `gene` / `cohort` que l'encodage catégoriel natif de HGB.
- **Pourquoi ça devrait marcher** : guide du bobathon (Day 2, étape 3) (proposition externe).
- **Test** : sonde `e021_feat_select` sur le fold 0 : `no_cat` (retire les 3 catégorielles) et `te_cat` (TargetEncoder sklearn cross-fitté, l'équivalent de ce que ferait skrub).
- **Critère de succès** : Δ < −0.02 sur le fold 0.
- **Risques / fuites** : un target encoding cross-fitté *par ligne* fuit à l'intérieur d'un même patient (les visites d'un patient se retrouvent dans les deux parties du cross-fit).
- **Coût** : 15 min.
- **Statut** : ABANDONNÉ. Fold 0 : sans catégorielles 2.7472 (Δ 0.000) → le modèle ne s'en sert plus (le rater est déjà absorbé par `s_rater` et la normalisation H04, gene/cohort par la trajectoire). TargetEncoder 2.8188 (+0.072, pire : fuite intra-patient dans le cross-fit par ligne, cf. risques). Il n'y a qu'une table, donc pas de jointure à faire, et les cardinalités sont faibles (2 cohortes, quelques gènes, 50 raters). skrub n'apporte rien au modèle ; il peut servir à présenter le pipeline dans le rapport skore, sans attendre de gain.

### H29 — Harmoniser les covariables fixes incohérentes intra-patient
- **Axe** : Structure
- **Hypothèse** : comme dans les dossiers réels (module cohort de PIE), des covariables censées être fixes (gène, âge au diagnostic, cohorte, sexe) varient d'une visite à l'autre chez un même patient → vote majoritaire intra-patient.
- **Test** : EDA, nombre de patients avec plus d'une valeur ou un NaN partiel, sur train et test.
- **Statut** : RÉFUTÉ (EDA). Train et test : 0 patient avec > 1 valeur et 0 NaN partiel pour `cohort`, `sexM`, `gene`, `age_at_diagnosis` ; les manquants sont tout-ou-rien par patient (gene : 1 868 patients train / 452 test ; age_at_diagnosis : 286 / 67). `age` est strictement croissant intra-patient. Rien à harmoniser. Le bruit est sur on/off (rater, délai), déjà modélisé.


---

## Brainstorm 2026-10-12 (e022b CV 2.778, top équipes ~2.2)

Constats transverses (`lab/eda/{dgp2,residual,measurement,missing,methods}.md`) :
- Erreur e022b = 63 % niveau patient, 24 % pente, 13 % reste ; forme de trajectoire parfaite → 2.21 au mieux. Trajectoire MAP déjà au plancher bruit/√n (sd post. 3.45 vs erreur 3.55).
- Modèle de mesure validé : rater multiplicatif pur commun on/off, sans interaction ; A_p constant ; tsi_off inutile ; pas de contamination groupée. Cohorte B = suivi court, même population.
- off_missing coûte par manque d'information (hors de l'intervalle des OFF mesurés : 3.59 après le dernier, 4.21 avant le premier), pas par biais.
- Générateur : target = rule(P), P cubique latente ; si P(t_i) < T_{i−1} alors T_i = 1.02·T_{i−1} (723/723 visites), plancher 0, arrondi 0.1. Les mesures suivent P, pas target. leaspy rejeté.
- Données limitantes : −0.08 à −0.10 par doublement du nombre de patients (le modèle final sur 100 % du train devrait faire ~−0.03 vs CV).
- Le bruit n'est pas i.i.d. (lié à la pente locale, échelle persistante) et HGB l'exploite : re-bruiter casse le modèle.
- Externe : dépôt officiel probabl-ai/bobathon-esilv sans générateur ; Kaggle privé ; AMP PD 2023 et rLMM Castro Mejia (MDS 2025) déjà couverts par nos modules.

### H30 — Réseau séquentiel par patient (BiGRU + Transformer), blend avec e022b
- **Axe** : Méthode
- **Hypothèse** : un réseau qui lit toutes les visites d'un patient d'un coup (séquence ≤ 12, têtes niveau + delta) corrige l'erreur de niveau/pente patient que HGB, visite par visite, laisse passer ; ses erreurs sont assez décorrélées de e022b (ρ 0.84–0.88).
- **Pourquoi ça devrait marcher** : fold 0 GRU 2.584, Transformer 2.571, moyenne 2.541 (e022b 2.745) ; fold 3 2.806 / 2.782 / 2.761 (e022b 2.905). Blend 0.7 réseaux + 0.3 e022b : −0.223 / −0.173. Contrôle MLP par visite, mêmes features : 2.704 → la séquence apporte −0.12.
- **Test (≤ 10 lignes)** : eNNN autonome : build e022b (`build_e011` + `cubic_hook(keep_quad=True)` + `add_temporal` + `add_long`), matrice dense de `scratch/meth/m3_seq.py` (fittée sur train) ; 5 seeds GRU + 5 seeds Transformer (H 128, 2 couches, AdamW 2e-3, OneCycle 25 epochs, batch 64 patients, epoch choisie sur 10 % des patients du train, perte niveau + 0.5·delta), reconstruction λ = 4. Puis `blend.py e022b eNNN` ; figer 0.7/0.3 si NNLS ∈ [0.6, 0.8] sur 5 folds.
- **Critère de succès** : blend ≤ −0.12 sur 5 folds, |Δ| > std ; réseaux seuls ≤ 2.70 ; baisse dernière visite, 1re visite, 4–5 visites, décile haut du target.
- **Risques / fuites** : préprocessing et epoch sur train seulement ; torch absent de `.venv` (installé dans un venv temporaire) → accord utilisateur pour l'ajouter ; reproductibilité CPU (seeds, algos déterministes) ; variance par seed forte (2.72–2.85) ; faisabilité de la soumission (temps de fit) à vérifier.
- **Coût** : moyen (~10 min/fold) · **Gain attendu** : fort
- **Statut** : CONFIRMÉE (2026-10-12). `e023_seq_nn` (réseaux seuls) CV 2.610, −0.169 ± 0.042 vs e022b (5/5) ; `b002_seq_e022b` (NNLS imbriquée, poids e023 0.71–0.73 sur les 5 folds) CV 2.579, −0.200 ± 0.033 vs e022b, folds 1/2/4 −0.200 vs 0/3 −0.198. Conditions bloquantes de la revue respectées (deterministic, 8 threads, seeds cv.SEED + s, check_order avant pack, poids par NNLS imbriquée ; fold 0 identique sur 3 exécutions). Baisse : dernière visite 3.46→3.25, 1re 2.88→2.67, off_missing 3.25→3.04, n 4–5 2.90→2.81. Reste : coût de la soumission (torch, ~40 min CPU).

- **Revue leak-reviewer (2026-10-12) : GO sous conditions.** (1) BLOQUANT : le 0.7 et les −0.223/−0.173 viennent d'une grille sur la valid (`m3_seq.py:97-100`, `m4_blend.py`) → optimistes ; poids uniquement par NNLS imbriquée de `lab/blend.py`, ne pas figer depuis `w_all`. (2) Archi/epochs/λ développés sur folds 0/3 → juger d'abord sur folds 1/2/4. (3) Pas de fuite (dense() et choix d'epoch sur train, par patient). (4) assert âge croissant avant `pack()`. (5) BLOQUANT soumission : torch dans `.venv` avec accord utilisateur, `use_deterministic_algorithms`, threads fixes, seeds `cv.SEED + s`, reproductibilité ×2.

### H31 — Transformer « amortized » sur colonnes harmonisées, 3e composante
- **Axe** : Ensemble
- **Hypothèse** : un Transformer sans MAP ni bloc temporel (y_v, w_v, on_n, off_n, s_rater, g_on, A_p, covariables) apprend sa propre trajectoire, avec des erreurs différentes (ρ 0.82).
- **Pourquoi ça devrait marcher** : fold 0 2.774 seul ; NNLS (H30, amortized, e022b) = (0.73, 0.21, 0.06) −0.266 vs −0.237 (poids fittés sur valid : optimiste).
- **Test (≤ 10 lignes)** : `m3_seq.py` feats=raw, 5 seeds, 40 epochs, expérience séparée ; `blend.py e022b eH30 eH31`.
- **Critère de succès** : apport ≥ −0.02 au blend H30, poids NNLS stable 0.15–0.3.
- **Risques / fuites** : aucune nouvelle. **Coût** : faible après H30 · **Gain attendu** : faible-moyen
- **Statut** : TODO (après H30)

### H32 — LightGBM à la place de HGB dans e022b
- **Axe** : Méthode
- **Hypothèse** : LightGBM régularisé (63 feuilles, lr 0.03 × 1500, colsample 0.3, subsample 0.8) bat HGB 500 × 0.05, pour 30× moins cher que HGB v2.
- **Pourquoi ça devrait marcher** : folds 0/3 : l2 −0.047 / −0.061, Huber(4) −0.063 / −0.044 ; HGB v2 porté sur e022b −0.057 (fold 0) en 2 500 s contre 86 s.
- **Test (≤ 10 lignes)** : e022b avec `hgb_predict` et `_hgb_delta` → `LGBMRegressor(random_state=cv.SEED)`, l2 puis Huber ; libomp via `brew install libomp` (ou `DYLD_LIBRARY_PATH` vers torch/lib, à documenter).
- **Critère de succès** : Δ ≤ −0.04 sur 5 folds, |Δ| > std ; si H30 adopté, garder seulement si apport ≥ 0.02 au blend (sondé −0.01 à −0.02).
- **Risques / fuites** : aucune ; dépendance libomp/LightGBM. **Coût** : faible · **Gain attendu** : moyen seul, faible avec H30
- **Statut** : TODO

- **Revue leak-reviewer (2026-10-12) : GO sous conditions.** l2 figé avant le run 5 folds (Huber = 2e expérience déclarée) ; `random_state=cv.SEED`, `deterministic=True`, `n_jobs` fixe ; préférer `brew install libomp` au DYLD vers torch.

### H33 — Règle ×1.02 du générateur appliquée à la cubique latente
- **Axe** : Structure
- **Hypothèse** : target = rule(P) : T_0 = P(t_0) ; T_i = P(t_i) si ≥ T_{i−1}, sinon 1.02·T_{i−1} ; plancher 0, arrondi 0.1. HGB n'apprend pas cette récursion.
- **Pourquoi ça devrait marcher** : 723/723 visites post-pic = round(1.02·T_{i−1}, 1) ; patients à queue post-pic = 31 % du SSE, biais −0.50 ; `rule(hb_post_full)` trajectoire 3.552 → 3.524 (fold 0), 3.544 → 3.491 (fold 1).
- **Test (≤ 10 lignes)** : e022b + par patient, dans l'ordre des âges : `R = rule(P̂)`, `R − P̂`, flag de déclenchement, nb de déclenchements, et `E[rule(P)]` sur ~200 tirages de la postérieure Laplace 4D de `cubic_map`.
- **Critère de succès** : Δ < −std(Δ) vs e022b ; baisse dernière visite, durée > 11 ans, patients toujours traités.
- **Risques / fuites** : aucune (constante du générateur, sans target). P̂ trop monotone (déclenche sur 1 % des visites contre 5–8 % en vrai) → combiner avec H34. **Coût** : faible · **Gain attendu** : faible-moyen
- **Statut** : TODO

- **Revue leak-reviewer (2026-10-12) : GO sous conditions (gain probablement faible).** 1.02 = identification exacte, pas un fit. Aucun coefficient choisi sur la valid : features HGB seulement. Exporter la covariance 4×4 par patient de `cubic_map` pour E[rule(P)] (`default_rng(cv.SEED)`, vectorisé sur les tirages). Tri explicite par âge, plancher 0, arrondi 0.1.

### H34 — Template de bruit et prior cubique fittés sur P latent
- **Axe** : Structure
- **Hypothèse** : les mesures sont générées depuis P ; fitter template SN et prior 4D sur le target monotonisé biaise le prior vers des courbes trop croissantes et contamine la queue gauche du bruit.
- **Pourquoi ça devrait marcher** : aux visites post-pic off/(s·P) 0.96–0.99 contre off/(s·target) 0.88 ; ON 0.557 avec P contre 0.517.
- **Test (≤ 10 lignes)** : dans `fit_predict`, sur train : P latent via `scratch/dgp2/lat.py` (fit rule(P(θ)) si RMSE < 0.06, sinon P = target) → `fit_skew_template`, `fit_cubic_prior` ; HGB restent sur target. Avec H33.
- **Critère de succès** : déclenchement de la règle sur P̂ valid ≥ 3 % ; Δ < −std(Δ) vs e022b.
- **Risques / fuites** : P vient du target train → strictement in-fold ; ~1 min de fit. **Coût** : moyen · **Gain attendu** : faible-moyen
- **Statut** : TODO

### H35 — Pipeline cible P latent + règle, blend avec e022b
- **Axe** : Ensemble
- **Hypothèse/Test** : e022b entraîné sur P puis règle appliquée (`scratch/dgp2/b4.py`), expérience autonome puis `blend.py`.
- **Pourquoi** : seul +0.023 / −0.013, 50/50 −0.012 / −0.024 (folds 0/1).
- **Critère de succès** : NNLS stable, Δ ≤ −0.015, |Δ| > std. **Coût** : moyen · **Gain attendu** : faible
- **Statut** : TODO

### H36 — Générateur de bruit conditionnel et augmentation de données
- **Axe** : Méthode
- **Hypothèse** : le bruit réel est lié à sa visite (pente locale, position, échelle patient) et HGB exploite ce lien ; un générateur qui le reproduit permettrait d'augmenter un modèle limité par les données.
- **Pourquoi ça devrait marcher** : HGB sur copies re-bruitées seules (fold 0) : identique 2.768, gigue 5 % 2.808, ε permutés intra-patient 3.077, re-tirés 3.41 ; courbe d'apprentissage 2.996 → 2.827 → 2.745 (25/50/100 %).
- **Test (≤ 10 lignes)** : `scratch/resid/p4.py` avec générateurs de plus en plus conditionnés (kNN sur pente locale, position, échelle |ε| patient, source, traité) ; arrêt quand « copies seules ≤ 2.85 » ; puis e022b + K = 2–3 copies par patient, features des copies calculées côté valid.
- **Critère de succès** : copies seules ≤ 2.85 (fold 0), puis Δ ≤ −0.05 sur 5 folds.
- **Risques / fuites** : copies hors des cross-fits internes (prior, Σ) ; générateur infidèle dégrade (+0.09 à +0.18 mesurés). **Coût** : fort · **Gain attendu** : moyen, incertain
- **Statut** : TODO

### H37 — Localisation du bruit dépendante de la pente locale
- **Axe** : Structure
- **Hypothèse** : log(off/(s·target)) croît avec la pente locale (médiane −0.118 → +0.077 d'octile bas à haut) ; ξ fixe biaise le niveau MAP (+1.47 octile bas, HGB ramène à +0.23).
- **Test (≤ 10 lignes)** : `cubic_map` avec loc = sv·(ξ + β·(pente_pass1 − b̄)), β OLS sur train, pente de passe 1 LOO côté train.
- **Critère de succès** : trajectoire fold 0 3.562 → ≤ 3.50, final −0.01 à −0.03 ; sinon abandonner.
- **Risques / fuites** : confusion avec niveau/durée (contrôler par bins de target). **Coût** : faible · **Gain attendu** : faible
- **Statut** : TODO

### H38 — Template OFF scindé traités / non traités
- **Axe** : Mesure (Rater + Pharmaco)
- **Hypothèse** : ~25–30 % des OFF de patients traités sont ×0.74 (effet résiduel) ; une seule source `off` crée une marche de −9 % à l'initiation.
- **Pourquoi** : log(off/(s·t)) traités − non traités −0.12 (intra-patient −0.096 ± 0.010) ; trajectoire fold 0 3.569 → 3.554.
- **Test (≤ 10 lignes)** : e022b avec source `off_tr` (on.notna()) dans fit_skew_template, cubic_map et skew_map ; variante B vraisemblance de mélange (π, c).
- **Critère de succès** : Δ < −std(Δ) vs e022b. **Coût** : faible · **Gain attendu** : faible (−0.005 à −0.015)
- **Statut** : TODO

### H39 — A_p : récupérer les paires on/off sans tsi_on
- **Axe** : Manquants
- **Hypothèse** : 2 962 paires (24 %) sont jetées faute de tsi_on ; avec g = gbar et un poids réduit, elles améliorent A_p (910 patients sans paire utilisable en ont une).
- **Pourquoi** : 1 − on/off de ces paires corr 0.39 avec A vrai ; résidu e022b corr 0.15 ; oracle A_p 2.778 → 2.728.
- **Test (≤ 10 lignes)** : `add_drug_amplitude` : a_v = (1 − on/off)/g_use, poids g_use²/σ²_cat (plateau / tsi < 1 h / NaN, fitté train), shrink k ≈ 0.40 fitté train, + A_p_se.
- **Critère de succès** : Δ −0.005 à −0.015, ≥ 4/5 folds ; off_missing en baisse. **Coût** : faible · **Gain attendu** : faible
- **Statut** : TODO

### H40 — Effet aléatoire patient sur les sources ON dans la MAP (fusion de 2 fiches)
- **Axe** : Mesure
- **Hypothèse** : l'erreur de A_p est commune aux visites ON d'un patient (sd log 0.117, 0.15–0.17 si n_av = 0) ; la MAP la traite comme i.i.d. et sur-pondère les visites ON.
- **Pourquoi** : GLS gaussien fold 0 trajectoire 4.171 → 4.152 ; après le dernier OFF, MSE 12.9 dont 11.6 niveau patient, corr(résidu dernier OFF, résidu après) 0.79. La correction plug-in dégrade (3.569 → 3.601) : passer par la vraisemblance.
- **Test (≤ 10 lignes)** : `cubic_map` 5D : u_p multiplicatif sur les sources ON, prior N(0, τ²) par bin de n_av (fitté train) ou τ_p = g·sd_post(A_p)/(1 − A_p·g) ; variante légère : variance ON inflatée de τ_p²/n_on.
- **Critère de succès** : Δ −0.01 à −0.02 ; zones après le dernier / avant le premier OFF. **Coût** : moyen · **Gain attendu** : faible
- **Statut** : TODO

### H41 — Écarts de mesure patient en features HGB
- **Axe** : Mesure
- **Test** : e022b + marche médiane de log y_v off entre visites traitées et non traitées + médiane de log(y_v ON / MAP fittée sur OFF seuls) par source (MAP supplémentaire avec w_v = 0 sur ON).
- **Critère de succès** : Δ < −std(Δ) sur 5 folds. **Coût** : faible · **Gain attendu** : faible (risque NOISE)
- **Statut** : TODO

---

<!-- Brainstorm 2026-10-13 (réf. e023 2.610 / b002 2.579, top LB ≈ 2.2). EDA : level.md, dgp3.md, rater.md,
methods2.md, ensemble2.md, missing2.md. Constat commun : 63–66 % de la MSE = erreur de niveau patient,
calibrée mais de signe imprévisible ; ampleur pilotée par l'incertitude sur A_p (patients ON-lourds). -->

### H42 — A_p a posteriori complet injecté dans e023 (fusion H39 + H41)
- **Review (2026-10-13)** : circularité bloquante pour la 2e passe — prior cubique utilise A_m (feat_hier.py:32) et sv vient de mu1 avec y_v ON (feat_cubic.py:108). Lancer d'abord H42a = config sondée seule.
- **Axe** : Mesure
- **Hypothèse** : l'erreur de niveau des patients ON-lourds vient de A_p, persistante sur tout le patient ; le réseau ne reçoit qu'un A_p construit sur les paires avec tsi.
- **Pourquoi ça devrait marcher** : 20 % des patients traités n'ont **aucun** tsi (MCAR, A_p = A0 partout) : e023 3.13 vs 2.50, MSE 7.68 vs 4.92 ; ramenés au niveau des autres → CV ≈ 2.48. Visites ON sans paire : 28 % des patients, 37 % du SSE niveau. H39 seul (g = gbar, poids 1, shrink 0.5) sur e022b fold 0 : 2.7448 → 2.7043 ; patients sans tsi 3.011 → 2.902, avant leur 1er OFF 5.00 → 3.71. Le recalage linéaire de l'OOF ne voit rien (corr 0.012) : le gain passe par les features.
- **Test (≤ 10 lignes)** : `add_drug_amplitude` : paires sans tsi avec g = gbar (`scratch/miss2/p2_h39.py`) ; 2e passe a_traj = médiane LOO de (1 − on/(s·T̂·g)) sur les visites ON-only, T̂ = trajectoire MAP **sans ON** (w_v = 0), cross-fittée in-fold ; combinaison variance inverse → A_p_post, sd_A_p, on_eq recalculé. D'abord e022b (5 folds), puis matrice e023 et blend.
- **Critère de succès** : Δ < −std(Δ) vs e022b puis vs e023 ; baisse de off_missing, patients sans tsi, n_off ≤ 2, décile haut.
- **Risques / fuites** : T̂ cross-fitté in-fold ; circularité si T̂ utilise déjà A_p. Shrink et poids fittés sur train.
- **Coût** : moyen (e022b ~rapide, e023 ~40 min) · **Gain attendu** : moyen (−0.02 à −0.04)
- **Statut** : H42a FAIT (e024, 2026-10-13) — CV 2.7664 vs 2.7781, Δ −0.011 ± 0.048 (3/5), NOISE ; fold 0 = sonde (−0.041), folds 1–4 −0.004 ± 0.052 (fold 2 +0.063). Gain ciblé réel sur patients sans tsi (−0.064, 4/5 folds) mais instable. H42b non lancé (autorisation requise) ; injection e023 à décider.

### H43 — Ensemble séquentiel élargi avec membres régularisés décorrélés
- **Axe** : Méthode
- **Hypothèse** : des réseaux régularisés autrement (dropout de features, dropout par bloc) apportent une diversité que des seeds neuves n'apportent pas.
- **Pourquoi ça devrait marcher** : fold 0, ρ résidus vs e023 0.933 (GRU fdrop 0.2) / 0.954 (TF bdrop 0.3) vs 0.970 (seeds). Blend b002 2.5211 → 2.5052 (+10 seeds), 2.4975 (+10 régularisés), 2.4927 (+20). Bruit d'un bag de 5 ≈ ±0.025 non apparié.
- **Test (≤ 10 lignes)** : eNNN = build/dense de e023 ; `feat_seq.train_predict` + options `fdrop` (Bernoulli par patient × colonne, ×1/(1−p)) et `bdrop` (bloc t_* / h-hb-hc-q / L_*, brut+pharmaco gardé) cf. `scratch/meth2/p1.py`. Membres : 5 GRU + 5 TF base, 5 GRU fdrop 0.2 ep 50, 5 TF bdrop 0.3 ; seeds `cv.SEED + s`. Puis `blend.py e022b eNNN`. Contrôle : e023 + 10 seeds neuves.
- **Critère de succès** : blend ≤ b002 − 0.015 sur 5 folds, |Δ| > std, jugé d'abord sur folds 1/2/4.
- **Risques / fuites** : aucune nouvelle. Coût torch ×2 (~80 min/fit à la soumission).
- **Coût** : moyen · **Gain attendu** : moyen (−0.02 à −0.03)
- **Statut** : TODO

### H44 — TF offset sur la trajectoire MAP + TF amortized (H31), composantes 3–4 du blend
- **Axe** : Ensemble
- **Hypothèse** : la diversité vient de la cible (offset `target − hb_post_full`) et des entrées (colonnes harmonisées seules) ; les deux sont quasi orthogonales (ρ 0.814 entre elles).
- **Pourquoi ça devrait marcher** : fold 0, poids NNLS sur une moitié des patients, éval sur l'autre (80 tirages) : offset −0.020 ± 0.008 (seul 2.561, ρ 0.933), amortized −0.024 ± 0.010 (ρ 0.807), les deux −0.038 ± 0.012 (poids e022b → 0.05). Même cible que e023 : poids 0.
- **Test (≤ 10 lignes)** : eNNN = e023 avec cible offset (base `hb_post_full` du build e022b), 5 TF (+5 GRU) ; eMMM = H31. `blend.py e023_seq_nn e022b_cubic_traj_quad eNNN eMMM --name b003_…` ; figer les poids si stables (écart < 0.1).
- **Critère de succès** : b003 ≤ b002 − 0.03 sur 5 folds, chaque composante ≥ 0.01. Baisse : 4–5 visites, off_missing > 60 %, cohorte B.
- **Risques / fuites** : fold 0 = fold de dev (optimiste). Poids par NNLS imbriquée seulement. Modèle soumis à 3 réseaux.
- **Coût** : moyen · **Gain attendu** : moyen (−0.03 à −0.04) ; recouvrement probable avec H43 à mesurer.
- **Statut** : CONFIRMÉE (2026-10-13), composante amortized à la limite. `e025_seq_offset` (5 TF, cible target − hb_post_full, **EP 25** : la sonde `s0_tf_all_off.pkl` est bit-identique à EP 25, pas 60 ; fold 0 2.5613 = sonde) CV 2.603, seul NOISE vs e023 (−0.007 ± 0.019). `e026_seq_amortized` (3 TF raw, 100 ep ; fold 0 2.7738 = sonde) CV 2.853. `b003_seq4` (NNLS imbriquée) CV 2.541, −0.038 ± 0.011 vs b002, 5/5 ; fold 0 −0.040, folds 1–4 −0.038 ± 0.013. Ablations appariées : sans offset +0.022 ± 0.006, sans amortized +0.0096 ± 0.005 (5/5, mais < 0.01). Poids par fold : e023 0.37–0.39, e025 0.37–0.39, e026 0.14–0.16, e022b 0.08–0.11 (figeables). Sous-groupes vs b002 : off_missing 3.040→2.999, sans tsi 2.680→2.648, n 4–5 2.807→2.783, p_off_miss > 0.6 3.531→3.513, cohorte B 2.894→2.891 (inchangée, critère non atteint). À soumettre : b003 (4 composantes) ou b003b (sans amortized, −0.029 vs b002) si le coût d'un 3e réseau pèse.

### H45 — Augmentation par masquage MCAR des délais de prise
- **Axe** : Manquants
- **Hypothèse** : le manque de tsi est tout-ou-rien par patient et MCAR ; des copies de patients avec tsi masqués sont des exemples fidèles (ε conservé, contrairement à H36).
- **Pourquoi ça devrait marcher** : quantiles de on/(s·T) identiques ±0.003 ; e022b fold 0 copies 30 % : 2.7448 → 2.7261 (sans tsi 3.011 → 2.934). Moyenne H39 + aug 100 % : 2.681 (−0.023 vs H39 seul).
- **Test (≤ 10 lignes)** : dans `fit_predict`, ajouter au train des copies de 30 % des patients avec tsi (nouveaux patient_id/Index, tsi à NaN), par-dessus H42. Folds 0/3 puis 5 folds.
- **Critère de succès** : Δ ≤ −0.01, |Δ| > std vs e022b+H42 ; autres patients ≤ +0.005.
- **Risques / fuites** : copie et original doivent rester dans le même groupe des GroupKFold internes (prior cubic_map, cross-fit HGB), ou exclure les copies des fits de prior/template. Redondance probable avec H42.
- **Coût** : moyen · **Gain attendu** : faible-moyen
- **Statut** : TODO

### H46 — Template de bruit propre à la 1re visite (absorbe H38)
- **Axe** : Structure
- **Hypothèse** : la visite 0 a un régime de bruit plus étroit et décalé vers le haut pour toutes les sources.
- **Pourquoi ça devrait marcher** : IQR log(off/(s·T)) 0.325 à v0 vs 0.47–0.50 à v1–v3 (patients cubiques exacts 0.36 vs 0.49), lié au rang, pas au temps. Template off ω 1.49 → 1.19. Trajectoire fold 0 3.5688 → 3.5387, v0 4.04 → 3.69 ; + off_tr 3.5370. Confirmé côté level.md (iqr 0.27 vs 0.45).
- **Test (≤ 10 lignes)** : e022b avec `harmonized_score` patché comme `scratch/dgp3/e11.py` (source `<src>_v0` si visit_rank == 0, `off_tr` si on présent) dans fit_skew_template, cubic_map, skew_map.
- **Critère de succès** : Δ < −std(Δ) vs e022b, first_visit en baisse ; apport au blend ≥ −0.01.
- **Risques / fuites** : aucune nouvelle. HGB/NN voient déjà visit_rank : gain final ≪ −0.030 possible.
- **Coût** : faible · **Gain attendu** : faible-moyen
- **Statut** : TODO

### H47 — Réseaux finaux sans holdout interne de 10 %
- **Axe** : Méthode
- **Hypothèse** : retirer 10 % des patients pour choisir l'epoch coûte ≈ −0.009 à un réseau limité par les données.
- **Pourquoi ça devrait marcher** : +0.061 par division par 2 des patients (GRU fold 0) ; epoch retenue souvent 20–24/25.
- **Test (≤ 10 lignes)** : e023 avec `inner = []`, epoch fixe 25 (variante : moyenne des sorties epochs 12–24), mêmes seeds, apparié à e023.
- **Critère de succès** : Δ ≤ −0.008 sur ≥ 4/5 folds. **Coût** : faible · **Gain attendu** : faible
- **Statut** : TODO

### H48 — Courbe PK dont la vitesse de montée dépend de A_p
- **Axe** : Pharmaco
- **Hypothèse** : g_p(t) = g(t·(1 + β(A_p − A0))) (EC50 propre au patient).
- **Pourquoi ça devrait marcher** : normalisé au plateau, 0.53 → 0.70 à 0.75–1 h du quintile bas au haut de A ; plat par ledd/durée/âge. e022b fold 0 −0.008, β = 0.83 (paires train, sans target).
- **Test (≤ 10 lignes)** : `add_drug_amplitude` : recalcul g_on / on_eq avec g_p après A_p (`scratch/miss2/p5_pkA.py`), avec H42.
- **Critère de succès** : Δ < −std(Δ) ; si > −0.02, négligeable (hors modèle soumis). **Coût** : faible · **Gain attendu** : faible
- **Statut** : TODO

### H49 — Pré-entraînement auto-supervisé masqué sur train + valid + test
- **Axe** : Méthode
- **Hypothèse** : le réseau apprend le modèle de mesure (rater, A_p, contamination) sur 56 % de patients en plus, sans target.
- **Pourquoi ça devrait marcher** : data-limited (−0.06 à −0.10 / doublement). Contre : le pseudo-labeling a donné ρ 0.99 (distillation) → espoir limité.
- **Test (≤ 10 lignes)** : dans `fit_predict`, pré-entraîner l'encodeur e023 à reconstruire 15–30 % des on/off masqués (perte log-ratio robuste) sur train + valid ; fine-tune têtes niveau + delta.
- **Critère de succès** : Δ < −std(Δ) vs e023 ; n = 4–5, n_off ≤ 2, cohorte B.
- **Risques / fuites** : transductif (légitime, sans target) ; la soumission doit pré-entraîner avec X_test. Coût ×1.5–2.
- **Coût** : fort · **Gain attendu** : faible-moyen (incertain)
- **Statut** : TODO

### H50 — Vraisemblance de l'initiation du traitement dans la MAP
- **Axe** : Structure
- **Hypothèse** : l'initiation est déclenchée par P(t) et la durée, indépendamment du bruit ; un terme logistique contraint P là où les mesures manquent.
- **Pourquoi ça devrait marcher** : jamais d'initiation à T < 15.8 ; 97–100 % si T ≥ 35 et dur ≥ 4 ; sonde `e8.py` full −0.0015, LOO 4.432 → 4.406.
- **Test (≤ 10 lignes)** : `cubic_map` + Σ_at-risk [y·η − log(1+e^η)], η = β·(1, P̂, dur, 1re visite), gradient/hessien analytiques, β fitté sur train.
- **Critère de succès** : Δ < −std(Δ) vs e022b. **Coût** : faible · **Gain attendu** : faible (risque NOISE)
- **Statut** : TODO

### H51 — Échelle du bruit OFF non traité selon le rang
- **Axe** : Structure
- **Pourquoi** : IQR non traités 0.50 → 0.63 de 1 à ≥ 4 OFF avant ; traités plat. Non sondé en MAP.
- **Test** : variante H46, ω/ξ off non traité par bin min(n_off_avant, 3) ; fold 0 d'abord, contrôle sur cubiques exacts (confusion règle ×1.02).
- **Critère de succès** : trajectoire fold 0 ≤ 3.53. **Coût** : faible · **Gain attendu** : faible
- **Statut** : TODO (après H46)

### H52 — Facteur rater localisé par le quantile 80
- **Axe** : Rater
- **Pourquoi** : bord droit net de log(off/target) ; erreur d'échantillonnage 0.0103 (q80) vs 0.0270 (médiane). Sonde OOF : e022b 2.7781 → 2.7730 (5/5), b002 −0.002, e023 −0.0007.
- **Test** : `fit_rater_scale` avec q80 recentré sur la convention médiane (recentrage obligatoire, cf. measurement.md).
- **Statut** : TODO — négligeable, uniquement si e022b est rerun pour une autre raison.

### H53 — e023 à 40 seeds
- **Axe** : Ensemble
- **Pourquoi** : MSE(N) ≈ 6.57 + 1.27/N → ≈ −0.01 dans le blend pour un coût ×4.
- **Statut** : TODO — priorité basse, largement couvert par H43.

---

<!-- Fiches de la branche bertrand (2026-10-13), renumérotées au merge : H30 → H58, H31 → H59, H32 → H60. -->

### H58 — Confiance accordée aux observations selon leur qualité (bruit par visite)
- **Axe** : Structure
- **Hypothèse** : au-delà de la source (off / on_t / on_nt) et du niveau, la qualité d'une mesure dépend de la visite : rater plus ou moins bruité, OFF mesuré tôt après la prise (effet résiduel), patient traité ou non, âge. Le bruit standardisé `z = (y_v − target)/(c0 + c1·target)` aurait une localisation m(x) et une échelle s(x) prédictibles, à injecter dans la MAP skew (cubique + quadratique).
- **Pourquoi ça devrait marcher** : EDA fold 0 (`lab/scratch/quality_eda.py`) : off, tsi_off < 12 h → moyenne z −0.26 vs −0.10 (≥ 16 h) ; sd z 0.87–0.89 sans ledd / tsi / on vs 0.95 avec ; on_t sd 1.12 → 0.97 avec l'âge. Mais dispersion par rater **non reproductible** (corrélation split-half 0.36 pour off, ~0 pour on) : la qualité rater n'est pas un signal.
- **Test** : e023_quality_noise = e022b + 2 HGB (max_leaf 15, min_leaf 200) sur covariables visite sans target (tsi_on/off, ledd, âge, durée, sexM, cohort, gene, rater, rang, NaN, g_on, source), cross-fit patient sur train ; template SN refitté sur (z − m)/s ; MAP avec loc = sv·(m + s·ξ), échelle sv·s·ω (`feat_quality.py`, argument `vis` de `cubic_map` / `skew_map`).
- **Critère de succès** : Δ < −std(Δ) vs e022b ; off_measured et tsi_off_present baissent.
- **Risques / fuites** : m, s appris sur target → dans fit_predict, cross-fit pour les lignes train.
- **Statut** : RÉFUTÉ. Fold 0 : échelle seule 2.7540 (≈ e022b 2.7551), loc + échelle 2.7465, loc seule 2.7211. Run complet loc seule : CV 2.7846, **+0.004 ± 0.023 vs e022b, 1/5 folds** (−0.034 / +0.008 / +0.026 / +0.005 / +0.017) → gain du fold 0 = chance. HGB utilise déjà directement tsi_off / ledd / âge ; corriger l'entrée de la trajectoire n'ajoute que de la variance. Piste restante, négligeable a priori : biais m(tsi_off) seul sur la source off (1 covariable, motivée par le processus générateur).


### H60 — Modèle séquentiel par patient, entrées brutes (BiGRU / Transformer)
- **Axe** : Trajectoire / Ensemble
- **Hypothèse** : un réseau qui lit toute la séquence de visites d'un patient (passé + futur, intervalles irréguliers) apprend lui-même débruitage, effet rater et trajectoire, avec des erreurs différentes de la pile HGB à features manuelles.
- **Pourquoi ça devrait marcher** : HGB prédit chaque visite indépendamment (lags ±2, résumés patient, λ fixe) ; e024_mlp_bag (MLP sur les mêmes features) reste corrélé 0.88 à e022b. Développé en parallèle de H30 (main), qui lit la matrice e022b dense au lieu des entrées brutes.
- **Test** : `lab/feat_seq_raw.py`. Par visite : 8 brutes standardisées (NaN → 0) + gap off−on, 7 masques NaN, embeddings rater (8) / gene (4) / cohort (2), encodage temporel T_* (dt_prev/next, first/last, since_first, to_last, age_rel, pos, n_vis, tsd). Projection 2 couches → BiGRU 2×64 (packed) → tête sur [h, h0]. MSE visite sur target standardisée, AdamW + OneCycle lr 2e-3, 60 epochs, batch 64 patients, dropout 0.4, wd 5e-2, bag 5 seeds. Aucune statistique apprise sur target hors fit_predict.
- **Critère de succès** : blend avec b002_e022b_mlp Δ < −std(Δ), 5/5.
- **Statut** : FAIT. e025_gru_raw seul 2.803 (NOISE vs e022b, folds variables) ; ρ résidus 0.77 vs e022b. **b003_e022b_mlp_gru (e022b 0.38 + e024_mlp_bag 0.19 + GRU 0.43) : 2.618, −0.109 ± 0.038 vs b002_e022b_mlp, 5/5.** Robuste aux seeds (e025s_gru_raw_seed100 : NOISE, même blend −0.107). Ablation temps : Δt voisins suffisent (e025t_gru_raw_tdt NOISE), sans encodage temporel WORSE +0.047 (e025u_gru_raw_tnone). Transformer e026_transformer_raw : 2.968, poids NNLS 0 avec le GRU. Coût ~18 min CV CPU (5 seeds). Suites possibles : complémentarité avec b003_seq4 (main, 2.541 ; à mesurer, OOF de main absents de ce clone), GRU niveau + delta.

### H61 — Prior cubique à queues lourdes sur la courbure
- **Axe** : Structure
- **Hypothèse** : les résidus des coefficients latents (patients ≥ 8 visites) ont des queues lourdes en c2/c3 (excès de kurtosis 1.92 / 4.88). Le prior gaussien (MinCovDet) de `cubic_map` sous-estime les trajectoires qui passent un maximum, d'où P̂ trop monotone et une règle ×1.02 sous-déclenchée.
- **Pourquoi ça devrait marcher** : générateur candidat (dgp4.md § 4) : MVN → part G 5.4 %, excès du pic 1.02 2.9 ; bootstrap des résidus → 10.7 % / 5.6 (réel 9.2 % / 5.7). P(trig) postérieur sous-calibré (≤ 0.01 prédit : 4.8 % observé). Sonde Σ_courbure × 1.5 (proxy grossier) : E[rule(P)] fold 0 3.5118 → 3.5048, P(trig) 3 → 4 %.
- **Test (≤ 10 lignes)** : `cubic_map` avec prior t multivarié (ν fitté sur les résidus train, EM / poids de Student dans la boucle Newton : Q → w_p Q, w_p = (ν + D)/(ν + δ'Qδ)) ; postérieure par importance sampling des tirages Laplace (poids prior t / gaussien) → r_* de e027.
- **Critère de succès** : trajectoire fold 0 < 3.50 et P(trig) calibré ; puis Δ < −std(Δ) vs e027 et apport au blend ≥ 0.01. **Coût** : moyen · **Gain attendu** : faible-moyen
- **Risques / fuites** : ν et Σ fittés sur train uniquement ; tirages seedés cv.SEED.
- **Statut** : TODO

### H62 — Vraisemblance du bruit OFF non i.i.d. (absorbe H51, remplace H46)
- **Axe** : Structure
- **Hypothèse** : le bruit OFF n'est pas i.i.d. par visite : sa dispersion croît avec le nombre d'OFF antérieures (pas avec le rang de visite seul) et les bruits de deux OFF consécutives sont négativement corrélés.
- **Pourquoi ça devrait marcher** : IQR log(off/(s·T)) des naïfs stricts 0.33 / 0.48 / 0.54 / 0.59 pour 0 / 1 / 2 / ≥ 3 OFF antérieures (à T > 24 : 0.24 / 0.41 / 0.48 / 0.57) ; 1re OFF à v > 0 0.313 ; ρ(v0, v1) −0.128 (folds −0.05 à −0.18), lag 1 ensuite −0.03 à −0.08, lag ≥ 2 ≈ 0 ; pas d'ancrage du target sur v0. H46 (template v0 seul) a échoué en final car il pondère v0 sans modéliser la corrélation avec v1 (v>0 dégradé 3.495 → 3.516).
- **Test (≤ 10 lignes)** : sonde fold 0 (`scratch/meas`) : MAP cubique avec ω_k = ω·f(min(k_off, 3)) et pseudo-vraisemblance bandée (paire consécutive gaussienne ρ = −0.1 sur les z) ; garder seulement si trajectoire v>0 ne se dégrade pas.
- **Critère de succès** : trajectoire fold 0 ≤ 3.53 sans perte sur v>0, puis Δ < −std(Δ) vs e022b. **Coût** : moyen · **Gain attendu** : faible
- **Risques / fuites** : aucune nouvelle (mesures seulement) ; confusion possible avec la règle ×1.02 → estimer hors visites censurées.
- **Statut** : TODO

### H63 — Extrapolation de la progression hors fenêtre OFF à partir des observations ON
- **Axe** : Trajectoire
- **Hypothèse** : les séries ON contiennent une information sur la progression latente qui n'est pas complètement exploitée lorsque les visites sortent de la fenêtre d'observation OFF.
- **Résultats** (sonde sur OOF de b003_seq4, sans réentraînement ; revue leak-reviewer GO sous conditions) :
  - corrélation pente ON corrigée / pente vraie : 0.20 ;
  - corrélation pente b003 / pente vraie : 0.84 ;
  - R² du correcteur HGB hors fenêtre OFF : −0.109 ;
  - correction principale : RMSE 2.5408 → 2.5562, dégradation sur les cinq folds ;
  - aucun gain apporté par les corrections linéaires ;
  - aucune information résiduelle prédictive identifiée.
- **Conclusion** : les observations ON sont déjà exploitées par les features longitudinales et les réseaux existants. Les nouvelles corrections testées ajoutent de la variance sans améliorer la reconstruction.
- **Rapport** : `lab/eda/h63_on_progression.md`
- **Statut** : ABANDONNÉ

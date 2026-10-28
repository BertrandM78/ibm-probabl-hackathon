# Ensembles et audit des erreurs de b003_seq4 (2026-10-13)

Mission ensembles / analyse d'erreurs, à partir des OOF existants seulement (aucun réseau réentraîné).
Scripts : `lab/scratch/ens3/blends.py`, `lab/scratch/ens3/audit.py` ; sorties brutes dans
`lab/scratch/ens3/{blends,audit}.log`, `blends.csv`, `slices.csv`, `residual_corr.csv`.

## Protocole

- 12 runs OOF alignés : même `Index`, folds = `lab/folds.csv` (hash 815b8eaa), même target, tout fini.
- b003_seq4 = NNLS imbriquée de e023_seq_nn, e025_seq_offset, e026_seq_amortized, e022b. Reproduction :
  NNLS imbriquée recalculée 2.5407 vs OOF fourni 2.5408. Les poids figés du LOG (0.38/0.38/0.15/0.09)
  donnent 2.5389, mais ils sont fittés sur les 5 folds : optimiste, non utilisé comme référence.
- Poids appris : NNLS par fold, fittés sur les OOF des 4 autres folds seulement (patients du fold évalué
  jamais vus), comme `lab/blend.py`. Contrôles : poids fixes sur une grille déclarée (0.1/0.2/0.3), tous
  rapportés. IC 95 % : bootstrap par patient (2000 tirages) du Δ global.

## 1. Blends

Δ = RMSE(blend) − RMSE(b003_seq4), par fold. Référence 2.5408 (folds 2.481 / 2.471 / 2.525 / 2.703 / 2.516).

| Blend | RMSE | Δ moyen ± std | Folds | Δ par fold | IC 95 % patient | Poids (moyenne par fold) |
|---|---|---|---|---|---|---|
| A b003 (contrôle) | 2.5407 | −0.000 ± 0.001 | 2/5 | — | [−0.001, +0.001] | e023 0.38, e025_off 0.38, e026_am 0.15, e022b 0.10 |
| **B b003 + GRU brut** | **2.5177** | **−0.023 ± 0.019** | **4/5** | −0.034 / +0.009 / −0.028 / −0.021 / −0.041 | [−0.034, −0.012] | e023 0.29, e025_off 0.34, e026_am 0.09, e022b 0.06, **GRU 0.22 (0.20–0.26)** |
| C b003 + MLP | 2.5407 | −0.000 ± 0.001 | 2/5 | — | [−0.001, +0.001] | MLP **0.00** |
| D b003 + GRU + MLP | 2.5177 | −0.023 ± 0.019 | 4/5 | = B | [−0.035, −0.012] | MLP 0.00 |
| E b003 + GRU 10 seeds | 2.5180 | −0.023 ± 0.021 | 4/5 | | [−0.033, −0.012] | gru10 0.23 |
| F b003 + GRU + TF brut | 2.5185 | −0.022 ± 0.019 | 4/5 | | [−0.033, −0.011] | TF brut 0.02 |
| G b003 + tous runs bertrand | 2.5179 | −0.023 ± 0.018 | 4/5 | | [−0.034, −0.011] | 10 composantes, aucun gain en plus |
| fixe 0.9·b003 + 0.1·GRU | 2.5236 | −0.017 ± 0.008 | **5/5** | −0.022 / −0.004 / −0.019 / −0.015 / −0.026 | | — |
| fixe 0.8·b003 + 0.2·GRU | 2.5163 | −0.025 ± 0.017 | 4/5 | −0.034 / +0.002 / −0.028 / −0.020 / −0.042 | | — |
| fixe 0.7·b003 + 0.3·GRU | 2.5191 | −0.022 ± 0.025 | 4/5 | | | — |
| fixe 0.9·b003 + 0.1·MLP | 2.5485 | +0.008 ± 0.004 | 0/5 | | | — |

- **Seul le GRU brut (e025_gru_raw) apporte quelque chose.** Gain réel (IC hors de 0, |Δ| > std), mais
  ≈ 0.02 : à la limite du seuil « négligeable » du protocole. Le fold 1 se dégrade en NNLS (+0.009) :
  c'est le fold où le GRU seul est le plus faible relativement (2.837 vs b003 2.471).
- Le MLP e024 et le Transformer brut e026_transformer_raw n'apportent plus rien : leur diversité est
  déjà couverte par les réseaux séquentiels de main. 10 seeds GRU au lieu de 5 : rien.
- Le gain du GRU est localisé là où le OFF est observé : 58 % de la baisse de SSE sur les visites à OFF
  mesuré, 35 % entre deux OFF, ~7 % hors de la fenêtre OFF (avant le premier, après le dernier).

## 2. Corrélations des erreurs (résidus visite)

| | b003 | e023_seq_nn | e025_seq_offset | e026_seq_amortized | e022b | GRU brut | MLP | TF brut |
|---|---|---|---|---|---|---|---|---|
| b003_seq4 | 1 | 0.973 | 0.976 | 0.890 | 0.910 | **0.828** | 0.913 | 0.797 |
| e023_seq_nn | | 1 | 0.925 | 0.817 | 0.857 | 0.802 | 0.882 | 0.758 |
| e025_seq_offset | | | 1 | 0.825 | 0.872 | 0.793 | 0.888 | 0.764 |
| e026_seq_amortized | | | | 1 | 0.822 | 0.770 | 0.808 | 0.764 |
| e022b | | | | | 1 | 0.768 | 0.884 | 0.772 |
| GRU brut | | | | | | 1 | 0.776 | 0.889 |

RMSE seuls : e025_seq_offset 2.603, e023_seq_nn 2.610, e022b 2.780, GRU brut 2.803, MLP 2.841,
e026_seq_amortized 2.853, TF brut 2.968. Matrice complète : `lab/scratch/ens3/residual_corr.csv`.

## 3. Diagnostic de b003_seq4

**L'erreur est une erreur de trajectoire patient, pas de visite.** Décomposition des résidus par
patient sur une base orthonormée {1, x, x²} (x = âge centré) :

| Composante | b003 | attendu si bruit i.i.d. |
|---|---|---|
| niveau | **66.1 %** | 13.2 % |
| pente | **23.6 %** | 13.2 % |
| courbure | 9.1 % | 13.2 % |
| locale (visite) | **1.2 %** | 60.5 % |

Même profil pour chaque composante et pour le GRU brut. Oracle : retirer le résidu moyen par patient
ramène 2.541 à **1.479**.

**Où se concentre l'erreur** (RMSE, part du SSE) :

| Tranche | RMSE | Part SSE |
|---|---|---|
| OFF mesuré (58 % des visites) | 2.14 | 41 % |
| entre deux OFF | 2.52 | 24 % |
| hors fenêtre OFF (18 % des visites) | 3.37–5.05 | **36 %** |
| — après le dernier OFF | 3.37 | 25 % |
| — avant le premier OFF | 3.81 | 8 % |
| — patient sans aucun OFF (65 patients) | 5.05 | 3 % |
| dernière visite | 3.21 | 20 % |
| part OFF manquant patient > 60 % | 3.51 | 37 % |
| durée > 12 ans | 3.56 | 13 % |
| cohorte B | 2.89 | 14 % |
| jamais ON (non traités) | 1.14 | 1.5 % |

- Dernière visite : RMSE 2.65 si elle a un OFF, puis 3.18 (≤ 1 an après le dernier OFF), 3.83 (2–4 ans),
  4.42 (> 4 ans), 5.35 si le patient n'a aucun OFF.
- Erreur de niveau par patient (RMS) selon le nombre de OFF mesurés : 4.70 (0), 3.83 (1), 2.85 (2),
  2.0 (3–4), 1.1–1.9 (≥ 5). Les patients à ≤ 2 OFF (1 070 patients, 12 % des visites) font **28 % du SSE**.
- **Limite d'information, pas de modèle.** L'erreur de niveau des patients à ≤ 2 OFF est corrélée à
  0.89–0.93 entre b003 et le GRU brut (entrées totalement différentes) et 0.95 avec e022b ; à ≥ 5 OFF,
  0.78. Ajouter des modèles n'y changera rien : il faut extraire plus d'information des données.
- Intervalles : légère sur-prédiction en début (1re visite biais −0.14, dt_next > 2 ans −0.22) et
  sous-prédiction en fin (dernière +0.10, dt_prev > 2 ans +0.18) : pentes un peu trop plates. Biais²
  ≈ 0.02 face à un MSE ≈ 7–10 : effet négligeable.
- **Calibration correcte.** Résidu moyen patient par quintile de niveau **prédit** : +0.05 / −0.06 /
  +0.08 / −0.07 / −0.05 (≈ 0). Le biais apparent par quintile de target (−0.45 … +0.60) est un effet de
  régression vers la moyenne. Les covariables patient ne prédisent pas le résidu (R² cross-fitté −0.08).

**Patients où tous les modèles échouent** (top 10 % d'erreur pour les 6 modèles à la fois) :
221 patients (0.006 attendus si indépendants), **29 % du SSE**. Profil vs autres : 2.8 OFF mesurés
(vs 4.7), 59 % de OFF manquants (vs 39 %), 6.9 visites (vs 8.0), durée initiale 2.4 ans (vs 1.5),
pente vraie 3.3/an (vs 2.5), target moyen 46 (vs 35), 31 % cohorte B (vs 20 %), 99 % traités.

## 4. Comparaison avec l'analyse préliminaire sur e022b

| Constat | e022b (préliminaire) | b003_seq4 |
|---|---|---|
| Décomposition niveau / pente / courbure / locale | 63 / 24 / 11 / 1.6 % | 66 / 24 / 9 / 1.2 % — confirmé |
| Patients en échec conjoint | 269, 29 % du SSE (3 modèles) | 221, 29 % du SSE (6 modèles) — confirmé |
| Profil (peu de OFF, sévères, pente forte, cohorte B) | oui | identique — confirmé |
| « 79–82 % sous-prédits » | interprété comme un biais | **corrigé** : artefact de sélection sur la target, calibration ≈ 0 par niveau prédit |
| Résidu prédictible par covariables | R² −0.07 | R² −0.08 — confirmé |
| Gain du GRU brut | −0.15 vs e022b | **−0.023** vs b003 : les réseaux de main captent déjà l'essentiel |
| Gain du MLP | −0.05 vs e022b | **0** (poids nul) |

## 5. Hypothèses (classées par gain potentiel)

IDs H63–H65 : H61 et H62 sont déjà pris sur `origin/lab-victor`. Non reportées dans IDEAS.md (fichier partagé).

### H63 — Extrapolation hors de la fenêtre OFF à partir de la série ON (rang 1)
- **Mécanisme** : après le dernier OFF (et avant le premier), niveau et pente ne sont informés que par
  les ON via A_p et par la pente de population. L'erreur croît avec la distance au dernier OFF : les
  modèles n'utilisent pas assez la dynamique ON, qui reste observée.
- **Indice** : hors fenêtre = 18 % des visites, 36 % du SSE ; dernière visite 2.65 → 4.42 selon les
  années depuis le dernier OFF ; erreurs de niveau corrélées à 0.9 entre modèles d'entrées différentes.
- **Test minimal (OOF, minutes)** : sur les visites hors fenêtre, régresser le résidu de b003 sur des
  quantités ON sans target (pente de on_eq_n après le dernier OFF − pente prédite, ratio ON/préd.,
  années depuis le dernier OFF). HGB cross-fitté par fold. Seuil : R² > 0.02 → feature pour e023/e025.
- **Gain potentiel** : −0.02 à −0.05 si R² ≥ 0.05 (la zone pèse 36 % du SSE). **Coût** : faible (sonde),
  puis moyen (réentraîner les réseaux). **Fuite** : régression de résidus cross-fittée par fold ;
  features ON sans target. **Lien** : complémentaire de H40/H42 (A_p, incertitude ON), qui visent le niveau.

### H64 — Poids d'ensemble par zone d'information OFF (rang 2)
- **Mécanisme** : le GRU brut est utile là où le OFF est observé et inutile en extrapolation ; un poids
  unique est un compromis.
- **Indice** : 93 % du gain du GRU dans les zones OFF mesuré / entre deux OFF, ~7 % hors fenêtre.
- **Test minimal (OOF, minutes)** : NNLS imbriquée avec poids distincts par zone {OFF mesuré, entre,
  hors fenêtre}, sur les composantes de B. Critère : Δ vs B < −std, ≥ 4/5 folds.
- **Gain potentiel** : −0.003 à −0.01 au-delà de B. **Coût** : très faible. **Fuite** : zone définie
  par X seul ; poids fittés hors fold. **Risque** : H54 (sélection par patient) était nulle ; ici 3 zones
  seulement, donc variance faible.

### H65 — Correction de pente au voisinage des bords (rang 3)
- **Mécanisme** : pentes patient légèrement trop rétrécies vers la population.
- **Indice** : biais −0.14 à la 1re visite, +0.10 à la dernière ; ±0.2 aux grands intervalles.
- **Test minimal (OOF)** : rotation des prédictions autour de la moyenne patient d'un facteur β fitté
  hors fold ; critère Δ < −std.
- **Gain potentiel** : faible (biais² ≈ 0.02 face à un MSE ≈ 7–10, donc ≤ −0.005). **Coût** : très faible.
  **Fuite** : β fitté hors fold.

## 6. Recommandation

1. **Meilleur blend validé : B = b003_seq4 + e025_gru_raw** (NNLS imbriquée, 2.518, −0.023 ± 0.019,
   4/5, IC patient [−0.034, −0.012]). Contrôle à poids fixe plus robuste : **0.9·b003 + 0.1·GRU**
   (2.524, −0.017 ± 0.008, **5/5**). Le gain est au seuil « négligeable » : ne l'intégrer au modèle soumis
   que si le coût d'un 4e réseau (~18 min CV, CPU) est acceptable. Il faut le test.csv de e022b et des
   réseaux de main pour produire la soumission (`lab/blend.py`).
2. **Arrêter d'empiler des modèles** : MLP, Transformer brut et seeds supplémentaires ont un poids nul.
3. **Priorité scientifique : l'information manquante, pas le modèle.** 66 % de l'erreur est un niveau
   patient, partagé par tous les modèles chez les patients pauvres en OFF. Sonder H63 en premier (minutes,
   OOF), puis H40/H42 côté coéquipiers ; H64 et H65 sont des ajustements d'ensemble de faible portée.

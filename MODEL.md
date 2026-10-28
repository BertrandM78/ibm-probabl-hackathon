# Le modèle retenu : `b006_seq4_gruref`

**RMSE CV (hors-fold, par patient) : 2.510** — contre 16.51 pour la moyenne constante (−85 %) et 10.2 pour le `off` brut.

Ce document explique ce que fait le modèle soumis, comment il fonctionne et surtout pourquoi chaque brique existe.
Tout le détail chiffré des 50+ expériences est dans [lab/LOG.md](lab/LOG.md) ; les constats sur le générateur de données
sont dans [lab/eda/](lab/eda/).

---

## 1. En une phrase

> Un **blend NNLS de 5 modèles** : un gradient boosting posé sur une **trajectoire latente cubique par patient** (estimée par MAP
> avec un bruit de mesure skew-normal), et **quatre réseaux séquentiels** (Transformers et BiGRU) qui lisent toute la séquence
> de visites d'un patient, dont un GRU volontairement « naïf » sur les entrées brutes pour la diversité.

| Composant | Run | Poids NNLS | Rôle |
|---|---|---|---|
| Réseaux séq. (BiGRU + Transformer) sur features e022b | `e030_seq_nn_full` | 0.26 | Meilleur modèle seul, lit trajectoire + contexte |
| Transformer en **offset** sur la trajectoire MAP | `e031_seq_offset_full` | 0.36 | Corrige la trajectoire MAP au lieu de tout réapprendre |
| Transformer « amortized » (colonnes harmonisées seules) | `e032_seq_amortized_full` | 0.10 | Erreurs très décorrélées des autres |
| Boosting sur trajectoire cubique | `e022b_cubic_traj_quad` | 0.07 | Base solide et rapide |
| **BiGRU sur entrées brutes** | `e033_gru_raw_ref` | 0.21 | Aucun a priori : diversité maximale |

Poids : [lab/runs/b006_seq4_gruref/metrics.json](lab/runs/b006_seq4_gruref/metrics.json). Ils sont appris par moindres carrés non négatifs
sur les prédictions hors-fold ([lab/blend.py](lab/blend.py)).

> **Note.** Les fichiers `e030`–`e034` ne sont pas versionnés dans ce clone (seuls leurs runs et `metrics.json` le sont).
> Ce sont les variantes « full » des recettes `e023`/`e025`/`e026` (voir
> [e023_seq_nn.py](lab/experiments/e023_seq_nn.py), [e025_seq_offset.py](lab/experiments/e025_seq_offset.py),
> [e026_seq_amortized.py](lab/experiments/e026_seq_amortized.py)) : les descriptions ci-dessous suivent ces fichiers.
> Un blend voisin, `b007_seq4_grusim`, atteint 2.501 en CV mais n'a pas été soumis.

---

## 2. Le problème, et ce qu'il impose

Pour chaque visite, il faut prédire le **vrai OFF** (`target`), un score moteur MDS-UPDRS débiaisé, alors qu'on observe
seulement des mesures déformées :

- `off` : manquant à 42 %, bruité, dépend du **rater** et du délai depuis la prise ;
- `on` : le score sous médicament, plus souvent présent mais déformé par la **courbe pharmacodynamique** de la lévodopa ;
- `ledd`, `time_since_intake_*`, âge, durée de maladie, etc., avec beaucoup de manquants.

Trois faits structurent tout le modèle :

1. **Les données sont synthétiques** : il existe un générateur qu'on peut retrouver par rétro-ingénierie
   ([lab/eda/dgp.md](lab/eda/dgp.md)). Modéliser le processus vaut mieux qu'empiler des features.
2. **Le test contient toutes les visites de chaque patient** (le holdout est par patient, pas par visite). Les informations
   intra-patient *sans target* (lags, leads, moyennes de `on`/`off`) sont donc légitimes : on peut regarder le passé *et* le futur
   du patient.
3. **Le vrai OFF est une courbe lisse par patient.** C'est le fait qui change tout (section 3).

---

## 3. Ce que l'EDA a révélé sur le générateur

Résumé de [lab/eda/dgp.md](lab/eda/dgp.md) :

- Le `target` d'un patient est un **polynôme cubique de la durée de maladie**, exact à l'arrondi 0.1 près pour 67 % des patients
  (8+ visites), et cubique + un « coude » pour le reste. Il **ne décroît jamais**.
- Les coefficients sont quasi **indépendants des covariables** (R² ≤ 0.05) : la forme de la courbe est propre au patient.
- Le `target` est **latent** : son écart à la courbe n'est corrélé à aucun bruit de mesure.
- Le bruit de mesure est **multiplicatif par rater** (50 raters, mêmes en train et test) et **asymétrique** : bord droit net,
  queue gauche. Une loi **skew-normale** le décrit presque aussi bien que n'importe quel estimateur non paramétrique.
- Les valeurs ON se déduisent de OFF via une courbe pharmacodynamique **commune** à tous les patients (constante avec l'âge).
- L'initiation du traitement est déclenchée par le niveau du target, ce qui donne un signal dans les manquants de `on`/`ledd`.

Conséquence : si on arrive à estimer proprement la **courbe latente de chaque patient** à partir de ses mesures bruitées, on est
très proche de la réponse. Le reste est de l'apprentissage de résidus.

---

## 4. Architecture pas à pas

```
mesures brutes (on, off, ledd, tsi, rater…)
        │
        ▼
 ① Débiaisage par visite ───────────────  facteur rater + courbe PD  →  score harmonisé y_v
        │
        ▼
 ② Trajectoire latente par patient ─────  MAP cubique hiérarchique, bruit skew-normal  →  hb_post_*, c0..c3
        │
        ▼
 ③ Features contextuelles ──────────────  lags/leads, surprises y_v − trajectoire, forme du patient
        │
        ├──────────────► Boosting (HGB)  ──────────────────────────────►  e022b
        │
        ├──► Réseaux séquentiels niveau + delta ───────────────────────►  e030 / e031 / e032
        │
        └──► BiGRU sur colonnes brutes ────────────────────────────────►  e033
                                                                              │
                                              ④ Blend NNLS hors-fold  ◄───────┘
                                                        │
                                              prédiction finale (RMSE 2.510)
```

### ① Débiaisage de chaque visite

Avant toute modélisation, chaque mesure est ramenée à une échelle commune
([feat_rater_scale.py](lab/feat_rater_scale.py), [feat_drug.py](lab/feat_drug.py), [feat_harmonized.py](lab/feat_harmonized.py)) :

- **Facteur rater** `s_r` = médiane de `off/target` par rater, rétrécie vers la médiane globale (20 pseudo-visites).
  Il s'applique de façon multiplicative à `on` et `off` (*pourquoi :* le bruit des raters est multiplicatif, 50 facteurs ≈ log-normaux).
- **Courbe de médicament commune** `g(tsi_on)` (Emax sur courbe de Bateman : absorption rapide puis décroissance) et
  **amplitude `A_p` propre au patient**. Elle permet de convertir un `on` en son équivalent OFF, `on_eq`
  (*pourquoi :* on sait que la courbe PD est identique pour tous, seule l'amplitude varie).
- **Score harmonisé `y_v`** : le `off` normalisé s'il est présent, sinon le `on_eq` normalisé, avec la **source** retenue
  (`off`, `on_t`, `on_nt`) et sa variance de bruit estimée sur le train.

Effet : un OFF absent n'est plus un trou, c'est une mesure indirecte de qualité connue.

### ② Trajectoire latente cubique par patient (cœur du modèle tabulaire)

C'est la pièce qui apporte le plus ([feat_skew.py](lab/feat_skew.py), [feat_cubic.py](lab/feat_cubic.py)).

Pour chaque patient on estime `c0 + c1·x + c2·x² + c3·x³` (x = âge centré), en maximum a posteriori :

- **Vraisemblance** : `y_v = trajectoire + σ_v · z`, avec `z` **skew-normal** dont les paramètres sont appris *par source de mesure*
  (la mesure OFF est plus fiable que ON sans temps de prise). La convolution d'une skew-normale et d'une gaussienne reste
  skew-normale, donc le calcul est exact.
- **A priori hiérarchique** : moyenne des coefficients prédite par de petits boosting sur les covariables du patient, covariance
  résiduelle robuste (MinCovDet, projetée définie positive). Un patient à 4 visites emprunte de la force à la population ;
  un patient à 12 visites s'en éloigne.
- **Optimisation** : Newton amorti vectorisé sur tous les patients, avec une version *leave-one-out* par visite pour que la
  trajectoire d'une visite n'ait pas vu sa propre mesure (utile pour calculer des « surprises » honnêtes).

*Pourquoi cubique et skew-normal ?* Parce que le générateur est cubique et que le bruit a un bord droit net : une moyenne ou une
droite perd l'information. C'est ce qui fait passer le RMSE de 4.07 à 3.23 (puis 2.78 avec le terme quadratique conservé).

### ③ Features contextuelles

- **Temporelles** ([feat_temporal.py](lab/feat_temporal.py)) : valeurs précédentes/suivantes (lags 1–2, leads 1–2), écarts, tendance.
- **Surprises et forme** ([feat_long.py](lab/feat_long.py)) : `y_v − trajectoire LOO`, et la forme de ces surprises par patient
  (max, min, écart-type, asymétrie, médiane). *Pourquoi :* le bruit a un bord droit net, donc `max(surprise)` borne le niveau du
  patient par le bas.
- **Manquants conservés** : on n'impute pas, la missingness est un signal (traitement initié, OFF rarement mesuré).

### ④ Les modèles

**Boosting (`e022b`).** `HistGradientBoostingRegressor` sur toutes les features ci-dessus, cible en **delta** (voir plus bas).
Rapide, robuste, base solide (2.78 seul).

**Réseaux séquentiels (`e030`, `e031`, `e032`).**
Un patient = une séquence de visites. Un réseau (BiGRU 2 couches ou Transformer pré-norm 2 couches, H = 128) lit la séquence
entière et produit deux têtes ([feat_seq.py](lab/feat_seq.py)) :

- **niveau** `A_t` (le target de la visite) ;
- **delta** `D_t` (la variation vs la visite précédente).

Les deux sont recombinés par un **moindre carré par patient** ([feat_temporal.py `reconstruct`](lab/feat_temporal.py)) :

```
L* = argmin_L  Σ_t (L_t − A_t)²  +  λ · Σ_{t>0} (L_t − L_{t−1} − D_t)²        λ = 4
```

*Pourquoi :* le target est une courbe lisse et croissante ; prédire les variations d'une visite à l'autre et les réconcilier avec
les niveaux réduit le bruit indépendant de chaque prédiction. C'est une régularisation structurelle, pas du lissage ad hoc.

Les variantes diffèrent par leur entrée et leur cible :

| Variante | Entrée | Cible | Pourquoi l'avoir gardée |
|---|---|---|---|
| `e030` | matrice e022b densifiée | niveau + delta | meilleur modèle seul (2.61), exploite tout |
| `e031` (offset) | idem | `target − hb_post_full` | le réseau corrige la trajectoire MAP au lieu de la réapprendre (−0.022 en blend) |
| `e032` (amortized) | colonnes harmonisées seules, pas de MAP | niveau + delta | « apprend la MAP » lui-même, erreurs très décorrélées |
| `e033` (GRU brut) | colonnes brutes + masques NaN + embeddings rater/gène/cohorte | niveau | **aucun** module métier ; seul ≈ 2.80, mais ρ des résidus ≈ 0.77 avec les autres |

Détails d'entraînement : AdamW (2e-3), OneCycle, perte MSE masquée, dropout, gradient clipping. L'**epoch est choisi sur 10 % des
patients du train** (jamais sur le fold évalué), et chaque variante est un **bag de 5 seeds**.

### ⑤ Blend NNLS

Les cinq prédictions sont combinées par moindres carrés non négatifs. Les poids sont appris sur 4 folds d'OOF et évalués sur le 5ᵉ
(donc le RMSE annoncé est honnête) ; la soumission finale utilise les poids ajustés sur tout l'OOF. Ils sont stables d'un fold à
l'autre (amplitude ≤ 0.03).

*Pourquoi le blend gagne :* ces modèles se trompent **différemment**. Mesuré sur les résidus, la corrélation vaut 0.77 entre le GRU
brut et le boosting, 0.86 entre le réseau e023 et le boosting, contre 0.88 entre deux réseaux de même famille. Moyenner des erreurs
décorrélées réduit la variance sans augmenter le biais.

---

## 5. Pourquoi ça marche : la progression du RMSE

| Étape | Run | RMSE CV | Idée |
|---|---|---|---|
| Moyenne constante | `e000_dummy` | 16.51 | référence |
| Boosting sur brut, NaN natifs | `e001_hgb_raw` | 7.58 | la missingness est déjà un signal |
| + contexte intra-patient | `e002_hgb_basic_context` | 4.58 | le test contient toutes les visites du patient |
| + facteur rater, amplitude du médicament | `e004_drug_amplitude` | 4.07 | débiaiser avant de modéliser |
| + score harmonisé, trajectoire hiérarchique | `e007_hier_traj` | 3.55 | une courbe latente par patient |
| + bruit skew-normal par source | `e012_traj_skew` | 3.23 | la bonne loi de bruit |
| + features temporelles, cible en delta | `e017_skew_temporal_delta` | 3.07 | lisser la séquence |
| + surprises et forme | `e019_long_surp_shape` | 2.81 | le bord droit du bruit borne le niveau |
| + trajectoire cubique | `e022b_cubic_traj_quad` | 2.78 | la vraie forme du générateur |
| Réseaux séquentiels | `e023_seq_nn` | 2.61 | lire toute la séquence d'un coup |
| Blend de 4 modèles | `b003_seq4` | 2.541 | erreurs décorrélées |
| **+ GRU sur entrées brutes** | **`b006_seq4_gruref`** | **2.510** | diversité maximale |

La leçon : les gros sauts viennent de **modéliser le mécanisme** (étapes ①–②, de 7.6 à 2.8), puis d'**exploiter la structure
séquentielle** (réseaux, de 2.8 à 2.6), puis de **diversifier** (blend, de 2.6 à 2.5).

Où le modèle est meilleur ou moins bon (sous-groupes de `metrics.json`) : RMSE 2.11 quand l'OFF est mesuré contre 2.97 quand
il manque ; 1.62 quand le ON manque ; 2.81 quand `ledd` est présent contre 1.88 quand il manque (patients non traités, plus faciles).

---

## 6. Rigueur : pourquoi ce score est crédible

- **Aucune fuite par le target.** Tout ce qui est appris sur `target` (facteur rater, courbe PD, bruit skew-normal, prior
  hiérarchique, scaler, choix d'epoch) est ajusté **dans** `fit_predict`, sur le train du fold uniquement. Le code a été relu par
  l'agent `leak-reviewer` avant les runs coûteux.
- **Aucune fuite par groupe.** Folds `GroupKFold` par `patient_id` (5 folds fixes, [lab/folds.csv](lab/folds.csv)) ; le test est
  un holdout par patient.
- **Features intra-patient légitimes.** Elles n'utilisent jamais le target, seulement les mesures des autres visites du même
  patient, que le test fournit.
- **Règle de décision.** Comparaison appariée fold par fold : un gain < std(Δ) est du bruit, un gain < ~0.02 est noté sans complexifier
  le modèle. Le leaderboard public n'a servi qu'à vérifier que CV et LB corrèlent.
- **Reproductibilité.** Seeds fixées (`cv.SEED`), PyTorch en mode déterministe.

## 7. Ce qui n'a pas fonctionné (et pourquoi c'est utile)

Documenté dans [lab/LOG.md](lab/LOG.md) et [lab/IDEAS.md](lab/IDEAS.md) :

- **Règle de censure du générateur** (`e027_rule_post`, `e029_latent_prior`) : gain individuel négligeable (< 0.02) et 0 apport au
  blend, car le prior gaussien ne capte pas les queues lourdes des coefficients.
- **Template de bruit séparé à la 1ʳᵉ visite** (`e028`) : pire (+0.007).
- **Transformer sur entrées brutes** (`e026_transformer_raw`) : nettement moins bon que le GRU (+0.166), poids NNLS nul.
- **Biais fin par rater, stacking par patient, pseudo-labeling, tuning poussé des réseaux** : sous le seuil de bruit.

## 8. Reproduire

```bash
.venv/bin/python lab/blend.py e030_seq_nn_full e022b_cubic_traj_quad e031_seq_offset_full \
    e032_seq_amortized_full e033_gru_raw_ref --name b006_seq4_gruref
.venv/bin/python lab/submit.py b006_seq4_gruref --key b006_seq4_gruref
```

Prérequis : chaque composante doit avoir été exécutée avec `lab/cv.py … --test` (OOF + prédictions test). Coût dominant : les
réseaux (≈ 25 à 100 s par seed et par fold sur CPU, ~6 min par fit pour `e023`).

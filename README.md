<p align="center">
  <img src="docs/figures/probabl-logo.svg" alt="Probabl" height="90">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="docs/figures/ibm-logo.svg" alt="IBM" height="55">
</p>

<h1 align="center">Parkinson : retrouver le « vrai OFF »</h1>

<p align="center">
  <em>Prédire et corriger le biais d'évaluation du score moteur MDS-UPDRS —<br>
  Bobathon ESILV × Probabl × IBM · Kaggle <a href="https://www.kaggle.com/competitions/bobathon-esilv/">bobathon-esilv</a></em>
</p>

<p align="center">
  <img alt="RMSE CV" src="https://img.shields.io/badge/RMSE%20CV-2.510-2ea44f">
  <img alt="Baseline" src="https://img.shields.io/badge/dummy-16.51-lightgrey">
  <img alt="Réduction" src="https://img.shields.io/badge/erreur%20r%C3%A9duite%20de-85%20%25-blue">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12%2B-3776ab">
</p>

---

## Le problème

Chez les patients parkinsoniens, le score moteur MDS-UPDRS dépend fortement de la prise de **lévodopa**.
Le score **OFF** (sans médicament actif) est le plus informatif sur la maladie, mais il est rarement
mesuré, souvent bruité, et dépend du rater et du délai depuis la dernière prise.

**Objectif :** pour chaque visite, prédire le **vrai OFF** (score débiaisé, `target`) à partir des mesures
ON/OFF observées, de la dose de lévodopa, des délais de prise et de l'historique du patient.

- **Métrique :** RMSE au niveau visite.
- **Holdout :** par patient (aucun patient commun entre train et test).
- **Données :** synthétiques. 44 590 visites / 5 576 patients en train, 11 013 visites / 1 395 patients en test,
  avec beaucoup de valeurs manquantes (`off` 42 %, `on` 30 %, `ledd` 37 %, `time_since_intake_off` 79 %).

Fond scientifique : [docs/CONTEXT.md](docs/CONTEXT.md).

## Ce que contient ce dépôt

Ce dépôt est un projet **Bob + skore** qui va de l'idée à la soumission : brainstorming → hypothèse →
benchmark → évaluation → ensemble → rapport Skore Hub → soumission Kaggle. Bob pilote toute la chaîne, avec un
protocole d'évaluation unique pour que chaque run soit comparable, et soumet directement le meilleur.

```
brainstorm ──▶ IDEAS.md ──▶ review (fuites) ──▶ experiment ──▶ cv.py ──▶ LOG.md / results.csv
 (analysts)    (priorités)    (leak-reviewer)    (experimenter)  (protocole    │
                                                                  verrouillé)  ▼
                                    blend NNLS ──▶ submit.py (EstimatorReport skore + CSV) ──▶ Kaggle
```

| Élément | Rôle |
|---|---|
| [lab/cv.py](lab/cv.py) | Protocole d'évaluation **verrouillé** : 5 folds `GroupKFold` par patient ([lab/folds.csv](lab/folds.csv)), RMSE OOF |
| [lab/experiments/](lab/experiments/) | Une hypothèse = un fichier `eNNN_*.py` (`fit_predict`, `DESCRIPTION`, `FEATURES`) |
| [lab/results.csv](lab/results.csv) · [lab/LOG.md](lab/LOG.md) | Tableau automatique et journal commenté de tous les runs |
| [lab/IDEAS.md](lab/IDEAS.md) | Backlog d'hypothèses priorisées (gain attendu / effort) |
| [lab/eda/](lab/eda/) | Constats d'EDA par axe (pharmaco, rater, manquants, trajectoire, générateur…) |
| [lab/blend.py](lab/blend.py) | Blends NNLS calculés sur les prédictions hors-fold |
| [lab/submit.py](lab/submit.py) | Soumission : EstimatorReport poussé sur Skore Hub + `submissions/<clé>.csv` validé |
| [Bob.md](Bob.md) · [.bob/](.bob/) | Contexte du projet, règles, agents (`analyst`, `experimenter`, `leak-reviewer`), skills (`/cv`, `/idea`, `/brainstorm`, `/review`, `/status`, `/submit`, `/submit-prep`) et garde-fous |

## Résultats

Plus de 50 runs évalués avec le même protocole. Progression de la RMSE CV (hors-fold, par patient) :

| Étape | Run | RMSE CV |
|---|---|---|
| Moyenne constante | `e000_dummy` | 16.51 |
| Gradient boosting sur colonnes brutes (NaN natifs) | `e001_hgb_raw` | 7.58 |
| + contexte intra-patient (visite précédente/suivante, moyennes) | `e002_hgb_basic_context` | 4.58 |
| + facteur rater et amplitude du médicament par patient | `e004_drug_amplitude` | 4.07 |
| + score harmonisé et trajectoire hiérarchique par patient | `e007_hier_traj` | 3.55 |
| + trajectoire MAP à bruit skew-normal (par source de mesure) | `e012_traj_skew` | 3.23 |
| + features temporelles, cible en delta | `e017_skew_temporal_delta` | 3.07 |
| + surprises et forme de la courbe par patient | `e019_long_surp_shape` | 2.81 |
| + trajectoire cubique hiérarchique | `e022b_cubic_traj_quad` | 2.78 |
| Réseaux séquentiels par patient (BiGRU / Transformer) | `e023_seq_nn` | 2.61 |
| Blend NNLS de 4 modèles | `b003_seq4` | 2.541 |
| **Blend NNLS de 5 modèles (+ GRU sur entrées brutes)** | **`b006_seq4_gruref`** | **2.510** |

Détail et verdicts (gain / bruit / pire) dans [lab/LOG.md](lab/LOG.md).

### Ce qui a marché

1. **Modéliser le processus générateur** plutôt qu'empiler des features : courbe pharmacodynamique du médicament,
   amplitude propre au patient, échelle propre au rater.
2. **Une trajectoire latente par patient** (hiérarchique, puis cubique), estimée par MAP avec un bruit
   skew-normal propre à chaque source de mesure, fournit au boosting un excellent a priori.
3. **Exploiter la séquence des visites** : le test contient toutes les visites de chaque patient, donc lags,
   leads et réseaux séquentiels sont légitimes (aucune fuite par le `target`).
4. **Ensembler des modèles décorrélés** (boosting sur trajectoire + réseaux séquentiels + GRU sur entrées brutes) via NNLS hors-fold.

### Ce qui n'a pas aidé

Plusieurs axes ont été explorés puis abandonnés faute de gain au-dessus du bruit : biais fin par rater,
stacking par patient, pseudo-labeling, tuning poussé des réseaux, MLP seul, Transformer « amortized » seul.
Les constats sont documentés dans [lab/eda/](lab/eda/) et [lab/IDEAS.md](lab/IDEAS.md).

## Rigueur méthodologique

- **Règle de décision :** comparaison appariée fold par fold contre une référence. Un gain dont |Δ| < std(Δ)
  est du bruit ; un gain < ~0.02 RMSE est noté mais ne justifie pas de complexifier le modèle soumis.
- **Anti-fuite :** tout ce qui est appris sur `target` (courbes PK, biais rater, encodages, stacking, scalers)
  est fitté *dans* `fit_predict`, sur le train uniquement ; jamais de split au niveau visite.
- **Ne jamais optimiser sur le leaderboard public**, utilisé uniquement pour vérifier que CV et LB corrèlent.
- Un agent critique (`leak-reviewer`) relit chaque hypothèse coûteuse avant le run.

## Utilisation

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r lab/requirements.txt

# test rapide sur un fold (non loggé)
.venv/bin/python lab/cv.py lab/experiments/e002_hgb_basic_context.py --folds 0

# run complet : 5 folds + prédictions test, loggé dans results.csv / LOG.md
.venv/bin/python lab/cv.py lab/experiments/e022b_cubic_traj_quad.py --test --ref e002_hgb_basic_context

# blend NNLS hors-fold
.venv/bin/python lab/blend.py e023_seq_nn e022b_cubic_traj_quad e025_seq_offset e026_seq_amortized --name b003_seq4

# soumission : EstimatorReport sur Skore Hub + submissions/<clé>.csv (--no-hub pour un essai à blanc)
.venv/bin/python lab/submit.py b006_seq4_gruref --key b006_seq4_gruref

# vérifier un CSV de soumission
python3 lab/check_submission.py submissions/<clé>.csv
```

Notes : LightGBM demande `brew install libomp` sur macOS (sinon `HistGradientBoostingRegressor`) ;
les réseaux utilisent PyTorch. Les données Kaggle vont dans `data/` (lecture seule).

## Règles du concours

Tout le code est écrit et exécuté dans **Bob**, l'évaluation passe par **skore**. Chaque upload Kaggle doit
mentionner dans sa description l'URL d'un EstimatorReport Skore Hub (`https://skore.probabl.ai/…`), avec une
nouvelle clé de rapport par soumission : `/submit` (`lab/submit.py`) pousse ce rapport, écrit
`submissions/<clé>.csv`, le valide et l'inscrit dans le suivi CV ↔ leaderboard de [lab/LOG.md](lab/LOG.md).
Règles : [competition rules](https://www.kaggle.com/competitions/bobathon-esilv/rules).

## Parcours du bobathon

| Étape | Guide | Contenu |
|---|---|---|
| 1 | [SETUP.md](docs/SETUP.md) | Kaggle, équipe, Python, workspace Skore Hub, installation de skore |
| 2 | [DAY1.md](docs/DAY1.md) | Données, EDA, baseline dummy poussée sur Skore Hub |
| 3 | [DAY2.md](docs/DAY2.md) | Modèles, validation croisée groupée par patient, soumissions Kaggle |

Autres documents : [CONTEXT.md](docs/CONTEXT.md) (contexte scientifique et pièges de modélisation),
[workflow.md](docs/workflow.md).

<p align="center"><sub>Données synthétiques à des fins pédagogiques · Bobathon ESILV</sub></p>

# Workflow agentique du lab

Ce dépôt est le projet **Bob + skore** du hackathon, de l'idée à la soumission Kaggle :

- le **sujet officiel** (`docs/`, `data/`, `tests/`) ;
- le **laboratoire** (`lab/`, `.bob/`, `Bob.md`), où Bob explore, évalue, ensemble et soumet.

Le principe : **une seule chaîne, de bout en bout**. Chaque hypothèse est évaluée avec le même protocole verrouillé ; le meilleur run est poussé tel quel sur Skore Hub (EstimatorReport) et son CSV est uploadé sur Kaggle avec l'URL du rapport.

```
 brainstorm ──► priorisation ──► critique ──► implémentation ──► évaluation ──► ensemble ──► soumission ──► Kaggle
 /brainstorm     IDEAS.md        /review      experimenter        /cv (cv.py)    blend.py     /submit       CSV + URL skore
```

## 1. Les trois piliers

| Pilier | Fichiers | Rôle |
|---|---|---|
| **Contexte permanent** | `Bob.md` (+ `.bob/rules/`) | Problème, métrique, pièges, protocole, soumission. Chargé au début de chaque conversation et de chaque agent. |
| **Mémoire du lab** | `lab/LOG.md`, `lab/IDEAS.md`, `lab/results.csv` | Ce qui a été testé, ce qui reste à tester, les chiffres. Relus à chaque session. |
| **Benchmark verrouillé** | `lab/cv.py`, `lab/folds.csv` | Une seule façon d'évaluer, donc tous les résultats sont comparables. |

## 2. Le benchmark verrouillé

Le holdout Kaggle est **par patient** : une CV au niveau visite fuirait et donnerait des scores trop optimistes. `lab/cv.py` impose donc :

- 5 folds `GroupKFold` sur `patient_id` (mélangés avec `random_state=42`), sauvegardés dans `lab/folds.csv` ;
- une seule interface pour toute expérience :

```python
def fit_predict(train, valid) -> np.ndarray   # train contient target, valid non
```

`valid` contient toutes les visites des patients mis de côté (comme `X_test`), donc les features intra-patient (visite précédente/suivante, moyennes) sont légitimes. Tout ce qui est appris à partir du `target` doit être fitté **dans** `fit_predict`, sur `train` seulement.

Pour chaque run complet, le script produit :

| Sortie | Contenu |
|---|---|
| `lab/runs/<exp>/oof.csv` | prédictions hors-fold, par visite |
| `lab/runs/<exp>/metrics.json` | RMSE global, par fold, par sous-groupe (`off`/`ledd` présent ou non, cohorte, première visite…), par rater |
| `lab/runs/<exp>/test.csv` | prédictions test (option `--test`), format `Index,target` |
| `lab/results.csv` | une ligne par run, pour comparer |
| `lab/LOG.md` | une ligne brouillon à compléter par une interprétation |

**Règle de décision** : le run est comparé au meilleur run existant, fold par fold (comparaison appariée). Si |Δ| est inférieur à l'écart-type de Δ, c'est du bruit (`NOISE`). Un gain net mais inférieur à ~0.02 RMSE est noté mais ne justifie pas de complexifier le modèle soumis.

Le leaderboard public ne sert qu'à vérifier que la CV locale y est corrélée (tableau en tête de `LOG.md`) ; on ne l'optimise jamais.

`cv.py`, `folds.csv` et `data/` ne se modifient pas sans accord explicite : un garde-fou les protège.

## 3. Boucle de travail

1. **Démarrer** : `/status` résume le top des runs, les hypothèses ouvertes et l'écart CV ↔ leaderboard.
2. **Brainstormer** : `/brainstorm` lance en parallèle un agent `analyst` par axe (pharmaco, trajectoire, manquants, rater, structure, méthode). Chacun fait de l'EDA ciblée, regarde où le meilleur modèle se trompe, et renvoie 2 à 4 fiches d'hypothèse au format imposé. Les fiches sont fusionnées dans `lab/IDEAS.md`. `/idea <texte>` ajoute une idée isolée.
3. **Prioriser** : classement par gain attendu / effort dans le tableau de `IDEAS.md`. Utiliser le plan mode avant une grosse expérience.
4. **Critiquer** : `/review HNN` lance l'agent `leak-reviewer` (lecture seule) qui cherche fuites, features indisponibles au test et incohérences, *avant* tout run coûteux.
5. **Implémenter** : un agent `experimenter` par hypothèse crée `lab/experiments/eNNN_<slug>.py`, le valide sur un fold (`--folds 0`), puis lance le run complet. Plusieurs hypothèses en parallèle se font dans des worktrees git séparés (nécessite un dépôt git).
6. **Évaluer** : `/cv eNNN` lance la CV, lit `metrics.json`, complète la ligne de `LOG.md` et met à jour le statut dans `IDEAS.md`.
7. **Combiner** : on garde les 2 ou 3 meilleurs runs peu corrélés et on les mélange avec `lab/blend.py` (poids non négatifs appris sur les OOF, scorés hors-fold).
8. **Soumettre** : `/submit eNNN <clé-skore>` lance `lab/submit.py` : le run est enveloppé dans un estimateur sklearn (skore refitte tout ce qui est appris sur le target), un EstimatorReport est calculé sur le holdout patient = fold 0 du lab (son RMSE doit égaler celui du run), poussé sur Skore Hub, et `submissions/<clé>.csv` est écrit et validé. On uploade ce CSV sur Kaggle avec l'URL du rapport dans la description, puis on reporte le LB public dans `LOG.md`.

## 3 bis. Soumission (`lab/submit.py`)

```bash
.venv/bin/python lab/submit.py e022b_cubic_traj_quad --no-hub            # essai à blanc, rien n'est poussé
.venv/bin/python lab/submit.py b003_seq4 --key b003_seq4                 # EstimatorReport sur Skore Hub + CSV
.venv/bin/python lab/submit.py b003_seq4 --key b003_seq4_v2 --cv         # + CrossValidationReport (5 folds du lab)
```

| Étape | Détail |
|---|---|
| Estimateur | expérience ou composantes d'un blend (poids NNLS de `runs/bNNN/metrics.json`) ; `fit` stocke le train, `predict` appelle `fit_predict` |
| Rapport | EstimatorReport, train = folds 1–4, test = fold 0 (patients disjoints) ; contrôle : RMSE = fold 0 du lab |
| Hub | `project.put(<clé>, report)`, projet `bobathon-esilv`, identifiants `.skore` (`scripts/skore-agent`) |
| CSV | `submissions/<clé>.csv` = prédictions test du run (fit sur tout le train), validé par `check_submission.py` |
| Suivi | ligne ajoutée au tableau « Leaderboard Kaggle » de `LOG.md`, LB public à compléter |

Règle du concours : une nouvelle clé par soumission, et l'URL `https://skore.probabl.ai/…` dans la Submission Description, sinon la ligne est invalide.

## 4. Agents (`.bob/agents/`)

| Agent | Outils | Rôle |
|---|---|---|
| `analyst` | lecture, Bash, écriture dans `lab/eda/` | EDA d'un axe, renvoie des fiches d'hypothèse (n'écrit pas dans `IDEAS.md` pour éviter les conflits entre analystes parallèles). |
| `experimenter` | lecture, Bash, écriture/édition | Implémente et évalue **une** hypothèse via `cv.py` ; ne touche jamais à `cv.py`, `folds.csv`, `data/`. |
| `leak-reviewer` | lecture seule | Relecteur critique : fuites target/groupe, disponibilité au test, bugs d'ordre de lignes. Rend un verdict go / no-go. |

## 5. Commandes (`.bob/skills/`)

| Commande | Action |
|---|---|
| `/status` | État du lab en une dizaine de lignes |
| `/brainstorm [axes]` | Analystes en parallèle → `IDEAS.md` → critique des meilleures fiches |
| `/idea <texte>` | Ajoute une hypothèse formatée au backlog |
| `/review <eNNN\|HNN\|chemin>` | Critique de fuites avant un run ou une soumission |
| `/cv <exp> [--test] [--ref]` | Lance la CV standard et documente le résultat |
| `/submit <exp> [clé-skore] [--cv]` | Rapport skore sur Skore Hub + `submissions/<clé>.csv` |
| `/submit-prep <exp\|csv> [--compare]` | Valide le format Kaggle, compare deux fichiers de prédictions |

## 6. Garde-fous automatiques

Scripts dans `.bob/hooks/`, rendus obligatoires par la règle `.bob/rules/lab.md` :

- **`guard_protected.py`** : avant toute écriture, bloque `data/` et `lab/folds.csv` (code de sortie 2).
- **`check_new_predictions.py`** : après une commande qui écrit des prédictions, valide les `lab/runs/*/test.csv` récents (en-tête `Index,target`, nombre de lignes, index identiques à `sample_submission.csv`, pas de NaN, valeurs dans [0, 132]).
- **`lab/submit.py`** valide lui-même chaque CSV de `submissions/` avant de le déclarer prêt.

## 7. Carte des fichiers

```
Bob.md                     contexte permanent
docs/workflow.md           ce document
lab/
  cv.py                    protocole de CV verrouillé
  folds.csv                patient_id → fold (ne pas modifier)
  features.py              helpers de features partagés
  feat_<slug>.py           helpers propres à une hypothèse
  blend.py                 blends sur OOF (NNLS hors-fold)
  submit.py                rapport skore sur Skore Hub + CSV Kaggle
  check_submission.py      validation du format Kaggle (stdlib)
  experiments/eNNN_*.py    une hypothèse = un fichier
  runs/<exp>/              oof, metrics, test (régénérable, ignoré par git)
  results.csv              tableau comparatif auto-généré
  LOG.md                   mémoire : une ligne par run + suivi CV ↔ leaderboard
  IDEAS.md                 backlog d'hypothèses et priorisation
  eda/                     constats d'EDA par axe
  scratch/                 scripts jetables (ignoré par git)
.bob/
  rules/                   setup gate + contexte du projet (injectés à chaque conversation)
  agents/                  analyst, experimenter, leak-reviewer
  skills/                  commandes slash du projet + skills skore
  hooks/                   garde-fous
submissions/               CSV Kaggle (ignoré par git)
```

## 8. Conventions

- Expériences numérotées `e000`, `e001`… ; blends `b001`… ; hypothèses `H01`… ; une ligne de `LOG.md` par run.
- Seeds : toujours `cv.SEED`.
- Les NaN ne sont pas imputés par défaut : la missingness est un signal (modèles NaN-natifs).
- `Index` et `patient_id` ne sont jamais des features.
- Notes en français, code et commentaires en anglais.

## 9. Prise en main rapide

```bash
# environnement (une fois)
python3 -m venv .venv && .venv/bin/python -m pip install -r lab/requirements.txt

# évaluer une expérience
.venv/bin/python lab/cv.py lab/experiments/e002_hgb_basic_context.py --test

# puis, dans Bob
/status
/brainstorm
/submit e002_hgb_basic_context 01_hgb_context
```

Points de départ : `e000_dummy` (plancher, CV RMSE 16.51), `e001_hgb_raw` (7.58), `e002_hgb_basic_context` (4.58). LightGBM nécessite `brew install libomp` sur macOS ; sans lui, `HistGradientBoostingRegressor` suffit.

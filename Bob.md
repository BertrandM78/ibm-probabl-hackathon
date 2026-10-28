# Bobathon ESILV — projet Bob

Ce dépôt est le projet **Bob + skore** du bobathon, de l'exploration à la soumission Kaggle :
brainstorming → benchmark → test → évaluation → rapport skore → soumission.

- Les soumissions Kaggle partent **d'ici** : `lab/submit.py` pousse un EstimatorReport sur Skore Hub
  et écrit `submissions/<clé>.csv` (règle du concours : code écrit et exécuté dans Bob, évaluation skore).
- La règle « setup gate » (`.bob/rules/AGENTS.md`) s'applique : les checks de `tests/test_setup.py`
  passent avant tout le reste.
- `docs/` = sujet officiel (CONTEXT.md pour le fond scientifique). `data/` = données, en lecture seule.

## Problème

Prédire, pour chaque visite, le **true OFF** (score moteur MDS-UPDRS débiaisé, `y_train.target`).
Métrique : **RMSE au niveau visite**. Holdout Kaggle **par patient** (aucun patient commun train/test).

Données **synthétiques**. Processus générateur annoncé :
- effet de la lévodopa ∝ concentration sanguine : absorption rapide puis décroissance vers 0 ;
  courbe pharmacodynamique **constante avec l'âge** ;
- biais du OFF mesuré : subjectivité du rater, valeurs manquantes/erronées, délai depuis la prise,
  OFF rarement mesuré. ON améliore typiquement de 50–100 % par rapport à OFF.

| Fait | Valeur |
|------|--------|
| Train | 44 590 visites, 5 576 patients (4 à 12 visites, médiane 7) |
| Test | 11 013 visites, 1 395 patients |
| Colonnes | `Index, patient_id, cohort (A/B), sexM, gene, age_at_diagnosis, age, ledd, time_since_intake_on, time_since_intake_off, rater_id, on, off` |
| Manquants (train) | `off` 42 %, `on` 30 %, `ledd` 37 %, `gene` 32 %, `tsi_on` 46 %, `tsi_off` 79 %, `age_at_diagnosis` 5 % — toujours au moins un de on/off |
| Target | moyenne 37.5, std 16.5, min 0, max 109.5 |
| `off` brut vs target | corr 0.87, RMSE 10.2 → biais structuré |
| `time_since_intake_on` | 0–6.3 h (méd. 1.6) · `time_since_intake_off` : 6.9–24.7 h (méd. 13.8) |
| Raters | 50, les mêmes en train et test |
| Ordre | fichiers triés par patient puis âge croissant |

`time_since_diagnosis` n'existe **pas** dans les fichiers : le dériver (`age - age_at_diagnosis`).

## Protocole d'évaluation (verrouillé)

- Tout modèle passe par **`lab/cv.py`** : 5 folds `GroupKFold(shuffle=True, random_state=42)` sur
  `patient_id`, sauvegardés dans `lab/folds.csv`. Ne jamais modifier `cv.py` (constantes, métrique),
  `lab/folds.csv` ni `data/` sans accord explicite de l'utilisateur (`.bob/hooks/guard_protected.py`).
- Une expérience = `lab/experiments/eNNN_<slug>.py` qui définit
  `fit_predict(train, valid) -> np.ndarray` + `DESCRIPTION` + `FEATURES`.
  `train` contient `target` ; `valid` = toutes les visites des patients held-out, **sans** `target`.
- Sorties : `lab/runs/<exp>/{oof.csv, metrics.json, test.csv}`, une ligne dans `lab/results.csv`
  et une ligne brouillon dans `lab/LOG.md` (colonne Notes à compléter).
- **Règle de décision** : comparaison appariée fold par fold contre la référence. Un gain dont
  |Δ| < std(Δ) est du bruit. Un gain net mais < ~0.02 RMSE est négligeable : le noter, ne pas
  complexifier le modèle soumis pour lui.
- Le leaderboard public sert seulement à vérifier que CV ↔ LB corrèlent (tableau en haut de LOG.md).
  Ne jamais optimiser sur le LB.

```bash
.venv/bin/python lab/cv.py lab/experiments/eNNN_x.py --folds 0     # rapide, non loggé
.venv/bin/python lab/cv.py lab/experiments/eNNN_x.py --test        # run complet + prédictions test
.venv/bin/python lab/cv.py lab/experiments/eNNN_x.py --ref e002_hgb_basic_context
.venv/bin/python lab/blend.py eAAA eBBB --name b001_x               # blend NNLS hors-fold
.venv/bin/python lab/submit.py eNNN_x --key <clé>                   # skore Hub + submissions/<clé>.csv
python3 lab/check_submission.py submissions/<clé>.csv [--compare autre.csv]
```

## Soumission Kaggle

`lab/submit.py <eNNN | bNNN> [--key <clé>] [--cv] [--no-hub]` :
1. enveloppe le run (expérience, ou composantes d'un blend avec leurs poids NNLS) dans un estimateur
   sklearn dont `fit` stocke le train et `predict` appelle `fit_predict` → skore refitte tout ce qui
   est appris sur `target` ;
2. EstimatorReport sur le holdout patient = fold 0 du lab ; son RMSE doit égaler `per_fold[0]` de
   `metrics.json` (sinon le code a changé depuis le run) ; `--cv` ajoute un CrossValidationReport
   sur les 5 folds du lab (clé `<clé>_cv`) ;
3. `project.put(<clé>, report)` sur Skore Hub (projet `bobathon-esilv`, identifiants `.skore`) ;
4. `submissions/<clé>.csv` = `lab/runs/<run>/test.csv` (fit sur tout le train), validé par
   `check_submission.py` ;
5. ajoute une ligne au tableau « Leaderboard Kaggle » de `lab/LOG.md`.

Une **nouvelle clé par soumission**. L'URL `https://skore.probabl.ai/…` affichée par `put` va dans la
**Submission Description** Kaggle, sinon la soumission est invalide. Upload : `submissions/<clé>.csv`.
Un run doit avoir été fait avec `--test` (ou par `blend.py`) avant d'être soumis.

## Pièges connus

- **Fuite par le target** : tout ce qui est appris sur `target` (courbe PK, biais rater, target
  encoding, stacking, scaler) se fitte **dans** `fit_predict`, sur `train` uniquement.
- **Fuite par groupe** : jamais de split ni de statistique au niveau visite.
- Les features **intra-patient** sans target (lags, leads, moyennes patient de on/off/ledd,
  interpolation) sont **légitimes** : le test fournit toutes les visites de chaque patient.
- `Index` et `patient_id` ne sont pas des features (`features.to_model_matrix` les retire).
- `rater_id`, `gene`, `cohort` sont des chaînes : passer en `category` et aligner les niveaux
  (`features.align_categories`).
- Ne pas imputer les NaN par défaut : la missingness est un signal (modèles NaN-natifs).
- Seeds : utiliser `cv.SEED` partout.
- LightGBM nécessite `brew install libomp` sur cette machine (sinon HistGradientBoosting).

## Organisation

```
Bob.md                    ce fichier
lab/cv.py                 protocole verrouillé            lab/folds.csv     folds patient → fold
lab/features.py           helpers partagés                lab/feat_*.py     helpers par hypothèse
lab/experiments/eNNN_*.py une hypothèse = un fichier      lab/blend.py      blends sur OOF
lab/submit.py             rapport skore + CSV Kaggle      submissions/      CSV à uploader (ignoré par git)
lab/runs/<exp>/           oof, metrics, test (régénérable) lab/results.csv  tableau auto
lab/LOG.md                mémoire : 1 ligne / run + LB    lab/IDEAS.md      backlog d'hypothèses
lab/eda/<axe>.md          constats d'EDA                  lab/scratch/      scripts jetables
.bob/rules/               règles injectées à chaque conversation
.bob/agents/              analyst, experimenter, leak-reviewer
.bob/skills/              /cv /idea /brainstorm /review /status /submit /submit-prep (+ skills skore)
.bob/hooks/               garde-fous (fichiers protégés, format des prédictions)
```

Environnement : `.venv/bin/python` (dépendances dans `lab/requirements.txt` + `requirements.txt`).

## Boucle de travail

1. **Début de session** : lire `lab/LOG.md` et `lab/IDEAS.md` (ou `/status`).
2. **Idées** : `/brainstorm` (un agent `analyst` par axe, en parallèle) ou `/idea <texte>`.
3. **Priorisation** : gain attendu / effort, dans le tableau d'IDEAS.md. Planifier avant une
   expérience lourde.
4. **Critique** : `/review HNN` avant tout run coûteux.
5. **Implémentation** : un agent `experimenter` par hypothèse (en worktree git si plusieurs en
   parallèle), évaluation via `/cv`.
6. **Ensemble** : garder les 2–3 meilleurs runs décorrélés → `lab/blend.py`.
7. **Soumission** : `/review` puis `/submit eNNN <clé>` → rapport sur Skore Hub + `submissions/<clé>.csv`,
   upload Kaggle avec l'URL du rapport, LB public reporté dans `lab/LOG.md`.

## État actuel

Meilleur run : voir `lab/results.csv` (au départ `e002_hgb_basic_context`, CV RMSE 4.58).
Leviers attendus, par ordre : (a) modèle explicite de l'effet médicament selon le délai depuis la
prise, (b) trajectoire par patient, (c) gestion fine des NaN et des raters.

## Style

Réponses et notes en français. Code et commentaires en anglais, concis, dans le style de `lab/cv.py`.

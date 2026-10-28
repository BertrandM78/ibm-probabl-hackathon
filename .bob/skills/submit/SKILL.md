---
name: submit
description: Turn the best lab run (experiment or blend) into an official Kaggle submission - skore EstimatorReport pushed to Skore Hub plus submissions/<key>.csv. Use when the user types /submit or wants to submit a model on Kaggle.
argument-hint: <eNNN | bNNN> [clé-skore] [--cv]
---

Soumission Kaggle d'un run du lab, via `lab/submit.py` (procédure détaillée dans `Bob.md`).

1. Résous le run (`e022b` → `e022b_cubic_traj_quad`, nom complet si plusieurs runs partagent le numéro).
   Clé = 2e argument, sinon le nom du run. Une **nouvelle clé par soumission** : si
   `submissions/<clé>.csv` existe ou si la clé figure déjà dans le tableau « Leaderboard Kaggle » de
   `lab/LOG.md`, demande une autre clé (ex. suffixe `_v2`).
2. Prérequis : `lab/runs/<run>/metrics.json` et `test.csv` existent (sinon `/cv <run> --test`, ou
   `lab/blend.py` pour un blend, après accord : un run peut être long). `.skore` présent (sinon
   `scripts/skore-agent`, cf. `docs/SETUP.md`). Lance `/review <run>` si ce n'est pas déjà fait.
3. Teste sans rien pousser si le run n'a jamais été soumis :
   `.venv/bin/python lab/submit.py <run> --key <clé> --no-hub` (supprime ensuite le `*_dryrun.csv`).
   Le RMSE de l'EstimatorReport doit égaler le fold 0 du lab ; sinon arrête-toi et cherche ce qui a
   changé dans le code depuis le run.
4. Run réel : `.venv/bin/python lab/submit.py <run> --key <clé>` (`--cv` si l'utilisateur veut aussi le
   CrossValidationReport : 5 fits de plus). En arrière-plan s'il dure > 2 min (blends à réseaux : ~15 min
   par composante et par fit).
5. Restitue en ≤ 6 lignes : URL Skore Hub (à coller dans la Submission Description Kaggle), fichier
   `submissions/<clé>.csv`, RMSE holdout vs fold 0 lab, CV lab. Rappelle de reporter le LB public dans
   `lab/LOG.md` une fois l'upload fait.

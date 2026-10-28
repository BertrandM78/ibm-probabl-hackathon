---
name: submit-prep
description: Validate a prediction CSV against the Kaggle format (Index,target, row count, no NaN, range) and optionally compare it to a lab run. Use when the user types /submit-prep or before uploading a submission.
argument-hint: <eNNN | chemin.csv> [--compare <autre.csv | eNNN>]
---

1. Résous `$ARGUMENTS` : `eNNN` → `lab/runs/<exp>/test.csv` (le générer avec `cv.py --test` s'il manque).
   `--compare eNNN` → `lab/runs/<exp>/test.csv`.
2. Lance `python3 lab/check_submission.py <fichier> [--compare <autre>]`.
3. Si on compare `submissions/<clé>.csv` au run dont il est issu : un RMSE entre fichiers > 1e-6 signifie
   que le CSV ne vient pas de ce run (mauvais run, run relancé depuis) ; refaire `/submit` avec une nouvelle clé.
4. Rappelle que l'upload Kaggle se fait avec `submissions/<clé>.csv` et l'URL Skore Hub du rapport
   dans la Submission Description.

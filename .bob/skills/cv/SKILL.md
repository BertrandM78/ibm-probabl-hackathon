---
name: cv
description: Run the locked patient-grouped CV on a lab experiment and log the result. Use when the user types /cv or asks to evaluate an experiment.
argument-hint: <eNNN_name | path> [--test] [--ref eXXX] [--folds 0]
---

Évalue une expérience avec le protocole verrouillé.

1. Résous `$ARGUMENTS` vers `lab/experiments/<exp>.py` (accepte `e005` ou le chemin complet).
   Sans argument : la dernière expérience modifiée dans `lab/experiments/`.
2. Lance `.venv/bin/python lab/cv.py lab/experiments/<exp>.py <options>` (ajoute `--test` par défaut
   pour un run complet, sauf si `--folds` est demandé). Lance-le en arrière-plan s'il dure > 2 min.
3. Lis `lab/runs/<exp>/metrics.json` : compare les sous-groupes à la référence.
4. Remplace `_à compléter_` dans la ligne de `lab/LOG.md` par une phrase d'interprétation.
   Si le gain est réel mais < 0.02 de RMSE, écris-le (gain statistiquement net mais négligeable).
5. Mets à jour le statut de l'hypothèse dans `lab/IDEAS.md` (`FAIT (eNNN)`).
6. Réponds en 4 lignes max : RMSE, Δ ± std vs ref, verdict, sous-groupe le plus impacté.
